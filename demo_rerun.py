"""Visualise one GLidE-SLAM run in Rerun.

Reads the artefacts a ``mono_tum`` run leaves in its working directory
(``FrameTrajectory.txt``, ``KeyFrameTrajectory.txt``, ``frame_types.txt``,
``gpuTimings.csv``) plus the TUM sequence itself, aligns the monocular estimate
onto the ground truth with a Sim(3) Umeyama fit, and logs the whole thing to a
Rerun recording.

Two upstream quirks shape the loading code:

* ``FrameTrajectory.txt`` carries a stale timestamp on every direct frame,
  because ``Tracking::Track`` stores ``mCurrentFrame.mTimeStamp`` and
  ``mCurrentFrame`` is only rebuilt on indirect frames.  Row ``k`` of the file
  is always image ``first + k`` of the sequence, so the true timestamp and the
  matching RGB file are recovered by index rather than by timestamp.
* ``frame_types.txt`` prints its timestamp column with the default ostream
  precision, which on TUM stamps collapses to ``1.34185e+09`` for the whole
  sequence.  Its rows are therefore joined to the trajectory by order, and the
  join is checked against the direct/indirect pattern that the stale timestamps
  themselves encode.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro

from bench import load_sequence_timestamps, restamp_trajectory

# Direct-tracking intrinsics from Examples/Monocular/GLideConfig.yaml, which match
# Examples/Monocular/TUM3.yaml (TUM RGB-D freiburg3).
FX, FY, CX, CY = 535.4, 539.2, 320.1, 247.6
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

DIRECT_COLOR = (255, 0, 255)
INDIRECT_COLOR = (0, 255, 255)
GT_COLOR = (128, 128, 128)
EST_COLOR = (255, 160, 0)

MAX_ASSOCIATION_DT_S = 0.020


@dataclass
class RerunConfig:
    """How the recording is delivered."""

    application_id: str = "glide-slam"
    """Rerun application id."""
    spawn: bool = False
    """Spawn a native viewer and stream to it."""
    headless: bool = False
    """Do not spawn or connect to any viewer; use together with --save."""
    save: Path | None = None
    """Write the recording to this .rrd file."""
    connect: bool = False
    """Connect to an already running viewer over gRPC."""


@dataclass
class Config:
    """Inputs for the demo."""

    run_dir: Path = Path("out/runs/glide-1")
    """Directory a mono_tum run was executed in."""
    data_dir: Path = Path("data/rgbd_dataset_freiburg3_long_office_household")
    """TUM RGB-D sequence directory."""
    image_stride: int = 1
    """Log every Nth RGB frame (poses and metrics are always logged)."""
    jpeg_quality: int = 80
    """JPEG quality used when re-encoding the TUM PNGs."""
    rr_config: RerunConfig = field(default_factory=RerunConfig)


@dataclass
class Trajectory:
    """A TUM-format trajectory: timestamps, positions and xyzw quaternions."""

    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass
class FrameTypeRow:
    """One row of ``frame_types.txt``; the timestamp column is unusable."""

    frame_id: int
    is_direct: bool
    chi2: float


@dataclass
class ImageAssociation:
    """Maps every trajectory row back onto the image that produced it."""

    first_image: int
    image_indices: np.ndarray
    timestamps: np.ndarray
    is_fresh: np.ndarray
    """True where the file's own timestamp was refreshed, i.e. an indirect frame."""


@dataclass
class Alignment:
    """Sim(3) transform taking estimate coordinates into ground-truth ones."""

    rotation: np.ndarray
    translation: np.ndarray
    scale: float

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_tum_trajectory(path: Path, *, sort: bool = True) -> Trajectory:
    """Load a ``t tx ty tz qx qy qz qw`` file.

    ``sort`` must stay off for ``FrameTrajectory.txt``: its rows are in image
    order but its timestamps repeat, so sorting would shuffle direct poses.
    """
    rows: list[list[float]] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(x) for x in parts[:8]])
    if not rows:
        return Trajectory(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4)))
    data = np.asarray(rows, dtype=np.float64)
    if sort:
        data = data[np.argsort(data[:, 0])]
    return Trajectory(data[:, 0], data[:, 1:4], data[:, 4:8])


def load_frame_types(path: Path) -> list[FrameTypeRow]:
    """Read ``frame_types.txt`` in file order."""
    rows: list[FrameTypeRow] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) < 4:
            continue
        try:
            rows.append(
                FrameTypeRow(frame_id=int(fields[0]), is_direct=fields[1].strip() == "direct", chi2=float(fields[2]))
            )
        except ValueError:
            continue
    return rows


def load_gpu_timings(path: Path) -> dict[int, dict[str, float]]:
    """Map frame id -> {stage: ms} from ``gpuTimings.csv``.

    ``track`` rows carry two undeclared extra columns, so only the first three
    fields are read.
    """
    out: dict[int, dict[str, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) < 3:
            continue
        try:
            out.setdefault(int(fields[1]), {})[fields[0].strip()] = float(fields[2])
        except ValueError:
            continue
    return out


def load_rgb_index(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load ``rgb.txt`` into aligned timestamp and relative-path arrays."""
    names: list[str] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    names.append(parts[1])
    return np.asarray(load_sequence_timestamps(path) if path.is_file() else [], dtype=np.float64), names


def associate_to_images(raw_timestamps: np.ndarray, rgb_stamps: np.ndarray) -> ImageAssociation:
    """Recover the true image index and timestamp of every trajectory row.

    Row 0 is the initialising frame, which is always indirect, so its stored
    timestamp is genuine and locates the whole run in the sequence.  Every later
    row advances by exactly one image.  The reconstruction is verified on the
    rows whose stored timestamp did change: there it must reproduce the file.
    """
    index_of = {round(float(s), 6): i for i, s in enumerate(rgb_stamps)}
    first = index_of.get(round(float(raw_timestamps[0]), 6))
    if first is None:
        raise SystemExit("the first trajectory timestamp is not in rgb.txt; cannot locate the run")
    n = raw_timestamps.shape[0]
    if first + n > rgb_stamps.shape[0]:
        raise SystemExit(f"trajectory runs past the end of the sequence ({first} + {n} > {rgb_stamps.shape[0]})")

    image_indices = first + np.arange(n)
    timestamps = rgb_stamps[image_indices]
    is_fresh = np.ones(n, dtype=bool)
    is_fresh[1:] = raw_timestamps[1:] != raw_timestamps[:-1]

    bad = int(np.count_nonzero(np.abs(raw_timestamps[is_fresh] - timestamps[is_fresh]) > 1e-6))
    if bad:
        raise SystemExit(f"timestamp reconstruction failed on {bad} of {int(is_fresh.sum())} refreshed rows")
    return ImageAssociation(first_image=first, image_indices=image_indices, timestamps=timestamps, is_fresh=is_fresh)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert one xyzw quaternion into a 3x3 rotation matrix."""
    x, y, z, w = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each entry of ``query`` return the index of the closest ``reference``."""
    if reference.size < 2 or query.size == 0:
        return np.zeros(query.shape[0], dtype=np.int64)
    idx = np.searchsorted(reference, query)
    idx = np.clip(idx, 1, reference.size - 1)
    left, right = reference[idx - 1], reference[idx]
    return np.where(np.abs(query - left) <= np.abs(query - right), idx - 1, idx)


def umeyama_sim3(source: np.ndarray, target: np.ndarray) -> Alignment:
    """Least-squares Sim(3) fit mapping ``source`` onto ``target`` (Umeyama 1991)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    src_c, tgt_c = source - mu_s, target - mu_t
    cov = (tgt_c.T @ src_c) / source.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    rotation = u @ s @ vt
    var_src = float((src_c**2).sum() / source.shape[0])
    scale = float(np.trace(np.diag(d) @ s) / var_src) if var_src > 0 else 1.0
    translation = mu_t - scale * (rotation @ mu_s)
    return Alignment(rotation=rotation, translation=translation, scale=scale)


def evo_ape_rmse(gt: Path, est: Path) -> float | None:
    """Run ``evo_ape tum <gt> <est> -as`` and return its RMSE."""
    if not est.is_file():
        return None
    try:
        proc = subprocess.run(
            ["evo_ape", "tum", str(gt.resolve()), str(est.resolve()), "-as"], capture_output=True, text=True
        )
    except FileNotFoundError:
        # evo lives in the pixi environment; `pixi run demo` puts it on PATH.
        print("evo_ape not found on PATH; skipping the independent ATE check")
        return None
    if proc.returncode != 0:
        return None
    if m := re.search(r"^\s*rmse\s+([0-9.eE+-]+)", proc.stdout, re.MULTILINE):
        return float(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def make_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="Map and trajectories", contents=["/world/**"]),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/world/camera/image", name="Camera"),
                rrb.TimeSeriesView(
                    origin="/metrics",
                    name="Tracking",
                    contents=[
                        "/metrics/chi2",
                        "/metrics/is_direct",
                        "/metrics/ate_error_m",
                        "/metrics/direct_utilization_pct",
                    ],
                ),
                rrb.TimeSeriesView(origin="/metrics/gpu", name="GPU stage time (ms)"),
                row_shares=[3, 2, 2],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=False,
    )


def start_recording(cfg: RerunConfig, blueprint: rrb.Blueprint) -> None:
    rr.init(cfg.application_id, spawn=cfg.spawn and not cfg.headless)
    if cfg.save is not None:
        cfg.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(cfg.save, default_blueprint=blueprint)
    elif cfg.connect and not cfg.headless:
        rr.connect_grpc(default_blueprint=blueprint)
    elif cfg.spawn and not cfg.headless:
        rr.send_blueprint(blueprint)


def log_static_scene(
    ground_truth: Trajectory,
    aligned_positions: np.ndarray,
    is_direct: np.ndarray,
    keyframes: Trajectory,
    aligned_kf: np.ndarray,
    alignment: Alignment,
) -> None:
    """Log everything that does not change over time."""
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "/world/ground_truth",
        rr.LineStrips3D([ground_truth.positions], colors=[GT_COLOR], radii=0.004),
        static=True,
    )
    rr.log("/world/estimate", rr.LineStrips3D([aligned_positions], colors=[EST_COLOR], radii=0.004), static=True)

    colors = np.where(is_direct[:, None], np.array(DIRECT_COLOR), np.array(INDIRECT_COLOR)).astype(np.uint8)
    rr.log("/world/estimate/frames", rr.Points3D(aligned_positions, colors=colors, radii=0.008), static=True)

    for i in range(len(keyframes)):
        rotation = alignment.rotation @ quaternion_xyzw_to_matrix(keyframes.quaternions_xyzw[i])
        rr.log(f"/world/keyframes/{i:04d}", rr.Transform3D(translation=aligned_kf[i], mat3x3=rotation), static=True)
        rr.log(
            f"/world/keyframes/{i:04d}",
            rr.Pinhole(
                focal_length=[FX, FY],
                principal_point=[CX, CY],
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                image_plane_distance=0.03,
            ),
            static=True,
        )


def main(cfg: Config) -> None:
    run_dir = cfg.run_dir.resolve()
    data_dir = cfg.data_dir.resolve()

    raw_estimate = load_tum_trajectory(run_dir / "FrameTrajectory.txt", sort=False)
    keyframes = load_tum_trajectory(run_dir / "KeyFrameTrajectory.txt")
    ground_truth = load_tum_trajectory(data_dir / "groundtruth.txt")
    frame_types = load_frame_types(run_dir / "frame_types.txt")
    gpu_timings = load_gpu_timings(run_dir / "gpuTimings.csv")
    rgb_stamps, rgb_names = load_rgb_index(data_dir / "rgb.txt")

    if len(raw_estimate) == 0:
        raise SystemExit(f"no estimated poses in {run_dir / 'FrameTrajectory.txt'}")
    if len(ground_truth) == 0:
        raise SystemExit(f"no ground truth in {data_dir / 'groundtruth.txt'}")
    if rgb_stamps.size == 0:
        raise SystemExit(f"no images listed in {data_dir / 'rgb.txt'}")

    association = associate_to_images(raw_estimate.timestamps, rgb_stamps)
    estimate = Trajectory(association.timestamps, raw_estimate.positions, raw_estimate.quaternions_xyzw)
    n = len(estimate)

    # The stale-timestamp pattern already says which rows were direct; frame_types
    # supplies chi2 and the GPU frame id.  Row j of frame_types is trajectory row
    # j + 1, because row 0 is the initialising frame and is never typed.
    is_direct = ~association.is_fresh
    chi2 = np.full(n, np.nan)
    frame_ids = np.full(n, -1, dtype=np.int64)
    type_agreement = "no frame_types.txt"
    if frame_types:
        m = min(len(frame_types), n - 1)
        agree = sum(1 for j in range(m) if frame_types[j].is_direct == bool(is_direct[j + 1]))
        type_agreement = f"{agree}/{m} rows ({100.0 * agree / m:.1f} %)" if m else "empty"
        for j in range(m):
            chi2[j + 1] = frame_types[j].chi2
            frame_ids[j + 1] = frame_types[j].frame_id

    # Associate estimate -> ground truth by nearest timestamp.
    gt_idx = nearest_indices(estimate.timestamps, ground_truth.timestamps)
    dt = np.abs(estimate.timestamps - ground_truth.timestamps[gt_idx])
    associated = dt <= MAX_ASSOCIATION_DT_S
    if associated.sum() < 3:
        raise SystemExit(f"only {associated.sum()} poses associate within {MAX_ASSOCIATION_DT_S * 1e3:.0f} ms")

    alignment = umeyama_sim3(estimate.positions[associated], ground_truth.positions[gt_idx[associated]])
    aligned_positions = alignment.apply_points(estimate.positions)
    per_pose_error = np.full(n, np.nan)
    per_pose_error[associated] = np.linalg.norm(
        aligned_positions[associated] - ground_truth.positions[gt_idx[associated]], axis=1
    )
    ours_rmse = float(np.sqrt(np.nanmean(per_pose_error[associated] ** 2)))

    # evo must see the repaired timestamps, otherwise it associates the direct
    # poses against the wrong ground-truth samples.  bench.py writes the repaired
    # file next to the raw one, so it is only rebuilt here when it is missing.
    restamped = run_dir / "FrameTrajectory_restamped.txt"
    if restamped.is_file():
        restamp_source = f"{restamped.name} (written by bench.py)"
    else:
        report = restamp_trajectory(run_dir / "FrameTrajectory.txt", data_dir / "rgb.txt", restamped)
        restamp_source = f"{restamped.name} (rebuilt here)" if report["restamp_ok"] else "unavailable"
    evo_rmse = evo_ape_rmse(data_dir / "groundtruth.txt", restamped)
    evo_rmse_raw = evo_ape_rmse(data_dir / "groundtruth.txt", run_dir / "FrameTrajectory.txt")

    aligned_kf = alignment.apply_points(keyframes.positions) if len(keyframes) else np.zeros((0, 3))

    start_recording(cfg.rr_config, make_blueprint())
    log_static_scene(ground_truth, aligned_positions, is_direct, keyframes, aligned_kf, alignment)

    direct_seen = 0
    images_logged = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]

    for i in range(n):
        rr.set_time("frame", sequence=int(association.image_indices[i]))
        rr.set_time("timestamp", timestamp=float(estimate.timestamps[i]))

        direct_seen += int(is_direct[i])
        kind = "direct" if is_direct[i] else "indirect"

        rotation = alignment.rotation @ quaternion_xyzw_to_matrix(estimate.quaternions_xyzw[i])
        rr.log("/world/camera", rr.Transform3D(translation=aligned_positions[i], mat3x3=rotation))
        rr.log(
            "/world/camera",
            rr.Pinhole(
                focal_length=[FX, FY],
                principal_point=[CX, CY],
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                image_plane_distance=0.15,
            ),
        )

        if i % cfg.image_stride == 0:
            image_path = data_dir / rgb_names[int(association.image_indices[i])]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                ok, buffer = cv2.imencode(".jpg", image, encode_params)
                if ok:
                    rr.log("/world/camera/image", rr.EncodedImage(contents=buffer.tobytes(), media_type="image/jpeg"))
                    images_logged += 1

        if np.isfinite(chi2[i]):
            rr.log("/metrics/chi2", rr.Scalars(float(chi2[i])))
        rr.log("/metrics/is_direct", rr.Scalars(float(is_direct[i])))
        rr.log("/metrics/direct_utilization_pct", rr.Scalars(100.0 * direct_seen / (i + 1)))
        if np.isfinite(per_pose_error[i]):
            rr.log("/metrics/ate_error_m", rr.Scalars(float(per_pose_error[i])))
        for stage, value in gpu_timings.get(int(frame_ids[i]), {}).items():
            rr.log(f"/metrics/gpu/{stage}", rr.Scalars(value))
        rr.log("/frame_type", rr.TextLog(f"image {int(association.image_indices[i])}: {kind}"))

    direct_total = int(is_direct.sum())
    median_tracking_ms = None
    log_path = run_dir / "run.log"
    if log_path.is_file():
        if m := re.search(r"median tracking time:\s*([0-9.eE+-]+)", log_path.read_text(errors="replace")):
            median_tracking_ms = float(m.group(1)) * 1000.0

    print("")
    print("=== GLidE-SLAM run summary ===")
    print(f"run dir                  : {run_dir}")
    print(f"images in sequence       : {rgb_stamps.size}")
    print(f"estimated poses          : {n}  (images {association.first_image}..{int(association.image_indices[-1])})")
    print(f"sequence tracked         : {100.0 * n / rgb_stamps.size:.1f} %")
    print(f"associated to GT (<=20ms): {int(associated.sum())}")
    print(f"keyframes                : {len(keyframes)}")
    print(f"direct frames            : {direct_total} of {n} = {100.0 * direct_total / n:.1f} %")
    print(f"frame_types agreement    : {type_agreement}")
    print(f"restamped trajectory     : {restamp_source}")
    print(f"Sim(3) scale             : {alignment.scale:.6f}")
    print(f"ATE RMSE (ours)          : {ours_rmse:.6f} m")
    print(f"ATE RMSE (evo_ape -as)   : {'n/a' if evo_rmse is None else f'{evo_rmse:.6f} m'}")
    if evo_rmse is not None:
        print(f"ATE RMSE difference      : {abs(evo_rmse - ours_rmse):.2e} m")
    print(f"ATE RMSE (raw timestamps): {'n/a' if evo_rmse_raw is None else f'{evo_rmse_raw:.6f} m'}  <- wrong, for reference only")
    print(f"median tracking time     : {'n/a' if median_tracking_ms is None else f'{median_tracking_ms:.1f} ms'}")
    print(f"RGB images logged        : {images_logged} (stride {cfg.image_stride}, jpeg q{cfg.jpeg_quality})")
    if cfg.rr_config.save is not None:
        print(f"recording                : {cfg.rr_config.save}")


if __name__ == "__main__":
    main(tyro.cli(Config))
