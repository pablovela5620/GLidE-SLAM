"""Benchmark GLidE-SLAM against an ORB-SLAM2-style indirect-only baseline.

Every run happens in its own directory under ``out/runs/<mode>-<n>/`` because the
``mono_tum`` binary resolves ``Examples/Monocular/GLideConfig.yaml``, ``shaders/``
and all of its output files relative to the process working directory.

The two modes differ only in ``DirTrackParams.maxFramesDirect``:

* ``glide`` keeps the committed value of 10, so the GPU direct tracker runs.
* ``orb`` sets it to 0, which makes ``Tracking::SwitchToIndirect`` report a wide
  baseline on every frame and therefore falls back to indirect ORB tracking.

No upstream source file is modified; only the per-run copy of the config changes.

One post-processing step is unavoidable.  ``Tracking::Track`` stores
``mCurrentFrame.mTimeStamp`` for every pose it appends to the frame trajectory,
but ``mCurrentFrame`` is only rebuilt on indirect frames (``src/Tracking.cc``
~line 279), so each direct frame inherits the previous indirect frame's
timestamp.  ``FrameTrajectory.txt`` therefore carries stale timestamps for every
direct pose, which makes a timestamp-based ATE meaningless.  ``restamp_trajectory``
rewrites the file with the true per-image timestamps and verifies the
reconstruction against the rows that did get a fresh timestamp.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SEQUENCE_DIR = REPO_ROOT / "data" / "rgbd_dataset_freiburg3_long_office_household"
VOCAB = REPO_ROOT / "Vocabulary" / "ORBvoc.txt"
SETTINGS = REPO_ROOT / "Examples" / "Monocular" / "TUM3.yaml"
GLIDE_CONFIG = REPO_ROOT / "Examples" / "Monocular" / "GLideConfig.yaml"
MONO_TUM = REPO_ROOT / "Examples" / "Monocular" / "mono_tum"
SHADERS = REPO_ROOT / "shaders"
OUT_DIR = REPO_ROOT / "out"
RUNS_DIR = OUT_DIR / "runs"

# A run that tracked more than this share of the sequence is treated as complete.
# Runs below it stop producing poses partway through and their reported medians are
# not comparable, because ``mono_tum`` averages over the whole sequence regardless.
COVERAGE_FULL_PCT = 95.0

# Paper (Table II) reference numbers for the tum3 sequence on a Radxa Zero 3W.
PAPER: dict[str, dict[str, float]] = {
    "radxa": {
        "orb_median_ms": 131.0,
        "glide_median_ms": 58.8,
        "speedup": 2.2,
        "orb_ate_m": 0.009,
        "glide_ate_m": 0.016,
        "direct_utilization_pct": 69.0,
        "gpu_pre_ms": 43.7,
        "gpu_trk_ms": 39.1,
    },
    "laptop": {
        "gpu_pre_ms": 2.5,
        "gpu_trk_ms": 3.5,
    },
}


@dataclass
class GpuStage:
    """Median/mean wall time of one GPU stage, in milliseconds."""

    count: int
    median_ms: float
    mean_ms: float


@dataclass
class RunResult:
    """Everything measured for a single ``mono_tum`` invocation."""

    mode: str
    index: int
    run_dir: str
    exit_code: int
    wall_time_s: float
    max_frames_direct: int
    log_timing: int = 1
    gl_renderer: str | None = None
    gl_version: str | None = None
    median_tracking_ms: float | None = None
    mean_tracking_ms: float | None = None
    frames_logged: int = 0
    frames_direct: int = 0
    frames_indirect: int = 0
    direct_utilization_pct: float | None = None
    gpu_track_fail_count: int = 0
    images_in_sequence: int = 0
    frame_trajectory_poses: int = 0
    keyframe_trajectory_poses: int = 0
    first_tracked_image: int | None = None
    last_tracked_image: int | None = None
    tracking_coverage_pct: float | None = None
    restamp_ok: bool = False
    restamp_fresh_rows: int = 0
    restamp_mismatches: int = 0
    ate_rmse_frame_m: float | None = None
    ate_rmse_frame_restamped_m: float | None = None
    ate_rmse_keyframe_m: float | None = None
    gpu_stages: dict[str, GpuStage] = field(default_factory=dict)
    gpu_pre_ms: float | None = None
    gpu_trk_ms: float | None = None


# --------------------------------------------------------------------------- #
# config handling
# --------------------------------------------------------------------------- #


def _rewrite_config(text: str, *, render: int, log_timing: int, max_frames_direct: int) -> str:
    """Return ``text`` with the three knobs this benchmark varies replaced."""
    substitutions = {
        r"^Viewer\.render:.*$": f"Viewer.render: {render}",
        r"^Viewer\.logTiming:.*$": f"Viewer.logTiming: {log_timing}",
        r"^DirTrackParams\.maxFramesDirect:.*$": f"DirTrackParams.maxFramesDirect: {max_frames_direct}",
    }
    for pattern, replacement in substitutions.items():
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"could not patch {pattern!r} in GLideConfig.yaml")
    return text


def prepare_run_dir(run_dir: Path, *, max_frames_direct: int, render: bool, log_timing: int) -> None:
    """Create a self-contained working directory for one ``mono_tum`` run."""
    (run_dir / "Examples" / "Monocular").mkdir(parents=True, exist_ok=True)
    config = _rewrite_config(
        GLIDE_CONFIG.read_text(),
        render=1 if render else 0,
        log_timing=log_timing,
        max_frames_direct=max_frames_direct,
    )
    (run_dir / "Examples" / "Monocular" / "GLideConfig.yaml").write_text(config)

    shaders_link = run_dir / "shaders"
    if shaders_link.is_symlink() or shaders_link.exists():
        if shaders_link.is_symlink():
            shaders_link.unlink()
        else:
            shutil.rmtree(shaders_link)
    shaders_link.symlink_to(SHADERS)

    # frame_types.txt is opened in append mode, so a stale file would merge runs.
    for stale in (
        "frame_types.txt",
        "gpuTimings.csv",
        "FrameTrajectory.txt",
        "FrameTrajectory_restamped.txt",
        "KeyFrameTrajectory.txt",
    ):
        (run_dir / stale).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def parse_log(log_path: Path) -> dict[str, object]:
    """Pull the renderer strings and tracking-time summary out of ``run.log``."""
    out: dict[str, object] = {"gpu_track_fail_count": 0}
    if not log_path.is_file():
        return out
    text = log_path.read_text(errors="replace")
    if m := re.search(r"^GL Renderer\s*:\s*(.+)$", text, re.MULTILINE):
        out["gl_renderer"] = m.group(1).strip()
    if m := re.search(r"^GL Version \(string\)\s*:\s*(.+)$", text, re.MULTILINE):
        out["gl_version"] = m.group(1).strip()
    if m := re.search(r"median tracking time:\s*([0-9.eE+-]+)", text):
        out["median_tracking_ms"] = float(m.group(1)) * 1000.0
    if m := re.search(r"mean tracking time:\s*([0-9.eE+-]+)", text):
        out["mean_tracking_ms"] = float(m.group(1)) * 1000.0
    if m := re.search(r"Images in the sequence:\s*(\d+)", text):
        out["images_in_sequence"] = int(m.group(1))
    out["gpu_track_fail_count"] = len(re.findall(r"GPU track\(\) FAIL", text))
    return out


def load_sequence_timestamps(rgb_txt: Path) -> list[float]:
    """Timestamps of every image listed in ``rgb.txt``, in sequence order."""
    stamps: list[float] = []
    for line in rgb_txt.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stamps.append(float(line.split()[0]))
    return stamps


def restamp_trajectory(traj_path: Path, rgb_txt: Path, out_path: Path) -> dict[str, object]:
    """Rewrite ``FrameTrajectory.txt`` with the true per-image timestamps.

    ``Tracking::Track`` appends ``mCurrentFrame.mTimeStamp`` for every pose, but
    ``mCurrentFrame`` is only rebuilt on indirect frames, so a direct pose keeps
    the previous indirect frame's timestamp.  Row ``k`` of the file is always
    image ``first + k`` of the sequence; ``first`` comes from row 0, whose
    timestamp is genuine because the initialising frame is always indirect.

    The reconstruction is checked before it is used: on every row that does
    carry a fresh timestamp, the stored and reconstructed values must agree.
    """
    report: dict[str, object] = {"restamp_ok": False, "restamp_fresh_rows": 0, "restamp_mismatches": 0}
    if not traj_path.is_file() or not rgb_txt.is_file():
        return report

    rows = [line.split() for line in traj_path.read_text(errors="replace").splitlines() if line.strip()]
    stamps = load_sequence_timestamps(rgb_txt)
    if len(rows) < 2 or not stamps:
        return report

    index_of = {round(s, 6): i for i, s in enumerate(stamps)}
    first = index_of.get(round(float(rows[0][0]), 6))
    if first is None or first + len(rows) > len(stamps):
        return report

    raw = [float(r[0]) for r in rows]
    fresh = [True] + [raw[k] != raw[k - 1] for k in range(1, len(raw))]
    mismatches = sum(1 for k in range(len(raw)) if fresh[k] and abs(raw[k] - stamps[first + k]) > 1e-6)

    report["restamp_fresh_rows"] = sum(fresh)
    report["restamp_mismatches"] = mismatches
    report["first_tracked_image"] = first
    report["last_tracked_image"] = first + len(rows) - 1
    report["tracking_coverage_pct"] = 100.0 * len(rows) / len(stamps)
    if mismatches:
        print(f"  restamp check failed: {mismatches} timestamp mismatches in {traj_path.name}", file=sys.stderr)
        return report

    out_path.write_text(
        "".join(f"{stamps[first + k]:.6f} " + " ".join(r[1:8]) + "\n" for k, r in enumerate(rows))
    )
    report["restamp_ok"] = True
    return report


def parse_frame_types(path: Path) -> dict[str, object]:
    """Count direct vs indirect frames in ``frame_types.txt``."""
    direct = indirect = 0
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines()[1:]:
            fields = line.split(",")
            if len(fields) < 2:
                continue
            if fields[1].strip() == "direct":
                direct += 1
            elif fields[1].strip() == "indirect":
                indirect += 1
    total = direct + indirect
    return {
        "frames_logged": total,
        "frames_direct": direct,
        "frames_indirect": indirect,
        "direct_utilization_pct": (100.0 * direct / total) if total else None,
    }


def parse_gpu_timings(path: Path) -> dict[str, GpuStage]:
    """Group ``gpuTimings.csv`` rows by function name.

    The ``track`` rows carry two extra columns (chi2, ok) that the header does
    not declare, so only the first three fields are read.
    """
    buckets: dict[str, list[float]] = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines()[1:]:
            fields = line.split(",")
            if len(fields) < 3:
                continue
            try:
                buckets.setdefault(fields[0].strip(), []).append(float(fields[2]))
            except ValueError:
                continue
    return {
        name: GpuStage(count=len(v), median_ms=statistics.median(v), mean_ms=statistics.fmean(v))
        for name, v in buckets.items()
        if v
    }


def count_poses(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())


def run_evo_ape(gt: Path, est: Path) -> float | None:
    """Return the Sim(3)-aligned ATE RMSE in metres, or None when evo fails."""
    if not est.is_file() or count_poses(est) < 2:
        return None
    proc = subprocess.run(
        ["evo_ape", "tum", str(gt.resolve()), str(est.resolve()), "-as"],
        capture_output=True,
        text=True,
        cwd=str(est.parent),
    )
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()[-2:]
        print(f"  evo_ape failed for {est.name}: {detail}", file=sys.stderr)
        return None
    if m := re.search(r"^\s*rmse\s+([0-9.eE+-]+)", proc.stdout, re.MULTILINE):
        return float(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #


def execute_run(mode: str, index: int, *, render: bool, run_name: str | None, log_timing: int) -> RunResult:
    """Run ``mono_tum`` once and collect every metric for it."""
    max_frames_direct = 10 if mode == "glide" else 0
    suffix = "" if log_timing else "-notiming"
    name = run_name or f"{mode}{suffix}-{index}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_run_dir(run_dir, max_frames_direct=max_frames_direct, render=render, log_timing=log_timing)

    cmd = [str(MONO_TUM), str(VOCAB), str(SETTINGS), str(SEQUENCE_DIR)]
    log_path = run_dir / "run.log"
    print(f"[{name}] {' '.join(cmd)}  (cwd={run_dir})", flush=True)

    start = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=str(run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            log.write(chunk)
            log.flush()
        exit_code = proc.wait()
    wall = time.monotonic() - start
    print(f"[{name}] exit={exit_code} wall={wall:.1f}s", flush=True)

    result = RunResult(
        mode=mode,
        index=index,
        run_dir=str(run_dir),
        exit_code=exit_code,
        wall_time_s=wall,
        max_frames_direct=max_frames_direct,
        log_timing=log_timing,
    )
    for key, value in parse_log(log_path).items():
        setattr(result, key, value)
    for key, value in parse_frame_types(run_dir / "frame_types.txt").items():
        setattr(result, key, value)

    result.gpu_stages = parse_gpu_timings(run_dir / "gpuTimings.csv")
    pyramid = result.gpu_stages.get("imagePyramid")
    pre = result.gpu_stages.get("preCompute")
    trk = result.gpu_stages.get("track")
    if pyramid and pre:
        result.gpu_pre_ms = pyramid.median_ms + pre.median_ms
    if pyramid and trk:
        result.gpu_trk_ms = pyramid.median_ms + trk.median_ms

    result.frame_trajectory_poses = count_poses(run_dir / "FrameTrajectory.txt")
    result.keyframe_trajectory_poses = count_poses(run_dir / "KeyFrameTrajectory.txt")

    restamped = run_dir / "FrameTrajectory_restamped.txt"
    for key, value in restamp_trajectory(
        run_dir / "FrameTrajectory.txt", SEQUENCE_DIR / "rgb.txt", restamped
    ).items():
        setattr(result, key, value)

    gt = SEQUENCE_DIR / "groundtruth.txt"
    result.ate_rmse_frame_m = run_evo_ape(gt, run_dir / "FrameTrajectory.txt")
    result.ate_rmse_frame_restamped_m = run_evo_ape(gt, restamped)
    result.ate_rmse_keyframe_m = run_evo_ape(gt, run_dir / "KeyFrameTrajectory.txt")

    (run_dir / "result.json").write_text(json.dumps(_to_jsonable(result), indent=2))
    return result


def _to_jsonable(result: RunResult) -> dict[str, object]:
    data = asdict(result)
    data["gpu_stages"] = {k: asdict(v) for k, v in result.gpu_stages.items()}
    return data


def load_results() -> list[RunResult]:
    """Read back every ``result.json`` written by earlier invocations."""
    results: list[RunResult] = []
    if not RUNS_DIR.is_dir():
        return results
    fields = {f for f in RunResult.__dataclass_fields__ if f != "gpu_stages"}
    for path in sorted(RUNS_DIR.glob("*/result.json")):
        raw = json.loads(path.read_text())
        stages = {k: GpuStage(**v) for k, v in raw.pop("gpu_stages", {}).items()}
        results.append(RunResult(**{k: v for k, v in raw.items() if k in fields}, gpu_stages=stages))
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _is_full_coverage(result: RunResult) -> bool:
    """True when the run tracked more than ``COVERAGE_FULL_PCT`` of the sequence."""
    return result.tracking_coverage_pct is not None and result.tracking_coverage_pct > COVERAGE_FULL_PCT


def _summarize(runs: list[RunResult]) -> dict[str, float | None]:
    """Median of every headline metric over ``runs``."""
    return {
        "runs": float(len(runs)),
        "median_tracking_ms": _median([r.median_tracking_ms for r in runs]),
        "mean_tracking_ms": _median([r.mean_tracking_ms for r in runs]),
        "direct_utilization_pct": _median([r.direct_utilization_pct for r in runs]),
        "tracking_coverage_pct": _median([r.tracking_coverage_pct for r in runs]),
        "ate_rmse_frame_m": _median([r.ate_rmse_frame_m for r in runs]),
        "ate_rmse_frame_restamped_m": _median([r.ate_rmse_frame_restamped_m for r in runs]),
        "ate_rmse_keyframe_m": _median([r.ate_rmse_keyframe_m for r in runs]),
        "gpu_pre_ms": _median([r.gpu_pre_ms for r in runs]),
        "gpu_trk_ms": _median([r.gpu_trk_ms for r in runs]),
    }


def build_report(results: list[RunResult]) -> tuple[str, dict[str, object]]:
    """Render ``metrics.md`` text and the matching JSON payload."""
    by_mode: dict[str, list[RunResult]] = {}
    for r in results:
        if r.mode in ("glide", "orb") and r.log_timing == 1 and Path(r.run_dir).name == f"{r.mode}-{r.index}":
            by_mode.setdefault(r.mode, []).append(r)

    renderers = sorted({r.gl_renderer for r in results if r.gl_renderer})
    images = max((r.images_in_sequence for r in results), default=0)
    # Name the machine the runs happened on. The renderer comes from the same
    # `GL Renderer` line of run.log that the header below quotes, so the two can
    # never disagree; --report over an empty out/runs has none, and the hostname
    # stands alone.
    title: str = f"GLidE-SLAM on {socket.gethostname()}"
    if renderers:
        title += f" ({renderers[0]})"

    lines: list[str] = [f"# {title}", ""]
    lines += [
        f"Sequence: `rgbd_dataset_freiburg3_long_office_household` (TUM RGB-D, {images} images)",
        f"GL renderer: `{renderers[0] if renderers else 'unknown'}`",
        "",
        "## Per-run results",
        "",
        "| run | exit | wall s | typed frames | direct % | traj poses | coverage % | median track ms | mean track ms "
        "| ATE restamped m | ATE raw m | ATE KF m | imagePyramid ms | preCompute ms | track ms | Pre ms | Trk ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x.mode, x.log_timing, x.index)):
        stage = lambda n: _fmt(r.gpu_stages[n].median_ms, 2) if n in r.gpu_stages else "n/a"  # noqa: E731
        lines.append(
            f"| `{Path(r.run_dir).name}` | {r.exit_code} | {r.wall_time_s:.1f} | {r.frames_logged} | "
            f"{_fmt(r.direct_utilization_pct)} | {r.frame_trajectory_poses} | {_fmt(r.tracking_coverage_pct)} | "
            f"{_fmt(r.median_tracking_ms)} | {_fmt(r.mean_tracking_ms)} | "
            f"{_fmt(r.ate_rmse_frame_restamped_m, 4)} | {_fmt(r.ate_rmse_frame_m, 4)} | "
            f"{_fmt(r.ate_rmse_keyframe_m, 4)} | "
            f"{stage('imagePyramid')} | {stage('preCompute')} | {stage('track')} | "
            f"{_fmt(r.gpu_pre_ms, 2)} | {_fmt(r.gpu_trk_ms, 2)} |"
        )

    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    lost_runs: list[RunResult] = []
    for mode, runs in by_mode.items():
        full = [r for r in runs if _is_full_coverage(r)]
        lost_runs += [r for r in runs if not _is_full_coverage(r)]
        summary[mode] = {"all_runs": _summarize(runs), "full_coverage_runs": _summarize(full)}

    def cell(mode: str, group: str, key: str, digits: int = 1) -> str:
        return _fmt(summary.get(mode, {}).get(group, {}).get(key), digits)

    def metric_row(label: str, key: str, digits: int, paper_orb: str, paper_glide: str) -> str:
        return (
            f"| {label} | {cell('orb', 'all_runs', key, digits)} | {cell('orb', 'full_coverage_runs', key, digits)} "
            f"| {cell('glide', 'all_runs', key, digits)} | {cell('glide', 'full_coverage_runs', key, digits)} "
            f"| {paper_orb} | {paper_glide} |"
        )

    speedups: dict[str, float | None] = {}
    for group in ("all_runs", "full_coverage_runs"):
        orb_ms = summary.get("orb", {}).get(group, {}).get("median_tracking_ms")
        glide_ms = summary.get("glide", {}).get(group, {}).get("median_tracking_ms")
        speedups[group] = orb_ms / glide_ms if orb_ms and glide_ms else None

    paper = PAPER["radxa"]
    lines += [
        "",
        "## Median across runs, next to the paper's Radxa Zero 3W numbers (tum3)",
        "",
        "Every figure is the median over the runs of that mode. The `full cov.` columns keep only the runs",
        f"that tracked more than {COVERAGE_FULL_PCT:.0f} % of the sequence; the runs they drop are listed under",
        "*Runs that lost tracking* below. Read the `all runs` columns only together with that list.",
        "",
        "| metric | ORB baseline: all runs | ORB baseline: full cov. | GLidE: all runs | GLidE: full cov. "
        "| paper: ORB (Radxa Zero 3W) | paper: GLidE (Radxa Zero 3W) |",
        "|---|---|---|---|---|---|---|",
        metric_row("runs", "runs", 0, "-", "-"),
        metric_row("median frame time (ms)", "median_tracking_ms", 1, f"{paper['orb_median_ms']:.1f}", f"{paper['glide_median_ms']:.1f}"),
        metric_row("mean frame time (ms)", "mean_tracking_ms", 1, "-", "-"),
        f"| speedup (ORB / GLidE) | - | - | {_fmt_speedup(speedups['all_runs'])} "
        f"| {_fmt_speedup(speedups['full_coverage_runs'])} | - | {paper['speedup']:.1f}x |",
        metric_row("sequence tracked (%)", "tracking_coverage_pct", 1, "-", "-"),
        metric_row("ATE RMSE, frame traj., restamped (m)", "ate_rmse_frame_restamped_m", 4, f"{paper['orb_ate_m']:.3f}", f"{paper['glide_ate_m']:.3f}"),
        metric_row("ATE RMSE, frame traj., raw file (m)", "ate_rmse_frame_m", 4, "-", "-"),
        metric_row("ATE RMSE, keyframe traj. (m)", "ate_rmse_keyframe_m", 4, "-", "-"),
        metric_row("direct utilization (%)", "direct_utilization_pct", 1, "-", f"{paper['direct_utilization_pct']:.1f}"),
        metric_row("GPU Pre (ms)", "gpu_pre_ms", 2, "-", f"{paper['gpu_pre_ms']:.1f}"),
        metric_row("GPU Trk (ms)", "gpu_trk_ms", 2, "-", f"{paper['gpu_trk_ms']:.1f}"),
        "",
        f"Paper laptop reference for the same GPU stages: Pre {PAPER['laptop']['gpu_pre_ms']} ms, Trk {PAPER['laptop']['gpu_trk_ms']} ms.",
        "",
        "`Pre` = median(imagePyramid) + median(preCompute); `Trk` = median(imagePyramid) + median(track), mirroring Table II of the paper.",
        "",
        "## Runs that lost tracking",
        "",
    ]

    if lost_runs:
        lines += [
            f"These runs stopped producing poses before the end of the sequence ({images} images). `mono_tum`",
            "keeps processing the remaining images, which cost almost nothing once there is nothing to track,",
            "so a lost run reports a much *lower* median frame time than a complete one.",
            "",
            "| run | last tracked image | poses | coverage % | median track ms | GPU gate fails |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(lost_runs, key=lambda x: (x.mode, x.index)):
            last = "n/a" if r.last_tracked_image is None else str(r.last_tracked_image)
            lines.append(
                f"| `{Path(r.run_dir).name}` | {last} | {r.frame_trajectory_poses} | "
                f"{_fmt(r.tracking_coverage_pct)} | {_fmt(r.median_tracking_ms)} | {r.gpu_track_fail_count} |"
            )
    else:
        lines.append(f"None: every benchmarked run tracked more than {COVERAGE_FULL_PCT:.0f} % of the sequence.")

    lines += [
        "",
        "## How to read these numbers",
        "",
        "1. **Coverage first, then timing.** `median tracking time` is printed by upstream `mono_tum`, which",
        "   sorts the per-image times of **all** images in the sequence and takes the middle one, including",
        "   the images processed after tracking was lost. A run that loses tracking early therefore reports a",
        "   lower median than one that tracks throughout, which is why the table carries both an `all runs`",
        "   and a `full cov.` column and why the lost runs are named above. Neither mode reaches 100 %:",
        "   monocular initialisation consumes the first 28 images, so a complete run covers 98.9 %.",
        "2. **`ATE ... restamped` is the trustworthy accuracy figure.** `Tracking::Track` stores",
        "   `mCurrentFrame.mTimeStamp` with every pose, and `mCurrentFrame` is only rebuilt on indirect frames,",
        "   so every direct pose in `FrameTrajectory.txt` inherits the previous indirect frame's timestamp.",
        "   `evo_ape` associates by timestamp, so the raw-file column matches direct poses against the wrong",
        "   ground-truth samples. `bench.py` rewrites the timestamps (`FrameTrajectory_restamped.txt`) after",
        "   verifying the reconstruction against every row that did receive a fresh timestamp.",
        "3. **The ORB baseline is not pure ORB-SLAM2.** It is produced config-only, by setting",
        "   `DirTrackParams.maxFramesDirect: 0`, which makes `Tracking::SwitchToIndirect` see a wide baseline",
        "   on every frame, so `mbUseDirectTracking` is false and the indirect pipeline runs. Two caveats. The",
        "   GPU direct tracker still runs on every frame and its result is discarded, so the baseline pays GPU",
        "   cost that upstream ORB-SLAM2 would not, made synchronous by `Viewer.logTiming: 1`; its timing is an",
        "   upper bound. And `SwitchToIndirect` returns false outright while the local mapper is stopped, which",
        "   `LoopClosing::CorrectLoop` does around the loop-closure global bundle adjustment, so the baseline",
        "   still shows a short burst of direct frames there (the non-zero `direct %` in the per-run table).",
        "   No upstream source file is modified.",
        "",
    ]

    payload: dict[str, object] = {
        "runs": [_to_jsonable(r) for r in results],
        "coverage_full_threshold_pct": COVERAGE_FULL_PCT,
        "summary_median_across_runs": summary,
        "speedup_orb_over_glide": speedups,
        "lost_runs": [
            {
                "run": Path(r.run_dir).name,
                "last_tracked_image": r.last_tracked_image,
                "poses": r.frame_trajectory_poses,
                "tracking_coverage_pct": r.tracking_coverage_pct,
                "median_tracking_ms": r.median_tracking_ms,
                "gpu_track_fail_count": r.gpu_track_fail_count,
            }
            for r in sorted(lost_runs, key=lambda x: (x.mode, x.index))
        ],
        "paper_reference": PAPER,
        "gl_renderers": renderers,
    }
    return "\n".join(lines), payload


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["glide", "orb"], default="glide")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--render", action="store_true", help="keep the native SDL viewer on (Viewer.render: 1)")
    parser.add_argument(
        "--log-timing",
        type=int,
        choices=[0, 1],
        default=1,
        help="Viewer.logTiming; 1 writes gpuTimings.csv but inserts glFinish() around the GPU stages",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="index of the first run, so a single run of a finished batch can be repeated",
    )
    parser.add_argument("--run-name", default=None, help="override the run directory name (single run only)")
    parser.add_argument("--report", action="store_true", help="aggregate every existing run into out/metrics.{md,json}")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.report:
        for path, what in ((MONO_TUM, "binary"), (VOCAB, "vocabulary"), (SEQUENCE_DIR, "sequence")):
            if not path.exists():
                print(f"missing {what}: {path}", file=sys.stderr)
                return 1
        if "DISPLAY" not in os.environ:
            print("warning: DISPLAY is unset; the GLidE engine needs an X display", file=sys.stderr)
        for n in range(args.start_index, args.start_index + args.runs):
            result = execute_run(
                args.mode, n, render=args.render, run_name=args.run_name, log_timing=args.log_timing
            )
            if result.exit_code != 0:
                print(f"run {n} failed with exit code {result.exit_code}; see {result.run_dir}/run.log", file=sys.stderr)
                return result.exit_code

    results = load_results()
    if not results:
        print("no runs found", file=sys.stderr)
        return 1
    text, payload = build_report(results)
    (OUT_DIR / "metrics.md").write_text(text)
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    if args.report:
        print(text)
    print(f"wrote {OUT_DIR / 'metrics.md'} and {OUT_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
