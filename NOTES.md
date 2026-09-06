# Packaging and reproduction notes

How GLidE-SLAM was made to build and run through [pixi](https://pixi.prefix.dev) on a
Raspberry Pi 5, what it measured there, and every upstream quirk that got in the way.

| | |
|---|---|
| host | Raspberry Pi 5 Model B Rev 1.1, BCM2712, 4 cores, 8 GB |
| OS | Debian GNU/Linux 12 (bookworm), kernel 6.12.96+rpt-rpi-2712, aarch64 |
| GPU | VideoCore VII, Mesa 24.2.8-1~bpo12+rpt5, driver `v3d` |
| GL renderer reported by the engine | `V3D 7.1.10.2`, vendor `Broadcom`, `OpenGL ES 3.1 Mesa 24.2.8-1~bpo12+rpt5`, GLSL ES 3.10 |
| pixi | 0.77.1 |
| source snapshot | `pablovela5620/GLidE-SLAM` @ `13f5d9fe8d12bd0f3943be0a399891052b64d5cd` |
| upstream `capsMD/GLidE-SLAM` | `dfd51fb15ad204ecd563dada160f7245f2fdbbe8` (README only; it carries no source) |
| sequence | TUM RGB-D `rgbd_dataset_freiburg3_long_office_household`, 2585 images |

## Decisions

**pixi tasks, not a pixi-build package.** Every dependency the project needs — OpenCV 4,
Eigen 3.4, SDL2, GLM, the glvnd headers and loader, plus the Python side — is already on
conda-forge for `linux-aarch64`, and DBoW2 and g2o are vendored under `Thirdparty/` and are
built in-tree by the upstream `build.sh`. Wrapping that in a `pixi-build` backend would mean
re-expressing a build that already works, for no gain. The `pixi.toml` tasks call the same
`cmake` commands `build.sh` does, so the packaging adds an environment and a task graph and
changes nothing about how the project compiles.

**Shell guards, not pixi `inputs`/`outputs`.** Every build artefact (`Thirdparty/*/lib/*.so`,
`lib/libGLidE_SLAM.so`, `Examples/Monocular/mono_tum`, `Vocabulary/ORBvoc.txt`) is gitignored,
and pixi's file-change cache does not handle gitignored paths reliably. Each build task is
therefore `test -f <artefact> || (cmake … && cmake --build …)`. `pixi run build` on an
already-built tree takes about 0.08 s.

**The GPU driver comes from the host, not from pixi.** conda-forge ships the glvnd loader
(`libglvnd`, `libegl-devel`, `libgles-devel`) but no `v3d` or Panfrost user-space driver, so
an all-conda-forge stack would fall back to software rendering, which would defeat the point
of the exercise. `pixi.toml` sets `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d`
so the loader picks up the host ICD `50_mesa.json` → `libEGL_mesa.so.0`. This is the one
deliberate exception to "everything from pixi"; the run log proves it worked, because the
engine reports `V3D 7.1.10.2` rather than `llvmpipe` or `softpipe`.

**`fr3_long_office_household`.** It is the `tum3` sequence of the paper's Table II, so the
measurements line up with the published ones, and its intrinsics are exactly the ones already
hard-coded in `Examples/Monocular/GLideConfig.yaml` (fx 535.4, fy 539.2, cx 320.1, cy 247.6).
It is also the longest of the three, which exposes the tracking-stability behaviour recorded
below. The depth stream is stripped from the mirror because the demo runs monocular.

**No simplecv.** `demo_rerun.py` uses `tyro`, `rerun`, `numpy` and `cv2` directly, plus two
loader helpers imported from the sibling `bench.py` so the timestamp repair exists in exactly
one place.

## Commands

```bash
# everything at once: build, fetch the sequence, run on the GPU, write the recording
pixi run demo

# the individual steps
pixi run build                        # DBoW2, g2o, vocabulary, libGLidE_SLAM.so, mono_tum
pixi run _download-data               # hf download pablovela5620/glide-slam-example
pixi run run-glide                    # one GLidE run   -> out/runs/glide-1/
pixi run run-orb                      # one baseline run -> out/runs/orb-1/
pixi run bench                        # 5 + 5 runs -> out/metrics.md, out/metrics.json
pixi run demo-upstream                # one GLidE run with the native SDL viewer
pixi run eval                         # evo_ape against the TUM ground truth

# view the recording
pixi run rerun out/glide-tum3.rrd

# rebuild only the report from the runs already on disk
python bench.py --report

# repeat one run of a finished batch, keeping its index (here out/runs/orb-5)
python bench.py --mode orb --runs 1 --start-index 5
```

The exact `mono_tum` invocation `bench.py` issues, with absolute paths and the run directory
as the working directory:

```bash
cd out/runs/glide-1
<repo>/Examples/Monocular/mono_tum \
  <repo>/Vocabulary/ORBvoc.txt \
  <repo>/Examples/Monocular/TUM3.yaml \
  <repo>/data/rgbd_dataset_freiburg3_long_office_household
```

### Data preparation and the Hugging Face mirror

The sequence was taken from TUM, unpacked, and the depth stream removed:

```bash
curl -O https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar -xzf rgbd_dataset_freiburg3_long_office_household.tgz -C data
rm -rf data/rgbd_dataset_freiburg3_long_office_household/depth \
       data/rgbd_dataset_freiburg3_long_office_household/depth.txt
```

`rgb/`, `rgb.txt`, `groundtruth.txt` and `accelerometer.txt` are byte-for-byte the originals.
The mirror was then created and uploaded:

```bash
hf repo create pablovela5620/glide-slam-example --repo-type dataset
hf upload pablovela5620/glide-slam-example \
  data/rgbd_dataset_freiburg3_long_office_household \
  rgbd_dataset_freiburg3_long_office_household --repo-type dataset
hf upload pablovela5620/glide-slam-example README.md README.md --repo-type dataset
```

<https://huggingface.co/datasets/pablovela5620/glide-slam-example>, 2590 files (2585 PNGs, the
three index/ground-truth files, the dataset card and `.gitattributes`), 1.2 GB. The revision the
numbers in this file were produced against is
**`3cf50295d0b6bcdd5177d95334843b11dca162bd`**; `hf upload` split the folder over 8 commits.

The round trip was checked, not assumed. The same command the `_download-data` task runs was
pointed at a scratch directory and the result compared with the local copy:

```bash
hf download pablovela5620/glide-slam-example --repo-type dataset --local-dir /tmp/hf-verify
diff -r --brief /tmp/hf-verify/rgbd_dataset_freiburg3_long_office_household \
                data/rgbd_dataset_freiburg3_long_office_household   # no output
```

`diff` reported nothing, so every one of the 2588 files came back identical. `hf download` also
drops the dataset card into `data/README.md`; that is harmless, and `data/` is gitignored.

## Upstream edits

None to the project's own code. Everything this branch adds or changes is packaging:

```
$ git diff origin/main --name-status
M	.gitignore
A	NOTES.md
M	README.md
A	bench.py
A	demo_rerun.py
A	pixi.lock
A	pixi.toml
```

Seven files, all of them packaging: nothing under `src/`, `include/`, `shaders/`,
`Thirdparty/`, `CMakeLists.txt` or `Examples/` is touched. (`--name-status` rather than
`--stat`, because `NOTES.md`'s own line count changes every time this file is edited.)

Everything the benchmark varies is a per-run **copy** of `Examples/Monocular/GLideConfig.yaml`
written into the run directory; the committed config is never touched.

## Gotchas

**Every runtime path is relative to the working directory.** `System::System` opens
`Examples/Monocular/GLideConfig.yaml` (`src/System.cc:118`) and the engine opens
`shaders/<name>.{vert,frag,comp}`, all relative to the process CWD, and all outputs
(`FrameTrajectory.txt`, `KeyFrameTrajectory.txt`, `frame_types.txt`, `gpuTimings.csv`) land
there too. `bench.py` therefore gives every run its own directory containing a patched copy of
the config and a symlink to the repo's `shaders/`, and passes the binary, the vocabulary, the
settings file and the sequence as absolute paths.

**`frame_types.txt` is opened in append mode.** `static std::ofstream logFile("frame_types.txt",
std::ios::app)` in `src/Tracking.cc:816`. Running twice in the same directory silently
concatenates the runs — the copy committed at the repo root has 88 header rows in it, from 88
merged runs. `bench.py` deletes the file before every run.

**An X display is required even with `Viewer.render: 0`.** The engine unconditionally creates
two SDL windows and calls `eglCreateWindowSurface` on `sysInfo.info.x11.window`, so the run
fails without a display no matter what the viewer flag says. `pixi.toml` sets
`SDL_VIDEODRIVER=x11`, `DISPLAY=:0`, `XAUTHORITY=$HOME/.Xauthority` and
`XDG_RUNTIME_DIR=/run/user/1000` for the local Wayfire/Xwayland session. Two windows appear on
the Pi's screen during every run; that is expected.

**`-march=native` and the g2o Eigen flag.** The root `CMakeLists.txt` compiles with
`-O3 -march=native`, so the binaries are tied to the machine that built them; a fresh clone has
to rebuild rather than reuse artefacts from another host. g2o needs
`-DCMAKE_CXX_FLAGS=-DEIGEN_DONT_ALIGN_STATICALLY`, matching upstream `build.sh`, otherwise the
aligned-allocation assumptions clash on aarch64. The vendored CMake files also predate CMake 4,
so both `Thirdparty` builds get `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.

**`printf` output is block-buffered when stdout is a pipe.** The `GL Vendor` / `GL Renderer`
banner from `GLideEngine::printVersions()` is written with `printf`, while the coloured
progress messages go through `Logger`. When `run.log` is a pipe the banner does not appear
until the buffer flushes, which can be at process exit. It is always in the finished log;
do not conclude from a partial log that the renderer was never printed.

**`frame_types.txt` timestamps are unusable.** The writer streams a `double` with the default
ostream precision, so every TUM timestamp in the file collapses to `1.34185e+09`. Rows must be
joined to the trajectory by order, not by timestamp.

**Frame IDs are not image indices.** After commit `3d25f7b` `FrameDirect` draws from
`Frame::nNextId` as well, so an indirect image consumes two IDs and a direct one consumes one.
The IDs in `frame_types.txt` and the `frame` column of `gpuTimings.csv` are consistent with
each other — that is what makes the GPU-stage join work — but they are not sequence positions.

**`FrameTrajectory.txt` carries stale timestamps on every direct frame.** This is the one that
actually changes the numbers. `Tracking::Track` appends `mCurrentFrame.mTimeStamp` for every
pose (`src/Tracking.cc:715`), but `mCurrentFrame` is only rebuilt on indirect frames
(`src/Tracking.cc:472`); on a direct frame only `mCurrentDirectFrame` is constructed, and
the pose is copied into `mCurrentFrame` without its timestamp. Each direct pose therefore
inherits the previous indirect frame's timestamp, and the file contains runs of up to
`maxFramesDirect` identical timestamps carrying different poses. `evo_ape` associates by
timestamp, so on the raw file it matches most GLidE poses against the wrong ground-truth
samples and reports an ATE several times too large.

The chain in the source is short. The pose is appended together with
`mCurrentFrame.mTimeStamp` at `src/Tracking.cc:715`, and `System::SaveTrajectoryTUM`
(`src/System.cc:376-381`) writes that list out, skipping the frames marked lost. But
`mCurrentFrame` is rebuilt from the image only on the indirect branch
(`src/Tracking.cc:472`); on a direct frame only `mCurrentDirectFrame` is built with the true
stamp (`src/Tracking.cc:406`) and line 456 copies **just the pose** into `mCurrentFrame`:

```cpp
// src/Tracking.cc:456 — direct branch, pose only
mCurrentFrame.SetPose(mCurrentDirectFrame.mTcw);
// src/Tracking.cc:472 — indirect branch, whole frame incl. timestamp
mCurrentFrame = Frame(mImGray, mCurrentTimestamp, mpORBextractorLeft, ...);
// src/Tracking.cc:715 — what actually gets written
mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
```

`out/runs/glide-1/FrameTrajectory.txt` has 2557 pose rows but only 538 distinct timestamps,
one repeated 13 times. Rows 42-49 of the raw file against the same rows of
`FrameTrajectory_restamped.txt` (translation truncated), next to `rgb.txt`:

| row | raw timestamp | restamped timestamp | image in `rgb.txt` | tx |
|---|---|---|---|---|
| 42 | 1341847983.034693 | 1341847983.034693 | 1341847983.034693.png | -0.036388304 |
| 43 | 1341847983.034693 | 1341847983.066707 | 1341847983.066707.png | -0.022226302 |
| 44 | 1341847983.034693 | 1341847983.102757 | 1341847983.102757.png | -0.036388304 |
| 45 | 1341847983.034693 | 1341847983.134717 | 1341847983.134717.png | -0.040586226 |
| 46 | 1341847983.034693 | 1341847983.166651 | 1341847983.166651.png | -0.045060635 |
| 47 | 1341847983.034693 | 1341847983.202653 | 1341847983.202653.png | -0.049570609 |
| 48 | 1341847983.034693 | 1341847983.234717 | 1341847983.234717.png | -0.052922305 |
| 49 | 1341847983.034693 | 1341847983.266753 | 1341847983.266753.png | -0.055214144 |

Eight different poses, one timestamp. Row 42 is the indirect frame that refreshed it, so its
value is unchanged; rows 43-49 are the direct frames that inherited it.

Row `k` of the file is always image `first + k` of the sequence, and row 0 is the initialising
frame, which is always indirect and so has a genuine timestamp. `bench.restamp_trajectory`
uses that to write `FrameTrajectory_restamped.txt`, and refuses to do so unless the
reconstruction reproduces the file exactly on every row whose stored timestamp did change --
538 such rows in glide-1. On every run recorded here that check passed with zero mismatches.
Both the raw and the repaired ATE are reported; the repaired one is the meaningful figure.

The same defect gives a free, independent signal: a row is a direct frame exactly when its
stored timestamp repeats the previous row's. `demo_rerun.py` derives the direct/indirect
colouring from that and cross-checks it against `frame_types.txt`; agreement was 2556/2556
rows on the run used for the recording.

**`mono_tum`'s "median tracking time" covers the whole sequence.** It sorts the per-image
times of all `nImages` images and prints the middle one, including images processed after
tracking was lost, which cost almost nothing. A run that loses tracking early therefore reports
a *lower* median than one that tracks throughout. Never read that number without the coverage
figure next to it.

**`Viewer.logTiming: 1` changes what it measures.** The timing branches in
`src/GLideEngine.cpp` (~1440-1490) call `glFinish()` after `buildPyramid` and `preCompute` to
get a wall time per stage, which serialises GPU work that is otherwise asynchronous. Both modes
are benchmarked with it on, so the comparison is internally consistent, but the absolute
numbers are not what the system does with timing off.

**The config-only ORB baseline is not pure ORB-SLAM2, and it is not 100 % indirect either.**
`DirTrackParams.runDirectTracking` is never read by any code — `src/GLideUtils.cpp` reads
`DirTrackParams.maxFramesDirect` and not that key — so the only lever is `maxFramesDirect: 0`.
With `mMinFrames` at 0 that makes `bWideBaseline` true on every frame, so `SwitchToIndirect`
returns true and `mbUseDirectTracking` stays false. Two things spoil the purity of that
baseline.

First, `NeedNewDirectRef` (`src/Tracking.cc:778`) fires every frame, so the GPU still runs
`imagePyramid`, `preCompute` and `track` and the result is simply thrown away: the baseline's
`gpuTimings.csv` is not empty. The baseline is "GLidE with the direct result always rejected",
which costs more than upstream ORB-SLAM2 would, so its timing is an upper bound.

Second, `SwitchToIndirect` returns `false` outright, before any baseline test, when the local
mapper is stopped or has been asked to stop (`src/Tracking.cc:729-730`), and
`LoopClosing::CorrectLoop` stops it (`src/LoopClosing.cc:409`) for the pose-graph correction
and again for the global bundle adjustment (`src/LoopClosing.cc:667`). During that window the
direct pose is accepted no matter what `maxFramesDirect` says, so the baseline's
`frame_types.txt` does contain `direct` rows: 10, 11, 15, 9 and 14 of 2556 in `orb-1` .. `orb-5`
(0.35 - 0.59 %), each one or two short blocks of consecutive frame ids in the 4300-4460 range,
which is where the log prints `Loop detected!` and `Starting Global Bundle Adjustment`. A
faithful ORB-SLAM2 baseline would need a source change, which was out of scope here.

**Tracking stability varies between identical runs, and the losses all land in the same
place.** The same binary, config and sequence tracked to the last image (2584, 98.9 % of the
sequence: monocular initialisation eats the first 28) in glide-1, glide-2 and glide-4, and lost
tracking for good in glide-3 (last pose at image 1010, 38.0 % coverage) and glide-5 (image 1037,
39.1 %). All five ORB-baseline runs went through. The window is not
arbitrary: measured on the TUM ground truth over a 0.1 s sliding window, the camera's angular
rate over images 1001-1036 reaches 42.2 deg/s, the maximum of the whole sequence, against a
sequence-wide median of 12.0 deg/s, and the median inside images 1000-1045 is 31.5 deg/s. That
is roughly 1.4 deg per frame at 30 Hz, about 13 px of image motion at `fx = 535`, while the
direct tracker searches with `searchRadius: 5` and per-level `maxShift: [3,4,5,6]` px, so only
the coarsest pyramid levels can still bracket the shift. Both lost runs were deep in a
direct-only stretch when it hit: glide-3's last eight typed frames (images 1003-1010) are all
`direct`, chi2 creeping 0.00291 -> 0.00332, and seven of glide-5's last eight (images
1031-1037) likewise. Neither
tripped the emergency threshold (`chi2 > 0.004`, `src/Tracking.cc:733`) or the degraded-quality
one (0.0035), so the tracker never handed control back early.

The two runs then died in slightly different ways. glide-3 has 982 typed frames and 983 poses:
the loss is at image 1011, where the indirect fallback failed both `TrackWithMotionModel` and
`TrackReferenceKeyFrame` (`src/Tracking.cc:477-484`) and so never reached `LogFrameType`.
glide-5 has 1010 typed frames and 1010 poses, one typed frame more than it saved: its last
typed row is an `indirect` frame at image 1038 whose `TrackLocalMap` failed
(`src/Tracking.cc:578-579`), which sets `mState = LOST` (`src/Tracking.cc:603`) and makes
`SaveTrajectoryTUM` drop that pose. After the loss `LogFrameType` is unreachable — it sits on
the `bOK` path — which is why `frame_types.txt` stops dead at the loss while `gpuTimings.csv`
runs one frame further. `Relocalization` then never succeeded for the remaining ~1550 images.
The tail of the log shows three `g2o ... 0 vertices to optimize` lines from those attempts, but
that message on its own proves nothing: glide-2 printed it three times and still tracked to the
end.

The survivors crossed the same window with more indirect frames in it — glide-1 has an indirect
frame at image 1040 with chi2 0.00447, just under the 0.0045 acceptance gate in
`GLideEngine::track` (`src/GLideEngine.cpp:920`). Which frames go direct is not deterministic:
`SwitchToIndirect` reads `mpLocalMapper->isStopped()` and `KeyframesInQueue()`
(`src/Tracking.cc:729-747`), so the split depends on how the mapping thread is scheduled across
the Pi's four cores. That makes the spread a property of this snapshot — its most recent commits
are `tuning motion model filter for indirect tracking, now it works, maybe too relaxed` and
`fixed global ID for indirect/direct frames` — and not of the packaging. `out/metrics.md`
therefore reports every median twice, over all runs and over the full-coverage runs alone, and
names the lost runs with the last image they tracked.

## Reproduction

`pixi run bench` runs the sequence five times in each mode and then
`python bench.py --report`, which writes `out/metrics.md` and `out/metrics.json`. What follows is
that report, minus its own commentary; the reading instructions are in the Gotchas above.
Both modes ran with `Viewer.render: 0` and `Viewer.logTiming: 1`, one after the other, nothing
else on the machine.

### Per-run results

| run | exit | wall s | typed frames | direct % | traj poses | coverage % | median track ms | mean track ms | ATE restamped m | ATE raw m | ATE KF m | imagePyramid ms | preCompute ms | track ms | Pre ms | Trk ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `glide-1` | 0 | 133.0 | 2556 | 79.0 | 2557 | 98.9 | 16.9 | 26.3 | 0.0950 | 0.0938 | 0.0733 | 8.11 | 10.37 | 20.65 | 18.47 | 28.76 |
| `glide-2` | 0 | 134.4 | 2556 | 78.4 | 2557 | 98.9 | 17.0 | 27.0 | 0.0335 | 0.0355 | 0.0161 | 8.03 | 10.41 | 20.71 | 18.45 | 28.75 |
| `glide-3` | 0 | 124.7 | 982 | 74.0 | 983 | 38.0 | 2.4 | 12.3 | 0.0297 | 0.0294 | 0.0173 | 7.96 | 10.44 | 20.44 | 18.40 | 28.40 |
| `glide-4` | 0 | 134.0 | 2556 | 78.2 | 2557 | 98.9 | 17.2 | 27.1 | 0.0315 | 0.0331 | 0.0128 | 8.07 | 10.44 | 20.78 | 18.50 | 28.84 |
| `glide-5` | 0 | 124.8 | 1010 | 74.1 | 1010 | 39.1 | 2.2 | 12.6 | 0.0381 | 0.0387 | 0.0227 | 7.98 | 10.45 | 20.50 | 18.43 | 28.49 |
| `orb-1` | 0 | 229.1 | 2556 | 0.4 | 2557 | 98.9 | 74.7 | 76.5 | 0.0173 | 0.0190 | 0.0133 | 8.26 | 12.40 | 22.12 | 20.65 | 30.37 |
| `orb-2` | 0 | 228.4 | 2556 | 0.4 | 2557 | 98.9 | 74.5 | 76.2 | 0.0140 | 0.0148 | 0.0111 | 8.26 | 12.39 | 22.06 | 20.66 | 30.32 |
| `orb-3` | 0 | 227.4 | 2556 | 0.6 | 2557 | 98.9 | 74.1 | 75.8 | 0.0151 | 0.0172 | 0.0116 | 8.25 | 12.40 | 22.09 | 20.64 | 30.33 |
| `orb-4` | 0 | 229.1 | 2556 | 0.4 | 2557 | 98.9 | 74.9 | 76.5 | 0.0120 | 0.0129 | 0.0098 | 8.24 | 12.39 | 21.95 | 20.63 | 30.19 |
| `orb-5` | 0 | 228.5 | 2556 | 0.5 | 2557 | 98.9 | 74.5 | 76.2 | 0.0242 | 0.0248 | 0.0197 | 8.25 | 12.39 | 21.94 | 20.64 | 30.19 |

### Median across runs, next to the paper's Radxa Zero 3W numbers (tum3)

Every figure is the median over the runs of that mode. The `full cov.` columns keep only the runs
that tracked more than 95 % of the sequence; the runs they drop are listed under
*Runs that lost tracking* below. Read the `all runs` columns only together with that list.

| metric | ORB baseline: all runs | ORB baseline: full cov. | GLidE: all runs | GLidE: full cov. | paper: ORB (Radxa Zero 3W) | paper: GLidE (Radxa Zero 3W) |
|---|---|---|---|---|---|---|
| runs | 5 | 5 | 5 | 3 | - | - |
| median frame time (ms) | 74.5 | 74.5 | 16.9 | 17.0 | 131.0 | 58.8 |
| mean frame time (ms) | 76.2 | 76.2 | 26.3 | 27.0 | - | - |
| speedup (ORB / GLidE) | - | - | 4.40x | 4.39x | - | 2.2x |
| sequence tracked (%) | 98.9 | 98.9 | 98.9 | 98.9 | - | - |
| ATE RMSE, frame traj., restamped (m) | 0.0151 | 0.0151 | 0.0335 | 0.0335 | 0.009 | 0.016 |
| ATE RMSE, frame traj., raw file (m) | 0.0172 | 0.0172 | 0.0355 | 0.0355 | - | - |
| ATE RMSE, keyframe traj. (m) | 0.0116 | 0.0116 | 0.0173 | 0.0161 | - | - |
| direct utilization (%) | 0.4 | 0.4 | 78.2 | 78.4 | - | 69.0 |
| GPU Pre (ms) | 20.64 | 20.64 | 18.45 | 18.47 | - | 43.7 |
| GPU Trk (ms) | 30.32 | 30.32 | 28.75 | 28.76 | - | 39.1 |

Paper laptop reference for the same GPU stages: Pre 2.5 ms, Trk 3.5 ms.

`Pre` = median(imagePyramid) + median(preCompute); `Trk` = median(imagePyramid) + median(track), mirroring Table II of the paper.

### Runs that lost tracking

These runs stopped producing poses before the end of the sequence (2585 images). `mono_tum`
keeps processing the remaining images, which cost almost nothing once there is nothing to track,
so a lost run reports a much *lower* median frame time than a complete one.

| run | last tracked image | poses | coverage % | median track ms | GPU gate fails |
|---|---|---|---|---|---|
| `glide-3` | 1010 | 983 | 38.0 | 2.4 | 17 |
| `glide-5` | 1037 | 1010 | 39.1 | 2.2 | 13 |


### The Rerun recording

```bash
pixi run demo          # or, once the run exists:
python -u demo_rerun.py --run-dir out/runs/glide-1 --rr-config.headless --rr-config.save out/glide-tum3.rrd
pixi run rerun out/glide-tum3.rrd
```

From `out/runs/glide-1`: 2557 poses over images 28-2584, 188 keyframes, 2019 direct frames
(79.0 %), 2557 RGB images at JPEG quality 80 with stride 1. The file is 102 MiB (`rerun rrd
stats`: 1702 chunks, 24885 rows, 213 entity paths, 101.1 MiB compressed IPC). Two checks run
with it: the direct/indirect pattern derived from the stale timestamps agreed with
`frame_types.txt` on 2556 of 2556 rows, and the demo's own Sim(3)-aligned ATE RMSE of
0.095023 m matched `evo_ape tum ... -as` on `FrameTrajectory_restamped.txt` to 3.9e-07 m.
glide-1 is the least accurate of the three complete runs (0.095 m against 0.032 m and 0.031 m
for glide-2 and glide-4); `--run-dir out/runs/glide-2` shows the better end of the spread.

## Fresh-clone timing

Timed on the machine in the table at the top, from a tree stripped back to what a fresh clone
contains: `.pixi/`, `build/`, `Thirdparty/*/{build,lib}`, `lib/`, the three
`Examples/Monocular/mono_*` binaries, the extracted `Vocabulary/ORBvoc.txt` and `data/` were all
deleted first, and `out/` was moved aside. **The pixi package cache (`~/.cache/rattler`) was
warm**, so the environment was installed from cached packages and not downloaded; a machine that
has never seen these packages will need a few minutes more.

```
$ time pixi run demo
...
FRESH_CLONE_REAL=560.377 s USER=705.692 SYS=47.436
```

**9 min 20 s wall for everything**: environment, compile, dataset, SLAM run and recording. Only
the total is measured; the split below comes from polling the log every ~2 min, so read it as
+/- 30 s.

| phase | approx. wall |
|---|---|
| `pixi install` from the warm cache + DBoW2 + g2o + vocabulary + `libGLidE_SLAM.so` + `mono_tum` (`-j4`) | ~2 min 15 s |
| `hf download` of the 1.2 GB sequence | ~4 min |
| one GLidE run over 2585 images | 133.6 s (measured by `bench.py`) |
| `demo_rerun.py` -> 102 MiB `out/glide-tum3.rrd` | ~1 min |

That run tracked images 29-2584 (98.9 %), agreed with `frame_types.txt` on 2555 of 2555 rows and
scored an ATE RMSE of 0.031951 m, matching `evo_ape` to 4.2e-07 m. `-O3 -march=native` means the
binaries cannot be moved to another machine; every clone compiles its own.
