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
The IDs in `frame_types.txt` and the `frame` column of `gpuTimings.csv` come from that one
counter, but they are not sequence positions, and for the same image they are not always the
same ID — see the next gotcha.

**`gpuTimings.csv` is one ID behind `frame_types.txt` on every indirect frame.** The `frame`
column is `GLideEngine::m_sourceFrameID`, and only `updateNewFrame` ever writes it
(`src/GLideEngine.cpp:1519`), so a GPU row carries the ID of the `FrameDirect` built for that
image at `src/Tracking.cc:281`. When the direct result is rejected the image is re-tracked
indirectly: `mCurrentFrame` is rebuilt at `src/Tracking.cc:472`, taking the next
`Frame::nNextId`, and `LogFrameType` records **that** ID. `updateRefFrame` does not touch
`m_sourceFrameID` at all, and a new direct reference can only be built from an indirect frame's
map points (`Tracking::updateDirectReference`), so every `preCompute` row is exactly one ID
short of its typed frame. glide-1 says so unambiguously: 0 of 774 `preCompute` rows match a
typed ID, all 768 that match `id + 1` land on an `indirect` row, while `id - 1` matches a
meaningless mix of 601 `direct` and 167 `indirect` rows; the 6 leftovers are warm-up rows
stamped with frame 0. `demo_rerun.join_gpu_timings` therefore joins a row to its own ID when
that ID is typed and to `id + 1` otherwise, and prints the coverage per stage. Before that fix
`/metrics/gpu/preCompute` was silently missing from the recording, and 527 `track` rows — the
frames whose direct result was rejected — were dropped as well.

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
stats`: 1835 chunks, 26468 rows, 216 entity paths — 203 in the recording, 13 in the blueprint —
101.5 MiB compressed IPC). Three checks run with it: the direct/indirect pattern derived from
the stale timestamps agreed with `frame_types.txt` on 2556 of 2556 rows, the GPU rows joined
onto typed frames at 2546/2546 (`track`), 3314/3320 (`imagePyramid`) and 768/774
(`preCompute`), and the demo's own Sim(3)-aligned ATE RMSE of 0.095023 m matched
`evo_ape tum ... -as` on `FrameTrajectory_restamped.txt` to 3.9e-07 m. The right-hand column of
the blueprint is one view per quantity — camera, ATE error, photometric chi2, GPU stage times —
because chi2 (~0.003), ATE (~0.1 m) and a 0-100 percentage share no usable y range;
`is_direct` and `direct_utilization_pct` are still logged and reachable from the streams
panel.
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

## Desktop (RTX 3060) reproduction

The same branch, the same sequence and the same tasks on an x86-64 desktop, to check that the
packaging is not Raspberry-Pi-shaped. It is not, but getting there needed one build-flag fix
that only x86-64 exposes, and the accuracy numbers came out worse than the Pi's rather than
better. Both are recorded below.

| | |
|---|---|
| host | `pablo-ubuntu`, 16 cores, 62 GB |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, x86-64 |
| GPU | NVIDIA GeForce RTX 3060, 12 GB, driver 595.84 |
| GL renderer reported by the engine | `NVIDIA GeForce RTX 3060/PCIe/SSE2`, vendor `NVIDIA Corporation`, `OpenGL ES 3.2 NVIDIA 595.84`, GLSL ES 3.20 |
| pixi | 0.77.1 |
| X display | `:1` (`XAUTHORITY=/run/user/1000/gdm/Xauthority`), Xorg running on the 3060 |
| sequence | the same TUM RGB-D `rgbd_dataset_freiburg3_long_office_household`, 2585 images |

The banner the engine prints, from `out/runs/glide-1/run.log`:

```
GL Vendor              : NVIDIA Corporation
GL Renderer            : NVIDIA GeForce RTX 3060/PCIe/SSE2
GL Version (string)    : OpenGL ES 3.2 NVIDIA 595.84
GL Version (integeger) : 3.2
GLSL Version           : OpenGL ES GLSL ES 3.20
```

So the `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d` trick carries over unchanged:
on this host the same directory holds `10_nvidia.json` next to `50_mesa.json`, and the loader
picks the NVIDIA ICD. Nothing about the EGL/SDL path needed touching. `Viewer: Shutting down.`
is followed by `Xlib: extension "NV-GLX" missing on display ":1"`, which the NVIDIA GLX layer
prints on teardown; it is harmless and the process still exits 0.

### What the manifest needed

`platforms` gained `linux-64`; `DISPLAY`, `XAUTHORITY` and `XDG_RUNTIME_DIR` moved into
`[target.linux-aarch64.activation.env]`, because they describe the Pi's Wayfire session and
nothing else; `SDL_VIDEODRIVER` and `__EGL_VENDOR_LIBRARY_DIRS` stayed shared. `pixi lock`
solves both platforms from one manifest and `pixi lock --check` is clean.

The build tasks now say `-j$(nproc)` instead of `-j4`. pixi runs task commands through a shell
that performs command substitution, so this expands per host: the desktop build logs
`cmake --build build -j16`.

### The one real portability bug: g2o and Eigen disagree on x86-64

The first `pixi run demo` on this machine died like this, 14.5 s in, immediately after
`New Map created with 164 points`:

```
[glide-1] exit=-11 wall=14.5s
run 1 failed with exit code -11; see .../out/runs/glide-1/run.log
```

`-11` is SIGSEGV, and it is not a GPU problem. Under gdb:

```
Thread 1 "mono_tum" received signal SIGSEGV, Segmentation fault.
0x00007fffedea53be in __GI___libc_free (mem=0x41) at ./malloc/malloc.c:3368
#1  g2o::VertexSE3Expmap::~VertexSE3Expmap() ... /Thirdparty/g2o/lib/libg2o.so
#2  g2o::HyperGraph::clear() ...
#3  g2o::OptimizableGraph::~OptimizableGraph() ...
#4  ORB_SLAM2::Optimizer::BundleAdjustment(...) ... /lib/libGLidE_SLAM.so
#5  ORB_SLAM2::Optimizer::GlobalBundleAdjustemnt(...)
#6  ORB_SLAM2::Tracking::CreateInitialMapMonocular()
```

`free()` is handed `0x41`, so a pointer was read out of memory that never held one. The cause is
a build-flag split the two trees have always had, which only x86-64 makes visible. The root
`CMakeLists.txt:3` compiles the SLAM library with `-DEIGEN_DONT_VECTORIZE
-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT`, while the vendored g2o was built only with
`-DEIGEN_DONT_ALIGN_STATICALLY` (upstream `build.sh`, carried into `pixi.toml`). Both trees also
compile `-O3 -march=native`. Compiling a probe against g2o's headers with each flag set in turn
shows what that costs:

| flags | `EIGEN_MAX_STATIC_ALIGN_BYTES` | `EIGEN_MAX_ALIGN_BYTES` |
|---|---|---|
| g2o: `-DEIGEN_DONT_ALIGN_STATICALLY -march=native` | 0 | **32** |
| SLAM lib: `-DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT -march=native` | 0 | **0** |

Class layouts agree — `sizeof(VertexSE3Expmap)` is 256 with `alignof` 8 either way — so this is
not a size mismatch. `EIGEN_MAX_ALIGN_BYTES` is what selects Eigen's allocator: at 0 it falls
through to plain `malloc`/`free`, and above 16 it switches to an over-aligned path with its own
bookkeeping. `Optimizer::BundleAdjustment` `new`s its vertices in the SLAM library and
`HyperGraph::clear()` deletes them inside libg2o, so the two halves of every Eigen buffer's
lifetime use different allocators, and the first global bundle adjustment corrupts the heap.

`-march=native` is what decides it. On x86-64 it turns on AVX, which lifts
`EIGEN_IDEAL_MAX_ALIGN_BYTES` to 32 for anything not compiled `EIGEN_DONT_VECTORIZE`. aarch64
tops out at 16, where both sides take the plain `malloc`/`free` path regardless, which is why
the Pi never noticed and why five Pi runs of each mode came back clean.

The fix is in `pixi.toml`, not in any source file: `G2O_EIGEN_FLAGS` is empty by default and set
in `[target.linux-64.activation.env]` to `-DEIGEN_DONT_VECTORIZE
-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT`, which `_build-g2o` appends to g2o's `CMAKE_CXX_FLAGS`.
Both sides then agree on `EIGEN_MAX_ALIGN_BYTES = 0`, and `mono_tum` runs to the end. It is
scoped to `linux-64` deliberately: adding `-DEIGEN_DONT_VECTORIZE` on aarch64 would change g2o's
codegen on the Pi and invalidate every number above for no benefit, since the mismatch is
harmless there. The Pi build is byte-for-byte what it was.

This is a latent defect in the upstream build system, not in the packaging. Any x86-64 host that
builds `build.sh` as written hits it.

### Wall time

Measured phase by phase on a warm `~/.cache/rattler`, with `DISPLAY=:1` exported by hand:

| phase | wall |
|---|---|
| `pixi install` (warm cache) | 8.7 s |
| `pixi run build` — DBoW2, g2o, vocabulary, `libGLidE_SLAM.so`, `mono_tum`, `-j16` | 1 min 31.7 s |
| `hf download` of the 1.2 GB sequence | 3 min 21.8 s |
| `pixi run demo` on a built tree with the data present | 2 min 20.7 s |
| — of which the GLidE run itself | 124.8 s |
| — of which `demo_rerun.py` -> 42 MiB `out/glide-tum3.rrd` | ~16 s |
| **total** | **7 min 23 s** |

Against 9 min 20 s for the same sequence of steps on the Pi. The gap is almost entirely compile
time (`-j16` against `-j4`); the download dominates both and depends on the link, not the host.
The build figure was taken before the g2o flag fix above; g2o rebuilds on its own in 23.2 s, so
it is still representative.

### Per-run results

`pixi run bench`, five runs of each mode, 26 min 15 s in total, every run exit 0. GL renderer
`NVIDIA GeForce RTX 3060/PCIe/SSE2`.

| run | exit | wall s | typed frames | direct % | traj poses | coverage % | median track ms | mean track ms | ATE restamped m | ATE raw m | ATE KF m | imagePyramid ms | preCompute ms | track ms | Pre ms | Trk ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `glide-1` | 0 | 129.1 | 2557 | 75.5 | 2558 | 99.0 | 11.5 | 21.3 | 1.1089 | 1.1081 | 1.1106 | 1.56 | 2.38 | 13.81 | 3.94 | 15.37 |
| `glide-2` | 0 | 124.4 | 1011 | 75.4 | 1011 | 39.1 | 1.2 | 9.7 | 0.0343 | 0.0344 | 0.0198 | 1.55 | 2.39 | 13.83 | 3.94 | 15.39 |
| `glide-3` | 0 | 124.0 | 969 | 75.2 | 969 | 37.5 | 1.0 | 9.3 | n/a | 0.0694 | 0.0678 | 1.56 | 2.44 | 13.72 | 4.00 | 15.28 |
| `glide-4` | 0 | 124.3 | 1006 | 75.1 | 1006 | 38.9 | 1.1 | 9.6 | 0.0491 | 0.0469 | 0.0404 | 1.57 | 2.47 | 13.93 | 4.05 | 15.51 |
| `glide-5` | 0 | 129.9 | 2557 | 78.2 | 2558 | 99.0 | 11.5 | 21.3 | 0.1621 | 0.1656 | 0.0804 | 1.55 | 2.40 | 13.69 | 3.95 | 15.24 |
| `orb-1` | 0 | 184.8 | 2556 | 0.7 | 2557 | 98.9 | 54.4 | 58.6 | 0.0122 | 0.0147 | 0.0099 | 1.65 | 2.69 | 11.61 | 4.34 | 13.26 |
| `orb-2` | 0 | 183.6 | 2556 | 0.4 | 2557 | 98.9 | 53.8 | 58.2 | 0.0324 | 0.0329 | 0.0305 | 1.64 | 2.69 | 11.61 | 4.33 | 13.25 |
| `orb-3` | 0 | 184.0 | 2556 | 0.7 | 2557 | 98.9 | 54.0 | 58.3 | 0.0144 | 0.0158 | 0.0115 | 1.64 | 2.69 | 11.61 | 4.33 | 13.25 |
| `orb-4` | 0 | 185.2 | 2556 | 0.5 | 2557 | 98.9 | 54.3 | 58.8 | 0.0137 | 0.0144 | 0.0114 | 1.65 | 2.69 | 11.63 | 4.34 | 13.28 |
| `orb-5` | 0 | 182.4 | 2556 | 0.7 | 2557 | 98.9 | 53.8 | 57.6 | 0.0127 | 0.0141 | 0.0107 | 1.64 | 2.69 | 11.63 | 4.34 | 13.27 |

### Median across runs

`full cov.` keeps only the runs that tracked more than 95 % of the sequence. On this host that
is 2 of the 5 GLidE runs, so read the GLidE columns with the loss table below in hand.

| metric | ORB baseline: all runs | ORB baseline: full cov. | GLidE: all runs | GLidE: full cov. |
|---|---|---|---|---|
| runs | 5 | 5 | 5 | 2 |
| median frame time (ms) | 54.0 | 54.0 | 1.2 | 11.5 |
| mean frame time (ms) | 58.3 | 58.3 | 9.7 | 21.3 |
| speedup (ORB / GLidE) | - | - | 44.06x | 4.70x |
| sequence tracked (%) | 98.9 | 98.9 | 39.1 | 99.0 |
| ATE RMSE, frame traj., restamped (m) | 0.0137 | 0.0137 | 0.1056 | 0.6355 |
| ATE RMSE, frame traj., raw file (m) | 0.0147 | 0.0147 | 0.0694 | 0.6368 |
| ATE RMSE, keyframe traj. (m) | 0.0114 | 0.0114 | 0.0678 | 0.5955 |
| direct utilization (%) | 0.7 | 0.7 | 75.4 | 76.9 |
| GPU Pre (ms) | 4.34 | 4.34 | 3.95 | 3.94 |
| GPU Trk (ms) | 13.26 | 13.26 | 15.37 | 15.31 |

The `all runs` GLidE columns are the artefact the Gotchas warn about, in its purest form: three
of five runs lost tracking early, so the median of `median frame time` is 1.2 ms and the
apparent speedup is 44x. Neither number means anything.

### Runs that lost tracking

| run | last tracked image | poses | coverage % | median track ms | GPU gate fails |
|---|---|---|---|---|---|
| `glide-2` | 1037 | 1011 | 39.1 | 1.2 | 15 |
| `glide-3` | 995 | 969 | 37.5 | 1.0 | 12 |
| `glide-4` | 1032 | 1006 | 38.9 | 1.1 | 9 |

Three of five, against two of five on the Pi, and all three die in the same images 995-1037
window as the Pi's `glide-3` and `glide-5`. That window is a property of the sequence, not of the
host: it holds the sequence's maximum angular rate, 42.2 deg/s, which is measured and explained
under *Tracking stability varies between identical runs* above. Faster hardware does not help,
because the failure is a search-radius limit in the direct tracker, not a deadline miss.

`glide-3` is also the first run on either machine to trip `bench.restamp_trajectory`'s guard: 6
of its rows disagreed with the reconstruction, so `bench.py` refused to write
`FrameTrajectory_restamped.txt` and reported `n/a` rather than an ATE it could not stand behind.
That is the check working as designed.

### Three-way comparison

Every GLidE column is the `full cov.` median; ORB is the all-runs median, which on both machines
equals its full-coverage median. The paper laptop column is the published Table II figure for
`tum3` on an RTX 4060 mobile.

| metric | paper laptop, RTX 4060 mobile | desktop, RTX 3060 | Raspberry Pi 5, VideoCore VII |
|---|---|---|---|
| ORB median frame time (ms) | 9.6 | 54.0 | 74.5 |
| GLidE median frame time (ms) | 5.0 | 11.5 | 17.0 |
| speedup (ORB / GLidE) | 1.9x | 4.70x | 4.39x |
| ATE RMSE, ORB (m) | 0.010 | 0.0137 | 0.0151 |
| ATE RMSE, GLidE (m) | 0.014 | 0.6355 | 0.0335 |
| direct utilization (%) | 69.3 | 76.9 | 78.4 |
| GPU Pre (ms) | 2.5 | 3.94 | 18.47 |
| GPU Trk (ms) | 3.5 | 15.31 | 28.76 |

Three things stand out, and only the first is flattering.

**The GPU stages do not scale evenly.** From Pi to desktop, `imagePyramid` goes 8.11 -> 1.56 ms
(5.2x) and `preCompute` 10.37 -> 2.38 ms (4.4x), but `track` only 20.65 -> 13.81 ms (1.5x). A
3060 against a VideoCore VII is a far larger gap than 1.5x in any throughput sense, so `track` is
not throughput-bound here. It is a chain of many small dispatches per frame, serialised by the
`glFinish()` calls that `Viewer.logTiming: 1` inserts (see the Gotchas), so what it measures is
dispatch and synchronisation latency, which a bigger GPU does not shrink. The consequence is
visible in the table: the desktop's GPU `Trk` of 15.31 ms is *above* its own ORB baseline's 13.26
ms, and still 4.4x the paper laptop's 3.5 ms on a card of a similar class. Whatever the laptop
figure was measured with, it was not this configuration.

**The end-to-end speedup grows as the host gets slower.** 1.9x on the laptop, 4.70x here, 4.39x
on the Pi. The direct path's cost is dominated by fixed GPU latency, so it barely moves between
hosts, while the indirect baseline's ORB extraction and bundle adjustment scale with the CPU.
The slower the CPU, the more the direct path wins.

**Accuracy got worse, not better.** The two complete GLidE runs scored 1.109 m and 0.162 m,
against 0.032-0.095 m for the Pi's three complete runs; the ORB baseline is unchanged at 0.014 m
against the Pi's 0.015 m, so the sequence, the data and the evaluation are all fine, and it is
the direct path alone that degraded. `glide-1`'s 1.109 m is not a small drift, it is a broken
map. This is the same non-determinism as the loss window: `Tracking::SwitchToIndirect` reads
`mpLocalMapper->isStopped()` and `KeyframesInQueue()` (`src/Tracking.cc:729-747`), so which
frames go direct depends on how the mapping thread is scheduled, and 16 cores schedule it very
differently from 4 — the tracking thread is never starved here, so it runs ahead and takes the
direct branch in places the Pi never could. Two complete runs is far too small a sample to put a
number on the spread; what the desktop shows is that the spread exists and is wide, which the Pi
already suggested. It is a property of this source snapshot, not of the packaging.

### The recording and the viewer

`out/glide-tum3.rrd` was written by the `demo` task from its own GLidE run, before the benchmark
overwrote `out/runs/glide-1`. That run tracked images 28-1078 (40.7 % — it is one of the runs
that hit the loss window), 1051 poses, 89 keyframes, 790 direct frames (75.2 %), ATE RMSE
0.028816 m matching `evo_ape -as` to 4.7e-07 m, `frame_types.txt` agreement 1050/1050 rows, GPU
joins 99.5 % / 98.4 % / 99.9 %. The file is 42 MiB against the Pi run's 102 MiB, because it
carries 1051 images rather than 2557.

A headless viewer serving it was checked over the timeline with a 120-frame sweep of the `frame`
timeline; the sweep reported the range as `28..1078`, which is exactly the run's tracked span,
and all four views (camera, ATE error, photometric chi2, GPU stage times) render at both ends and
in the middle.

### What was not verified

Only `linux-64` and `linux-aarch64` are in `platforms`, and only these two hosts were run. The
`eval` and `demo-upstream` tasks were not exercised on the desktop.

The first desktop report was generated while `bench.py` still hardcoded
`# GLidE-SLAM on Raspberry Pi 5 (VideoCore VII / v3d)` as the title of `out/metrics.md`, so it
named the wrong machine; only the `GL renderer:` line beneath it came from the runs. `b081c65`
builds the title from `socket.gethostname()` and that same renderer string, so the two cannot
disagree, and falls back to the hostname alone when no run has been parsed. The regenerated
report opens `# GLidE-SLAM on pablo-ubuntu (NVIDIA GeForce RTX 3060/PCIe/SSE2)`. Regenerating it
changed line 1 and nothing else — `out/metrics.json` came back byte-identical — so every number
in the tables above is the one the benchmark produced; they are copied from that report below
its title.
