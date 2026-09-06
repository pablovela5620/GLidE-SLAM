# GLidE-SLAM

**GL-accelerated Indirect-Direct Embedded SLAM**

A hybrid visual SLAM system combining ORB-SLAM2's sparse backend with GPU-accelerated direct photometric tracking for embedded platforms.

## Features

- **Hybrid Tracking**: Sparse indirect (ORB-SLAM2) backend with direct photometric tracking for intermediate frames
- **GPU Acceleration**: OpenGL ES 3.1 compute shaders for photometric tracking pipeline
- **Embedded-First**: Targets ARM platforms (Raspberry Pi 5, Radxa Zero 3W) with vendor-agnostic GPU support
- **No Pangolin**: Custom EGL/SDL-based viewer for embedded deployment
- **Real-time**: 3-4x speedup over CPU-only direct tracking on tested platforms

## Installation

### Quick Install (Debian 12 / Ubuntu 22.04+)
```bash
./install_dependencies.sh
./build.sh
```

### Manual Installation

#### Prerequisites

**Debian 12 (Bookworm) / Ubuntu 22.04+**
```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libeigen3-dev \
  libopencv-dev \
  libsdl2-dev \
  libboost-all-dev \
  libglm-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev
```

#### GPU Requirements

- **OpenGL ES 3.1+** capable GPU
- For embedded platforms: Mali, VideoCore, or similar with compute shader support

Verify GPU support:
```bash
eglinfo | grep "OpenGL ES"
# Should show: OpenGL ES 3.1 or higher
```

#### Build
```bash
# Clone repository
git clone https://github.com/capsMD/GLidE-SLAM.git
cd GLidE-SLAM

# Build everything
./build.sh
```

This will:
1. Build DBoW2 in `Thirdparty/DBoW2/`
2. Build g2o in `Thirdparty/g2o/`
3. Extract vocabulary file
4. Build `libGLidE_SLAM.so` in `lib/`
5. Build example executables in `Examples/Monocular/`

#### Embedded Platforms (ARM)

**Tested on:**
- Radxa Zero 3W (RK3566, Mali-G52)
- Raspberry Pi 5 (BCM2712, VideoCore VII)

**Additional notes:**
- Mesa 24.x+ recommended for best Panfrost (Mali) support
- Kernel 6.1+ for optimal ARM GPU drivers

## Run with pixi

### Tested platforms

| platform | machine | GPU / driver | GL renderer string | GLidE median frame time (full-coverage runs) | ORB median frame time | status |
|---|---|---|---|---|---|---|
| `linux-aarch64` | Raspberry Pi 5 (BCM2712) | VideoCore VII, Mesa 24.2.8 `v3d` | `V3D 7.1.10.2` | 17.0 ms | 74.5 ms | works |
| `linux-64` | `pablo-ubuntu` desktop | NVIDIA GeForce RTX 3060, driver 595.84 | `NVIDIA GeForce RTX 3060/PCIe/SSE2` | 11.5 ms | 54.0 ms | works (needs the `linux-64` g2o Eigen flags, see `NOTES.md`) |

`pixi.lock` solves both platforms from one manifest; see `NOTES.md` for the ATE, tracking-loss
and GPU-stage caveats behind these numbers.

[pixi](https://pixi.prefix.dev) builds the project, fetches the sample sequence and runs the
demo without installing anything system-wide. Every dependency except the GPU driver comes
from conda-forge; the driver has to be the host's Mesa, because conda-forge ships no `v3d`
(VideoCore) or Panfrost user-space driver. `pixi.toml` therefore points the glvnd loader at
`/usr/share/glvnd/egl_vendor.d`.

```bash
pixi run demo
```

That single task builds DBoW2, g2o and `libGLidE_SLAM.so`, extracts the ORB vocabulary,
downloads `rgbd_dataset_freiburg3_long_office_household` from the Hugging Face Hub, runs
`mono_tum` on the GPU and writes a Rerun recording to `out/glide-tum3.rrd`. Open it with:

```bash
pixi run rerun out/glide-tum3.rrd
```

Other tasks:

| task | what it does |
|---|---|
| `pixi run build` | build the vendored dependencies and `mono_tum` (no-op once built) |
| `pixi run run-glide` | one GLidE run into `out/runs/glide-1/` |
| `pixi run run-orb` | one indirect-only baseline run into `out/runs/orb-1/` |
| `pixi run bench` | five runs of each mode, then `out/metrics.md` and `out/metrics.json` |
| `pixi run demo-upstream` | one GLidE run with the native SDL viewer (`Viewer.render: 1`) |
| `pixi run eval` | `evo_ape` against the TUM ground truth |

The runs need an X display, because the engine always creates its two SDL windows and calls
`eglCreateWindowSurface` on the X11 handle even when `Viewer.render` is 0. `pixi.toml` sets
`DISPLAY`, `XAUTHORITY` and `XDG_RUNTIME_DIR` **only for `linux-aarch64`**, where they describe
the Raspberry Pi 5's local Wayfire session. On any other host — including every `linux-64` one —
export them yourself before calling pixi, pointing at a session that actually has a GPU behind
it:

```bash
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority   # adjust to your session
pixi run demo
```

`SDL_VIDEODRIVER` and `__EGL_VENDOR_LIBRARY_DIRS` are shared by both platforms: the same ICD
directory resolves to Mesa `v3d` on the Pi and to NVIDIA on a desktop, so the GPU driver always
comes from the host and never from conda-forge.

See `NOTES.md` for the measured numbers on a Raspberry Pi 5 and on an RTX 3060 desktop, for the
three-way comparison against the paper's laptop figures, and for the upstream quirks the
benchmark has to work around.

## Usage

### TUM Dataset Example
```bash
./Examples/Monocular/mono_tum \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/TUM1.yaml \
  /path/to/rgbd_dataset_freiburg1_xyz
```

### KITTI Dataset Example
```bash
./Examples/Monocular/mono_kitti \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/KITTI00-02.yaml \
  /path/to/dataset/sequences/00
```

## License

- **Core SLAM framework** (derived from ORB-SLAM2): GPL-3.0 (see License-gpl.txt)
- **GLidE Engine** (GPU acceleration): Proprietary (see source file headers)
  - Free for academic/research use with citation
  - Commercial licensing available

See LICENSE.txt and Dependencies.md for details.

## Citation

If you use GLidE-SLAM in academic work, please cite:
```bibtex
[Your IROS 2026 paper citation - to be added]
```

## Acknowledgments

This work builds upon [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) by Raúl Mur-Artal and Juan D. Tardós.
