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
`DISPLAY`, `XAUTHORITY` and `XDG_RUNTIME_DIR` for the local desktop session; override them
in the environment if yours differ.

See `NOTES.md` for the measured numbers on a Raspberry Pi 5, and for the upstream quirks the
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
