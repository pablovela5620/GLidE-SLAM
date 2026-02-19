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
