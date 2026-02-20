# GLidE-SLAM Dependencies

## Quick Install (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config wget \
  libopencv-dev \
  libeigen3-dev \
  libglm-dev \
  libsdl2-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  libboost-all-dev
```

---

# List of Known Dependencies

## GLidE-SLAM (derived from ORB-SLAM2)

This document lists all code and libraries included in GLidE-SLAM that are not the property of the GLidE-SLAM authors.

## Code in `src` and `include` folders

### From ORB-SLAM2

* **ORBextractor.cc**  
  Modified version of orb.cpp from OpenCV library (BSD licensed)

* **PnPsolver.h, PnPsolver.cc**  
  Modified version of epnp by Vincent Lepetit (FreeBSD)  
  Also found in [OpenCV](https://github.com/opencv/opencv) and [OpenGV](https://github.com/laurentkneip/opengv)

* **ORBmatcher::DescriptorDistance** in ORBmatcher.cc  
  From: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel  
  Public domain

### GLidE-SLAM Additions

* **GLideEngine.cpp/h**  
  Proprietary GPU acceleration code (see source headers for licensing)

* **ImageHandler.cpp/h, GLideUtils.cpp/h**  
  GLidE-SLAM custom code (GPL-3.0 unless otherwise noted)

* **Thirdparty/glad/**  
  [GLAD](https://github.com/Dav1dde/glad) OpenGL loader (MIT/Public Domain)

## Code in Thirdparty folder

* **DBoW2**  
  Modified version of [DBoW2](https://github.com/dorian3d/DBoW2) and [DLib](https://github.com/dorian3d/DLib)  
  BSD licensed

* **g2o**  
  Modified version of [g2o](https://github.com/RainerKuemmerle/g2o)  
  BSD licensed

## Library Dependencies

* **OpenCV** - Image processing and features (BSD license)
* **Eigen3** - Linear algebra (MPL2 for v3.1.1+, LGPLv3 for earlier)
* **SDL2** - Window/input management for viewer (Zlib license)
* **OpenGL ES 3.1** - GPU compute shaders (vendor-specific, typically MIT/Apache)
* **EGL** - OpenGL ES context creation (vendor-specific)
* **Boost** - Utilities (Boost Software License)
* **GLM** - OpenGL Mathematics library for shader math (MIT license, header-only)

## Original ORB-SLAM2 Authors

GLidE-SLAM is derived from ORB-SLAM2 by:
- Raúl Mur-Artal
- Juan D. Tardós
- J. M. M. Montiel
- Dorian Gálvez-López (DBoW2)