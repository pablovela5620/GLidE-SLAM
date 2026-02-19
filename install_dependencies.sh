#!/bin/bash
set -e

echo "Installing GLidE-SLAM dependencies for Debian 12 / Ubuntu 22.04+"
echo "============================================================="

# Core build tools
echo "Installing build essentials..."
sudo apt install -y \
  build-essential \
  cmake \
  git

# GLidE-SLAM dependencies
echo "Installing libraries..."
sudo apt install -y \
  libeigen3-dev \
  libopencv-dev \
  libsdl2-dev \
  libboost-all-dev \
  libglm-dev \
  libegl1-mesa-dev \
  libgles2-mesa-dev

echo ""
echo "✓ All dependencies installed!"
echo ""
echo "Next steps:"
echo "  ./build.sh    # Build GLidE-SLAM"
