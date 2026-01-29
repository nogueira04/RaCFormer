#!/bin/bash

set -e

echo "=== Jetson Orin AGX Environment Setup ==="
echo ""

if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: This doesn't appear to be a Jetson device"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

echo "Jetson info:"
cat /etc/nv_tegra_release 2>/dev/null || echo "Could not read tegra release"
echo ""

echo "System CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found - ensure JetPack CUDA is installed"
echo ""

echo "=== Step 1: Creating conda environment ==="
conda env create -f environment_jetson.yml || conda env update -f environment_jetson.yml

echo ""
echo "=== Step 2: Activate environment ==="
echo "Run: conda activate racformerfix"
echo ""

echo "=== Step 3: Install PyTorch for Jetson ==="
echo ""
echo "Downloading and installing PyTorch 2.3 for JetPack 6 (L4T R36.x)..."

wget -O torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl

wget -O torchvision-0.18.0-cp310-cp310-linux_aarch64.whl \
    https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl

pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

rm -f torch-2.3.0-cp310-cp310-linux_aarch64.whl torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

echo "PyTorch installation complete."
echo ""

echo "=== Step 4: Install OpenMMLab packages ==="
echo ""

sudo apt-get install -y libssl-dev

pip install mmengine

# Build mmcv 2.1.0 from source (takes 20-40 min on Orin AGX)
# NOTE: Must use v2.1.0+ for PyTorch 2.3 compatibility (v2.0.0 fails to compile)
# NOTE: Must use --no-build-isolation so mmcv can find torch during build
# NOTE: Don't use -e (editable), it breaks __version__ attribute
echo "Building mmcv 2.1.0 from source (this takes a while)..."
git clone --branch v2.1.0 https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-build-isolation .
cd ..

echo "Building mmdetection from source..."
git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install --no-build-isolation -v .
cd ..

echo "Building mmdetection3d from source..."
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install --no-build-isolation -v .
cd ..

echo "OpenMMLab packages installed."
echo ""

echo "=== Step 6: Build custom CUDA extensions ==="
cd models/csrc
python setup.py build_ext --inplace
cd ../..
echo "Custom CUDA extensions built."
echo ""

echo "=== Setup Complete ==="
echo ""
