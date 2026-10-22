#!/bin/bash
# Create the "gs3d" conda env for the FFGS-benchmark 3DGS baseline and build the CUDA extensions.
#
#   bash install_gs3d.sh [-n NAME | -p PREFIX] [--arch X.Y] [--skip_create]
#
#   -n NAME / -p PREFIX   env to create (default: -n gs3d); on a cluster, -p puts it in a writable dir
#   --arch X.Y            GPU compute capability to build for, e.g. 8.0 (A100, Polaris) or 8.9 (RTX 4090);
#                         default: detected with nvidia-smi (pass it explicitly on a node without GPUs)
#   --skip_create         reuse an existing env: only (re)install the pip pins and rebuild the extensions
#
# Steps: conda env from environment_gs3d.yml -> pip pins from requirements_gs3d.txt (torch from the
# PyTorch cu121 index) -> build diff-gaussian-rasterization, simple-knn, fused-ssim with the env's
# nvcc 12.1 and gcc 11 -> import / pip check (+ a tiny GPU test if a GPU is visible).
#
# Polaris (login node, which has internet; build for A100 explicitly):
#   module use /soft/modulefiles && module load conda
#   bash install_gs3d.sh -p /path/to/envs/gs3d --arch 8.0
#
# Needs the submodules checked out: git submodule update --init --recursive

set -eo pipefail

usage() { sed -n '2,20p' "$0"; exit 2; }

ENV_FLAG=(-n gs3d)
ARCH=""
SKIP_CREATE=0
while [ $# -gt 0 ]; do
    case "$1" in
        -n|-p)         ENV_FLAG=("$1" "$2"); shift 2 ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --skip_create) SKIP_CREATE=1; shift ;;
        -h|--help)     usage ;;
        *)             echo "unknown argument: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXTENSIONS=(submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim)
for ext in "${EXTENSIONS[@]}"; do
    if [ ! -f "$ext/setup.py" ]; then
        echo "ERROR: $ext is not checked out; run: git submodule update --init --recursive"
        exit 1
    fi
done

if ! command -v conda > /dev/null; then
    echo "ERROR: conda not found (on Polaris: module use /soft/modulefiles && module load conda)"
    exit 1
fi
eval "$(conda shell.bash hook)"

# -----------------------------------------------------------------------
# 1. conda env: python, nvcc 12.1 + headers, gcc/g++ 11, ninja
# -----------------------------------------------------------------------
if [ "$SKIP_CREATE" != "1" ]; then
    echo "[gs3d] creating env ${ENV_FLAG[*]} from environment_gs3d.yml"
    conda env create "${ENV_FLAG[@]}" -f environment_gs3d.yml
fi
# NOTE: activating by prefix also works for a named env's name
conda activate "${ENV_FLAG[1]}"
echo "[gs3d] active env: $CONDA_PREFIX"

# -----------------------------------------------------------------------
# 2. pip pins (torch/torchvision from the PyTorch cu121 index)
# -----------------------------------------------------------------------
python -m pip install -r requirements_gs3d.txt --extra-index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------
# 3. CUDA extensions, built with the env's toolchain for one GPU architecture
# -----------------------------------------------------------------------
if [ -z "$ARCH" ]; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)
    if [ -z "$ARCH" ]; then
        echo "ERROR: no GPU visible to detect the architecture; pass --arch (e.g. 8.0 for A100)"
        exit 1
    fi
    echo "[gs3d] detected GPU compute capability $ARCH"
fi

export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_CCBIN="$CXX"
export TORCH_CUDA_ARCH_LIST="$ARCH"
echo "[gs3d] nvcc: $(command -v nvcc) ($(nvcc --version | tail -n 1))"
echo "[gs3d] building for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST with $CXX"

# stale build/ dirs (e.g. built for another GPU) would otherwise be reused
for ext in "${EXTENSIONS[@]}"; do
    rm -rf "${ext:?}/build"
done
# --no-deps: never let an extension's setup.py touch the pins above
python -m pip install --no-build-isolation --no-deps "${EXTENSIONS[@]}"

# -----------------------------------------------------------------------
# 4. checks
# -----------------------------------------------------------------------
python -m pip check
python - <<'EOF'
import numpy, torch, PIL, cv2
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print(f"[gs3d] torch {torch.__version__} (CUDA {torch.version.cuda}), numpy {numpy.__version__}, "
      f"pillow {PIL.__version__}, opencv {cv2.__version__}")
if torch.cuda.is_available():
    pts = torch.rand(1000, 3, device="cuda")
    distCUDA2(pts)
    a = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    fused_ssim(a, torch.rand_like(a)).backward()
    print(f"[gs3d] GPU test passed on {torch.cuda.get_device_name(0)}")
else:
    print("[gs3d] no GPU visible: imports OK, GPU test skipped (run a short train.py on a compute node)")
EOF
echo "[gs3d] done: conda activate ${ENV_FLAG[1]}"
