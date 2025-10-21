FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# 系统基础层
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-11 g++-11 cmake ninja-build git wget curl \
    ca-certificates software-properties-common gnupg2 dirmngr \
    # OpenGL 和图形库
    libgl1-mesa-dev mesa-common-dev libglew-dev libglib2.0-0 \
    # neurad-studio 需要的系统库 (COLMAP 依赖)
    libatlas-base-dev libhdf5-dev \
    libprotobuf-dev protobuf-compiler \
    libboost-filesystem-dev libboost-graph-dev \
    libboost-program-options-dev libboost-system-dev libboost-test-dev \
    libcgal-dev libeigen3-dev libflann-dev libfreeimage-dev \
    libgflags-dev libmetis-dev libqt5opengl5-dev libsuitesparse-dev \
    qtbase5-dev \
    # 多媒体和工具
    ffmpeg vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python 3.11
# ============================================================
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 100

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# ============================================================
# CUDA 11.8
# ============================================================
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y \
    cuda-toolkit-11-8 \
    cuda-compiler-11-8 \
    cuda-nvcc-11-8 \
    cuda-libraries-dev-11-8 \
    libcublas-dev-11-8 \
    libcufft-dev-11-8 \
    libcurand-dev-11-8 \
    libcusolver-dev-11-8 \
    libcusparse-dev-11-8 \
    libnpp-dev-11-8 \
    && rm cuda-keyring_1.0-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/lib64/stubs:/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs:/usr/local/lib \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" \
    TCNN_CUDA_ARCHITECTURES="70;75;80;86;90" \
    FORCE_CUDA=1

# ============================================================
# Python 虚拟环境
# ============================================================
RUN python3.11 -m venv /opt/env
ENV PATH=/opt/env/bin:${PATH}

WORKDIR /workspace

# ============================================================
# 基础 Python 包
# ============================================================
RUN pip install --no-cache-dir --upgrade pip "setuptools<72.1.0" wheel packaging \
    pathtools promise pybind11 dill ninja

# ============================================================
# PyTorch 2.1.2 + CUDA 11.8 (稳定版本)
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# 关键依赖版本锁定 (解决所有冲突)
# ============================================================
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<1.25.0" \
    "scipy==1.11.4" \
    "jupyter-client<8.0.0"

# 创建约束文件
RUN printf "torch==2.1.2\ntorchvision==0.16.2\ntorchaudio==2.1.2\nnumpy>=1.24.0,<1.25.0\nscipy==1.11.4\njupyter-client<8.0.0\n" > /tmp/constraints.txt

# ============================================================
# Kaolin 0.17.0 (有 CUDA 11.8 的预编译 wheel)
# ============================================================
RUN pip install --no-cache-dir \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html \
    kaolin==0.17.0

# ============================================================
# 项目层 1: 3DGRUT
# ============================================================
COPY 3dgrut /workspace/3dgrut
RUN cd /workspace/3dgrut

# 临时移除 fused-ssim (最后单独安装)
RUN cd /workspace/3dgrut && \
    cp requirements.txt requirements.txt.bak && \
    sed -i '/fused-ssim/d' requirements.txt

RUN cd /workspace/3dgrut && \
    pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt

RUN cd /workspace/3dgrut && \
    pip install --no-build-isolation -e . -c /tmp/constraints.txt

# ============================================================
# 项目层 2: neurad-studio
# ============================================================
COPY neurad-studio /workspace/neurad-studio

# 安装 neurad-studio (不用 [dev],避免拉太多开发依赖)
RUN cd /workspace/neurad-studio && \
    pip install --no-build-isolation -e . -c /tmp/constraints.txt

# ============================================================
# 项目层 3: splatad (gsplat fork)
# ============================================================
COPY splatad /workspace/splatad
RUN cd /workspace/splatad &&
RUN cd /workspace/splatad && \
    BUILD_NO_CUDA=1 pip install --no-build-isolation -e .[dev] -c /tmp/constraints.txt

# ============================================================
# 项目层 4: gsplat_original
# ============================================================
# 官方 gsplat_original：安装到单独的 target，再重命名为 gsplat_original.gsplat
ARG GUT_TARGET=/opt/gsplat_original_pkg
COPY gsplat_original /workspace/gsplat_original

RUN cd /workspace/gsplat_original && \
    BUILD_NO_CUDA=1 pip install --no-build-isolation \
        --target ${GUT_TARGET} \
        . -c /tmp/constraints.txt && \
    mkdir -p ${GUT_TARGET}/gsplat_original && \
    mv ${GUT_TARGET}/gsplat ${GUT_TARGET}/gsplat_original/gsplat && \
    touch ${GUT_TARGET}/gsplat_original/__init__.py && \
    rm -rf ${GUT_TARGET}/gsplat-*.dist-info

ENV PYTHONPATH="${GUT_TARGET}:${PYTHONPATH}"

# viser 预热缓存
RUN python -c "import viser; viser.ViserServer()" 2>/dev/null || echo "Viser warmup skipped"

# ============================================================
# 最后安装 fused-ssim
# ============================================================
RUN pip uninstall -y fused-ssim fused-ssim-cuda 2>/dev/null || true
RUN pip install --no-build-isolation -v \
    git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157

# ============================================================
# 验证所有关键组件
# ============================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); assert '2.1.2' in torch.__version__"
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}'); assert numpy.__version__.startswith('1.24')"
RUN python -c "import scipy; print(f'SciPy: {scipy.__version__}'); assert scipy.__version__.startswith('1.11')"
RUN python -c "import kaolin; print('✓ Kaolin 0.17.0')"
RUN python -c "from fused_ssim import fused_ssim; print('✓ fused-ssim')"
RUN python -c "from nerfstudio.models.splatad import SplatADModel; print('✓ SplatAD')"
RUN python -c "import gsplat; print('✓ gsplat (splatad fork)')"

WORKDIR /workspace

CMD ["/bin/bash"]
