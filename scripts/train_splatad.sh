#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time=unlimited
#SBATCH --output=/staging/fisheye/mthesis/run_logs/splatad/%A_%a.out

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# ==================== 路径配置 ====================
NUSCENES_DATA_DIR="/datasets/nuscenes/v1.0"
SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/splatad.sif"
ORIGINAL_CODE_DIR="/workspaces/s0002322/src/neurad-studio"

# ==================== 验证源代码目录 ====================
if [ ! -d "$ORIGINAL_CODE_DIR" ]; then
    echo "ERROR: Source code directory does not exist: $ORIGINAL_CODE_DIR"
    exit 1
fi

echo "Source code: $ORIGINAL_CODE_DIR"
ls -la "$ORIGINAL_CODE_DIR" | head -5

# ==================== Git 信息 ====================
cd "$ORIGINAL_CODE_DIR" || exit
git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
cd - > /dev/null || exit

# ==================== 创建临时代码目录 ====================
experiment_name="splatad_nuscenes"
model_name="$(date +"%m-%d_%H-%M_%Y")_${experiment_name}_${git_commit}"

mkdir -p /staging/fisheye/mthesis/splatad/code

TEMP_CODE_DIR=$(mktemp -d "/staging/fisheye/mthesis/splatad/code/${git_commit}_${SLURM_JOB_ID}_XXXXXX")

if [ ! -d "$TEMP_CODE_DIR" ]; then
    echo "ERROR: Failed to create temp directory"
    exit 1
fi

echo "Copying code to: ${TEMP_CODE_DIR}"
cp -r "$ORIGINAL_CODE_DIR"/* "$TEMP_CODE_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy code"
    exit 1
fi

echo "Code copied successfully"

# 输出目录
out_dir="/staging/fisheye/mthesis/splatad/runs/${model_name}"

echo "Git commit: ${git_commit}"
echo "Temp code: ${TEMP_CODE_DIR}"
echo "Output: ${out_dir}"

# ==================== 创建输出目录 ====================
mkdir -p "$out_dir"
mkdir -p /staging/fisheye/mthesis/run_logs/splatad

# ==================== 运行 Singularity ====================
echo "Starting Singularity container..."

singularity exec --nv \
    --bind "${TEMP_CODE_DIR}":/workspace/neurad-studio \
    --bind "${NUSCENES_DATA_DIR}":/workspace/neurad-studio/data/nuscenes:ro \
    --bind "${out_dir}":/workspace/outputs \
    --pwd /workspace/neurad-studio \
    "$SINGULARITY_IMAGE" \
    bash -c "
    export TORCHDYNAMO_DISABLE=1
    export TORCH_COMPILE_DISABLE=1
    
    echo '=== Starting Training ==='
    python nerfstudio/scripts/train.py splatad \
        --output-dir /workspace/outputs \
        --experiment-name reproduce-adsplat-config \
        --vis tensorboard \
        --viewer.quit-on-train-completion True \
        --pipeline.model.init-opacities 0.005 \
        --pipeline.model.mcmc-min-opacity 0.005 \
        --pipeline.datamanager.cache-images cpu \
        --pipeline.datamanager.cache-lidars gpu \
        nuscenes-data \
        --data data/nuscenes \
        --sequence scene-0104
    "


exit_code=$?

echo ""
echo "Job finished at $(date) with exit code $exit_code"
echo "Temp code: ${TEMP_CODE_DIR}"
echo "Results: ${out_dir}"
echo ""
echo "To view tensorboard:"
echo "  tensorboard --logdir ${out_dir}"

exit $exit_code


