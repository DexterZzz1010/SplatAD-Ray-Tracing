#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time=unlimited
#SBATCH --output=/staging/fisheye/mthesis/run_logs/splatad/%A_%a.out

set -euo pipefail  # 失败就停,别继续瞎跑

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# ==================== 路径配置 ====================
NUSCENES_DATA_DIR="/datasets/nuscenes/v1.0"
SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/splatad.sif"
ORIGINAL_CODE_DIR="/workspaces/s0002322/src/neurad-studio"

# ==================== 验证 ====================
[ -d "$ORIGINAL_CODE_DIR" ] || { echo "ERROR: Code dir not found"; exit 1; }
[ -d "$NUSCENES_DATA_DIR" ] || { echo "ERROR: Data dir not found"; exit 1; }
[ -f "$SINGULARITY_IMAGE" ] || { echo "ERROR: Singularity image not found"; exit 1; }

# ==================== Git 信息 ====================
cd "$ORIGINAL_CODE_DIR"
git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
cd - > /dev/null

# ==================== 创建临时目录 ====================
experiment_name="splatad_nuscenes"
model_name="$(date +"%m-%d_%H-%M_%Y")_${experiment_name}_${git_commit}"

TEMP_CODE_DIR=$(mktemp -d "/staging/fisheye/mthesis/splatad/code/${git_commit}_${SLURM_JOB_ID}_XXXXXX")
out_dir="/staging/fisheye/mthesis/splatad/runs/${model_name}"

echo "Copying code: ${ORIGINAL_CODE_DIR} -> ${TEMP_CODE_DIR}"
cp -r "$ORIGINAL_CODE_DIR"/* "$TEMP_CODE_DIR"

mkdir -p "$out_dir"
mkdir -p /staging/fisheye/mthesis/run_logs/splatad

echo "Git commit: ${git_commit}"
echo "Temp code: ${TEMP_CODE_DIR}"
echo "Output: ${out_dir}"

# ==================== 关键修复: 用HERE document ====================
# 这才是正确的多行命令传递方式!
singularity exec --nv \
    --bind "${TEMP_CODE_DIR}":/workspace/neurad-studio \
    --bind "${NUSCENES_DATA_DIR}":/workspace/neurad-studio/data/nuscenes:ro \
    --bind "${out_dir}":/workspace/outputs \
    --pwd /workspace/neurad-studio \
    "$SINGULARITY_IMAGE" \
    /bin/bash <<'EOFSCRIPT'
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Verifying Data Path ==="
ls -lh /workspace/neurad-studio/data/ || echo "Data mount failed!"

echo ""
echo "=== Starting Training ==="
python nerfstudio/scripts/train.py splatad \
    --output-dir /workspace/outputs \
    --experiment-name nuscenes-config1-optimized \
    --vis tensorboard \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 30001 \
    --steps-per-eval-image 250 \
    --steps-per-save 1000 \
    --steps-per-eval-all-images 5000 \
    --pipeline.model.init-opacities 0.005 \
    --pipeline.model.mcmc-min-opacity 0.005 \
    --pipeline.model.strategy mcmc \
    --pipeline.model.mcmc-cap-max 6000000 \
    --pipeline.model.stop-split-at 20000 \
    --pipeline.model.verbose True \
    --pipeline.datamanager.cache-images cpu \
    --pipeline.datamanager.cache-lidars gpu \
    --optimizers.means.optimizer.lr 0.0002 \
    --optimizers.means.scheduler.lr-final 0.000002 \
    --viewer.num-rays-per-chunk 65536 \
    nuscenes-data \
    --data data/nuscenes \
    --sequence scene-0103 \
    --version v1.0-trainval \
    --cameras all \
    --add-missing-points True
EOFSCRIPT

exit_code=$?

echo ""
echo "Job finished at $(date) with exit code $exit_code"
echo "Temp code: ${TEMP_CODE_DIR}"
echo "Results: ${out_dir}"
echo ""
echo "To view tensorboard:"
echo "  tensorboard --logdir ${out_dir}"

exit $exit_code