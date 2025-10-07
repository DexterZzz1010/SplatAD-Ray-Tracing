#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time=unlimited
#SBATCH --output=/staging/fisheye/mthesis/run_logs/splatgut/%A_%a.out

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# ==================== 路径配置 ====================
NUSCENES_DATA_DIR="/datasets/nuscenes/v1.0"
SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/splatad.sif"

# 两个独立的代码目录
NEURAD_CODE_DIR="/workspaces/s0002322/src/neurad-studio"
THREEDGUT_CODE_DIR="/workspaces/s0002322/src/3dgrut"

# ==================== 验证 ====================
[ -d "$NEURAD_CODE_DIR" ] || { echo "ERROR: neurad-studio not found"; exit 1; }
[ -d "$THREEDGUT_CODE_DIR" ] || { echo "ERROR: 3dgrut not found"; exit 1; }
[ -d "$NUSCENES_DATA_DIR" ] || { echo "ERROR: Data dir not found"; exit 1; }
[ -f "$SINGULARITY_IMAGE" ] || { echo "ERROR: Singularity image not found"; exit 1; }

# ==================== Git 信息 ====================
cd "$NEURAD_CODE_DIR"
neurad_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

cd "$THREEDGUT_CODE_DIR"
gut_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
cd - > /dev/null

# ==================== 创建临时目录 ====================
experiment_name="splatgut_nuscenes"
model_name="$(date +"%m-%d_%H-%M_%Y")_${experiment_name}_ns${neurad_commit}_gut${gut_commit}"

# 只copy neurad-studio (会被修改)
TEMP_NEURAD_DIR=$(mktemp -d "/staging/fisheye/mthesis/splatad/code/ns${neurad_commit}_${SLURM_JOB_ID}_XXXXXX")
out_dir="/staging/fisheye/mthesis/splatad/runs/${model_name}"

echo "Copying neurad-studio: ${NEURAD_CODE_DIR} -> ${TEMP_NEURAD_DIR}"
cp -r "$NEURAD_CODE_DIR"/* "$TEMP_NEURAD_DIR"

mkdir -p "$out_dir"
mkdir -p /staging/fisheye/mthesis/run_logs/splatgut

echo "Git commits:"
echo "  neurad-studio: ${neurad_commit}"
echo "  3dgrut: ${gut_commit}"
echo "Temp neurad-studio: ${TEMP_NEURAD_DIR}"
echo "3dgrut (read-only): ${THREEDGUT_CODE_DIR}"
echo "Output: ${out_dir}"

# ==================== 关键修复: 绑定两个代码目录 ====================
singularity exec --nv \
    --bind "${TEMP_NEURAD_DIR}":/workspace/neurad-studio \
    --bind "${THREEDGUT_CODE_DIR}":/workspace/3dgrut:ro \
    --bind "${NUSCENES_DATA_DIR}":/workspace/neurad-studio/data/nuscenes:ro \
    --bind "${out_dir}":/workspace/outputs \
    --pwd /workspace/neurad-studio \
    "$SINGULARITY_IMAGE" \
    /bin/bash <<'EOFSCRIPT'
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# 添加3dgrut到Python路径
export PYTHONPATH="/workspace/3dgrut:${PYTHONPATH:-}"

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Verifying 3DGUT Installation ==="
python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')
print()
try:
    import threedgut_tracer
    print('[OK] threedgut_tracer found')
    print(f'     Location: {threedgut_tracer.__file__}')
except ImportError as e:
    print(f'[ERROR] threedgut_tracer not found: {e}')
    print('[WARN] Will fallback to rasterization')
"

echo ""
echo "=== Verifying Data Path ==="
ls -lh /workspace/neurad-studio/data/ || echo "Data mount failed!"

echo ""
echo "=== Starting SplatGUT Training ==="
python nerfstudio/scripts/train.py splatgut \
    --output-dir /workspace/outputs \
    --experiment-name nuscenes-gut-raytracing \
    --vis tensorboard \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 30001 \
    --steps-per-eval-image 250 \
    --steps-per-save 1000 \
    --steps-per-eval-all-images 5000 \
    \
    `# === 3DGUT 渲染设置 ===` \
    --pipeline.model.use-ray-tracing True \
    --pipeline.model.camera-model pinhole \
    --pipeline.model.k-buffer-size 32 \
    --pipeline.model.ut-alpha 1.0 \
    --pipeline.model.ut-beta 0.0 \
    \
    `# === SplatAD 优化设置 (继承) ===` \
    --pipeline.model.init-opacities 0.005 \
    --pipeline.model.mcmc-min-opacity 0.005 \
    --pipeline.model.strategy mcmc \
    --pipeline.model.mcmc-cap-max 6000000 \
    --pipeline.model.stop-split-at 20000 \
    --pipeline.model.verbose True \
    \
    `# === Datamanager 设置 ===` \
    --pipeline.datamanager.cache-images cpu \
    --pipeline.datamanager.cache-lidars gpu \
    \
    `# === Optimizer 设置 ===` \
    --optimizers.means.optimizer.lr 0.0002 \
    --optimizers.means.scheduler.lr-final 0.000002 \
    \
    `# === Viewer 设置 ===` \
    --viewer.num-rays-per-chunk 65536 \
    \
    `# === Dataset 设置 ===` \
    nuscenes-data \
    --data data/nuscenes \
    --sequence scene-0103 \
    --version v1.0-trainval \
    --cameras all \
    --add-missing-points True

echo ""
echo "=== Training completed ==="
EOFSCRIPT

exit_code=$?

echo ""
echo "Job finished at $(date) with exit code $exit_code"
echo "Temp neurad-studio: ${TEMP_NEURAD_DIR}"
echo "3dgrut (read-only): ${THREEDGUT_CODE_DIR}"
echo "Results: ${out_dir}"
echo ""
echo "To view tensorboard:"
echo "  tensorboard --logdir ${out_dir}"

exit $exit_code