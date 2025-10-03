#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/staging/fisheye/mthesis/run_logs/splatad/render_%A.out

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

# ====== 可按需修改 ======
NUSCENES_HOST_ROOT="/datasets/nuscenes"                  # 宿主机数据根：其下应有 v1.0/（含 samples/ sweeps/ maps/ 以及 v1.0-trainval/）
SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/splatad.sif"
ORIGINAL_CODE_DIR="/workspaces/s0002322/src/neurad-studio" # 源码（容器内只读）
RENDER_ROOT="/staging/fisheye/mthesis/splatad/render"    # 所有渲染输出根目录
# ======================

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

CHECKPOINT_PATH="${1:-}"
[[ -n "${CHECKPOINT_PATH}" ]] || die "Must provide checkpoint path (.ckpt file or run directory)"

# 统一为 run 目录
if [[ "${CHECKPOINT_PATH}" == *.ckpt ]]; then
  CHECKPOINT_DIR="$(dirname "$(dirname "${CHECKPOINT_PATH}")")"
else
  CHECKPOINT_DIR="${CHECKPOINT_PATH%/}"
fi

CONFIG_FILE="${CHECKPOINT_DIR}/config.yml"
[[ -f "${CONFIG_FILE}" ]] || die "config.yml not found under ${CHECKPOINT_DIR}"
[[ -d "${ORIGINAL_CODE_DIR}" ]] || die "Source code not found: ${ORIGINAL_CODE_DIR}"
[[ -d "${NUSCENES_HOST_ROOT}/v1.0" ]] || die "${NUSCENES_HOST_ROOT}/v1.0 not found (expected dataset root)"

# 从路径中提取时间戳（固定位置：.../splatad/YYYY-MM-DD_HHMMSS/...）
RUN_TS="$(printf '%s\n' "${CHECKPOINT_DIR}" | sed -E 's#^.*/splatad/([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6})(/.*)?$#\1#')"
[[ "${RUN_TS}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}$ ]] || die "Failed to extract timestamp from path: ${CHECKPOINT_DIR}"

RENDER_OUTPUT_DIR="${RENDER_ROOT}/${RUN_TS}"
mkdir -p "${RENDER_OUTPUT_DIR}"

log "Render Job"
log "  Run dir   : ${CHECKPOINT_DIR}"
log "  Timestamp : ${RUN_TS}"
log "  Output dir: ${RENDER_OUTPUT_DIR}"

log "Singularity bind preview:"
echo "  --bind ${NUSCENES_HOST_ROOT}/v1.0 : /workspace/neurad-studio/data/nuscenes:ro"
echo "  --bind ${CHECKPOINT_DIR}          : /workspace/checkpoint:ro"
echo "  --bind ${RENDER_OUTPUT_DIR}       : /workspace/renders"
echo "  Script   : $(realpath "$0")"

singularity exec --nv \
  --bind "${ORIGINAL_CODE_DIR}":/workspace/neurad-studio:ro \
  --bind "${NUSCENES_HOST_ROOT}/v1.0":/workspace/neurad-studio/data/nuscenes:ro \
  --bind "${CHECKPOINT_DIR}":/workspace/checkpoint:ro \
  --bind "${RENDER_OUTPUT_DIR}":/workspace/renders \
  --pwd /workspace/neurad-studio \
  "${SINGULARITY_IMAGE}" \
  bash -c '
    set -Eeuo pipefail
    IFS=$'\''\n\t'\''

    log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

    export TORCHDYNAMO_DISABLE=1
    export TORCH_COMPILE_DISABLE=1

    log "Env check"
    ls -d /workspace/neurad-studio/data/nuscenes
    ls /workspace/neurad-studio/data/nuscenes/v1.0-trainval/category.json
    ls /workspace/checkpoint/nerfstudio_models/*.ckpt | head -1

    log "Patch config: version=v1.0-trainval, load_dir=/workspace/checkpoint/nerfstudio_models"
    python3 - << "PY"
import yaml
from pathlib import Path

cfg_path = Path("/workspace/checkpoint/config.yml")
raw = cfg_path.read_text()

cfg = yaml.load(raw, Loader=yaml.UnsafeLoader)  # 返回对象树
dp  = cfg.pipeline.datamanager.dataparser
old_ver = getattr(dp, "version", None)
dp.version = "v1.0-trainval"

old_ld = getattr(cfg, "load_dir", None)
cfg.load_dir = Path("/workspace/checkpoint/nerfstudio_models")

print("dataparser.version:", old_ver, "->", dp.version)
print("load_dir:", old_ld, "->", cfg.load_dir)

out = Path("/tmp/config_fixed.yml")
out.write_text(yaml.dump(cfg, default_flow_style=False))
print("Wrote:", out)
PY

    log "Sanity"
    ls /workspace/checkpoint/nerfstudio_models/step-*.ckpt | head -3 || true
    ls /workspace/neurad-studio/data/nuscenes/v1.0-trainval/category.json

    log "Render"
    python nerfstudio/scripts/render.py dataset \
      --load-config /tmp/config_fixed.yml \
      --output-path /workspace/renders \
      --rendered-output-names rgb

    log "Done"
  '

log "Results: ${RENDER_OUTPUT_DIR}"
