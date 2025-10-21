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

extract_run_ts() {
  local path="$1"
  local ts

  # 1) 先找 YYYY-MM-DD_HHMMSS（你的 /splatgut/2025-10-16_151103 就是这种）
  ts="$(grep -Eo '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}' <<<"$path" | tail -n1 || true)"
  if [[ -n "$ts" ]]; then
    printf '%s\n' "$ts"
    return 0
  fi

  # 2) 再找 MM-DD_HH-MM_YYYY（你的 /runs/10-16_15-10_2025 就是这种），统一转成 YYYY-MM-DD_HHMMSS
  if [[ "$path" =~ ([0-9]{2})-([0-9]{2})_([0-9]{2})-([0-9]{2})_([0-9]{4}) ]]; then
    #         MM          DD          HH          MM          YYYY
    local MM="${BASH_REMATCH[1]}"
    local DD="${BASH_REMATCH[2]}"
    local HH="${BASH_REMATCH[3]}"
    local MI="${BASH_REMATCH[4]}"
    local YYYY="${BASH_REMATCH[5]}"
    printf '%04d-%02d-%02d_%02d%02d00\n' "$YYYY" "$MM" "$DD" "$HH" "$MI"
    return 0
  fi

  return 1
}

RUN_TS="$(extract_run_ts "${CHECKPOINT_DIR}")" || die "Failed to extract timestamp from path: ${CHECKPOINT_DIR}"

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

  # 1) 选择那个“真的有 torch 的”Python
  choose_py() {
    for p in python3 python; do
      if command -v "$p" >/dev/null 2>&1; then
        if "$p" - <<'\''PY'\'' >/dev/null 2>&1; then
import torch, sys, inspect
print(sys.executable)
print(torch.__version__, inspect.getfile(torch))
PY
          echo "$p"
          return 0
        fi
      fi
    done
    return 1
  }
  PYBIN="$(choose_py)" || { echo "ERROR: 未找到可用的 Python(带 torch)"; exit 1; }
  echo "[env] Using Python: $("$PYBIN" -c "import sys;print(sys.executable)")"

  # 2) 把增量依赖装到可写覆盖层，并加入到 PYTHONPATH
  export PYDEPS_DIR=/workspace/renders/.pydeps
  mkdir -p "$PYDEPS_DIR" /workspace/renders/.cache/pip
  export PIP_CACHE_DIR=/workspace/renders/.cache/pip
  "$PYBIN" -m pip install --no-cache-dir --target "$PYDEPS_DIR" pyyaml kaleido
  export PYTHONPATH="/workspace/neurad-studio:$PYDEPS_DIR:${PYTHONPATH:-}"

  log "Env check"
  ls -d /workspace/neurad-studio/data/nuscenes
  ls /workspace/neurad-studio/data/nuscenes/v1.0-trainval/category.json
  ls /workspace/checkpoint/nerfstudio_models/*.ckpt | head -1

  # 3) YAML 仅做字典级解析：禁止 UnsafeLoader 去 import python 对象
log "Patch config: version=v1.0-trainval, load_dir=/workspace/checkpoint/nerfstudio_models"
"$PYBIN" - <<'PYCFG'
import yaml
from pathlib import Path

# 覆盖未知标签构造器：任何未知 tag（含 python/object 等）都当普通数据解析
class NoPythonTagLoader(yaml.SafeLoader):
    pass

def _construct_any(loader, node):
    from yaml.nodes import MappingNode, SequenceNode, ScalarNode
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    # 兜底
    return loader.construct_object(node, deep=True)

NoPythonTagLoader.construct_undefined = _construct_any  # ← 关键一行

cfg_path = Path("/workspace/checkpoint/config.yml")
raw = cfg_path.read_text()
cfg = yaml.load(raw, Loader=NoPythonTagLoader)

# 兼容对象/字典两种结构
def getv(obj, key, default=None):
    try:
        return getattr(obj, key)
    except Exception:
        try:
            return obj.get(key, default)
        except Exception:
            return default

def setv(obj, key, value):
    try:
        setattr(obj, key, value)
    except Exception:
        if isinstance(obj, dict):
            obj[key] = value
        else:
            raise

pipeline    = getv(cfg, "pipeline", {})
datamanager = getv(pipeline, "datamanager", {})
dataparser  = getv(datamanager, "dataparser", {})

old_ver = getv(dataparser, "version", None)
setv(dataparser, "version", "v1.0-trainval")

old_ld = getv(cfg, "load_dir", None)
# 用字符串，避免再次序列化成 Python 对象标签
if isinstance(cfg, dict):
    cfg["load_dir"] = "/workspace/checkpoint/nerfstudio_models"
else:
    setattr(cfg, "load_dir", "/workspace/checkpoint/nerfstudio_models")

print("dataparser.version:", old_ver, "->", getv(dataparser, "version"))
print("load_dir:", old_ld, "->", getv(cfg, "load_dir"))

out = Path("/tmp/config_fixed.yml")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Wrote:", out)
PYCFG

  log "Sanity"
  ls /workspace/checkpoint/nerfstudio_models/step-*.ckpt | head -3 || true
  ls /workspace/neurad-studio/data/nuscenes/v1.0-trainval/category.json

  log "Render"
  "$PYBIN" nerfstudio/scripts/render.py dataset \
    --load-config /tmp/config_fixed.yml \
    --output-path /workspace/renders \
    --rendered-output-names rgb \
    --render-point-clouds True

  log "Done"
'

log "Results: ${RENDER_OUTPUT_DIR}"
