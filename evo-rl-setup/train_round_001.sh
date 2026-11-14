#!/usr/bin/env bash
# Run the formal round_001 stages 9 -> 10 -> 11. Never auto-delete existing runs.
set -Eeuo pipefail

DRY_RUN=false
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=true ;;
  *) echo "Usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/suchenyu/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2
unset OBS_INDICES

STAGE=preflight
trap 'rc=$?; echo "[$(date -Is)] FAILED: $STAGE (exit $rc); subsequent stages were not started." >&2; exit "$rc"' ERR

if ! "$DRY_RUN"; then
  for dir in checkpoints/value/round_001 checkpoints/acp/round_001; do
    if [[ -e "$dir" || -L "$dir" ]]; then
      echo "Output already exists: $dir. Resolve it before starting a fresh run." >&2
      exit 1
    fi
  done
  test -f data/lerobot/insert_black_rack_into_blue_slot/training_round_001/meta/info.json
  test -f checkpoints/pretrained/pi05_base/model.safetensors
  mkdir -p "$REPO_ROOT/outputs"
  LOG_DIR="$(mktemp -d "$REPO_ROOT/outputs/round_001_pipeline_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
  echo "Logs: $LOG_DIR"
fi

run_stage() {
  STAGE="$1"
  shift
  if "$DRY_RUN"; then
    printf '\n[%s] ' "$STAGE"
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  echo "[$(date -Is)] START: $STAGE"
  "$@" 2>&1 | tee "$LOG_DIR/$STAGE.log"
  echo "[$(date -Is)] FINISHED: $STAGE"
}

check_step() {
  if "$DRY_RUN"; then return; fi
  python - "$1" "$2" <<'CHECKPOINT'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "checkpoints/last"
assert json.loads((p / "training_state/training_step.json").read_text())["step"] == int(sys.argv[2])
assert (p / "pretrained_model/model.safetensors").is_file()
CHECKPOINT
}

run_stage 09_value accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=insert_black_rack_into_blue_slot/training_round_001 \
  --dataset.root=data/lerobot/insert_black_rack_into_blue_slot/training_round_001 \
  --dataset.video_backend=torchcodec \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.push_to_hub=false \
  --value.scheduler_decay_steps=10000 \
  --batch_size=16 \
  --steps=10000 \
  --save_freq=2500 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --output_dir="checkpoints/value/round_001" \
  --job_name="value_round_001"
check_step checkpoints/value/round_001 10000

run_stage 10_labels accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=insert_black_rack_into_blue_slot/training_round_001 \
  --dataset.root=data/lerobot/insert_black_rack_into_blue_slot/training_round_001 \
  --inference.checkpoint_path="checkpoints/value/round_001" \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field="complementary_info.value_round_001" \
  --acp.advantage_field="complementary_info.advantage_round_001" \
  --acp.indicator_field="complementary_info.acp_indicator_round_001" \
  --output_dir="checkpoints/value_infer/round_001" \
  --job_name="value_infer_round_001"

export OBS_INDICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
run_stage 11_acp accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id=insert_black_rack_into_blue_slot/training_round_001 \
  --dataset.root=data/lerobot/insert_black_rack_into_blue_slot/training_round_001 \
  --dataset.video_backend=torchcodec \
  --policy.type=pi05 \
  --policy.pretrained_path=checkpoints/pretrained/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.push_to_hub=false \
  --acp.enable=true \
  --acp.indicator_field="complementary_info.acp_indicator_round_001" \
  --acp.indicator_dropout_prob=0.3 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --output_dir="checkpoints/acp/round_001" \
  --job_name="pi05_acp_round_001"
check_step checkpoints/acp/round_001 30000

if ! "$DRY_RUN"; then
  echo "[$(date -Is)] All stages completed. ACP: checkpoints/acp/round_001/checkpoints/last/pretrained_model"
fi
