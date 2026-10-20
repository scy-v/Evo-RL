# Evo-RL 运行命令

按顺序执行；安装命令见 [commands.md](commands.md)。第 5 步起为后续循环模板，需先填写真实采集配置和本轮数据名称。详细约定见 [notes.md](notes.md)。

## 1. 进入环境

```bash
cd /home/suchenyu/Evo-RL
conda activate evo-rl
```

## 2. 准备 PI05 基座和 tokenizer

本机已完成，可跳过；重新准备时执行，HF 账号需有 PaliGemma 访问权限。

```bash
hf auth login
hf download lerobot/pi05_base \
  --revision a538eb273274eb30f126a118f39dbc0ee212c883 \
  --local-dir checkpoints/pretrained/pi05_base
hf download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' 'special_tokens_map.json' 'added_tokens.json' 'config.json'
```

## 3. SFT 启动前检查（CPU）

```bash
CUDA_VISIBLE_DEVICES='' python evo-rl-setup/check_stage1_readiness.py
```

## 4. 初始 PI05 SFT

四卡，每卡 batch 16，全局 64；30000 步，每 5000 步本地保存；W&B 仅上传日志。`run_01` 已存在，新实验需替换为新名称；已有实验不重复启动。

```bash
OBS_INDICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id=scylearning/insert_black_rack_into_blue_slot_20260916_v01 \
  --dataset.root=/home/suchenyu/.cache/huggingface/lerobot/scylearning/insert_black_rack_into_blue_slot_20260916_v01 \
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
  --acp.enable=false \
  --batch_size=16 \
  --num_workers=4 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --output_dir=checkpoints/sft/pi05_ur5e_insert_black_rack/run_01 \
  --job_name=pi05_ur5e_insert_black_rack_sft
```

SFT 权重：`checkpoints/sft/pi05_ur5e_insert_black_rack/run_01/checkpoints/last/pretrained_model`。

## 5. 配置本轮 rollout / 评测 / 人工纠正采集

填写 UR5e 采集配置；当前仓库没有现成 UR5e 驱动，此命令为接入后的入口模板。评测使用同一策略配置，并单独保存结果。

```bash
ROUND=round_01
PREVIOUS_REPO=scylearning/insert_black_rack_into_blue_slot_20260916_v01
ROLLOUT_CONFIG=/填写真实路径/ur5e_rollout.json
ROLLOUT_REPO=scylearning/insert_black_rack_into_blue_slot_${ROUND}_rollout
MERGED_REPO=scylearning/insert_black_rack_into_blue_slot_${ROUND}_merged
POLICY_CKPT=checkpoints/sft/pi05_ur5e_insert_black_rack/run_01/checkpoints/last/pretrained_model
ACP_INIT_CKPT="${POLICY_CKPT}"

lerobot-human-inloop-record \
  --config_path="${ROLLOUT_CONFIG}" \
  --policy.path="${POLICY_CKPT}" \
  --dataset.repo_id="${ROLLOUT_REPO}" \
  --dataset.root="/home/suchenyu/.cache/huggingface/lerobot/${ROLLOUT_REPO}" \
  --dataset.push_to_hub=false
```

## 6. 合并示范与本轮采集数据

两份数据须使用相同 schema，并有成功/失败标注。下一轮将输入列表替换为已积累数据和新采集数据。

```bash
lerobot-edit-dataset \
  --repo_id="${MERGED_REPO}" \
  --operation.type=merge \
  --operation.repo_ids="['${PREVIOUS_REPO}','${ROLLOUT_REPO}']" \
  --push_to_hub=false

lerobot-dataset-report --dataset "${MERGED_REPO}"
```

## 7. 训练 value model

按作者多卡模板，全局 batch 32；其余沿用 value 默认训练配置。此处不设置 OBS_INDICES，value 训练和推理均读取原始 state。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_value_train \
  --dataset.repo_id="${MERGED_REPO}" \
  --dataset.root="/home/suchenyu/.cache/huggingface/lerobot/${MERGED_REPO}" \
  --dataset.video_backend=torchcodec \
  --value.type=pistar06 \
  --value.device=cuda \
  --value.dtype=bfloat16 \
  --value.push_to_hub=false \
  --batch_size=8 \
  --save_freq=5000 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --output_dir="checkpoints/value/${ROUND}" \
  --job_name="value_${ROUND}"
```

## 8. 计算 value / advantage / indicator

该命令会把新字段写回本轮合并数据集。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id="${MERGED_REPO}" \
  --dataset.root="/home/suchenyu/.cache/huggingface/lerobot/${MERGED_REPO}" \
  --inference.checkpoint_path="checkpoints/value/${ROUND}" \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field="complementary_info.value_${ROUND}" \
  --acp.advantage_field="complementary_info.advantage_${ROUND}" \
  --acp.indicator_field="complementary_info.acp_indicator_${ROUND}" \
  --output_dir="checkpoints/value_infer/${ROUND}" \
  --job_name="value_infer_${ROUND}"
```

## 9. 训练 ACP 策略

从指定的 ACP_INIT_CKPT 初始化，读取上一步生成的 indicator；四卡全局 batch 64。

```bash
OBS_INDICES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id="${MERGED_REPO}" \
  --dataset.root="/home/suchenyu/.cache/huggingface/lerobot/${MERGED_REPO}" \
  --dataset.video_backend=torchcodec \
  --policy.type=pi05 \
  --policy.pretrained_path="${ACP_INIT_CKPT}" \
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
  --acp.indicator_field="complementary_info.acp_indicator_${ROUND}" \
  --acp.indicator_dropout_prob=0.3 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=30000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --output_dir="checkpoints/acp/${ROUND}" \
  --job_name="pi05_acp_${ROUND}"
```

## 10. 下一轮

先更新变量，再执行第 5 步的采集命令及第 6–9 步；不要重复第 5 步的变量初始化。

```bash
POLICY_CKPT="checkpoints/acp/${ROUND}/checkpoints/last/pretrained_model"
PREVIOUS_REPO="${MERGED_REPO}"
ROUND=round_02
ROLLOUT_REPO=scylearning/insert_black_rack_into_blue_slot_${ROUND}_rollout
MERGED_REPO=scylearning/insert_black_rack_into_blue_slot_${ROUND}_merged
```
