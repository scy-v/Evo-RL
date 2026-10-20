# 配置与运行约定

## 保留文件

- `commands.md`：普通 clone 和环境安装命令。
- `run_commands.md`：环境、模型准备、SFT、rollout、数据合并、value、advantage、ACP、下一轮的分步命令。
- `check_stage1_readiness.py`：唯一的 SFT CPU 启动前检查，不启动训练，不使用 GPU，结果只输出到终端。
- `requirements-installed.txt`、`conda-explicit.txt`：环境版本快照。
- `notes.md`：配置依据、数据约定及验证边界。

## 环境

仓库：`/home/suchenyu/Evo-RL`，原始提交 `ec9842d8a8dd0371b4b50021547552e4608ec5ee`。
Conda 环境：`/home/suchenyu/miniconda3/envs/evo-rl`；uv 只是向该环境安装依赖，没有另建 .venv。

Python 3.10，torch 2.7.1、torchvision 0.22.1、torchcodec 0.5、FFmpeg 7。
PI extra 使用仓库要求的定制 Transformers 4.53.3，分支 `fix/lerobot_openpi`，提交 `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`。
安装支持 CUDA 的包不等于使用 GPU；本次仅执行 CPU 检查，未启动任何训练。

`checkpoints` 是 `/data/suchenyu/evo-rl/checkpoints` 的软链接，模型实际保存在数据盘。
官方 PI05 基座固定 revision `a538eb273274eb30f126a118f39dbc0ee212c883`，与当前源码处理器兼容。官方 PaliGemma tokenizer 已缓存；重新下载需要有权限的 HF 登录。

## SFT 设置

核对基准是仓库 `docs/source/pi05.mdx` 的 SFT 示例；主 README 第 6 步是后续 ACP，不是初始普通 SFT。

核心设置与 SFT 示例及默认值一致：bfloat16、全参、编译开启、梯度检查点开启、冻结视觉关闭、仅训练 expert 关闭、PEFT 关闭、ACP/RA-BC/RTC 关闭；默认图像处理、tokenizer、QUANTILES 归一化、chunk=50、内部 state/action 上限 32、token 长度 200 均未改动。
优化器沿用默认 AdamW：lr=2.5e-5、betas=(0.9,0.95)、eps=1e-8、weight_decay=0.01、梯度裁剪=1.0；默认 warmup=1000、decay=30000、末端 lr=2.5e-6。总步数小于 30000 时仓库会自动缩放调度。

用户指定的适配：4 卡、每卡 batch 16（全局 64）、30000 步、每 5000 步保存、本机数据与输出地址。W&B 只上传日志：enable=true、disable_artifact=true；HF 模型上传关闭，模型仅在本地保存。

## 离线选维度

唯一新增功能在 DatasetConfig、datasets/factory.py、datasets/state_selection.py；PI05 模型、处理器、训练循环和依赖声明保持原版，旧 JAX 迁移适配已全部撤销。

`OBS_INDICES` 为从 1 开始的索引，按参数顺序选列，支持重排；也可用零基 `dataset.state_indices`，显式配置优先。数据加载完成后通过一次 HF map 得到整张新表，替换训练使用的表；取样和采 batch 时不再筛选。状态名称、维度和对应统计同步选择，count、动作、图像和原始数据文件不改。该功能要求 streaming=false。

原始数据有 48 维 state，当前 PI05 使用第 1–19 维；200 个 episode、67740 帧、15 Hz、两路图像、7 维动作。原有末端增量动作语义不变，未增加额外 delta transform。
推理与在线调用由调用方提供已准备好的 19 维输入，不通过 OBS_INDICES 筛选。

## 后续循环模板

run_commands 第 5 步起尚未进行机器人或 GPU 实测。当前仓库未包含可直接使用的 UR5e 驱动，ROLLOUT_CONFIG 不是已存在的配置文件，需要接入真实硬件、相机与遥操作；不能拿 SO 系列示例冒充 UR5e 配置。采集文件应和示范数据 schema 一致，且必须正确标记 episode_success；推理输入仍需由采集适配代码准备。

合并数据保留原始 schema，不落盘选维。value 命令按作者默认读取原始 state，训练和推理保持一致；只在 PI05 SFT/ACP 命令前内联 OBS_INDICES，不要全局 export 给 value 阶段。若后续也要求 value 只用选定维度，必须先同时适配其训练与推理入口，不能仅在训练时筛选。

value 默认使用 google/siglip-so400m-patch14-384 和 google/gemma-3-270m，首次运行会下载；Gemma 的访问权限与 PaliGemma 需要分别确认。value/ACP 需要带成功和失败标注的有效数据，未标注不能直接作为正常失败数据继续训练。

value inference 会把 value/advantage/indicator 写回本轮合并数据集。每轮名称、字段 tag 和输出目录独立；下轮合并上一轮累积数据与新 rollout，不能只用原示范和最新一轮而漏掉历史数据。

POLICY_CKPT 是 rollout 使用的当前策略；ACP_INIT_CKPT 是训练初始化来源，二者独立。模板初始设为 SFT checkpoint，之后默认保持该初始化；如需采用其他研究协议应显式调整。原 π*0.6/RECAP 的每轮预训练初始化规则不等于自动继承当前 rollout 策略；代码不会自动完成任何阶段切换。

## 已验证范围

SFT CPU 检查：整表 map 67740 行、实际样本归一化与反归一化、官方 tokenizer、基座权重名称与形状、梯度检查点设置、全参数可训练、与仓库 SFT 示例逐字段核对均通过。相关回归测试此前为 52 passed、2 skipped，跳过项依赖另一份未提供的数据集。

这不等于真实 GPU 训练通过：未实测 torch.compile 的 GPU 编译、四卡通信、每卡 batch 16 的显存容量、W&B 在线连接或机器人任务成功率。

当前 run_01 目录已存在，保留不动。CPU 检查会继续只读核对配置和数据，但以非零退出码提醒新实验更换输出目录；不会绕过训练程序原有的防覆盖检查。
