# Evo-RL 复现总结

## 1. PI05 全参 SFT 耗时

在4张A800、每卡batch 16（总64）、动作长度50的设置下，Evo-RL使用LeRobot的PyTorch PI05，目前比本机OpenPI历史训练的常见速度快约1.6倍。

| 实现 | 每步耗时 | 30000步耗时 |
|---|---:|---:|
| Evo-RL | 约1.65秒 | 预计约14小时，含首次编译 |
| OpenPI常见组（8次） | 平均2.63秒 | 实测平均21小时55分 |
| OpenPI较慢组（4次） | 平均3.94秒 | 实测平均32小时51分 |

Evo-RL开启`torch.compile(max-autotune)`，OpenPI也使用`jax.jit`，尚不能将速度差异单独归因于编译。
