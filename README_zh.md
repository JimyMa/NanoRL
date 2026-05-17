<h1 align="center">NanoRL</h1>

<h3 align="center">面向大模型的 Ray 编排 Off-Policy 强化学习框架</h3>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_zh.md">中文</a>
</p>

NanoRL 是一个面向大模型的训推协同强化学习框架。它把
[**megatron-core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)
训练、[**NanoDeploy**](https://github.com/DeepLink-org/NanoDeploy) 推理，以及
[**DLSlime**](https://github.com/DeepLink-org/NanoDeploy) 传输连接成一个
off-policy GRPO 闭环，支持 rollout 侧 logprobs、Ray 管理的 TrainActor、
FSDP/ZeRO-3、RDMA 权重同步、验证集评测，以及 HF 格式 checkpoint 导出。

## 你可以跑什么

| 路径               | 说明                                                                                                         | 入口                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| **训练示例**       | Ray 管理 megatron-core TrainActor，消费轨迹并执行 GRPO step                                                  | `bash scripts/m1_smoke.sh`                                            |
| **推理 / Rollout** | NanoDeploy workers 生成数学轨迹、打 verifier reward，并可返回 sampled-token logprobs                         | `python -m nanorl.cli rollout-only ...` 或 `bash scripts/m2_smoke.sh` |
| **RL 训练实践**    | 完整 off-policy GRPO：rollout logprobs 作为 `old_logprobs`，FSDP 训练、权重同步、验证集评测、checkpoint 保存 | `bash scripts/m3_fsdp_smoke.sh`                                       |

第三条是当前最重要、也已经验证过的端到端路径。在仓库内置的
`nanorl_weird_algebra_v1` 代数集合上，Qwen3-4B FSDP 训练在 500-step smoke
run 中将 held-out sampled reward 从 `0.4023` 提升到 `0.5625`。数据集、命令和
注意事项见 [docs/weird_algebra_validation.md](docs/weird_algebra_validation.md)。

## 快速开始

前置条件（当前集群上通常已经启动；如果缺组件，见
[docs/install.md](docs/install.md)）：

- NanoCtrl: `http://10.102.97.179:3000`，Redis: `127.0.0.1:6379`
- Ray cluster: `10.102.97.179:7078`
- RDMA HCA 可在 `/sys/class/infiniband` 下看到
- rollout 所在 `master_address` 节点和训练侧 `TRAIN_IP` 节点有空闲 GPU。脚本会
  通过 Ray 启动两侧 actor；执行脚本的 shell 只是 driver。

```bash
pip install -e .
```

### 1. 训练示例

如果你只想确认训练侧能跑，先跑最小 TrainActor 示例：

```bash
NANORL_LOG_LEVEL=INFO STEPS=5 TRAIN_GPU=0 bash scripts/m1_smoke.sh
```

这个路径用于检查 megatron-core TrainActor、dataloader 和 GRPO step 是否正常，
适合在调 rollout 或权重同步之前做基础确认。

### 2. 推理 / Rollout

如果你想先看模型生成、verifier reward 和轨迹格式，可以跑 rollout-only：

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 1 --no-rpc
```

如果要测试分布式 rollout service 路径：

```bash
bash scripts/m2_smoke.sh
```

GRPO 配置默认会请求 rollout 侧 logprobs。调试一致性时可以加
`--no-ship-logprobs`，让 trainer 使用 `current_logprobs.detach()` 作为 old policy，
此时 ratio 会被构造成 1，便于定位问题。

### 3. RL 训练实践

这是推荐的端到端实践路径。它会启动 NanoDeploy rollout workers，把 Ray
TrainActors 放到 `TRAIN_IP`，用 FSDP/ZeRO-3 训练 Qwen3-4B，把权重同步回推理侧，
在 held-out split 上评测，并保存 HF 格式 checkpoint：

```bash
LOG_DIR=/tmp/nanorl_weird_algebra_m3_fsdp
mkdir -p "$LOG_DIR"

NANORL_LOG_LEVEL=INFO \
PROMPTS=nanorl/configs/datasets/weird_algebra_train192.jsonl \
EVAL_PROMPTS=nanorl/configs/datasets/weird_algebra_test64.jsonl \
LIMIT_PROMPTS=192 \
EVAL_LIMIT=64 \
EVAL_EVERY=25 \
STEPS=500 \
SYNC_EVERY=2 \
ROUNDS=1000 \
NPROC=8 \
TRAIN_IP=10.102.98.166 \
LOG_DIR="$LOG_DIR" \
SAVE_DIR="$LOG_DIR/checkpoints" \
SAVE_EVERY=100 \
SAVE_FINAL=1 \
bash scripts/m3_fsdp_smoke.sh
```

只想快速确认链路时，可以先降低步数：

```bash
STEPS=20 EVAL_EVERY=5 TRAIN_IP=10.102.98.166 NPROC=8 \
bash scripts/m3_fsdp_smoke.sh
```

查看 TensorBoard：

```bash
tensorboard --logdir "$LOG_DIR/m3_fsdp_tb" --port 6006 --bind_all
```

smoke 脚本会默认通过 `--tb-dir` 写 TensorBoard events。JSONL 日志仍然保存在
`$LOG_DIR` 下，方便后续排查或生成静态诊断页面。

## Rollout 侧 Logprobs

`sampling.ship_logprobs: true` 会让 NanoDeploy 返回 sampled tokens 对应的
per-token logprobs。NanoRL 会把它们存到每条 `Trajectory` 中，训练侧作为
`old_logprobs` 使用：

- `old_lp=True` 表示推理侧 logprobs 已经送达训练侧。
- `logprob_to_old=mean/max` 表示当前训练 policy 和 rollout policy 的漂移。
- `kl_to_old` 是观察 off-policy 漂移更有用的指标。旧字段 `kl` / `kl_mean` 是
  reference-model KL，除非 `rl.kl_beta > 0`，否则只是诊断项。

## Checkpoint

`nanorl train` 可以保存 HF 格式 checkpoint。trainer 由 Ray 纳管：driver 可以在
别的节点执行，Ray 会把 TrainActors 放到 `--train-ip` 指定的训练节点。

```bash
python -m nanorl.cli train ... \
  --save-dir /tmp/nanorl_ckpts/my_run \
  --save-every 50 \
  --save-final
```

每次保存会写出 `step_XXXXXX/model.safetensors`、从源 HF 目录复制 tokenizer/config
文件，并记录 `nanorl_checkpoint.json`。Ray 模式下，路径位于 Ray TrainActor
实际运行的训练节点本地。

## 验证结果

Qwen3-4B-Instruct，1 个训练 rank + 4 个 NanoDeploy workers 的权重同步耗时如下：

| 项目                 | DDP train  | 2-rank FSDP train |
| -------------------- | ---------- | ----------------- |
| Train step           | 0.30 s     | 0.30 s            |
| Weight gather        | 0 s        | 6 s               |
| Rollout 侧 RDMA pull | 1.5 s      | 1.5 s             |
| Apply                | 0.85 s     | 1.5 s             |
| **每次同步总耗时**   | **约 5 s** | **约 9 s**        |

每次同步约传输 8 GB 权重。相比之前通过 Ray RPC fan-out 的约 65 s，同步链路降到约
13 分之一。

## 测试

```bash
pytest tests/                           # 单测 + 可跳过的 RDMA loopback
pytest tests/test_grpo_loss.py          # vendored GRPO 数学与 upstream 对齐
pytest tests/test_slime_rpc_loopback.py # RDMA → NanoCtrl → Redis 轨迹回环
pytest tests/test_megatron_to_hf.py     # HF↔Megatron 名字映射 round-trip
pytest tests/test_weight_manifest.py    # 两进程 RDMA 权重传输
```

`scripts/m1_smoke.sh`、`scripts/m2_smoke.sh`、`scripts/m3_smoke.sh` 和
`scripts/m3_fsdp_smoke.sh` 是跨进程集成测试；pytest 主要覆盖数学逻辑和组件契约。

## 文档

| 文档                                                                 | 适合阅读的场景                                        |
| -------------------------------------------------------------------- | ----------------------------------------------------- |
| [docs/install.md](docs/install.md)                                   | 新机器安装或排查缺失依赖                              |
| [docs/architecture.md](docs/architecture.md)                         | 理解 Ray、NanoDeploy、megatron-core、DLSlime 如何协同 |
| [docs/cli.md](docs/cli.md)                                           | 查看 CLI 参数和示例                                   |
| [docs/rollout.md](docs/rollout.md)                                   | Rollout 配置、JSONL 格式和 smoke 输出                 |
| [docs/training.md](docs/training.md)                                 | Ray TrainActors、DDP/FSDP、权重同步和 checkpoint 保存 |
| [docs/data_plane.md](docs/data_plane.md)                             | SlimeRPC 轨迹协议和 raw-RDMA 权重传输                 |
| [docs/weird_algebra_validation.md](docs/weird_algebra_validation.md) | 固定代数数据集和已验证的 Qwen3-4B FSDP 涨点实验       |
| [docs/troubleshooting.md](docs/troubleshooting.md)                   | 已遇到过的问题和修复方式                              |

## 已知限制

- **默认 `kl_beta = 0`。** Reference KL 数学上可用，但 PyTorch gradient-mode
  SDPA 和 no_grad mode 会选择不同 kernel，Qwen3 BF16 下 per-token logprobs 会漂移。
  如需重新启用 KL loss，需要固定 deterministic SDPA backend。
- **训练侧暂不支持 TP > 1 / PP > 1。** 当前多卡训练主要通过 FSDP/ZeRO-3。
- **暂未接入 MoE Qwen3.5-35B-A3B。** 权重 gather 和 shape 假设 dense Qwen3。
- **当前只有单一 math verifier。** 还没有 `--verifier` 参数。
- **暂不支持续训恢复。** checkpoint save 用于评测/导出，还没有 optimizer/RNG restore。
- **内置 prompts 是 smoke set，不是大规模 benchmark。** 想看模型能力，需要替换成更正式的 train/eval JSONL。
