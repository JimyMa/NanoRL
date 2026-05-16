"""Configuration schemas (pydantic) for NanoRL.

Mirrors fields in ``configs/qwen35_35b_grpo.yaml``. Each section maps to a
single component so the actor that needs it can take exactly its slice.
"""

from __future__ import annotations

from typing import Literal

import pydantic
import yaml


class ModelCfg(pydantic.BaseModel):
    hf_path: str
    tokenizer_path: str | None = None
    is_moe: bool = False


class OptimizerCfg(pydantic.BaseModel):
    name: Literal["adam", "sgd"] = "adam"
    lr: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0


class TrainCfg(pydantic.BaseModel):
    tp: int = 1
    pp: int = 1
    ep: int = 1
    micro_batch_size: int = 1
    global_batch_size: int = 8
    optimizer: OptimizerCfg = OptimizerCfg()
    off_policy_iters: int = 1
    seq_len: int = 4096
    bf16: bool = True
    # M3 follow-up: use Megatron-Core FSDP (ZeRO-3) instead of DDP. Requires
    # world_size > 1; per-rank we hold 1/N of each parameter and the gather
    # path triggers an all-gather before reading. With fsdp=False (the
    # default) we use DDP, which is what M1's single-rank path relies on.
    fsdp: bool = False
    # Sharding strategy when fsdp=True. "optim_grads_params" (3) is full
    # ZeRO-3; "optim_grads" (2) shards optimizer state + grads but not
    # parameters (less memory savings, simpler gather).
    fsdp_sharding_strategy: Literal[
        "no_shard", "optim", "optim_grads", "optim_grads_params"
    ] = "optim_grads_params"
    # Multi-rank launch metadata. The TrainActor expects RANK / WORLD_SIZE /
    # MASTER_ADDR / MASTER_PORT to be set in the env (torchrun / Ray); these
    # fields are only used by the spawning driver to validate the topology.
    world_size: int = 1


class InferCfg(pydantic.BaseModel):
    attention_tp: int = 1
    attention_dp: int = 1
    attention_sp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1
    ffn_dp: int = 1
    max_model_len: int = 4096
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.85
    mode: Literal["prefill", "decode", "hybrid"] = "hybrid"
    kvcache_block_size: int = 64
    loop_count: int = 1
    executor_backend: Literal["ray", "dlslime"] = "ray"
    enforce_eager: bool = False
    trust_remote_code: bool = False
    use_mega_moe: bool = False
    num_speculative_tokens: int = 0

    ray_address: str = "127.0.0.1:6379"
    master_address: str = "127.0.0.1:6006"
    nanoctrl_address: str | None = None
    nanoctrl_scope: str | None = None


class SamplingCfg(pydantic.BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    n: int = 8


class RLCfg(pydantic.BaseModel):
    algo: Literal["grpo"] = "grpo"
    group_size: int = 8
    clamp_eps_lower: float = 0.2
    clamp_eps_upper: float = 0.2
    kl_beta: float = 0.001
    entropy_weight: float = 0.0


class RayCfg(pydantic.BaseModel):
    train_bundles: int = 8
    infer_bundles: int = 8
    address: str | None = None  # auto


class DLSlimeCfg(pydantic.BaseModel):
    nanoctrl_url: str = "http://127.0.0.1:3000"
    rollout_alias_prefix: str = "rollout"
    train_alias_prefix: str = "train"
    ib_port: int = 1
    qp_num: int = 1


class NanoRLCfg(pydantic.BaseModel):
    model: ModelCfg
    train: TrainCfg = TrainCfg()
    infer: InferCfg = InferCfg()
    sampling: SamplingCfg = SamplingCfg()
    rl: RLCfg = RLCfg()
    ray: RayCfg = RayCfg()
    dlslime: DLSlimeCfg = DLSlimeCfg()
    weight_sync_every: int = 1
    num_steps: int = 100

    @classmethod
    def from_yaml(cls, path: str) -> "NanoRLCfg":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
