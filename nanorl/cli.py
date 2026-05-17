"""NanoRL command-line entrypoint.

Subcommands:

    nanorl rollout-only --cfg=... --prompts=PATH [--no-rpc] [--save-jsonl=PATH]
        Stand up RolloutEngine + TrajectoryService. Reads a JSONL of
        ``{"prompt": ..., "reference": ..., "group_id": ...}``; runs ``rounds``
        rollouts; serves them over SlimeRPC for any consumer to pull.

    nanorl train-only   --cfg=... --steps=N
        (placeholder) Drive a megatron-core TrainActor; pulls trajectories
        from a SlimeRPC producer and runs N GRPO steps. Lands with M1.

    nanorl train        --cfg=... --steps=N
        (placeholder) The full M3 loop. Lands after M1 + M2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("nanorl")


def _parse_prompts(path: str, limit: int | None = None) -> list:
    """Read a prompt set. Accepts NanoRL-native JSONL, DAPO chat-format
    JSONL, or a bundled name (``sample``, ``dapo``, ``aime``)."""
    from nanorl.data.datasets import load_prompts

    return load_prompts(path, limit=limit)


def _open_jsonl_writer(path: str | None):
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def _ray_env_vars(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment variables to forward into Ray actors."""
    env = {
        "NANORL_LOG_LEVEL": os.environ.get("NANORL_LOG_LEVEL", "INFO"),
    }
    for key, value in os.environ.items():
        if key.startswith("NANORL_DEBUG_"):
            env[key] = value
    if extra:
        env.update(extra)
    return env


def _checkpoint_path(save_dir: str, step: int) -> str:
    return str(Path(save_dir) / f"step_{step:06d}")


def _trajectory_to_jsonl_row(t, tokenizer=None) -> dict:
    """Pack a Trajectory for human inspection.

    With ``tokenizer`` provided, also decode prompt + response into text —
    which is what an operator actually wants to read. Token IDs are kept
    so the dump is round-trippable.
    """
    row = {
        "group_id": t.group_id,
        "reward": t.reward,
        "eos": t.eos,
        "reference": (t.meta or {}).get("reference", ""),
        "prompt_ids": t.prompt_ids,
        "response_ids": t.response_ids,
        "response_len": len(t.response_ids),
        "meta": t.meta,
    }
    if tokenizer is not None:
        row["prompt"] = tokenizer.decode(t.prompt_ids, skip_special_tokens=True)
        row["response"] = tokenizer.decode(t.response_ids, skip_special_tokens=True)
    return row


def _log_traj_samples(trajs, tokenizer, n=2, prompt_tail=200, response_head=800):
    """Pretty-print a couple of decoded trajectories so the operator can
    spot-check what the model is producing without grepping the dump file.

    Picks one positive-reward and one zero-reward sample if both exist,
    otherwise the first ``n`` rows. No-op if no tokenizer is available.
    """
    if tokenizer is None or not trajs:
        return
    pos = next((t for t in trajs if t.reward > 0), None)
    zero = next((t for t in trajs if t.reward == 0), None)
    chosen = [t for t in (pos, zero) if t is not None][:n] or list(trajs[:n])
    for t in chosen:
        prompt = tokenizer.decode(t.prompt_ids[-prompt_tail:], skip_special_tokens=True)
        resp = tokenizer.decode(
            t.response_ids[:response_head], skip_special_tokens=True
        )
        ref = (t.meta or {}).get("reference", "")
        logger.info(
            "[traj sample] group=%d reward=%.3f ref=%r len=%d eos=%s\n"
            "  PROMPT(tail): %s\n  RESPONSE(head): %s",
            t.group_id,
            t.reward,
            ref,
            len(t.response_ids),
            t.eos,
            prompt,
            resp,
        )


def _install_sigint_handler():
    """Translate Ctrl-C into a clean KeyboardInterrupt at the top level."""

    def _handle(signum, frame):
        logger.warning("caught signal %d, exiting", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _rollout_only(args: argparse.Namespace) -> int:
    """Stand up rollout engine + TrajectoryService; serve until killed."""
    from nanorl.actors.rollout import RolloutEngine
    from nanorl.config import NanoRLCfg
    from nanorl.data.trajectory_buffer import TrajectoryService
    from nanorl.rl.reward import MathVerifier

    _install_sigint_handler()
    cfg_path = str(Path(args.cfg).resolve())
    cfg = NanoRLCfg.from_yaml(cfg_path)

    if args.seed is not None:
        # Best-effort determinism; downstream samplers may still be stochastic.
        import random

        random.seed(args.seed)
        try:
            import numpy as np

            np.random.seed(args.seed)
        except ImportError:
            pass

    if args.top_p is not None:
        cfg.sampling.top_p = args.top_p
    if args.max_new_tokens is not None:
        cfg.sampling.max_new_tokens = args.max_new_tokens
    if args.ship_logprobs is not None:
        cfg.sampling.ship_logprobs = args.ship_logprobs

    items = _parse_prompts(args.prompts, limit=args.limit_prompts or None)
    if not items:
        logger.error("no valid prompts in %s", args.prompts)
        return 2
    logger.info(
        "loaded %d prompts (group_size=%d ship_logprobs=%s)",
        len(items),
        cfg.sampling.n,
        cfg.sampling.ship_logprobs,
    )

    if args.dry_run:
        logger.info("dry-run: config + prompts validated; not starting NanoInfra")
        return 0

    service = TrajectoryService(capacity=cfg.rl.group_size * len(items) * 4)
    verifier = MathVerifier()
    engine = RolloutEngine(cfg.model, cfg.infer, cfg.sampling, verifier, service)

    agent = None
    if not args.no_rpc:
        from dlslime import PeerAgent

        from nanorl.data.trajectory_buffer import run_rpc_server
        from nanorl.weights.transport import WeightTransportRollout

        consumer_alias = args.consumer_alias or f"{cfg.dlslime.train_alias_prefix}:0"
        producer_alias = args.producer_alias or f"{cfg.dlslime.rollout_alias_prefix}:0"
        agent = PeerAgent(nanoctrl_url=cfg.dlslime.nanoctrl_url, alias=producer_alias)
        # Let the trainer initiate the DLSlime connection. If the producer
        # sends qp_ready before the trainer alias has registered its stream,
        # NanoCtrl can drop that early readiness notification; the producer
        # then thinks it is connected while trainer rank 0 waits forever.
        # run_rpc_server waits for the consumer-initiated endpoint before
        # binding SlimeRPC.
        # M3: hook up the weight-update path. The transport needs the
        # *trainable* peer's alias (the consumer here doubles as the
        # train side) so `apply_weight_update` can pull from it.
        weight_transport = WeightTransportRollout(agent, train_alias=consumer_alias)
        # Reach into the engine's underlying LLM (RolloutEngine wraps it).
        # Setting the attribute via a small accessor keeps the engine
        # encapsulation intact for unit tests.
        service.attach_weight_path(engine.llm, weight_transport)
        run_rpc_server(
            agent,
            service,
            consumer_alias=consumer_alias,
        )
        logger.info(
            "rollout-only: rpc server pending producer=%s expecting consumer=%s "
            "(weight-sync RPC enabled)",
            producer_alias,
            consumer_alias,
        )

    jsonl_writer = _open_jsonl_writer(args.save_jsonl)
    # Tokenizer for the JSONL dump, the periodic in-log samples, AND the
    # Redis sink (which ships decoded text, not just ids).
    dump_tokenizer = (
        getattr(engine, "_tokenizer", None)
        if (jsonl_writer or args.print_traj_every or args.redis_url)
        else None
    )

    redis_sink = None
    if args.redis_url:
        try:
            from nanorl.data.redis_sink import RedisTrajectorySink

            redis_sink = RedisTrajectorySink(
                args.redis_url,
                key=args.redis_key,
                maxlen=args.redis_maxlen,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis sink disabled: %s", exc)
            redis_sink = None

    eval_items: list = []
    eval_writer = None
    if args.eval_prompts:
        from nanorl.data.datasets import load_prompts as _load_prompts

        eval_items = _load_prompts(
            args.eval_prompts, limit=args.eval_limit_prompts or None
        )
        if not eval_items:
            logger.warning(
                "eval prompt set %r is empty; eval pass disabled", args.eval_prompts
            )
        else:
            eval_writer = _open_jsonl_writer(args.eval_jsonl)
            logger.info(
                "held-out eval enabled: %d prompts, every %d rollout rounds",
                len(eval_items),
                args.eval_every,
            )

    try:
        for r in range(args.rounds):
            trajs, stats = engine.run(items, publish=not args.no_rpc)
            stats.log()
            logger.info("round=%d buffered=%d", r, service.buffered())
            if jsonl_writer is not None:
                for t in trajs:
                    jsonl_writer.write(
                        json.dumps(_trajectory_to_jsonl_row(t, dump_tokenizer)) + "\n"
                    )
            if redis_sink is not None:
                for t in trajs:
                    redis_sink.push(_trajectory_to_jsonl_row(t, dump_tokenizer))
            if args.print_traj_every and r % args.print_traj_every == 0:
                _log_traj_samples(trajs, dump_tokenizer, n=args.print_traj_n)
            if eval_items and r % args.eval_every == 0:
                # Held-out pass: same engine, separate prompt list, NOT
                # published to SlimeRPC (so train doesn't consume it).
                # Reuses the engine's sampling config — for a deterministic
                # measurement set the producer's config to low-temperature
                # before launch, or invoke ``nanorl eval`` separately.
                ev_trajs, ev_stats = engine.run(eval_items, publish=False)
                ev_rewards = [t.reward for t in ev_trajs]
                ev_mean = sum(ev_rewards) / max(1, len(ev_rewards))
                ev_pos = sum(1 for x in ev_rewards if x > 0)
                logger.info(
                    "[eval] round=%d n=%d mean_reward=%.4f pos=%d/%d",
                    r,
                    len(ev_trajs),
                    ev_mean,
                    ev_pos,
                    len(ev_rewards),
                )
                if eval_writer is not None:
                    eval_writer.write(
                        json.dumps(
                            {
                                "round": r,
                                "n": len(ev_trajs),
                                "mean_reward": ev_mean,
                                "pos_count": ev_pos,
                                "rewards": ev_rewards,
                            }
                        )
                        + "\n"
                    )
            if (
                not args.no_rpc
                and args.stop_after_buffered
                and service.buffered() >= args.stop_after_buffered
            ):
                logger.info(
                    "stopping after buffering %d trajectories", service.buffered()
                )
                break

        if args.serve_forever and agent is not None:
            logger.info("rollout-only: idle, serving published trajectories")
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        logger.info("interrupted; cleaning up")
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()
        if eval_writer is not None:
            eval_writer.close()
        if redis_sink is not None:
            redis_sink.close()
        if agent is not None:
            agent.shutdown()
    return 0


def _train_only(args: argparse.Namespace) -> int:
    """Drive a single-rank megatron-core TrainActor against a SlimeRPC source."""
    from nanorl.actors.train import TrainActor
    from nanorl.config import NanoRLCfg

    _install_sigint_handler()
    cfg = NanoRLCfg.from_yaml(args.cfg)

    if args.dry_run:
        # Just exercise config + model build path, no NanoCtrl, no NanoInfra.
        logger.info("dry-run: train-only config validated; not building TrainActor")
        return 0

    actor = TrainActor(
        cfg,
        producer_alias=args.producer_alias,
        consumer_alias=args.consumer_alias,
        master_port=args.master_port,
    )

    log_path = args.log_jsonl
    log_writer = _open_jsonl_writer(log_path)

    try:
        for s in range(args.steps):
            stats = actor.train_step()
            logger.info(
                "step=%d loss=%.4f mean_reward=%.3f mean_adv=%.3f tokens=%d elapsed=%.2fs",
                stats.step,
                stats.loss,
                stats.mean_reward,
                stats.mean_advantage,
                stats.response_tokens,
                stats.elapsed_s,
            )
            if log_writer is not None:
                log_writer.write(json.dumps(asdict(stats)) + "\n")
            if not (stats.loss == stats.loss):  # NaN check
                logger.error("loss is NaN at step %d; aborting", stats.step)
                return 3
    except KeyboardInterrupt:
        logger.info("interrupted; cleaning up")
    finally:
        if log_writer is not None:
            log_writer.close()
        actor.close()
    return 0


def _train(args: argparse.Namespace) -> int:
    """Full M3 loop: train_step + periodic gather_and_publish to a running rollout."""
    from nanorl.actors.train import TrainActor
    from nanorl.config import NanoRLCfg
    from nanorl.metrics import build_logger

    _install_sigint_handler()
    cfg_path = str(Path(args.cfg).resolve())
    cfg = NanoRLCfg.from_yaml(cfg_path)

    if args.weight_sync_every is not None:
        cfg.weight_sync_every = args.weight_sync_every

    if args.dry_run:
        logger.info("dry-run: train config validated; not building TrainActor")
        return 0

    # Metrics: rank 0 owns the sinks. Other ranks no-op to avoid duplicated
    # wandb runs / TB writes.
    is_rank_0 = int(os.environ.get("RANK", "0")) == 0
    metrics_logger = build_logger(
        jsonl_path=args.log_jsonl if is_rank_0 else None,
        wandb_project=getattr(args, "wandb_project", None) if is_rank_0 else None,
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_config={
            "cfg": args.cfg,
            "steps": args.steps,
            "weight_sync_every": cfg.weight_sync_every,
        },
        tb_dir=getattr(args, "tb_dir", None) if is_rank_0 else None,
    )

    actor = TrainActor(
        cfg,
        producer_alias=args.producer_alias,
        consumer_alias=args.consumer_alias,
        master_port=args.master_port,
    )

    try:
        for s in range(args.steps):
            stats = actor.train_step()
            logger.info(
                "step=%d loss=%.4f kl=%.4f kl_to_old=%.4f ratios=%.3f "
                "trunc_a=%.3f trunc_b=%.3f H=%.3f old_lp=%s "
                "logprob_to_old=%.3g/%.3g "
                "gnorm=%.3f reward=%.3f "
                "tokens=%d resp_len=%.0f elapsed=%.2fs",
                stats.step,
                stats.loss,
                stats.kl_mean,
                stats.kl_to_old,
                stats.ratios_mean,
                stats.truncated_above_rate,
                stats.truncated_below_rate,
                stats.entropy_mean,
                stats.old_logprobs_present,
                stats.logprob_to_old_mean,
                stats.logprob_to_old_max,
                stats.grad_norm,
                stats.mean_reward,
                stats.response_tokens,
                stats.response_length_mean,
                stats.elapsed_s,
            )
            metrics_logger.log_step(stats.step, asdict(stats))

            if cfg.weight_sync_every and (stats.step + 1) % cfg.weight_sync_every == 0:
                logger.info("triggering weight sync v=%d", stats.step + 1)
                t0 = time.time()
                sync_result = actor.gather_and_publish(version=stats.step + 1)
                if sync_result is not None:
                    logger.info(
                        "weight sync v=%d done in %.2fs: n=%d pull=%.2fs apply=%.2fs",
                        sync_result["version"],
                        time.time() - t0,
                        sync_result["n_tensors"],
                        sync_result["pull_s"],
                        sync_result["apply_s"],
                    )
                    metrics_logger.log_event(
                        "weight_sync",
                        {
                            "version": sync_result["version"],
                            "n_tensors": sync_result["n_tensors"],
                            "pull_s": sync_result["pull_s"],
                            "apply_s": sync_result["apply_s"],
                            "wall_s": sync_result.get("wall_s", time.time() - t0),
                        },
                    )
                else:
                    logger.info("weight sync v=%d (non-rank-0 path)", stats.step + 1)

            save_due = (
                args.save_dir
                and args.save_every
                and args.save_every > 0
                and (stats.step + 1) % args.save_every == 0
            )
            if save_due:
                save_path = _checkpoint_path(args.save_dir, stats.step + 1)
                save_result = actor.save_hf_checkpoint(save_path, step=stats.step + 1)
                if save_result is not None:
                    logger.info("checkpoint saved: %s", save_result)

            if not (stats.loss == stats.loss):  # NaN
                logger.error("loss is NaN at step %d; aborting", stats.step)
                return 3
    except KeyboardInterrupt:
        logger.info("interrupted; cleaning up")
    finally:
        if args.save_dir and args.save_final:
            try:
                save_path = _checkpoint_path(args.save_dir, getattr(actor, "_step", 0))
                save_result = actor.save_hf_checkpoint(
                    save_path, step=getattr(actor, "_step", 0)
                )
                if save_result is not None:
                    logger.info("final checkpoint saved: %s", save_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("final checkpoint save failed: %s", exc)
        metrics_logger.close()
        actor.close()
    return 0


def _train_ray(args: argparse.Namespace) -> int:
    """Launch one TrainActor per GPU as Ray actors on the requested train node."""
    from nanorl.config import NanoRLCfg
    from nanorl.metrics import build_logger

    _install_sigint_handler()
    cfg_path = str(Path(args.cfg).resolve())
    cfg = NanoRLCfg.from_yaml(cfg_path)
    if args.weight_sync_every is not None:
        cfg.weight_sync_every = args.weight_sync_every

    world_size = int(args.nproc or cfg.train.world_size)
    if world_size < 1:
        logger.error("train-ray requires --nproc or cfg.train.world_size >= 1")
        return 2
    if cfg.train.fsdp and world_size < 2:
        logger.error("cfg.train.fsdp=true requires train-ray world_size >= 2")
        return 2

    ray_address = args.ray_address or cfg.ray.address or cfg.infer.ray_address
    train_ip = args.train_ip
    if args.dry_run:
        logger.info(
            "dry-run: train-ray cfg=%s world_size=%d train_ip=%s ray=%s",
            cfg_path,
            world_size,
            train_ip,
            ray_address,
        )
        return 0

    import ray

    megatron_path = "/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM"
    existing_py = os.environ.get("PYTHONPATH", "")
    runtime_py = os.pathsep.join(
        p for p in (str(Path.cwd()), megatron_path, existing_py) if p
    )
    ray.init(
        address=ray_address,
        runtime_env={"env_vars": _ray_env_vars({"PYTHONPATH": runtime_py})},
    )

    from nanodeploy.engine.ray_utils import get_available_nodes_with_master_first
    from ray.util.placement_group import placement_group, remove_placement_group

    master_address = f"{train_ip}:{args.master_port}"
    nodes = get_available_nodes_with_master_first(master_address)
    target_node_id = nodes[0]["NodeID"]
    logger.info(
        "train-ray master node resolved: ip=%s node_id=%s",
        nodes[0].get("NodeManagerAddress"),
        target_node_id,
    )

    pg = placement_group(
        bundles=[{"CPU": 1.0, "GPU": 1.0} for _ in range(world_size)],
        strategy="STRICT_PACK",
        name=f"nanorl-train-{train_ip}-{args.master_port}",
        _soft_target_node_id=target_node_id,
    )
    ray.get(pg.ready())

    @ray.remote(num_cpus=1, num_gpus=1)
    class _RayTrainWorker:
        def __init__(
            self,
            *,
            rank: int,
            world_size: int,
            local_rank: int,
            master_addr: str,
            master_port: int,
            cfg_path: str,
            weight_sync_every: int | None,
            producer_alias: str | None,
            consumer_alias: str | None,
        ):
            import os as _os

            _os.environ["RANK"] = str(rank)
            _os.environ["WORLD_SIZE"] = str(world_size)
            _os.environ["LOCAL_RANK"] = str(local_rank)
            _os.environ["MASTER_ADDR"] = master_addr
            _os.environ["MASTER_PORT"] = str(master_port)

            from nanorl.actors.train import TrainActor
            from nanorl.config import NanoRLCfg

            self.rank = rank
            cfg = NanoRLCfg.from_yaml(cfg_path)
            cfg.train.world_size = world_size
            if weight_sync_every is not None:
                cfg.weight_sync_every = weight_sync_every
            self.actor = TrainActor(
                cfg,
                producer_alias=producer_alias,
                consumer_alias=consumer_alias,
                master_port=master_port,
            )

        def step(self, sync_version: int | None = None) -> dict:
            from dataclasses import asdict as _asdict

            stats = self.actor.train_step()
            sync_result = None
            if sync_version is not None:
                sync_result = self.actor.gather_and_publish(sync_version)
            return {
                "rank": self.rank,
                "stats": _asdict(stats) if self.rank == 0 else None,
                "sync_result": sync_result,
            }

        def save_checkpoint(self, path: str, step: int) -> dict | None:
            return self.actor.save_hf_checkpoint(path, step=step)

        def close(self) -> None:
            self.actor.close()

    logger.info(
        "launching %d Ray TrainActor(s) on %s via Ray %s",
        world_size,
        train_ip,
        ray_address,
    )
    workers = [
        _RayTrainWorker.options(placement_group=pg).remote(
            rank=r,
            world_size=world_size,
            local_rank=r,
            master_addr=train_ip,
            master_port=args.master_port,
            cfg_path=cfg_path,
            weight_sync_every=args.weight_sync_every,
            producer_alias=args.producer_alias,
            consumer_alias=args.consumer_alias,
        )
        for r in range(world_size)
    ]

    metrics_logger = build_logger(
        jsonl_path=args.log_jsonl,
        wandb_project=getattr(args, "wandb_project", None),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_config={
            "cfg": args.cfg,
            "steps": args.steps,
            "weight_sync_every": cfg.weight_sync_every,
            "train_ip": train_ip,
            "world_size": world_size,
        },
        tb_dir=getattr(args, "tb_dir", None),
    )

    try:
        for step_idx in range(args.steps):
            # Every rank must enter gather_and_publish together, so pass the
            # same sync_version to all workers when a sync is due.
            sync_version = None
            if cfg.weight_sync_every:
                # Rank 0's TrainStats.step is the source of truth, but all
                # workers progress lockstep from zero, so this loop index is
                # equivalent and lets us decide before dispatching RPCs.
                next_step = step_idx + 1
                if next_step % cfg.weight_sync_every == 0:
                    sync_version = next_step

            results = ray.get([w.step.remote(sync_version) for w in workers])
            rank0 = next(r for r in results if r["rank"] == 0)
            stats = rank0["stats"]
            logger.info(
                "step=%d loss=%.4f kl=%.4f kl_to_old=%.4f ratios=%.3f "
                "trunc_a=%.3f trunc_b=%.3f H=%.3f old_lp=%s "
                "logprob_to_old=%.3g/%.3g "
                "gnorm=%.3f reward=%.3f "
                "tokens=%d resp_len=%.0f elapsed=%.2fs",
                stats["step"],
                stats["loss"],
                stats["kl_mean"],
                stats["kl_to_old"],
                stats["ratios_mean"],
                stats["truncated_above_rate"],
                stats["truncated_below_rate"],
                stats["entropy_mean"],
                stats.get("old_logprobs_present", False),
                stats.get(
                    "logprob_to_old_mean",
                    stats.get("old_logprobs_abs_diff_mean", 0.0),
                ),
                stats.get(
                    "logprob_to_old_max",
                    stats.get("old_logprobs_abs_diff_max", 0.0),
                ),
                stats["grad_norm"],
                stats["mean_reward"],
                stats["response_tokens"],
                stats["response_length_mean"],
                stats["elapsed_s"],
            )
            metrics_logger.log_step(stats["step"], stats)

            if sync_version is not None and rank0["sync_result"] is not None:
                sync_result = rank0["sync_result"]
                metrics_logger.log_event(
                    "weight_sync",
                    {
                        "version": sync_result["version"],
                        "n_tensors": sync_result["n_tensors"],
                        "pull_s": sync_result["pull_s"],
                        "apply_s": sync_result["apply_s"],
                        "wall_s": sync_result.get("wall_s", 0.0),
                    },
                )
                logger.info("weight sync v=%d complete: %s", sync_version, sync_result)

            save_due = (
                args.save_dir
                and args.save_every
                and args.save_every > 0
                and (step_idx + 1) % args.save_every == 0
            )
            if save_due:
                save_path = _checkpoint_path(args.save_dir, step_idx + 1)
                save_results = ray.get(
                    [w.save_checkpoint.remote(save_path, step_idx + 1) for w in workers]
                )
                save_result = next((r for r in save_results if r is not None), None)
                logger.info("checkpoint saved: %s", save_result)

            if not (stats["loss"] == stats["loss"]):
                logger.error("loss is NaN at step %d; aborting", stats["step"])
                return 3
    except KeyboardInterrupt:
        logger.info("interrupted; cleaning up Ray train actors")
    finally:
        if args.save_dir and args.save_final:
            try:
                final_step = (
                    min(args.steps, step_idx + 1) if "step_idx" in locals() else 0
                )
                save_path = _checkpoint_path(args.save_dir, final_step)
                save_results = ray.get(
                    [w.save_checkpoint.remote(save_path, final_step) for w in workers]
                )
                save_result = next((r for r in save_results if r is not None), None)
                logger.info("final checkpoint saved: %s", save_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("final checkpoint save failed: %s", exc)
        metrics_logger.close()
        close_refs = []
        for w in workers:
            try:
                close_refs.append(w.close.remote())
            except Exception as exc:  # noqa: BLE001
                logger.warning("skip closing dead Ray train actor: %s", exc)
        if close_refs:
            try:
                ray.get(close_refs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ray train actor close failed: %s", exc)
        remove_placement_group(pg)
    return 0


def _consume_ray(args: argparse.Namespace) -> int:
    """Run the M2 trajectory consumer as a Ray actor on a requested node."""
    from nanorl.config import NanoRLCfg

    _install_sigint_handler()
    cfg_path = str(Path(args.cfg).resolve())
    cfg = NanoRLCfg.from_yaml(cfg_path)
    ray_address = args.ray_address or cfg.ray.address or cfg.infer.ray_address
    consumer_ip = args.consumer_ip

    if args.dry_run:
        logger.info(
            "dry-run: consume-ray cfg=%s consumer_ip=%s ray=%s batches=%d batch_size=%d",
            cfg_path,
            consumer_ip,
            ray_address,
            args.batches,
            args.batch_size,
        )
        return 0

    import ray

    existing_py = os.environ.get("PYTHONPATH", "")
    runtime_py = os.pathsep.join(p for p in (str(Path.cwd()), existing_py) if p)
    ray.init(
        address=ray_address,
        runtime_env={"env_vars": _ray_env_vars({"PYTHONPATH": runtime_py})},
    )

    from ray.util.placement_group import placement_group, remove_placement_group

    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    target_node = next(
        (node for node in alive_nodes if node.get("NodeManagerAddress") == consumer_ip),
        None,
    )
    if target_node is None:
        available_ips = ", ".join(
            sorted(str(node.get("NodeManagerAddress")) for node in alive_nodes)
        )
        raise RuntimeError(
            f"consumer_ip={consumer_ip!r} is not an alive Ray node; available={available_ips}"
        )
    target_node_id = target_node["NodeID"]
    logger.info(
        "consume-ray node resolved: ip=%s node_id=%s",
        target_node.get("NodeManagerAddress"),
        target_node_id,
    )

    pg = placement_group(
        bundles=[{"CPU": 1.0}],
        strategy="STRICT_PACK",
        name=f"nanorl-consume-{consumer_ip}-{int(time.time())}",
        _soft_target_node_id=target_node_id,
    )
    ray.get(pg.ready())

    @ray.remote(num_cpus=1)
    class _RayTrajectoryConsumer:
        def run(
            self,
            *,
            cfg_path: str,
            producer_alias: str,
            consumer_alias: str,
            batches: int,
            batch_size: int,
        ) -> list[dict]:
            import logging as _logging
            import os as _os
            import time as _time

            from dlslime import PeerAgent
            from transformers import AutoTokenizer

            from nanorl.config import NanoRLCfg
            from nanorl.data.data_loader import TrajectoryClient, TrajectoryClientCfg

            _logging.basicConfig(
                level=_os.environ.get("NANORL_LOG_LEVEL", "INFO"),
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            consumer_logger = _logging.getLogger("nanorl.consume_ray")

            cfg = NanoRLCfg.from_yaml(cfg_path)
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.tokenizer_path or cfg.model.hf_path
            )
            agent = PeerAgent(
                nanoctrl_url=cfg.dlslime.nanoctrl_url,
                alias=consumer_alias,
            )
            conn = agent.connect_to(
                producer_alias,
                ib_port=cfg.dlslime.ib_port,
                qp_num=cfg.dlslime.qp_num,
            )
            conn.wait()
            client = TrajectoryClient(
                agent,
                TrajectoryClientCfg(
                    producer_alias=producer_alias,
                    pull_size=batch_size,
                    prefetch_depth=2,
                ),
            )

            summaries: list[dict] = []
            try:
                for batch_idx in range(batches):
                    t0 = _time.time()
                    batch = client.next_batch(batch_size)
                    decoded = tokenizer.decode(
                        batch.tokens[0, : int(batch.seq_lengths[0])].tolist(),
                        skip_special_tokens=True,
                    )
                    row = {
                        "batch": batch_idx,
                        "B": int(batch.tokens.shape[0]),
                        "T": int(batch.tokens.shape[1]),
                        "mean_reward": float(batch.rewards.mean()),
                        "wait_s": _time.time() - t0,
                        "sample": decoded[:120],
                    }
                    summaries.append(row)
                    consumer_logger.info(
                        "batch=%d B=%d T=%d mean_reward=%.3f wait=%.1fs sample=%r",
                        row["batch"],
                        row["B"],
                        row["T"],
                        row["mean_reward"],
                        row["wait_s"],
                        row["sample"],
                    )
            finally:
                client.close()
                agent.shutdown()
            return summaries

    consumer = _RayTrajectoryConsumer.options(placement_group=pg).remote()
    try:
        summaries = ray.get(
            consumer.run.remote(
                cfg_path=cfg_path,
                producer_alias=args.producer_alias,
                consumer_alias=args.consumer_alias,
                batches=args.batches,
                batch_size=args.batch_size,
            )
        )
        for row in summaries:
            logger.info(
                "batch=%d B=%d T=%d mean_reward=%.3f wait=%.1fs sample=%r",
                row["batch"],
                row["B"],
                row["T"],
                row["mean_reward"],
                row["wait_s"],
                row["sample"],
            )
    finally:
        remove_placement_group(pg)
    return 0


def _eval(args: argparse.Namespace) -> int:
    """Standalone held-out eval — boots a RolloutEngine, runs an eval
    prompt set with overridden sampling, prints + JSONL-writes the
    aggregate report."""
    from nanorl.actors.rollout import RolloutEngine
    from nanorl.config import NanoRLCfg
    from nanorl.data.trajectory_buffer import TrajectoryService
    from nanorl.eval import EvalConfig, Evaluator, load_eval_prompts
    from nanorl.rl.reward import MathVerifier

    _install_sigint_handler()
    cfg = NanoRLCfg.from_yaml(args.cfg)
    items = load_eval_prompts(args.prompts)
    if not items:
        logger.error("no eval prompts loaded from %s", args.prompts)
        return 2
    if args.limit_prompts:
        items = items[: args.limit_prompts]

    if args.dry_run:
        logger.info("dry-run: %d eval prompts, EvalConfig validated", len(items))
        return 0

    # Eval engines don't need SlimeRPC — pass a sacrificial service.
    service = TrajectoryService(capacity=8)
    engine = RolloutEngine(cfg.model, cfg.infer, cfg.sampling, MathVerifier(), service)
    evaluator = Evaluator(engine)

    eval_cfg = EvalConfig(
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pass_threshold=args.pass_threshold,
    )
    report = evaluator.evaluate(items, eval_cfg)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("eval report written to %s", args.output)
    return 0


def _placeholder(name: str):
    def fn(args):
        logger.error("subcommand %r not implemented yet (lands with M1/M3)", name)
        return 1

    return fn


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("NANORL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(prog="nanorl")
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("rollout-only", help="run RolloutEngine + TrajectoryService")
    r.add_argument("--cfg", required=True, help="YAML config path")
    r.add_argument(
        "--prompts", required=True, help="JSONL with prompt/reference/group_id"
    )
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help="if >0, take only the first N prompts (smoke runs)",
    )
    r.add_argument("--seed", type=int, default=None)
    r.add_argument(
        "--top-p", type=float, default=None, help="override sampling.top_p from config"
    )
    r.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override sampling.max_new_tokens from config",
    )
    r.add_argument(
        "--ship-logprobs",
        dest="ship_logprobs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override sampling.ship_logprobs; use --no-ship-logprobs to disable",
    )
    r.add_argument(
        "--save-jsonl",
        default=None,
        help="append every Trajectory to this JSONL for offline inspection",
    )
    r.add_argument("--producer-alias", default=None)
    r.add_argument("--consumer-alias", default=None)
    r.add_argument(
        "--serve-forever",
        action="store_true",
        help="after all rounds, idle and keep serving the SlimeRPC queue",
    )
    r.add_argument(
        "--stop-after-buffered",
        type=int,
        default=0,
        help="exit once the SlimeRPC buffer reaches N (0 = never)",
    )
    r.add_argument(
        "--no-rpc",
        action="store_true",
        help="run inference + verifier only; do not start SlimeRPC server",
    )
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="load and validate config + prompts, then exit",
    )
    r.add_argument(
        "--print-traj-every",
        type=int,
        default=0,
        help="every N rounds, log a couple of decoded sample trajectories so "
        "you can spot-check what the model is producing (0 = never).",
    )
    r.add_argument(
        "--print-traj-n",
        type=int,
        default=2,
        help="how many trajectories to print per --print-traj-every tick.",
    )
    r.add_argument(
        "--eval-prompts",
        default=None,
        help="held-out prompt set (path or bundled name like 'sample' / "
        "'aime'). Every --eval-every rounds, runs a non-published pass "
        "and logs mean reward.",
    )
    r.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="rollout rounds between eval passes (used with --eval-prompts).",
    )
    r.add_argument(
        "--eval-limit-prompts",
        type=int,
        default=0,
        help="cap eval prompt count (0 = use all).",
    )
    r.add_argument(
        "--eval-jsonl",
        default=None,
        help="optional JSONL path for per-eval reward dumps.",
    )
    r.add_argument(
        "--redis-url",
        default=None,
        help="Async sink: stream every trajectory to a Redis stream (XADD). "
        "Example: redis://10.102.97.179:6379/0",
    )
    r.add_argument(
        "--redis-key",
        default="nanorl:trajectories",
        help="Stream key under --redis-url.",
    )
    r.add_argument(
        "--redis-maxlen",
        type=int,
        default=100000,
        help="Approximate stream length cap (XADD MAXLEN ~).",
    )
    r.set_defaults(func=_rollout_only)

    tr = sp.add_parser(
        "train", help="full M3 loop: train + periodic weight sync to rollout"
    )
    tr.add_argument("--cfg", required=True, help="YAML config path")
    tr.add_argument("--steps", type=int, default=10)
    tr.add_argument(
        "--weight-sync-every",
        type=int,
        default=None,
        help="override cfg.weight_sync_every (0 disables)",
    )
    tr.add_argument("--producer-alias", default=None)
    tr.add_argument("--consumer-alias", default=None)
    tr.add_argument("--master-port", type=int, default=29500)
    tr.add_argument("--log-jsonl", default=None)
    tr.add_argument(
        "--wandb-project",
        default=None,
        help="enable wandb logger (requires `pip install wandb`)",
    )
    tr.add_argument("--wandb-run-name", default=None)
    tr.add_argument(
        "--tb-dir",
        default=None,
        help="enable TensorBoard logger (requires `pip install tensorboard`)",
    )
    tr.add_argument(
        "--save-dir",
        default=None,
        help="directory for HF-format checkpoints; writes step_XXXXXX subdirs",
    )
    tr.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="save a HF-format checkpoint every N steps (0 disables)",
    )
    tr.add_argument(
        "--save-final",
        action="store_true",
        help="save a final HF-format checkpoint during shutdown",
    )
    tr.add_argument("--dry-run", action="store_true")
    tr.set_defaults(func=_train)

    tray = sp.add_parser(
        "train-ray",
        help="run multi-rank TrainActor workers as Ray actors on one train node",
    )
    tray.add_argument("--cfg", required=True, help="YAML config path")
    tray.add_argument("--steps", type=int, default=10)
    tray.add_argument(
        "--weight-sync-every",
        type=int,
        default=None,
        help="override cfg.weight_sync_every (0 disables)",
    )
    tray.add_argument("--producer-alias", default=None)
    tray.add_argument("--consumer-alias", default=None)
    tray.add_argument("--master-port", type=int, default=29500)
    tray.add_argument("--train-ip", default="10.102.98.154")
    tray.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="number of Ray TrainActor workers (default cfg.train.world_size)",
    )
    tray.add_argument(
        "--ray-address",
        default=None,
        help="Ray address for the driver (default cfg.ray.address or cfg.infer.ray_address)",
    )
    tray.add_argument("--log-jsonl", default=None)
    tray.add_argument(
        "--wandb-project",
        default=None,
        help="enable wandb logger (requires `pip install wandb`)",
    )
    tray.add_argument("--wandb-run-name", default=None)
    tray.add_argument(
        "--tb-dir",
        default=None,
        help="enable TensorBoard logger (requires `pip install tensorboard`)",
    )
    tray.add_argument(
        "--save-dir",
        default=None,
        help="directory for HF-format checkpoints; writes step_XXXXXX subdirs",
    )
    tray.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="save a HF-format checkpoint every N steps (0 disables)",
    )
    tray.add_argument(
        "--save-final",
        action="store_true",
        help="save a final HF-format checkpoint during shutdown",
    )
    tray.add_argument("--dry-run", action="store_true")
    tray.set_defaults(func=_train_ray)

    cr = sp.add_parser(
        "consume-ray",
        help="run the M2 trajectory consumer as a Ray actor on one node",
    )
    cr.add_argument("--cfg", required=True, help="YAML config path")
    cr.add_argument("--producer-alias", default="rollout:0")
    cr.add_argument("--consumer-alias", default="train:0")
    cr.add_argument("--batches", type=int, default=3)
    cr.add_argument("--batch-size", type=int, default=8)
    cr.add_argument("--consumer-ip", default="10.102.98.154")
    cr.add_argument(
        "--ray-address",
        default=None,
        help="Ray address for the driver (default cfg.ray.address or cfg.infer.ray_address)",
    )
    cr.add_argument("--dry-run", action="store_true")
    cr.set_defaults(func=_consume_ray)

    e = sp.add_parser(
        "eval", help="held-out eval: mean reward + pass@k on a prompt set"
    )
    e.add_argument("--cfg", required=True)
    e.add_argument(
        "--prompts",
        required=True,
        help="JSONL path or bundled name (e.g. 'sample_eval')",
    )
    e.add_argument("--n-samples", type=int, default=1)
    e.add_argument("--temperature", type=float, default=0.0)
    e.add_argument("--top-p", type=float, default=1.0)
    e.add_argument("--max-new-tokens", type=int, default=None)
    e.add_argument("--pass-threshold", type=float, default=0.5)
    e.add_argument("--limit-prompts", type=int, default=0)
    e.add_argument("--output", default=None, help="optional JSON path for full report")
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=_eval)

    t = sp.add_parser(
        "train-only", help="run TrainActor pulling trajectories over SlimeRPC"
    )
    t.add_argument("--cfg", required=True, help="YAML config path")
    t.add_argument("--steps", type=int, default=10)
    t.add_argument("--producer-alias", default=None)
    t.add_argument("--consumer-alias", default=None)
    t.add_argument("--master-port", type=int, default=29500)
    t.add_argument(
        "--log-jsonl",
        default=None,
        help="append per-step TrainStats as JSONL to this path",
    )
    t.add_argument(
        "--dry-run",
        action="store_true",
        help="load and validate config, then exit without building the model",
    )
    t.set_defaults(func=_train_only)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
