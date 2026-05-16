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


def _parse_prompts(path: str) -> list:
    """Read a JSONL file of prompts. Bad rows are skipped with a warning so a
    single typo doesn't tank the run.
    """
    from nanorl.actors.rollout import PromptItem

    items: list[PromptItem] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                items.append(
                    PromptItem(
                        prompt=str(row["prompt"]),
                        reference=str(row.get("reference", row.get("answer", ""))),
                        group_id=int(row.get("group_id", row.get("id", 0))),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("skipping line %d in %s: %s", lineno, path, exc)
    return items


def _open_jsonl_writer(path: str | None):
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def _trajectory_to_jsonl_row(t) -> dict:
    return {
        "prompt_ids": t.prompt_ids,
        "response_ids": t.response_ids,
        "reward": t.reward,
        "group_id": t.group_id,
        "eos": t.eos,
        "meta": t.meta,
    }


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
    cfg = NanoRLCfg.from_yaml(args.cfg)

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

    items = _parse_prompts(args.prompts)
    if not items:
        logger.error("no valid prompts in %s", args.prompts)
        return 2
    if args.limit_prompts:
        items = items[: args.limit_prompts]
    logger.info("loaded %d prompts (group_size=%d)", len(items), cfg.sampling.n)

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
        # The serve loop needs a fully-up RDMA endpoint; we hand the
        # connection object to run_rpc_server so it can `.wait()` before
        # binding the mailbox. Otherwise the producer races the (possibly
        # not-yet-started) consumer and serve raises immediately.
        connection = agent.connect_to(
            consumer_alias,
            ib_port=cfg.dlslime.ib_port,
            qp_num=cfg.dlslime.qp_num,
        )
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
            connection=connection,
        )
        logger.info(
            "rollout-only: rpc server pending producer=%s expecting consumer=%s "
            "(weight-sync RPC enabled)",
            producer_alias,
            consumer_alias,
        )

    jsonl_writer = _open_jsonl_writer(args.save_jsonl)

    try:
        for r in range(args.rounds):
            trajs, stats = engine.run(items, publish=not args.no_rpc)
            stats.log()
            logger.info("round=%d buffered=%d", r, service.buffered())
            if jsonl_writer is not None:
                for t in trajs:
                    jsonl_writer.write(json.dumps(_trajectory_to_jsonl_row(t)) + "\n")
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

    _install_sigint_handler()
    cfg = NanoRLCfg.from_yaml(args.cfg)

    if args.weight_sync_every is not None:
        cfg.weight_sync_every = args.weight_sync_every

    if args.dry_run:
        logger.info("dry-run: train config validated; not building TrainActor")
        return 0

    actor = TrainActor(
        cfg,
        producer_alias=args.producer_alias,
        consumer_alias=args.consumer_alias,
        master_port=args.master_port,
    )

    log_writer = _open_jsonl_writer(args.log_jsonl)
    try:
        for s in range(args.steps):
            stats = actor.train_step()
            logger.info(
                "step=%d loss=%.4f kl=%.4f mean_reward=%.3f mean_adv=%.3f tokens=%d elapsed=%.2fs",
                stats.step,
                stats.loss,
                stats.kl_mean,
                stats.mean_reward,
                stats.mean_advantage,
                stats.response_tokens,
                stats.elapsed_s,
            )
            if log_writer is not None:
                log_writer.write(json.dumps(asdict(stats)) + "\n")

            if cfg.weight_sync_every and (stats.step + 1) % cfg.weight_sync_every == 0:
                logger.info("triggering weight sync v=%d", stats.step + 1)
                t0 = time.time()
                sync_result = actor.gather_and_publish(version=stats.step + 1)
                # Non-zero ranks return None — they participate in the
                # collective gather but the actual publish + RPC is rank-0-only.
                if sync_result is not None:
                    logger.info(
                        "weight sync v=%d done in %.2fs: n=%d pull=%.2fs apply=%.2fs",
                        sync_result["version"],
                        time.time() - t0,
                        sync_result["n_tensors"],
                        sync_result["pull_s"],
                        sync_result["apply_s"],
                    )
                    if log_writer is not None:
                        log_writer.write(
                            json.dumps({"event": "weight_sync", **sync_result}) + "\n"
                        )
                else:
                    logger.info("weight sync v=%d (non-rank-0 path)", stats.step + 1)

            if not (stats.loss == stats.loss):  # NaN
                logger.error("loss is NaN at step %d; aborting", stats.step)
                return 3
    except KeyboardInterrupt:
        logger.info("interrupted; cleaning up")
    finally:
        if log_writer is not None:
            log_writer.close()
        actor.close()
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
    tr.add_argument("--dry-run", action="store_true")
    tr.set_defaults(func=_train)

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
