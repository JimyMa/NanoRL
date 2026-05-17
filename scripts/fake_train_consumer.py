"""Tiny SlimeRPC consumer that pulls from a running rollout engine.

Used to verify M2 end-to-end without a real TrainActor. Pulls batches from
the rollout side and prints summary stats. Decodes the first response per
batch using the configured tokenizer for a smell-test.

Usage:
    python scripts/fake_train_consumer.py --cfg configs/qwen3_4b_grpo.yaml \\
        --producer-alias rollout:0 --consumer-alias train:0 --batches 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger("fake_train_consumer")


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--producer-alias", default="rollout:0")
    p.add_argument("--consumer-alias", default="train:0")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    from dlslime import PeerAgent

    from nanorl.config import NanoRLCfg
    from nanorl.data.data_loader import TrajectoryClient, TrajectoryClientCfg
    from transformers import AutoTokenizer

    cfg = NanoRLCfg.from_yaml(args.cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_path or cfg.model.hf_path
    )

    agent = PeerAgent(nanoctrl_url=cfg.dlslime.nanoctrl_url, alias=args.consumer_alias)
    c2p = agent.connect_to(
        args.producer_alias, ib_port=cfg.dlslime.ib_port, qp_num=cfg.dlslime.qp_num
    )
    c2p.wait()
    client = TrajectoryClient(
        agent,
        TrajectoryClientCfg(
            producer_alias=args.producer_alias,
            pull_size=args.batch_size,
            prefetch_depth=2,
        ),
    )

    try:
        for b in range(args.batches):
            t0 = time.time()
            batch = client.next_batch(args.batch_size)
            decoded = tokenizer.decode(
                batch.tokens[0, : int(batch.seq_lengths[0])].tolist(),
                skip_special_tokens=True,
            )
            mean_r = float(batch.rewards.mean())
            logger.info(
                "batch=%d B=%d T=%d mean_reward=%.3f wait=%.1fs sample=%r",
                b,
                batch.tokens.shape[0],
                batch.tokens.shape[1],
                mean_r,
                time.time() - t0,
                decoded[:120],
            )
    finally:
        client.close()
        agent.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)
