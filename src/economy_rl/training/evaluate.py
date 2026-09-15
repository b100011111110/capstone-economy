import argparse
from pathlib import Path

import torch

from economy_rl.config import Config
from economy_rl.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained hierarchical PPO checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    config = Config.from_file(args.config)
    trainer = Trainer(config)
    checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.planner.load_state_dict(checkpoint["planner"])
    trainer.worker.load_state_dict(checkpoint["worker"])
    trainer.planner.eval()
    trainer.worker.eval()
    buffers = trainer.collect_rollout()
    print({"checkpoint": str(Path(args.checkpoint)), "worker_steps": len(buffers["worker"]), "planner_intervals": len(buffers["planner"])})
    trainer.worlds.close()


if __name__ == "__main__":
    main()
