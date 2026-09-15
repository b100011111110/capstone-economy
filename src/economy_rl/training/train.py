import argparse
import sys

from economy_rl.config import Config
from economy_rl.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical planner and worker PPO policies")
    parser.add_argument("--config", default="configs/default.json")
    # Some terminal launchers append a whitespace-only token to the command.
    arguments = [argument for argument in sys.argv[1:] if argument.strip()]
    args = parser.parse_args(arguments)
    Trainer(Config.from_file(args.config)).train()


if __name__ == "__main__":
    main()
