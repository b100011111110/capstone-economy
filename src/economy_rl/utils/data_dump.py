import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DataDump:
    """Append-only, spreadsheet-friendly training data for later analysis."""

    def __init__(self, output_dir: str):
        self.directory = Path(output_dir) / "data"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._files = {}
        self._writers = {}
        self._row_counts = {}

    def _write(self, filename: str, fields: Iterable[str], record: Dict[str, Any]) -> None:
        if filename not in self._writers:
            handle = (self.directory / filename).open("a", newline="")
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            self._files[filename] = handle
            self._writers[filename] = writer
        self._writers[filename].writerow(record)
        self._row_counts[filename] = self._row_counts.get(filename, 0) + 1
        if self._row_counts[filename] % 128 == 0:
            self._files[filename].flush()

    def step(self, record: Dict[str, Any]) -> None:
        self._write(
            "steps.csv",
            ("global_step", "world", "episode", "timestep", "planner_reward", "worker_total_reward", "worker_mean_reward", "reward_gap", "planner_action_value", "worker_action_mean", "worker_action_mode", "done", "planner_action"),
            record,
        )

    def planner_interval(self, record: Dict[str, Any]) -> None:
        self._write(
            "planner_intervals.csv",
            ("global_step", "world", "episode", "duration", "reward", "done", "planner_action"),
            record,
        )

    def episode(self, record: Dict[str, Any]) -> None:
        self._write(
            "episodes.csv",
            ("global_step", "world", "episode", "length", "planner_total_reward", "worker_total_reward", "worker_mean_reward"),
            record,
        )

    def update(self, record: Dict[str, Any]) -> None:
        self._write(
            "updates.csv",
            ("update", "global_step", "worker_loss", "worker_policy_loss", "worker_value_loss", "worker_entropy", "planner_loss", "planner_policy_loss", "planner_value_loss", "planner_entropy"),
            record,
        )

    def write_summary(self, summary: Dict[str, Any]) -> None:
        with (self.directory / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, default=float)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
