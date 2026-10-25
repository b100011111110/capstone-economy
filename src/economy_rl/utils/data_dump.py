import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


class DataDump:
    """Append-only, spreadsheet-friendly training data for later analysis."""

    AGENT_FIELDS = (
        "global_step",
        "world",
        "episode",
        "timestep",
        "agent_id",
        "action",
        "step_reward",
        "cumulative_reward",
        "coin",
        "wood",
        "stone",
        "houses",
        "labor",
        "step_labor",
        "effort_multiplier",
        "efficiency",
        "loyalty_leader_0",
        "loyalty_leader_1",
        "loyalty_leader_2",
        "loyalties",
        "preferred_vote",
    )

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

    def _write_many(self, filename: str, fields: Iterable[str], records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        if filename not in self._writers:
            handle = (self.directory / filename).open("a", newline="")
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            self._files[filename] = handle
            self._writers[filename] = writer
        self._writers[filename].writerows(records)
        self._row_counts[filename] = self._row_counts.get(filename, 0) + len(records)
        if self._row_counts[filename] % 256 < len(records):
            self._files[filename].flush()

    def agent_step(self, record: Dict[str, Any]) -> None:
        self._write("agent_steps.csv", self.AGENT_FIELDS, record)

    def agent_steps(self, records: Sequence[Dict[str, Any]]) -> None:
        self._write_many("agent_steps.csv", self.AGENT_FIELDS, records)

    def agent_episodes(self, records: Sequence[Dict[str, Any]]) -> None:
        self._write_many("agent_episodes.csv", self.AGENT_FIELDS, records)

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
            ("update", "global_step", "worker_loss", "worker_policy_loss", "worker_value_loss", "worker_entropy", "planner_loss", "planner_policy_loss", "planner_value_loss", "planner_entropy", "steps_per_second", "elapsed_seconds"),
            record,
        )

    def election(self, record: Dict[str, Any]) -> None:
        self._write(
            "elections.csv",
            ("global_step", "world", "episode", "timestep", "incumbent_id", "winner_id", "incumbent_reelected", "margin_of_victory", "vote_counts", "vote_shares", "cluster_vote_shares"),
            record,
        )

    def loyalty(self, record: Dict[str, Any]) -> None:
        self._write(
            "loyalty.csv",
            ("global_step", "world", "episode", "timestep", "incumbent_id", "leader_0_loyalty", "leader_1_loyalty", "leader_2_loyalty", "cluster_proportions"),
            record,
        )

    def write_summary(self, summary: Dict[str, Any]) -> None:
        with (self.directory / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, default=float)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
