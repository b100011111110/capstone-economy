import json
from pathlib import Path
from typing import Any, Dict


class HistoryWriter:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
