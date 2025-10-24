from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import EnvConfig
from .env import AgentNetDesktopEnv


def _iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class DatasetReplayer:
    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()

    def replay(
        self,
        jsonl_path: str | Path,
        task_id: Optional[str] = None,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        data = None
        if task_id is None:
            # First sample only
            data = next(iter(_iter_jsonl(jsonl_path)))
        else:
            for sample in _iter_jsonl(jsonl_path):
                if str(sample.get("task_id")) == str(task_id):
                    data = sample
                    break
        if data is None:
            raise ValueError("Task not found in JSONL")

        env = AgentNetDesktopEnv(self.config, task_id=data.get("task_id"))
        env.reset()

        traj = data.get("traj", [])
        if end is None:
            end = len(traj)
        for step in traj[start:end]:
            code = step["value"]["code"]
            env.step(code)
