#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agentnet_env.config import EnvConfig, LoggingLevel
from agentnet_env.replay import DatasetReplayer


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay AgentNet trajectories with robust logging"
    )
    p.add_argument("--jsonl", required=True, help="Path to AgentNet JSONL file")
    p.add_argument(
        "--task-id",
        default=None,
        help="Specific task_id to replay; defaults to first task",
    )
    p.add_argument("--monitor-index", type=int, default=0)
    p.add_argument("--log-dir", default="runs")
    p.add_argument(
        "--log-level",
        choices=[e.value for e in LoggingLevel],
        default=LoggingLevel.standard.value,
    )
    p.add_argument("--video", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = EnvConfig(
        monitor_index=args.monitor_index,
        log_dir=args.log_dir,
        logging_level=LoggingLevel(args.log_level),
        save_video=bool(args.video),
        dry_run=bool(args.dry_run),
    )
    DatasetReplayer(cfg).replay(args.jsonl, task_id=args.task_id)


if __name__ == "__main__":
    main()
