import json
import math
import os
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, Iterator, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text


console = Console()


def count_lines_in_file(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def get_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


@dataclass
class SchemaNode:
    types: set = field(default_factory=set)
    children: Dict[str, "SchemaNode"] = field(default_factory=dict)
    item: Optional["SchemaNode"] = None
    examples: List[Any] = field(default_factory=list)
    encounter_count: int = 0

    def add_example(self, value: Any, max_examples: int = 3) -> None:
        if len(self.examples) >= max_examples:
            return
        if get_json_type(value) in {"null", "bool", "int", "float", "str"}:
            self.examples.append(value)


class SchemaBuilder:
    def __init__(self) -> None:
        self.root = SchemaNode()
        self.sampled_records = 0

    def merge(self, node: SchemaNode, value: Any) -> None:
        node.encounter_count += 1
        t = get_json_type(value)
        node.types.add(t)
        node.add_example(value)
        if t == "dict":
            for key, child_val in value.items():
                child = node.children.get(key)
                if child is None:
                    child = SchemaNode()
                    node.children[key] = child
                self.merge(child, child_val)
        elif t == "list":
            if node.item is None:
                node.item = SchemaNode()
            for elem in value:
                self.merge(node.item, elem)

    def add_record(self, record: Dict[str, Any]) -> None:
        self.sampled_records += 1
        self.merge(self.root, record)

    def _render_node(self, label: str, node: SchemaNode) -> Tree:
        type_str = ", ".join(sorted(node.types)) if node.types else "unknown"
        label_text = Text.assemble(
            Text(str(label), style="bold"),
            Text(": "),
            Text("["),
            Text(type_str, style="cyan"),
            Text("] "),
            Text(f"(seen {node.encounter_count})", style="dim"),
        )
        tree = Tree(label_text)
        if node.examples:
            ex_preview = ", ".join(
                [
                    (
                        repr(e)
                        if not isinstance(e, str)
                        else e[:60] + ("…" if len(e) > 60 else "")
                    )
                    for e in node.examples
                ]
            )
            tree.add(f"examples: {ex_preview}")
        if node.children:
            for key, child in sorted(node.children.items()):
                subtree = self._render_node(key, child)
                tree.add(subtree)
        if node.item is not None:
            subtree = self._render_node("items", node.item)
            tree.add(subtree)
        return tree

    def render(self, title: str = "Schema") -> Panel:
        tree = self._render_node("root", self.root)
        return Panel(tree, title=title, border_style="cyan", box=box.ROUNDED)


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def describe_numeric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
        }
    vals = list(values)
    vals.sort()
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(vals[0]),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "max": float(vals[-1]),
    }


def add_numeric_row(table: Table, name: str, stats: Dict[str, float]) -> None:
    table.add_row(
        name,
        str(int(stats["count"])) if not math.isnan(stats["count"]) else "0",
        f"{stats['mean']:.3f}" if not math.isnan(stats["mean"]) else "nan",
        f"{stats['std']:.3f}" if not math.isnan(stats["std"]) else "nan",
        f"{stats['min']:.3f}" if not math.isnan(stats["min"]) else "nan",
        f"{stats['p25']:.3f}" if not math.isnan(stats["p25"]) else "nan",
        f"{stats['p50']:.3f}" if not math.isnan(stats["p50"]) else "nan",
        f"{stats['p75']:.3f}" if not math.isnan(stats["p75"]) else "nan",
        f"{stats['max']:.3f}" if not math.isnan(stats["max"]) else "nan",
    )


def build_numeric_table(title: str) -> Table:
    table = Table(title=title, show_lines=False, box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="bold")
    table.add_column("count", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")
    table.add_column("min", justify="right")
    table.add_column("p25", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p75", justify="right")
    table.add_column("max", justify="right")
    return table


@dataclass
class Aggregates:
    total_rows: int = 0
    total_steps: int = 0
    total_actions: int = 0
    step_correct_true: int = 0
    step_redundant_true: int = 0
    task_completed_true: int = 0

    alignment_scores: List[float] = field(default_factory=list)
    efficiency_scores: List[float] = field(default_factory=list)
    task_difficulties: List[float] = field(default_factory=list)
    traj_lengths: List[float] = field(default_factory=list)


def accumulate_from_record(aggr: Aggregates, record: Dict[str, Any]) -> None:
    aggr.total_rows += 1

    for key, collector in (
        ("alignment_score", aggr.alignment_scores),
        ("efficiency_score", aggr.efficiency_scores),
        ("task_difficulty", aggr.task_difficulties),
    ):
        val = record.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            collector.append(float(val))

    if isinstance(record.get("task_completed"), bool) and record["task_completed"]:
        aggr.task_completed_true += 1

    steps = record.get("traj")
    if isinstance(steps, list):
        aggr.traj_lengths.append(float(len(steps)))
        aggr.total_steps += len(steps)
        for step in steps:
            if not isinstance(step, dict):
                aggr.total_actions += 1
                continue
            value = step.get("value")
            action_counted = False
            if isinstance(value, dict):
                action = value.get("action")
                if isinstance(action, str) and action.strip():
                    aggr.total_actions += 1
                    action_counted = True
                last_step_correct = value.get("last_step_correct")
                last_step_redundant = value.get("last_step_redundant")
                if isinstance(last_step_correct, bool) and last_step_correct:
                    aggr.step_correct_true += 1
                if isinstance(last_step_redundant, bool) and last_step_redundant:
                    aggr.step_redundant_true += 1
            if not action_counted:
                aggr.total_actions += 1


def render_sample(record: Dict[str, Any]) -> Panel:
    top = Table(box=box.MINIMAL_DOUBLE_HEAD)
    top.add_column("field", style="bold cyan")
    top.add_column("value")

    def add_row_if_present(key: str) -> None:
        val = record.get(key)
        if val is not None:
            display = val
            if isinstance(val, str) and len(val) > 120:
                display = val[:120] + "…"
            top.add_row(key, str(display))

    for key in (
        "task_id",
        "instruction",
        "natural_language_task",
        "actual_task",
        "task_completed",
        "alignment_score",
        "efficiency_score",
        "task_difficulty",
    ):
        add_row_if_present(key)

    traj_table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    for col in (
        "index",
        "image",
        "observation",
        "thought",
        "action",
        "code",
        "last_step_correct",
        "last_step_redundant",
        "reflection",
    ):
        traj_table.add_column(col)

    steps = record.get("traj")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("index")
            image = step.get("image")
            value = step.get("value", {}) if isinstance(step.get("value"), dict) else {}

            def pick(k: str) -> str:
                v = value.get(k)
                if v is None:
                    return ""
                s = str(v)
                return s if len(s) <= 120 else s[:120] + "…"

            traj_table.add_row(
                str(idx) if idx is not None else "",
                str(image) if image is not None else "",
                pick("observation"),
                pick("thought"),
                pick("action"),
                pick("code"),
                str(value.get("last_step_correct", "")),
                str(value.get("last_step_redundant", "")),
                pick("reflection"),
            )

    grid = Table.grid(expand=True)
    grid.add_row(Panel(top, title="Top-level", border_style="green"))
    grid.add_row(Panel(traj_table, title="Trajectory (sample)", border_style="magenta"))
    return Panel(
        grid, title="Sample Trajectory", border_style="yellow", box=box.ROUNDED
    )


def analyze(paths: List[str]) -> None:
    total_rows = sum(count_lines_in_file(p) for p in paths)

    console.print(
        Panel(
            f"Total number of trajectories: [bold]{total_rows}[/bold]",
            title="Count",
            border_style="blue",
        )
    )

    schema = SchemaBuilder()
    samples_needed = 300
    sample_record: Optional[Dict[str, Any]] = None

    aggr = Aggregates()

    for path in paths:
        for record in stream_jsonl(path):
            if sample_record is None:
                sample_record = record
            if schema.sampled_records < samples_needed:
                schema.add_record(record)
            accumulate_from_record(aggr, record)

    console.print(schema.render(title="Inferred Schema (sampled)"))

    numeric_table = build_numeric_table("Numeric Metrics Summary")
    add_numeric_row(
        numeric_table, "alignment_score", describe_numeric(aggr.alignment_scores)
    )
    add_numeric_row(
        numeric_table, "efficiency_score", describe_numeric(aggr.efficiency_scores)
    )
    add_numeric_row(
        numeric_table, "task_difficulty", describe_numeric(aggr.task_difficulties)
    )
    add_numeric_row(
        numeric_table, "trajectory_length", describe_numeric(aggr.traj_lengths)
    )

    overall_table = Table(title="Overall", box=box.SIMPLE_HEAVY)
    overall_table.add_column("metric", style="bold")
    overall_table.add_column("value", justify="right")
    overall_table.add_row("total_rows", str(aggr.total_rows))
    overall_table.add_row("total_steps", str(aggr.total_steps))
    overall_table.add_row("total_actions", str(aggr.total_actions))
    overall_table.add_row("steps_correct_true", str(aggr.step_correct_true))
    overall_table.add_row("steps_redundant_true", str(aggr.step_redundant_true))
    overall_table.add_row("tasks_completed_true", str(aggr.task_completed_true))

    console.print(Panel(overall_table, title="Dataset Stats", border_style="white"))
    console.print(
        Panel(numeric_table, title="Numeric Distributions", border_style="white")
    )

    if sample_record is not None:
        console.print(render_sample(sample_record))
    else:
        console.print(
            Panel(
                "No sample record available.",
                title="Sample Trajectory",
                border_style="red",
            )
        )


if __name__ == "__main__":
    base = "/Users/asanshaygupta/Documents/Codes/websight-v2/AgentNet"
    paths = [
        os.path.join(base, "agentnet_ubuntu_5k.jsonl"),
        os.path.join(base, "agentnet_win_mac_18k.jsonl"),
    ]
    analyze(paths)
