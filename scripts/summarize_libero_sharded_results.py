#!/usr/bin/env python3
"""Summarize accelerated LIBERO sharded eval results.

This script is for the new LIBERO accelerated layout:

    results/libero_10_sharded/<checkpoint>/shard_0_of_3/eval.log
    results/libero_goal_sharded/<checkpoint>/shard_1_of_3/eval.log

It intentionally ignores LIBERO-Plus directories such as
`plus_libero_mix_sharded`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("results")
DEFAULT_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_90")

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUCCESS_RE = re.compile(r"Success:\s*(True|False)")
START_RE = re.compile(r"Starting task\s+(\d+),\s*episode\s+(\d+)")
SHARD_RE = re.compile(r"shard_(\d+)_of_(\d+)")
RESOLVED_RE = re.compile(r"resolved=\[(\d+),\s*(\d+)\)")
NUM_TRIALS_RE = re.compile(r'"num_trials_per_task":\s*(\d+)')
TOTAL_RATE_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
SERIOUS_ERROR_RE = re.compile(
    r"Policy server inference failed|CUDA out of memory|"
    r"KeyError: 'data'|RuntimeError:|ModuleNotFoundError:|FileNotFoundError:"
)


@dataclass
class EpisodeRecord:
    suite: str
    checkpoint: str
    shard: str
    shard_index: int | None
    num_shards: int | None
    task_id: int | None
    episode: int | None
    success: bool
    log_path: str


@dataclass
class SummaryRow:
    group: str
    episodes: int
    successes: int
    failures: int
    success_rate: float
    expected: int | None = None
    complete: bool | None = None
    errored: bool = False


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r", "\n"))


def strip_ignored_cleanup_tracebacks(text: str) -> str:
    """Drop noisy interpreter-shutdown cleanup tracebacks from robosuite/EGL."""
    lines = text.splitlines()
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("Exception ignored in:"):
            idx += 1
            while idx < len(lines) and not lines[idx].startswith(("07/", "20")):
                idx += 1
            continue
        kept.append(lines[idx])
        idx += 1
    return "\n".join(kept)


def parse_expected_episode_count(text: str) -> int | None:
    resolved = RESOLVED_RE.search(text)
    trials = NUM_TRIALS_RE.search(text)
    if resolved and trials:
        start, end = (int(resolved.group(1)), int(resolved.group(2)))
        return max(0, end - start) * int(trials.group(1))
    return None


def parse_log(log_path: Path, results_root: Path) -> tuple[list[EpisodeRecord], int | None, bool]:
    text = clean_text(log_path.read_text(errors="replace"))
    error_text = strip_ignored_cleanup_tracebacks(text)
    rel = log_path.relative_to(results_root)
    parts = rel.parts

    suite_dir = parts[0] if len(parts) >= 1 else "unknown"
    suite = suite_dir.removesuffix("_sharded")
    checkpoint = parts[1] if len(parts) >= 2 else "unknown"
    shard = parts[2] if len(parts) >= 3 else log_path.parent.name

    shard_index: int | None = None
    num_shards: int | None = None
    shard_match = SHARD_RE.fullmatch(shard)
    if shard_match:
        shard_index = int(shard_match.group(1))
        num_shards = int(shard_match.group(2))

    starts = [(int(match.group(1)), int(match.group(2))) for match in START_RE.finditer(text)]
    successes = [match.group(1) == "True" for match in SUCCESS_RE.finditer(text)]

    records: list[EpisodeRecord] = []
    for idx, success in enumerate(successes):
        task_id, episode = starts[idx] if idx < len(starts) else (None, None)
        records.append(
            EpisodeRecord(
                suite=suite,
                checkpoint=checkpoint,
                shard=shard,
                shard_index=shard_index,
                num_shards=num_shards,
                task_id=task_id,
                episode=episode,
                success=success,
                log_path=str(log_path),
            )
        )

    return records, parse_expected_episode_count(text), bool(SERIOUS_ERROR_RE.search(error_text))


def make_summary(group: str, records: list[EpisodeRecord], expected: int | None = None, errored: bool = False) -> SummaryRow:
    episodes = len(records)
    successes = sum(record.success for record in records)
    failures = episodes - successes
    success_rate = successes / episodes if episodes else 0.0
    complete = None if expected is None else episodes >= expected
    return SummaryRow(
        group=group,
        episodes=episodes,
        successes=successes,
        failures=failures,
        success_rate=success_rate,
        expected=expected,
        complete=complete,
        errored=errored,
    )


def group_records(records: Iterable[EpisodeRecord], key_name: str) -> dict[str, list[EpisodeRecord]]:
    grouped: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        key = getattr(record, key_name)
        grouped.setdefault(str(key), []).append(record)
    return grouped


def format_rate(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def print_table(title: str, rows: list[SummaryRow]) -> None:
    if not rows:
        return

    headers = ["group", "episodes", "successes", "failures", "success_rate", "expected", "complete", "errored"]
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            [
                row.group,
                str(row.episodes),
                str(row.successes),
                str(row.failures),
                format_rate(row.success_rate),
                "" if row.expected is None else str(row.expected),
                "" if row.complete is None else ("yes" if row.complete else "no"),
                "yes" if row.errored else "no",
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(rendered[idx]) for rendered in rendered_rows))
        for idx in range(len(headers))
    ]

    print(f"\n{title}")
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for rendered in rendered_rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(rendered)))


def write_csv(path: Path, rows: list[SummaryRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_episode_csv(path: Path, records: list[EpisodeRecord]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def find_logs(results_root: Path, suites: list[str]) -> list[Path]:
    log_paths: list[Path] = []
    for suite in suites:
        log_paths.extend(sorted((results_root / f"{suite}_sharded").glob("*/shard_*_of_*/eval.log")))
    return sorted(log_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Results root. Defaults to ./results.",
    )
    parser.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated LIBERO suites to include. Defaults to the common LIBERO suites.",
    )
    parser.add_argument("--csv", type=Path, help="Write summary rows to this CSV file.")
    parser.add_argument("--episodes-csv", type=Path, help="Write one row per episode to this CSV file.")
    parser.add_argument("--json", type=Path, help="Write all summary tables and episodes to this JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.root.expanduser()
    if not results_root.is_absolute():
        results_root = Path.cwd() / results_root

    if not results_root.exists():
        raise SystemExit(f"Results root does not exist: {results_root}")

    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    log_paths = find_logs(results_root, suites)
    if not log_paths:
        raise SystemExit(f"No new LIBERO sharded eval.log files found under: {results_root}")

    all_records: list[EpisodeRecord] = []
    expected_by_shard: dict[str, int | None] = {}
    errored_by_shard: dict[str, bool] = {}

    for log_path in log_paths:
        records, expected, errored = parse_log(log_path, results_root)
        all_records.extend(records)
        shard_group = str(log_path.relative_to(results_root).parent)
        expected_by_shard[shard_group] = expected
        errored_by_shard[shard_group] = errored

    overall_expected = sum(value or 0 for value in expected_by_shard.values())
    overall = make_summary("overall", all_records, overall_expected, any(errored_by_shard.values()))

    suite_rows = []
    for suite, records in sorted(group_records(all_records, "suite").items()):
        expected = sum(value or 0 for shard, value in expected_by_shard.items() if shard.startswith(f"{suite}_sharded/"))
        errored = any(value for shard, value in errored_by_shard.items() if shard.startswith(f"{suite}_sharded/"))
        suite_rows.append(make_summary(suite, records, expected, errored))

    checkpoint_rows = []
    for checkpoint, records in sorted(group_records(all_records, "checkpoint").items()):
        expected = sum(value or 0 for shard, value in expected_by_shard.items() if f"/{checkpoint}/" in shard)
        errored = any(value for shard, value in errored_by_shard.items() if f"/{checkpoint}/" in shard)
        checkpoint_rows.append(make_summary(checkpoint, records, expected, errored))

    shard_rows = []
    for shard_group, expected in sorted(expected_by_shard.items()):
        shard_records = [
            record
            for record in all_records
            if str(Path(f"{record.suite}_sharded") / record.checkpoint / record.shard) == shard_group
        ]
        shard_rows.append(make_summary(shard_group, shard_records, expected, errored_by_shard[shard_group]))

    print(f"Found {len(log_paths)} new LIBERO sharded eval.log files under {results_root}")
    print_table("Overall", [overall])
    print_table("By suite", suite_rows)
    print_table("By checkpoint", checkpoint_rows)
    print_table("By shard", shard_rows)

    summary_rows = [overall, *suite_rows, *checkpoint_rows, *shard_rows]
    if args.csv:
        write_csv(args.csv, summary_rows)
        print(f"\nWrote summary CSV: {args.csv}")

    if args.episodes_csv:
        write_episode_csv(args.episodes_csv, all_records)
        print(f"Wrote episode CSV: {args.episodes_csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(results_root),
            "suites": suites,
            "num_logs": len(log_paths),
            "overall": asdict(overall),
            "by_suite": [asdict(row) for row in suite_rows],
            "by_checkpoint": [asdict(row) for row in checkpoint_rows],
            "by_shard": [asdict(row) for row in shard_rows],
            "episodes": [asdict(record) for record in all_records],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
