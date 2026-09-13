#!/usr/bin/env python3
"""Session memory CLI for cross-session knowledge sharing in Codex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]{2,}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def safe_json_loads(line: str) -> dict | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ""
    return out.strip()


def detect_repo_root() -> str:
    root = run_git(["rev-parse", "--show-toplevel"])
    return root or os.getcwd()


def detect_branch() -> str:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return branch or "unknown"


def repo_slug(repo_root: str) -> str:
    root = os.path.abspath(repo_root)
    basename = os.path.basename(root)
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:10]
    return f"{basename}_{digest}"


def default_db_path(repo_root: str) -> Path:
    base = Path(os.environ.get("CODEX_SESSION_MEMORY_DIR", "~/.codex/session_memory")).expanduser()
    return base / f"{repo_slug(repo_root)}.jsonl"


def coalesce_text(details: str, details_file: str) -> str:
    if details and details_file:
        raise ValueError("Use either --details or --details-file, not both")
    if details_file:
        return Path(details_file).read_text(encoding="utf-8")
    return details


@dataclass
class Entry:
    id: str
    created_at: str
    repo: str
    branch: str
    topic: str
    summary: str
    details: str
    tags: list[str]
    files: list[str]
    decisions: list[str]
    next_steps: list[str]
    source: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "repo": self.repo,
            "branch": self.branch,
            "topic": self.topic,
            "summary": self.summary,
            "details": self.details,
            "tags": self.tags,
            "files": self.files,
            "decisions": self.decisions,
            "next_steps": self.next_steps,
            "source": self.source,
        }


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Entry) -> None:
        with self.db_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.as_dict(), ensure_ascii=True) + "\n")

    def load(self) -> list[dict]:
        if not self.db_path.exists():
            return []
        entries: list[dict] = []
        for line in self.db_path.read_text(encoding="utf-8").splitlines():
            item = safe_json_loads(line)
            if item is not None:
                entries.append(item)
        return entries


def parse_date(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def within_days(created_at: str, days: int | None) -> bool:
    if days is None:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return parse_date(created_at) >= cutoff


def matches_tags(entry: dict, wanted: set[str]) -> bool:
    if not wanted:
        return True
    tags = {t.lower() for t in entry.get("tags", []) if isinstance(t, str)}
    return wanted.issubset(tags)


def score_entry(entry: dict, query_tokens: set[str]) -> float:
    haystack = " ".join(
        [
            str(entry.get("topic", "")),
            str(entry.get("summary", "")),
            str(entry.get("details", "")),
            " ".join(entry.get("tags", []) or []),
            " ".join(entry.get("decisions", []) or []),
            " ".join(entry.get("next_steps", []) or []),
            " ".join(entry.get("files", []) or []),
        ]
    )
    tokens = to_tokens(haystack)
    overlap = len(tokens & query_tokens)
    if not query_tokens:
        overlap = 1

    age_days = max(0.0, (datetime.now(timezone.utc) - parse_date(str(entry.get("created_at", "")))).total_seconds() / 86400.0)
    recency = 1.0 / (1.0 + age_days / 14.0)
    return overlap + 0.2 * recency


def filter_and_rank(entries: Iterable[dict], query: str, top_k: int, days: int | None, tags: list[str]) -> list[dict]:
    query_tokens = to_tokens(query)
    wanted_tags = {t.lower().strip() for t in tags if t.strip()}

    candidates: list[tuple[float, dict]] = []
    for entry in entries:
        created_at = str(entry.get("created_at", ""))
        if not within_days(created_at, days):
            continue
        if not matches_tags(entry, wanted_tags):
            continue
        score = score_entry(entry, query_tokens)
        if query_tokens and score <= 0.0:
            continue
        candidates.append((score, entry))

    candidates.sort(key=lambda x: (x[0], str(x[1].get("created_at", ""))), reverse=True)
    return [entry for _, entry in candidates[:top_k]]


def print_human(entries: list[dict]) -> None:
    if not entries:
        print("No memory matched.")
        return

    for idx, entry in enumerate(entries, start=1):
        created_at = entry.get("created_at", "")
        topic = entry.get("topic", "")
        summary = entry.get("summary", "")
        tags = ", ".join(entry.get("tags", []) or [])
        print(f"[{idx}] {created_at} | {topic}")
        print(f"  Summary: {summary}")
        if tags:
            print(f"  Tags: {tags}")
        files = entry.get("files", []) or []
        if files:
            print(f"  Files: {', '.join(files)}")
        decisions = entry.get("decisions", []) or []
        if decisions:
            print("  Decisions:")
            for decision in decisions:
                print(f"    - {decision}")
        next_steps = entry.get("next_steps", []) or []
        if next_steps:
            print("  Next steps:")
            for step in next_steps:
                print(f"    - {step}")
        print()


def cmd_add(args: argparse.Namespace) -> int:
    repo_root = detect_repo_root()
    db_path = Path(args.db).expanduser() if args.db else default_db_path(repo_root)
    details = coalesce_text(args.details, args.details_file)

    entry = Entry(
        id=str(uuid.uuid4()),
        created_at=utc_now_iso(),
        repo=repo_root,
        branch=detect_branch(),
        topic=args.topic.strip(),
        summary=args.summary.strip(),
        details=details.strip(),
        tags=[t.strip().lower() for t in args.tag if t.strip()],
        files=[f.strip() for f in args.file if f.strip()],
        decisions=[d.strip() for d in args.decision if d.strip()],
        next_steps=[n.strip() for n in args.next_step if n.strip()],
        source=args.source.strip(),
    )

    store = MemoryStore(db_path)
    store.append(entry)

    print(f"Saved memory to: {db_path}")
    print(f"Entry id: {entry.id}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    repo_root = detect_repo_root()
    db_path = Path(args.db).expanduser() if args.db else default_db_path(repo_root)
    store = MemoryStore(db_path)
    entries = filter_and_rank(store.load(), args.query, args.top_k, args.since_days, args.tag)

    if args.json:
        print(json.dumps(entries, ensure_ascii=True, indent=2))
    else:
        print_human(entries)
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    repo_root = detect_repo_root()
    db_path = Path(args.db).expanduser() if args.db else default_db_path(repo_root)
    store = MemoryStore(db_path)
    entries = filter_and_rank(store.load(), args.query, args.top_k, args.since_days, args.tag)

    if not entries:
        print("No prior memory found. Proceed with fresh context.")
        return 0

    print("Cross-session context brief:")
    for idx, entry in enumerate(entries, start=1):
        print(f"{idx}. {entry.get('topic', '')} ({entry.get('created_at', '')})")
        print(f"   Summary: {entry.get('summary', '')}")

        decisions = entry.get("decisions", []) or []
        if decisions:
            print("   Decisions:")
            for decision in decisions[:3]:
                print(f"   - {decision}")

        next_steps = entry.get("next_steps", []) or []
        if next_steps:
            print("   Next:")
            for step in next_steps[:3]:
                print(f"   - {step}")

        files = entry.get("files", []) or []
        if files:
            print(f"   Files: {', '.join(files[:5])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex session memory utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Append a session memory entry")
    p_add.add_argument("--topic", required=True, help="Topic title")
    p_add.add_argument("--summary", required=True, help="Short summary")
    p_add.add_argument("--details", default="", help="Detailed notes")
    p_add.add_argument("--details-file", default="", help="Load details from file")
    p_add.add_argument("--tag", action="append", default=[], help="Tag (repeatable)")
    p_add.add_argument("--file", action="append", default=[], help="Related file path (repeatable)")
    p_add.add_argument("--decision", action="append", default=[], help="Decision made (repeatable)")
    p_add.add_argument("--next-step", action="append", default=[], help="Next step (repeatable)")
    p_add.add_argument("--source", default="codex-session", help="Source label")
    p_add.add_argument("--db", default="", help="Override jsonl DB path")
    p_add.set_defaults(func=cmd_add)

    p_search = sub.add_parser("search", help="Search memory entries")
    p_search.add_argument("--query", default="", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Maximum items")
    p_search.add_argument("--since-days", type=int, default=None, help="Only entries in recent days")
    p_search.add_argument("--tag", action="append", default=[], help="Tag filter (repeatable)")
    p_search.add_argument("--json", action="store_true", help="Print JSON")
    p_search.add_argument("--db", default="", help="Override jsonl DB path")
    p_search.set_defaults(func=cmd_search)

    p_brief = sub.add_parser("brief", help="Print short briefing for new sessions")
    p_brief.add_argument("--query", default="", help="Focus query for briefing")
    p_brief.add_argument("--top-k", type=int, default=3, help="Maximum items")
    p_brief.add_argument("--since-days", type=int, default=60, help="Only entries in recent days")
    p_brief.add_argument("--tag", action="append", default=[], help="Tag filter (repeatable)")
    p_brief.add_argument("--db", default="", help="Override jsonl DB path")
    p_brief.set_defaults(func=cmd_brief)

    return parser


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
