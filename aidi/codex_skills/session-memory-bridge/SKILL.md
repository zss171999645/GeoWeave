---
name: session-memory-bridge
description: Persist and recall project knowledge across Codex sessions by saving structured memory entries to a local JSONL store, then generating startup briefs from prior work. Use when a session should inherit decisions, TODOs, experiment outcomes, debugging findings, or file-level context from previous sessions.
---

# Session Memory Bridge

Use this skill to make cross-session context explicit and recoverable.

## Quick start
1. At session start, run:
   - `python3 scripts/session_memory.py brief --query "<task>" --top-k 5`
2. While working (optional checkpoints), run:
   - `python3 scripts/session_memory.py add --topic "<topic>" --summary "<summary>" --tag checkpoint`
3. At session end, run one final write:
   - `python3 scripts/session_memory.py add --topic "<task result>" --summary "<what changed>" --decision "<decision>" --next-step "<next>" --file "<path>"`

## Workflow
### 1) Bootstrap new session context
- Use `brief` to load relevant memories for the current task.
- If the brief is noisy, narrow by `--tag` and `--since-days`.
- If needed, inspect raw results with `search --json`.

### 2) Capture durable updates
- Use `add` whenever a meaningful decision or milestone is reached.
- Keep `summary` short and factual.
- Put detailed technical notes into `--details` or `--details-file`.
- Prefer consistent tags, such as: `bugfix`, `experiment`, `config`, `infra`, `decision`, `handoff`.

### 3) Prepare cross-session handoff
- Before ending the session, write one final entry that includes:
  - what was changed
  - key decisions and constraints
  - concrete next steps
  - touched file paths

## Command reference
- `add`: append one structured memory entry.
- `search`: retrieve entries by query/tags/time window.
- `brief`: produce compact startup context for a new session.

## Storage
- Default DB path: `~/.codex/session_memory/<repo_name>_<repo_hash>.jsonl`
- Override with env var: `CODEX_SESSION_MEMORY_DIR`
- Override per command: `--db <path>`

For field schema and retrieval conventions, read `references/memory-schema.md`.
