# Memory Schema

## Entry fields
- `id`: UUID.
- `created_at`: UTC ISO-8601 timestamp.
- `repo`: Git repo root path.
- `branch`: Git branch name at write time.
- `topic`: Short title for the memory chunk.
- `summary`: One-line key result.
- `details`: Optional longer notes.
- `tags`: Lowercase category tags.
- `files`: Related file paths.
- `decisions`: Durable decisions made in this session.
- `next_steps`: Explicit follow-up actions.
- `source`: Source label (default `codex-session`).

## Tag conventions
- `decision`: Architecture or workflow decision.
- `experiment`: Benchmark/ablation/training result.
- `bugfix`: Root cause and fix summary.
- `config`: Config expansion, override, or runtime knobs.
- `handoff`: End-of-session summary for next session.

## Retrieval conventions
- Start with `brief` for quick context hydration.
- Use `search --json` when you need exact payloads.
- Add `--since-days` to avoid stale context.
- Use repeated `--tag` to narrow semantic scope.
