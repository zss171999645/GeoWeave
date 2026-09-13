#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash aidi/scripts/ops/sync_host_groups_to_container.sh [--allow-non-whitelist] <container> [user]

Default user: feng01.zhou

This script syncs host supplementary groups of <user> into the container and
ensures the user is added to those groups in /etc/group.
EOF
}

ALLOW_NON_WHITELIST=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-non-whitelist)
      ALLOW_NON_WHITELIST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -lt 1 || ${#ARGS[@]} -gt 2 ]]; then
  usage
  exit 1
fi

CONTAINER="${ARGS[0]}"
TARGET_USER="${ARGS[1]:-feng01.zhou}"
WHITELIST=("vggt-zf-5090" "vggt-zf-5090-fixed" "vggt-zf-1201")

if [[ "$ALLOW_NON_WHITELIST" -eq 0 ]]; then
  ALLOWED=0
  for name in "${WHITELIST[@]}"; do
    if [[ "$CONTAINER" == "$name" ]]; then
      ALLOWED=1
      break
    fi
  done
  if [[ "$ALLOWED" -ne 1 ]]; then
    echo "[ERROR] Container '$CONTAINER' is not in whitelist: ${WHITELIST[*]}" >&2
    echo "[ERROR] Refusing to modify non-whitelisted container." >&2
    exit 2
  fi
fi

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
  echo "[ERROR] Container '$CONTAINER' is not running." >&2
  exit 3
fi

if ! id "$TARGET_USER" >/dev/null 2>&1; then
  echo "[ERROR] User '$TARGET_USER' not found on host." >&2
  exit 4
fi

if ! docker exec "$CONTAINER" python3 -V >/dev/null 2>&1; then
  echo "[ERROR] python3 is required in container '$CONTAINER'." >&2
  exit 5
fi

TMP_GROUP_PAIRS="$(mktemp)"
trap 'rm -f "$TMP_GROUP_PAIRS"' EXIT

while IFS= read -r gid; do
  [[ -z "$gid" ]] && continue
  group_line="$(getent group "$gid" || true)"
  if [[ -z "$group_line" ]]; then
    echo "[WARN] Skipping gid=$gid because host group entry is missing."
    continue
  fi
  group_name="${group_line%%:*}"
  printf '%s:%s\n' "$gid" "$group_name" >> "$TMP_GROUP_PAIRS"
done < <(id -G "$TARGET_USER" | tr ' ' '\n')

if [[ ! -s "$TMP_GROUP_PAIRS" ]]; then
  echo "[ERROR] No valid host group entries collected for '$TARGET_USER'." >&2
  exit 6
fi

HOST_GROUP_PAIRS_B64="$(base64 < "$TMP_GROUP_PAIRS" | tr -d '\n')"

docker exec -i \
  -e "HOST_GROUP_PAIRS_B64=${HOST_GROUP_PAIRS_B64}" \
  "$CONTAINER" python3 - "$TARGET_USER" <<'PY'
import base64
import os
import sys
from pathlib import Path

target_user = sys.argv[1]
incoming = []
decoded_pairs = base64.b64decode(os.environ["HOST_GROUP_PAIRS_B64"]).decode("utf-8")
for raw in decoded_pairs.splitlines():
    raw = raw.strip()
    if not raw:
        continue
    gid_s, name = raw.split(":", 1)
    incoming.append((int(gid_s), name))

group_path = Path("/etc/group")
original_lines = group_path.read_text(encoding="utf-8").splitlines()

entries = []
for line in original_lines:
    if not line or line.startswith("#"):
        entries.append({"kind": "raw", "raw": line})
        continue
    parts = line.split(":")
    if len(parts) != 4:
        entries.append({"kind": "raw", "raw": line})
        continue
    name, passwd, gid_s, members_s = parts
    try:
        gid = int(gid_s)
    except ValueError:
        entries.append({"kind": "raw", "raw": line})
        continue
    members = [m for m in members_s.split(",") if m]
    entries.append(
        {
            "kind": "group",
            "name": name,
            "passwd": passwd,
            "gid": gid,
            "members": members,
        }
    )


def rebuild_maps():
    by_name = {}
    by_gid = {}
    for idx, ent in enumerate(entries):
        if ent["kind"] != "group":
            continue
        by_name[ent["name"]] = idx
        by_gid[ent["gid"]] = idx
    return by_name, by_gid


for gid, name in incoming:
    by_name, by_gid = rebuild_maps()

    if name in by_name:
        idx = by_name[name]
        existing_gid = entries[idx]["gid"]
        if existing_gid != gid:
            raise SystemExit(
                f"[ERROR] Name conflict in container /etc/group: "
                f"group '{name}' has gid={existing_gid}, expected gid={gid}"
            )
        if target_user not in entries[idx]["members"]:
            entries[idx]["members"].append(target_user)
        continue

    if gid in by_gid:
        idx = by_gid[gid]
        existing_name = entries[idx]["name"]
        if existing_name != name:
            raise SystemExit(
                f"[ERROR] GID conflict in container /etc/group: "
                f"gid={gid} belongs to '{existing_name}', expected '{name}'"
            )
        if target_user not in entries[idx]["members"]:
            entries[idx]["members"].append(target_user)
        continue

    entries.append(
        {
            "kind": "group",
            "name": name,
            "passwd": "x",
            "gid": gid,
            "members": [target_user],
        }
    )

output_lines = []
for ent in entries:
    if ent["kind"] == "raw":
        output_lines.append(ent["raw"])
    else:
        members_s = ",".join(dict.fromkeys(ent["members"]))
        output_lines.append(
            f"{ent['name']}:{ent['passwd']}:{ent['gid']}:{members_s}"
        )

group_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
print(f"[OK] Updated /etc/group for user: {target_user}")
PY

echo "[INFO] Host groups for ${TARGET_USER}:"
id "$TARGET_USER"

echo "[INFO] Container groups for ${TARGET_USER}:"
docker exec "$CONTAINER" bash -lc "su ${TARGET_USER} -c 'id'"

echo "[OK] Group sync completed for container '${CONTAINER}'."
