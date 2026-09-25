#!/usr/bin/env bash
# PostToolUse hook: format the file Claude just edited. Never blocks.
file_path=$(jq -r '.tool_input.file_path // empty')
[ -z "$file_path" ] && exit 0
root=${CLAUDE_PROJECT_DIR:-$PWD}

case "$file_path" in
  "$root"/backend/*.py)
    (cd "$root/backend" && uv run --quiet ruff format "$file_path" && uv run --quiet ruff check --fix --quiet "$file_path") >/dev/null 2>&1 || true ;;
  "$root"/frontend/src/*.ts|"$root"/frontend/src/*.tsx|"$root"/frontend/src/*.css)
    (cd "$root/frontend" && npx --no-install prettier --write "$file_path") >/dev/null 2>&1 || true ;;
esac
exit 0
