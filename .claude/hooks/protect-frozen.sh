#!/usr/bin/env bash
# PreToolUse hook: block Claude's file tools from touching frozen or secret files.
# - backend/ml/holdouts/*  frozen holdout manifests (only README.md is editable;
#                           new manifests are written by `ml.holdout freeze`)
# - data/artifacts/* model artifacts (written only by backend/ml/registry.py)
# - .env files       secrets (.env.example is allowed)
# Exit code 2 blocks the tool call and shows the message to Claude.
set -euo pipefail

file_path=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
[ -z "$file_path" ] && exit 0

rel=${file_path#"${CLAUDE_PROJECT_DIR:-$PWD}/"}
base=$(basename "$rel")

case "$rel" in
  backend/ml/holdouts/README.md) exit 0 ;;
  backend/ml/holdouts/*)
    echo "Blocked: $rel is a frozen holdout manifest. Never edit or regenerate it; freeze a new versioned manifest via the freeze script instead." >&2
    exit 2 ;;
  data/artifacts/*)
    echo "Blocked: $rel is a model artifact. Artifacts are written only by backend/ml/registry.py." >&2
    exit 2 ;;
esac

case "$base" in
  .env.example) exit 0 ;;
  .env|.env.*)
    echo "Blocked: $rel holds secrets. Edit .env.example instead and let the developer set real values." >&2
    exit 2 ;;
esac

exit 0
