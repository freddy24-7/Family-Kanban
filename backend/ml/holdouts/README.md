# Frozen holdout manifests

Each file here lists the topic IDs (plus a content hash) of a frozen evaluation set.
Once written, a manifest is **never edited or regenerated** — that is what makes
metrics comparable across model versions. Add a new versioned manifest instead.
Edits are blocked by a Claude Code hook (`.claude/hooks/protect-frozen.sh`), and
`ml/holdout.py` refuses to overwrite an existing manifest. They live inside `backend/`
so the production image (which retrains in Phase 7) ships with them.
