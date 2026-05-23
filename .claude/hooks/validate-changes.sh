#!/usr/bin/env bash
# .claude/hooks/validate-changes.sh
# PostToolUse hook for Edit|Write. Runs the MalmoRL Tier 1 validator
# silently after edits to malmo/rl/**/*.py or malmo/rl/**/*.xml.
# Always exits 0 — PostToolUse hooks should not block, only inform.

set -u

# Read the tool payload from stdin and extract the edited file path.
payload=$(cat 2>/dev/null || true)
file_path=$(printf '%s' "$payload" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
ti = d.get("tool_input", {}) or {}
print(ti.get("file_path", "") or "")
' 2>/dev/null || true)

# Only validate when an .py / .xml file under malmo/rl/ was changed.
if ! printf '%s' "$file_path" | grep -qE 'malmo/rl/.*\.(py|xml)$'; then
    exit 0
fi

# Run from the project root so the validator's relative paths resolve.
cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0

# Run Tier 1 quietly. Silent on pass, terse on fail.
output=$(python3 malmo/rl/utils/validate.py --tier 1 --quiet 2>&1)
status=$?

if [ $status -ne 0 ]; then
    printf 'MalmoRL post-edit validator (Tier 1) flagged issues after editing %s:\n' "$file_path"
    printf '%s\n' "$output"
    printf '\nRun `@change-validator` for the full Tier 1+2 sweep, or route the listed failures to the responsible specialist agent.\n'
fi

exit 0
