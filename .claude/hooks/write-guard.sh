#!/bin/bash
# PostToolUse on Write|Edit. Checks two repo conventions at write time.
#   1. No em dashes anywhere in the repo: prose, comments, docstrings.
#   2. The status line path imports neither pandas nor numpy at module level.
# Warns, never blocks. The model gets additionalContext, the user gets a systemMessage.
# The em dash is matched by hex escape so this script does not trip its own check.

# The repo root. Claude Code sets CLAUDE_PROJECT_DIR for hooks; the fallback keeps the
# script working when it is run by hand, since it lives in <repo>/.claude/hooks.
repo="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
payload=$(cat)
file=$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')

[ -n "$file" ] || exit 0
case "$file" in "$repo"/*) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0

rel=${file#"$repo"/}
warnings=""

em=$(printf '\xe2\x80\x94')
if LC_ALL=C grep -q "$em" "$file"; then
  n=$(LC_ALL=C grep -c "$em" "$file")
  lines=$(LC_ALL=C grep -n "$em" "$file" | head -3 | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
  warnings="${warnings}${rel}: ${n} em dash(es) (line ${lines}). This repo has zero by convention; use a hyphen or an en dash. "
fi

case "$rel" in
  src/ccdrift/texts.py|src/ccdrift/state.py|src/ccdrift/cli.py|src/ccdrift/status.py)
    heavy=$(grep -nE '^(import|from)[[:space:]]+(pandas|numpy)\b' "$file" | tr '\n' ' ')
    if [ -n "$heavy" ]; then
      warnings="${warnings}${rel}: pandas or numpy imported at module level (${heavy}). ccdrift status --short runs on every status-line refresh; defer the import into the command branch in cli.py instead. "
    fi
    ;;
esac

[ -n "$warnings" ] || exit 0
jq -n --arg m "$warnings" '{systemMessage: $m, hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $m}}'
