#!/bin/bash
# Stop hook. src/ccdrift/logs.py parsing changes silently do nothing unless
# PARSER_VERSION in src/ccdrift/history.py is bumped: History.update skips
# transcripts it thinks are unchanged, so the store keeps its stale rows.

repo="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$repo" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

git diff HEAD --quiet -- src/ccdrift/logs.py && exit 0
git diff HEAD -- src/ccdrift/history.py | grep -qE '^\+PARSER_VERSION' && exit 0

msg="src/ccdrift/logs.py changed with no PARSER_VERSION bump in src/ccdrift/history.py. If parse_file output changed at all, including a new CANDIDATES path, bump PARSER_VERSION and add a numbered comment line. Otherwise every transcript on disk keeps its old rows and the change appears to do nothing."
jq -n --arg m "$msg" '{systemMessage: $m}'
