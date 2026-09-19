#!/bin/bash
# Stop hook. Fires a macOS notification when the turn ends, so a long run does
# not need watching. Always exits 0; a failed notification never blocks Claude.
osascript -e 'display notification "Turn finished in ccdrift" with title "Claude Code"' >/dev/null 2>&1
exit 0
