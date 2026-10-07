#!/bin/sh
# The ccdrift plugin's SessionStart hook. It hands Claude Code's payload to `ccdrift hook
# session-start`, which prints the status line's verdict and starts a check when one is due.
# The lines below are the only words ccdrift prints from outside texts.py: they speak when
# ccdrift itself can't run. It always exits 0, so a session never waits on a failure here.

# uv tool and pipx install to ~/.local/bin, which a Desktop app launched from the Dock may not
# have on PATH; it wins over any other ccdrift on PATH, such as a development checkout's.
bin="$HOME/.local/bin/ccdrift"
if [ ! -x "$bin" ]; then
    # PATH by hand, absolute folders only: Claude Code runs hooks in the project's folder, so an
    # empty or relative entry, which `command -v` would search, could run the project's own ccdrift.
    bin=""
    set -f
    IFS=:
    for dir in $PATH; do
        case "$dir" in
            /*) if [ -f "$dir/ccdrift" ] && [ -x "$dir/ccdrift" ]; then bin="$dir/ccdrift"; break; fi ;;
        esac
    done
    unset IFS
    set +f
fi
if [ -z "$bin" ]; then
    echo '{"systemMessage": "ccdrift: not installed. Run: uv tool install ccdrift"}'
    exit 0
fi
"$bin" hook session-start 2>/dev/null
status=$?
if [ "$status" -eq 2 ]; then
    # argparse's exit for a command it doesn't know: a ccdrift older than the hook
    echo '{"systemMessage": "ccdrift: the plugin needs ccdrift 0.26.0 or later. Run: uv tool upgrade ccdrift"}'
elif [ "$status" -ne 0 ]; then
    echo '{"systemMessage": "ccdrift: the session-start hook failed. Run: ccdrift hook session-start"}'
fi
exit 0
