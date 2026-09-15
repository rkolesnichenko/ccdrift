"""Log-writing helpers and day fixtures shared by the package and lab tests."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
DAY = 86400


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def text(n: int) -> dict:
    return {"type": "text", "text": "x" * n}


def thinking(signature_chars: int) -> dict:
    # Claude Code stores thinking blocks without their text, only the signature.
    return {"type": "thinking", "thinking": "", "signature": "s" * signature_chars}


def line(mid, block, *, ts, sid="s1", out=100, cache_read=0, cache_creation=0,
         model="claude-opus-5", sidechain=False):
    """One JSONL line as Claude Code writes it: a single content block, with the
    response's message.id and usage repeated on every line of that response."""
    msg = {"role": "assistant", "model": model, "content": [block],
           "usage": {"input_tokens": 10, "output_tokens": out,
                     "cache_read_input_tokens": cache_read,
                     "cache_creation_input_tokens": cache_creation}}
    rec = {"type": "assistant", "timestamp": ts, "sessionId": sid,
           "isSidechain": sidechain, "message": msg}
    if mid is not None:
        msg["id"] = mid
        rec["requestId"] = f"req_{mid}"
    return rec


def response(mid, *blocks, ts, **kw):
    """All lines of one API response."""
    return [line(mid, b, ts=ts, **kw) for b in blocks]


def prompt(ts, sid="s1", sidechain=False):
    """A prompt typed by the user, which opens a new turn."""
    return {"type": "user", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain,
            "message": {"role": "user", "content": "next request"}}


def tool_result(ts, sid="s1", sidechain=False):
    """A tool result, which continues the current turn."""
    return {"type": "user", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain,
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def compact_boundary(ts, sid="s1"):
    return {"type": "system", "subtype": "compact_boundary", "timestamp": ts, "sessionId": sid}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def daily_turns(days):
    """Turn-level rows for bin_metrics, one dict per day mapping a per-turn
    column to its values; metric columns left out are 0."""
    df = pd.concat([pd.DataFrame({"day": f"2026-09-{i + 1:02d}", **cols})
                    for i, cols in enumerate(days)], ignore_index=True)
    for col in ("thinking_fraction", "prompt_cache_read_ratio", "is_haiku"):
        if col not in df:
            df[col] = 0.0
    return df


def prompt_turn_days(misses_per_day, rng):
    """40 new-prompt turns per day: hits just under 1.0, the rest cold."""
    return [{"prompt_cache_read_ratio": [0.05] * k + [rng.uniform(0.99, 1.0) for _ in range(40 - k)]}
            for k in misses_per_day]


# One cold turn on 4 of 14 days: 4 misses in 560 turns.
MOSTLY_CLEAN_CACHE = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]


QUIET = {"is_haiku": [0.0] * 400}
HAIKU = {"is_haiku": [1.0] * 20 + [0.0] * 380}  # z of about 12 against QUIET days


def busy_days(path, days, per_day, prompts=True, cache_read=0, cache_creation=0):
    """One main-thread session a day from Sep 1: `per_day` responses a minute
    apart, each after a prompt unless prompts=False."""
    for d in range(days):
        records = []
        for k in range(per_day):
            ts = at(d * DAY + 60 * k)
            records += [prompt(ts, sid=f"s{d}")] if prompts else []
            records.append(line(f"m{d}-{k}", text(40), ts=ts, sid=f"s{d}",
                                cache_read=cache_read, cache_creation=cache_creation))
        write(path / f"s{d}.jsonl", records)
