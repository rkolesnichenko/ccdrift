"""Log-writing helpers and day fixtures shared by the package and lab tests."""

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

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
         model="claude-opus-5", sidechain=False, version=None, entrypoint=None, effort=None,
         cache_1h=None, cache_5m=None, thinking_logged=None, speed=None, service_tier=None):
    """One JSONL line as Claude Code writes it: a single content block, with the
    response's message.id and usage repeated on every line of that response.
    Fields left as None are left out, as older Claude Code versions do."""
    usage = {"input_tokens": 10, "output_tokens": out,
             "cache_read_input_tokens": cache_read,
             "cache_creation_input_tokens": cache_creation}
    if cache_1h is not None or cache_5m is not None:
        usage["cache_creation"] = {"ephemeral_1h_input_tokens": cache_1h or 0,
                                   "ephemeral_5m_input_tokens": cache_5m or 0}
    if thinking_logged is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_logged}
    for name, value in (("speed", speed), ("service_tier", service_tier)):
        if value is not None:
            usage[name] = value
    msg = {"role": "assistant", "model": model, "content": [block], "usage": usage}
    rec = {"type": "assistant", "timestamp": ts, "sessionId": sid,
           "isSidechain": sidechain, "message": msg}
    for name, value in (("version", version), ("entrypoint", entrypoint), ("effort", effort)):
        if value is not None:
            rec[name] = value
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


def compact_boundary(ts, sid="s1", trigger=None, pre_tokens=None, version=None):
    rec = {"type": "system", "subtype": "compact_boundary", "timestamp": ts, "sessionId": sid}
    if trigger is not None or pre_tokens is not None:
        rec["compactMetadata"] = {"trigger": trigger, "preTokens": pre_tokens}
    if version is not None:
        rec.update(version=version, entrypoint="cli", isSidechain=False)
    return rec


def turn_duration(ts, duration_ms, message_count, sid="s1", uuid=None, version="2.1.226", entrypoint="cli"):
    """The record Claude Code writes when a main-thread turn ends."""
    rec = {"type": "system", "subtype": "turn_duration", "timestamp": ts, "sessionId": sid,
           "isSidechain": False, "isMeta": False, "entrypoint": entrypoint, "version": version,
           "durationMs": duration_ms, "messageCount": message_count}
    if uuid is not None:
        rec["uuid"] = uuid
    return rec


def stop_hook_summary(ts, hook_count, errors=(), durations=(), sid="s1", uuid=None, version="2.1.226",
                      entrypoint="cli"):
    """The record Claude Code writes after running stop hooks. Each hook's command is
    logged too; ccdrift must not keep it."""
    infos = [{"command": "/home/someone/secret-hook.sh", "durationMs": d} for d in durations]
    infos += [{"command": "/home/someone/secret-hook.sh"}] * max(0, hook_count - len(durations))
    rec = {"type": "system", "subtype": "stop_hook_summary", "timestamp": ts, "sessionId": sid,
           "isSidechain": False, "isMeta": False, "entrypoint": entrypoint, "version": version,
           "hookCount": hook_count, "hookInfos": infos, "hookErrors": list(errors),
           "preventedContinuation": False, "stopReason": "", "hasOutput": False}
    if uuid is not None:
        rec["uuid"] = uuid
    return rec


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def damage_responses_table(path):
    """Replace a history store's responses table with one whose rows can't be read."""
    db = sqlite3.connect(path)
    db.execute("DROP TABLE responses")
    db.execute("CREATE TABLE responses (key INTEGER PRIMARY KEY, file_id INTEGER)")
    db.commit()
    db.close()


def nth_day(i):
    """The ISO date i days after Sep 1, 2026."""
    return (date(2026, 9, 1) + timedelta(days=i)).isoformat()


def daily_turns(days):
    """Turn-level rows for bin_metrics, one dict per day from Sep 1 mapping a
    per-turn column to its values; metric columns left out are 0."""
    df = pd.concat([pd.DataFrame({"day": nth_day(i), **cols})
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


def main_thread_days(path, days, per_day=60):
    """One CLI main-thread session a day from Sep 1: `per_day` responses a minute
    apart, each after a prompt and read 90% from the 1-hour cache. Each entry of
    `days` can set that day's `version` (default "2.1.226"), `haiku` (how many
    responses come from Haiku, default 0), `misses` (how many of the day's last
    responses miss the cache, writing 1000 tokens, default 0), `tier` ("1h" or "5m"
    cache writes, default "1h") and `effort` (default "xhigh")."""
    for d, spec in enumerate(days):
        tier = spec.get("tier", "1h")
        records = []
        for k in range(per_day):
            ts = at(d * DAY + 60 * k)
            model = "claude-haiku-4-5" if k < spec.get("haiku", 0) else "claude-opus-5"
            read, written = (0, 1000) if k >= per_day - spec.get("misses", 0) else (900, 100)
            records += [prompt(ts, sid=f"s{d}"),
                        line(f"m{d}-{k}", text(40), ts=ts, sid=f"s{d}", model=model,
                             cache_read=read, cache_creation=written,
                             cache_1h=written if tier == "1h" else 0, cache_5m=written if tier == "5m" else 0,
                             version=spec.get("version", "2.1.226"), entrypoint="cli",
                             effort=spec.get("effort", "xhigh"))]
        write(path / f"s{d}.jsonl", records)
