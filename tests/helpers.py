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
         cache_1h=None, cache_5m=None, thinking_logged=None, speed=None, service_tier=None,
         agent_type=None, stop_reason=None, miss_reason=None,
         skill=None, plugin=None, mcp_server=None, branch=None, extra=None):
    """One JSONL line as Claude Code writes it: a single content block, with the
    response's message.id, usage and diagnostics repeated on every line of that response.
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
    if stop_reason is not None:
        msg["stop_reason"] = stop_reason
    if miss_reason is not None:
        msg["diagnostics"] = {"cache_miss_reason": {"type": miss_reason}}
    rec = {"type": "assistant", "timestamp": ts, "sessionId": sid,
           "isSidechain": sidechain, "message": msg}
    for name, value in (("version", version), ("entrypoint", entrypoint), ("effort", effort),
                       ("attributionAgent", agent_type), ("attributionSkill", skill),
                       ("attributionPlugin", plugin), ("attributionMcpServer", mcp_server),
                       ("gitBranch", branch)):
        if value is not None:
            rec[name] = value
    if mid is not None:
        msg["id"] = mid
        rec["requestId"] = f"req_{mid}"
    # Top-level keys ccdrift does not read, so a fixture can carry a field Claude Code
    # has started logging.
    if extra:
        rec.update(extra)
    return rec


def deep_line(mid, depth, *, ts, **kw):
    """line() for a tool call whose input nests `depth` lists deep, as JSONL text:
    json.dumps can't write a depth near the recursion limit, so the nesting is spliced in."""
    rec = line(mid, {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"deep": "NESTING"}}, ts=ts, **kw)
    return json.dumps(rec).replace('"NESTING"', "[" * depth + "]" * depth) + "\n"


def decode_limit():
    """The shallowest list nesting json.loads refuses with RecursionError on this Python,
    called from here: about 1,000 on 3.10, where the recursion limit applies, and about
    10,000 on 3.13, whose decoder has a C stack limit of its own."""
    low, high = 1, 1_000_000
    while low < high:
        mid = (low + high) // 2
        try:
            json.loads("[" * mid + "]" * mid)
            low = mid + 1
        except RecursionError:
            high = mid
    return low


def response(mid, *blocks, ts, **kw):
    """All lines of one API response."""
    return [line(mid, b, ts=ts, **kw) for b in blocks]


# The banners Claude Code writes for a failed request, by the kind ccdrift reads.
BANNER_TEXTS = {
    "overloaded": "API Error: 529 Overloaded. This is a server-side issue, usually temporary.",
    "slept": "API Error: Your computer went to sleep mid-response. The response above may be incomplete.",
    "stream": "API Error: The response stopped arriving. The response above may be incomplete.",
    "other": "API Error: something new Claude Code says.",
}


def api_error(ts, sid="s1", kind="overloaded", sidechain=False, version=None, uuid=None):
    """Claude Code's error banner for a failed request: an assistant record of its own,
    with no API call behind it."""
    rec = {"type": "assistant", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain,
           "isApiErrorMessage": True, "error": "server_error", "uuid": uuid or f"err-{sid}-{ts}",
           "message": {"role": "assistant", "model": "<synthetic>", "stop_reason": "stop_sequence",
                       "content": [{"type": "text", "text": BANNER_TEXTS[kind]}]}}
    if kind == "overloaded":
        rec["apiErrorStatus"] = 529
    if version is not None:
        rec.update(version=version, entrypoint="cli")
    return rec


def no_response_stub(ts, sid="s1"):
    """The synthetic record Claude Code writes for a turn it didn't answer: the same
    shape as a banner, but no failure."""
    return {"type": "assistant", "timestamp": ts, "sessionId": sid, "isSidechain": False,
            "isApiErrorMessage": False, "uuid": f"stub-{sid}-{ts}",
            "message": {"role": "assistant", "model": "<synthetic>", "stop_reason": "stop_sequence",
                        "content": [{"type": "text", "text": "No response requested."}]}}


def retry_record(ts, sid="s1", version=None, attempt=1):
    """The system record Claude Code writes when it retries a request by itself: one per
    attempt, `attempt` saying which of them this is. `attempt=None` leaves the number
    out, as transcripts older than it did."""
    rec = {"type": "system", "subtype": "api_error", "level": "error", "timestamp": ts, "sessionId": sid,
           "uuid": f"retry-{sid}-{ts}", "source": "request_retry", "maxRetries": 10,
           "retryInMs": 567, "error": {"message": "Connection error.", "connection": {"code": "ECONNRESET"}}}
    if attempt is not None:
        rec["retryAttempt"] = attempt
    if version is not None:
        rec.update(version=version, entrypoint="cli", isSidechain=False)
    return rec


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


def cost_state(ts, usage, sid="s1", start=None):
    """The record Claude Code writes with what a session cost. It carries no timestamp,
    uuid, version, entrypoint or isSidechain: only a session id and an epoch-millisecond
    start time. `usage` maps a model to its counts and cost."""
    return {"type": "cost-state", "sessionId": sid,
            "startTime": start if start is not None else int(datetime.fromisoformat(
                ts.replace("Z", "+00:00")).timestamp() * 1000),
            "totalCostUSD": round(sum(u.get("costUSD", 0.0) for u in usage.values()), 8),
            "hasUnknownModelCost": False,
            "modelUsage": {model: {"inputTokens": u.get("input", 0), "outputTokens": u.get("output", 0),
                                   "cacheCreationInputTokens": u.get("cache_creation", 0),
                                   "cacheReadInputTokens": u.get("cache_read", 0),
                                   "thinkingTokens": u.get("thinking", 0),
                                   "webSearchRequests": u.get("web", 0),
                                   "costUSD": u.get("costUSD", 0.0)}
                           for model, u in usage.items()}}


# Text no record may leave in ccdrift's tables: listing, prompt and CLAUDE.md text, and a
# path naming a project folder.
PRIVATE_TEXT = "SECRET"
PRIVATE_PATH = "/home/someone/private-project"


def _sized(n: int) -> str:
    """Text of exactly `n` characters that carries PRIVATE_TEXT when it has room."""
    return PRIVATE_TEXT[:n].ljust(n, "x")


def attachment(ts, record, sid="s1", version="2.1.267", sidechain=False, rendered=True):
    """One attachment record as Claude Code writes it before a session's first response,
    wrapping `record`, one of the builders below. From 2.1.263 most of them also carry the
    text Claude Code put in the prompt, as `rendered`."""
    rec = {"type": "attachment", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain, "entrypoint": "cli",
           "version": version, "cwd": PRIVATE_PATH, "gitBranch": "main", "userType": "external",
           "parentUuid": None, "uuid": f"att-{sid}-{ts}-{record['type']}", "attachment": record}
    if rendered:
        rec["rendered"] = [{"content": _sized(40)}]
    return rec


def skill_listing(names, chars=2000, initial=True):
    return {"type": "skill_listing", "content": _sized(chars), "isInitial": initial, "names": list(names),
            "skillCount": len(names)}


def deferred_tools(added, removed=(), line_chars=20):
    """The deferred tools, MCP tools named `mcp__<server>__<tool>`, each on a line of its own."""
    return {"type": "deferred_tools_delta", "addedLines": [_sized(line_chars) for _ in added],
            "addedNames": list(added), "removedNames": list(removed), "readdedNames": [], "pendingMcpServers": [],
            "needsAuthMcpServers": [], "failedMcpServers": [], "wireHiddenNames": []}


def agent_listing(added, removed=(), line_chars=30, initial=True):
    return {"type": "agent_listing_delta", "addedTypes": list(added), "removedTypes": list(removed),
            "addedLines": [_sized(line_chars) for _ in added], "isInitial": initial, "showConcurrencyNote": False}


def mcp_instructions(added, removed=(), block_chars=500):
    return {"type": "mcp_instructions_delta", "addedNames": list(added), "removedNames": list(removed),
            "addedBlocks": [_sized(block_chars) for _ in added]}


def instructions(*chars):
    """CLAUDE.md files, one of each size in `chars`, from 2.1.263."""
    return {"type": "instructions", "files": [{"path": f"{PRIVATE_PATH}/CLAUDE-{i}.md", "type": "project",
                                               "content": _sized(n)} for i, n in enumerate(chars)]}


def prompt_snapshot(*chars, tools=None):
    """The system prompt, in parts of each size in `chars`, from 2.1.267. The snapshot
    before the first response has no `tools`; the one Claude Code writes right after it
    carries the tool definitions, `tools` mapping each name to its description's size."""
    rec = {"type": "prompt_snapshot", "systemPrompt": [_sized(n) for n in chars], "cliPrefix": _sized(30)}
    if tools is not None:
        rec["tools"] = [{"name": name, "description": _sized(n), "schema": {"type": "object", "properties": {}}}
                        for name, n in tools.items()]
    return rec


def tool_use(tool_id, name):
    """A tool call as one content block of a response."""
    return {"type": "tool_use", "id": tool_id, "name": name, "input": {"command": PRIVATE_TEXT}}


def hook_record(ts, event, tool_id, tool, kind="hook_success", sid="s1", version="2.1.261", sidechain=False):
    """The attachment Claude Code writes when a hook of `event` ran on a tool call: a
    success, a non-blocking error (exit code 1) or a blocking error. Its command, output
    and content can name private paths."""
    if kind == "hook_blocking_error":
        record = {"type": kind, "hookEvent": event, "hookName": f"{event}:{tool}", "toolUseID": tool_id,
                  "blockingError": {"blockingError": PRIVATE_TEXT, "command": f"{PRIVATE_PATH}/hook.sh"}}
    else:
        record = {"type": kind, "hookEvent": event, "hookName": f"{event}:{tool}", "toolUseID": tool_id,
                  "command": f"{PRIVATE_PATH}/hook.sh", "stdout": PRIVATE_TEXT, "stderr": "",
                  "exitCode": 0 if kind == "hook_success" else 1, "durationMs": 40}
        if kind == "hook_success":
            record["content"] = PRIVATE_TEXT
    rec = {"type": "attachment", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain, "entrypoint": "cli",
           "version": version, "cwd": PRIVATE_PATH, "gitBranch": "main", "userType": "external",
           "parentUuid": None, "uuid": f"hook-{sid}-{ts}-{event}-{tool_id}", "attachment": record}
    if sidechain:
        rec["agentId"] = "a1"
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


def main_thread_days(path, days, per_day=60, first_day=0):
    """One CLI main-thread session a day from Sep 1, or `first_day` days after it:
    `per_day` responses a minute apart, each after a prompt and read 90% from the
    1-hour cache. Each entry of `days` can set that day's `version` (default
    "2.1.226"), `haiku` (how many responses come from Haiku, default 0), `misses` (how
    many of the day's last responses miss the cache, writing 1000 tokens, default 0),
    `tier` ("1h" or "5m" cache writes, default "1h"), `effort` (default "xhigh"),
    `entrypoint` (default "cli"), `reason` (the cache-miss reason Claude Code recorded
    that day, default None), `reasons` (how many of the day's responses carry it,
    default 0) and `extra` (top-level keys ccdrift does not read, default None)."""
    for d, spec in enumerate(days, start=first_day):
        tier = spec.get("tier", "1h")
        records = []
        for k in range(per_day):
            ts = at(d * DAY + 60 * k)
            model = "claude-haiku-4-5" if k < spec.get("haiku", 0) else "claude-opus-5"
            read, written = (0, 1000) if k >= per_day - spec.get("misses", 0) else (900, 100)
            reason = spec.get("reason") if k < spec.get("reasons", 0) else None
            records += [prompt(ts, sid=f"s{d}"),
                        line(f"m{d}-{k}", text(40), ts=ts, sid=f"s{d}", model=model,
                             cache_read=read, cache_creation=written,
                             cache_1h=written if tier == "1h" else 0, cache_5m=written if tier == "5m" else 0,
                             version=spec.get("version", "2.1.226"), entrypoint=spec.get("entrypoint", "cli"),
                             effort=spec.get("effort", "xhigh"), miss_reason=reason, extra=spec.get("extra"))]
        write(path / f"s{d}.jsonl", records)


def failure_days(path, days, per_day=60):
    """One CLI main-thread session a day from Sep 1: `per_day` responses a minute apart,
    each after a prompt and read 90% from the cache, plus what each entry of `days`
    asks for: `errors` banners of `kind` (default "overloaded"), `slept` banners,
    `retries` retry records, and `truncated` responses that stop at the token limit.
    `version` sets the day's Claude Code version (default "2.1.226")."""
    for d, spec in enumerate(days):
        version = spec.get("version", "2.1.226")
        records = []
        for k in range(per_day):
            ts = at(d * DAY + 60 * k)
            records += [prompt(ts, sid=f"s{d}"),
                        line(f"m{d}-{k}", text(40), ts=ts, sid=f"s{d}", cache_read=900, cache_creation=100,
                             cache_1h=100, cache_5m=0, version=version, entrypoint="cli", effort="xhigh",
                             stop_reason="max_tokens" if k < spec.get("truncated", 0) else "end_turn")]
        after = d * DAY + 60 * per_day
        for j in range(spec.get("errors", 0)):
            records.append(api_error(at(after + j), sid=f"s{d}", kind=spec.get("kind", "overloaded"), version=version))
        for j in range(spec.get("slept", 0)):
            records.append(api_error(at(after + 100 + j), sid=f"s{d}", kind="slept", version=version))
        for j in range(spec.get("retries", 0)):
            records.append(retry_record(at(after + 200 + j), sid=f"s{d}", version=version))
        write(path / f"s{d}.jsonl", records)


def tool_loop_days(path, days, per_day=100, misses=0, subagent=False):
    """One CLI session a day from Sep 1 on 2.1.226, on the main thread or in a subagent
    of it: a prompt, then `per_day` turns a minute apart, each after a tool result and
    reading back what the one before had cached, with a second prompt halfway. The last
    `misses` turns of the last day miss the cache, reading nothing."""
    for d in range(days):
        records, cached = [], 0
        for k in range(per_day + 1):
            ts = at(d * DAY + 60 * k)
            opens = k in (0, per_day // 2)
            records.append(prompt(ts, sid=f"s{d}", sidechain=subagent) if opens
                           else tool_result(ts, sid=f"s{d}", sidechain=subagent))
            missed = d == days - 1 and k > per_day - misses
            read = 0 if k == 0 or missed else cached
            written = 1000 if k == 0 else 100 + (cached if missed else 0)
            records.append(line(f"{'a' if subagent else 'm'}{d}-{k}", text(40), ts=ts, sid=f"s{d}",
                                sidechain=subagent, cache_read=read, cache_creation=written, version="2.1.226",
                                entrypoint="cli"))
            cached = read + written
        write(path / (f"s{d}/subagents/agent-a.jsonl" if subagent else f"s{d}.jsonl"), records)


def hook_days_logs(path, failing_days, days, per_day=10):
    """`per_day` stop-hook summaries a day from Sep 1 in one transcript; on the day
    indexes in `failing_days` every hook reports an error."""
    records = [stop_hook_summary(at(d * DAY + 300 + k), 1, durations=(400,), uuid=f"hook-{d}-{k}",
                                 errors=("exit 1",) if d in failing_days else ())
               for d in range(days) for k in range(per_day)]
    write(path / "hooks.jsonl", records)
