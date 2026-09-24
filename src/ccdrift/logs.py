"""Find and parse Claude Code session transcripts into one row per API response."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

# NUL, line breaks, and the escapes that move a terminal's cursor or retitle its window.
# Defined in `texts`, which imports nothing heavy, and re-exported here: `schedule` and the
# parser have always read it from this module.
from ccdrift.texts import CONTROL_CHARS, LOG_LINES, no_transcripts_message


# ---------------------------------------------------------------------------
# Defensive field access
# ---------------------------------------------------------------------------
# Claude Code's JSONL schema has shifted across versions. Rather than hard-code
# one path, every field is resolved by trying a list of candidate dotted paths
# and taking the first that exists. If you discover a new variant, add it here:
# this is the single place schema drift is absorbed.

CANDIDATES: dict[str, list[str]] = {
    "role":              ["type", "message.role", "role"],
    "model":             ["message.model", "model"],
    "input_tokens":      ["message.usage.input_tokens", "usage.input_tokens"],
    "output_tokens":     ["message.usage.output_tokens", "usage.output_tokens"],
    "cache_creation":    ["message.usage.cache_creation_input_tokens",
                          "usage.cache_creation_input_tokens"],
    "cache_read":        ["message.usage.cache_read_input_tokens",
                          "usage.cache_read_input_tokens"],
    "cache_1h":          ["message.usage.cache_creation.ephemeral_1h_input_tokens",
                          "usage.cache_creation.ephemeral_1h_input_tokens"],
    "cache_5m":          ["message.usage.cache_creation.ephemeral_5m_input_tokens",
                          "usage.cache_creation.ephemeral_5m_input_tokens"],
    "thinking_logged":   ["message.usage.output_tokens_details.thinking_tokens",
                          "usage.output_tokens_details.thinking_tokens"],
    "speed":             ["message.usage.speed", "usage.speed"],
    "service_tier":      ["message.usage.service_tier", "usage.service_tier"],
    "effort":            ["effort"],
    "agent_type":        ["attributionAgent"],
    "version":           ["version"],
    "entrypoint":        ["entrypoint"],
    "timestamp":         ["timestamp", "message.timestamp", "createdAt"],
    "session_id":        ["sessionId", "session_id", "message.sessionId"],
    "content":           ["message.content", "content"],
    "is_sidechain":      ["isSidechain", "message.isSidechain", "is_sidechain"],
    "message_id":        ["message.id"],
    "request_id":        ["requestId", "request_id"],
    "uuid":              ["uuid"],
    "is_meta":           ["isMeta", "is_meta"],
    "subtype":           ["subtype"],
    "duration_ms":       ["durationMs"],
    "message_count":     ["messageCount"],
    "hook_count":        ["hookCount"],
    "hook_errors":       ["hookErrors"],
    "hook_infos":        ["hookInfos"],
    "prevented":         ["preventedContinuation"],
    "stop_reason":       ["message.stop_reason"],
    "miss_reason":       ["message.diagnostics.cache_miss_reason.type"],
    "attribution_skill":  ["attributionSkill"],
    "attribution_plugin": ["attributionPlugin"],
    "attribution_mcp":    ["attributionMcpServer"],
    "git_branch":         ["gitBranch"],
    "is_api_error":      ["isApiErrorMessage"],
    "api_error_status":  ["apiErrorStatus"],
    "retry_attempt":     ["retryAttempt"],
    "compact_trigger":   ["compactMetadata.trigger"],
    "compact_pre_tokens": ["compactMetadata.preTokens"],
    "cost_usage":        ["modelUsage"],
    "cost_start":        ["startTime"],
    # What a session started with, from the attachment records before its first response.
    # A delta names what it adds in addedNames, or addedTypes for agents, and carries its
    # text as addedLines, or addedBlocks for MCP instructions.
    "attachment_type":   ["attachment.type"],
    "is_initial":        ["attachment.isInitial"],
    "skill_names":       ["attachment.names"],
    "skill_listing":     ["attachment.content"],
    "added_names":       ["attachment.addedNames", "attachment.addedTypes"],
    "removed_names":     ["attachment.removedNames", "attachment.removedTypes"],
    "added_text":        ["attachment.addedLines", "attachment.addedBlocks"],
    "instruction_files": ["attachment.files"],
    "system_prompt":     ["attachment.systemPrompt"],
    "tool_definitions":  ["attachment.tools"],
    # A hook run on a tool call. hookName ("PreToolUse:Bash") carries the event too, and is
    # read only when hookEvent is missing.
    "hook_event":        ["attachment.hookEvent"],
    "hook_name":         ["attachment.hookName"],
    "tool_use_id":       ["attachment.toolUseID"],
}


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def field_get(obj: dict, logical: str, default: Any = None) -> Any:
    for path in CANDIDATES.get(logical, []):
        val = _dig(obj, path)
        if val is not None:
            return val
    return default


# How deep the census of a record's keys reaches. Below these the keys are per-call
# detail (`toolUseResult` alone carries over 40 in one person's logs), which would
# tally the shape of every tool result rather than the shape of a response.
CENSUS_NESTED = ("message", "message.usage")


# A CANDIDATES path cut to the depth the census walks, so the arrival rule doesn't
# report a path ccdrift reads through a deeper leaf: it reads
# message.usage.output_tokens_details.thinking_tokens, which the census records as
# message.usage.output_tokens_details.
def census_form(path: str) -> str:
    parts = path.split(".")
    if parts[0] != "message":
        return parts[0]
    return ".".join(parts[:3] if parts[1:2] == ["usage"] else parts[:2])


READ_PATHS = frozenset(census_form(p) for paths in CANDIDATES.values() for p in paths)


def record_paths(obj: dict) -> set[str]:
    """The dotted key paths one assistant record carries: its own keys, `message.*` and
    `message.usage.*`. Fifty of them in one person's logs over 25 Claude Code versions."""
    paths = set(obj)
    for prefix in CENSUS_NESTED:
        nested = _dig(obj, prefix)
        if isinstance(nested, dict):
            paths.update(f"{prefix}.{name}" for name in nested)
    return paths


def is_assistant(obj: dict) -> bool:
    role = field_get(obj, "role")
    # "type" variant carries "assistant"; "message.role" variant also "assistant".
    return role == "assistant"


# Thinking tokens per signature character. Claude Code stores thinking blocks
# without their text (or with a short summary), but each keeps its encrypted
# signature. Fitted on 20,896 real responses with a final output count: output
# tokens minus visible chars/4 ≈ 0.30 × signature chars (r = 0.97; per-model
# slopes 0.27–0.31). The stored thinking text tracks it poorly (r = 0.20).
TOKENS_PER_SIGNATURE_CHAR = 0.30


def content_chars(content: Any) -> tuple[int, int]:
    """Return (signature_chars, visible_chars) for one line's content: signature
    length of thinking blocks, and length of text plus tool-call input."""
    if isinstance(content, str):
        return 0, len(content)
    signature = visible = 0
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("thinking", "redacted_thinking", "reasoning"):
                sig = block.get("signature") or block.get("data") or ""
                if isinstance(sig, str):
                    signature += len(sig)
            elif btype == "text":
                txt = block.get("text") or ""
                if isinstance(txt, str):
                    visible += len(txt)
            elif btype == "tool_use":
                visible += len(json.dumps(block.get("input") or {}))
    return signature, visible


def mcp_tool_names(content: Any) -> list[str]:
    names: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "")
                if isinstance(name, str) and name.startswith("mcp__"):
                    names.append(name)
    return names


def has_tool_result(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


# Times a transcript line can carry: pandas holds 1677 to 2262 in nanoseconds.
MIN_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_TIME = datetime(2200, 1, 1, tzinfo=timezone.utc)


def parse_ts(raw: Any) -> Optional[datetime]:
    """The time `raw` names, in UTC; None when it names none, or one pandas can't hold."""
    dt = None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # epoch seconds or millis
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        try:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
        except (OverflowError, ValueError, OSError):
            return None
    elif isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    if dt is None or not MIN_TIME <= dt <= MAX_TIME:
        return None
    return dt


# ---------------------------------------------------------------------------
# Parsing → turn-level feature table
# ---------------------------------------------------------------------------

# Text fields that keep the first value logged across a response's lines, and
# token counts that keep the largest.
SETTING_FIELDS = ("version", "entrypoint", "effort", "speed", "service_tier", "agent_type")
# Where a response's work came from, and the branch it ran on. Kept apart from
# SETTING_FIELDS, which means a setting Claude Code chose for the request; a branch
# name is not one, and history.TEXT_COLUMNS is built from both.
ATTRIBUTION_FIELDS = ("attribution_skill", "attribution_plugin", "attribution_mcp", "git_branch")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation", "cache_read", "cache_1h", "cache_5m")

# What one model's slice of a cost-state record counts. `web_searches` is fitted as its
# own rate because it is billed per request, not per token: leaving it out makes
# claude-haiku-4-5 fit at 4.72% instead of 0.00%.
USAGE_COUNTS = ("input_tokens", "output_tokens", "cache_creation", "cache_read",
                "thinking_tokens", "web_searches")

# The keys Claude Code writes for each of USAGE_COUNTS inside modelUsage.
USAGE_KEYS = ("inputTokens", "outputTokens", "cacheCreationInputTokens", "cacheReadInputTokens",
              "thinkingTokens", "webSearchRequests")


# What a session started with, one row per transcript. The name sets are sorted lists and
# the sizes are characters; None means Claude Code logged no record of that part, so a
# part a version doesn't log reads as unknown rather than as empty. `deferred` holds the
# deferred tools, MCP ones named mcp__<server>__<tool>; `mcp` the MCP servers that sent
# instructions; `tools` the tools whose definitions were sent in full.
COMPONENT_SETS = ("skills", "deferred", "agents", "mcp", "tools")
COMPONENT_SIZES = ("skills_chars", "deferred_chars", "agents_chars", "mcp_chars", "claude_md_files",
                   "claude_md_chars", "system_chars", "tools_chars", "message_chars")
COMPONENT_COLUMNS = ("source_file", "session_id") + COMPONENT_SETS + COMPONENT_SIZES

# The delta records, by the component whose names and size they change.
COMPONENT_DELTAS = {"deferred_tools_delta": "deferred", "agent_listing_delta": "agents",
                    "mcp_instructions_delta": "mcp"}


def new_components(rel: str) -> dict:
    """A transcript's component row before any record is read: nothing logged, and no
    user text yet."""
    return {"source_file": rel, "session_id": None, **{name: None for name in COMPONENT_SETS + COMPONENT_SIZES},
            "message_chars": 0}


def _names(value: Any) -> Optional[set[str]]:
    """The names in a logged list; None when it isn't a list."""
    if not isinstance(value, list):
        return None
    return {name for name in map(_text, value) if name is not None}


def _size(value: Any) -> Optional[int]:
    """The characters in a logged text or list of texts; None when it is neither."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return sum(len(part) for part in value)
    return None


def read_attachment(row: dict, obj: dict, started: bool) -> None:
    """Fold one attachment record into a transcript's component row. Before the first
    response (`started` False) every kind counts: a delta adds and removes names and adds
    its size, unless it is initial, and a skills listing, CLAUDE.md record or system
    prompt replaces what came before. After it, only the tool definitions are read, from
    the first snapshot that carries them: from 2.1.267 Claude Code writes them in the
    snapshot right after the first response, describing that same first request. A
    record that isn't the shape Claude Code writes is left out. Only names and sizes are
    kept: never listing, prompt, CLAUDE.md or tool text, nor a CLAUDE.md path."""
    kind = field_get(obj, "attachment_type")
    if kind == "prompt_snapshot" and row["tools"] is None:
        definitions = field_get(obj, "tool_definitions")
        if isinstance(definitions, list) and definitions and all(isinstance(d, dict) for d in definitions):
            names = _names([d.get("name") for d in definitions])
            if names:
                row["tools"] = sorted(names)
                row["tools_chars"] = len(json.dumps(definitions, ensure_ascii=False))
    if started:
        return
    if kind in COMPONENT_DELTAS:
        part = COMPONENT_DELTAS[kind]
        added, removed = _names(field_get(obj, "added_names")), _names(field_get(obj, "removed_names", default=[]))
        size = _size(field_get(obj, "added_text", default=[]))
        if added is None or removed is None or size is None:
            return
        fresh = row[part] is None or field_get(obj, "is_initial") is True
        row[part] = sorted(((set() if fresh else set(row[part])) | added) - removed)
        row[f"{part}_chars"] = size + (0 if fresh else row[f"{part}_chars"])
    elif kind == "skill_listing":
        names, size = _names(field_get(obj, "skill_names")), _size(field_get(obj, "skill_listing"))
        if names is not None and size is not None:
            row["skills"], row["skills_chars"] = sorted(names), size
    elif kind == "instructions":
        files = field_get(obj, "instruction_files")
        sizes = [_size(f.get("content")) if isinstance(f, dict) else None for f in files] \
            if isinstance(files, list) else [None]
        if None not in sizes:
            row["claude_md_files"], row["claude_md_chars"] = len(sizes), sum(sizes)
    elif kind == "prompt_snapshot":
        size = _size(field_get(obj, "system_prompt"))
        if size is not None:
            row["system_chars"] = size


# The hook events whose runs are matched to tool calls, and the attachment kinds that
# mean a hook ran, whether it succeeded or not. Measured on 2026-09-24 over the owner's
# corpus: 127,647 hook_success records and 12 errors, 120,402 of them on these two events.
HOOK_EVENTS = ("PreToolUse", "PostToolUse")
HOOK_RAN = frozenset({"hook_success", "hook_non_blocking_error", "hook_blocking_error"})
COVERAGE_COLUMNS = ("source_file", "session_id", "day", "version", "entrypoint", "is_sidechain", "event", "tool",
                    "calls", "hooked")


def tool_uses(content: Any) -> list[tuple[str, str]]:
    """The id and name of each tool call in one line's content."""
    if not isinstance(content, list):
        return []
    return [(block["id"], block["name"]) for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
            and isinstance(block.get("id"), str) and isinstance(block.get("name"), str)]


def hook_tool(name: str) -> Optional[str]:
    """A tool as hook coverage counts it: a built-in tool by its name, an MCP tool by its
    server, as mcp__<server>, since a hook matches a server's tools alike."""
    clean = _text(name)
    if clean is None or not clean.startswith("mcp__"):
        return clean
    return "mcp__" + clean[len("mcp__"):].split("__", 1)[0]


def read_hook(hooked: dict[str, set[str]], obj: dict) -> None:
    """Note the tool call a hook ran on, by event. Only its event and the tool call's id
    are read, never its command, output or content, which can name private paths."""
    if field_get(obj, "attachment_type") not in HOOK_RAN:
        return
    event = field_get(obj, "hook_event")
    if not isinstance(event, str):
        name = field_get(obj, "hook_name")
        event = name.split(":", 1)[0] if isinstance(name, str) else None
    tool_id = field_get(obj, "tool_use_id")
    if event in hooked and isinstance(tool_id, str):
        hooked[event].add(tool_id)


@dataclass
class ParsedFile:
    """One transcript's responses, turn durations, hook runs, compactions and per-model
    cost records by key, the transcript's first session id, line counts for --verbose,
    the key census (field_census, field_days) of its responses, and what its session
    started with (components, one row), and hook coverage (hook_coverage: for each day,
    version, entrypoint, thread, hook event and tool, [tool calls, calls a hook ran on])."""
    responses: dict[str, dict] = field(default_factory=dict)
    durations: dict[str, dict] = field(default_factory=dict)
    hook_runs: dict[str, dict] = field(default_factory=dict)
    compactions: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, dict] = field(default_factory=dict)
    field_census: dict[tuple[str, str, str], int] = field(default_factory=dict)
    field_days: dict[tuple[str, str], int] = field(default_factory=dict)
    model_usage: dict[str, dict] = field(default_factory=dict)
    components: dict = field(default_factory=dict)
    hook_coverage: dict[tuple, list[int]] = field(default_factory=dict)
    session_id: Optional[str] = None
    lines: int = 0
    bad_json: int = 0
    assistant_lines: int = 0


def jsonl_files(source: Path) -> list[tuple[Path, str]]:
    """Transcripts under `source` with their paths relative to it, POSIX style,
    sorted by that path. When a response appears in several transcripts the first
    in this order owns it, so every reader must use the same order."""
    if source.is_file():
        return [(source, source.name)]
    return sorted(((fp, fp.relative_to(source).as_posix()) for fp in source.rglob("*.jsonl")),
                  key=lambda item: item[1])


def iter_jsonl_files(source: Path) -> Iterable[Path]:
    for fp, _ in jsonl_files(source):
        yield fp


def default_source(environ: Mapping[str, str] = os.environ) -> Path:
    """Where Claude Code keeps session transcripts: $CLAUDE_CONFIG_DIR/projects when
    that variable is set, otherwise ~/.claude/projects."""
    config = environ.get("CLAUDE_CONFIG_DIR")
    return (Path(config).expanduser() if config else Path.home() / ".claude") / "projects"


def _text(value: Any) -> Optional[str]:
    """`value` when it is text, without what SQLite, notifications, --exec or a terminal
    can't take: a lone surrogate (JSON "\\ud800") reads as "?", and control characters
    are dropped. Agent names come from agent files in a repository or plugin."""
    if not isinstance(value, str):
        return None
    return CONTROL_CHARS.sub("", value).encode("utf-8", "replace").decode("utf-8") or None


# Claude Code's own banners for a failed request, matched by the words they open
# with; one it words differently counts as "other". None of the text is kept.
BANNERS = (("API Error: 529 Overloaded", "overloaded"),
           ("API Error: Your computer went to sleep", "slept"),
           ("API Error: The response stopped arriving", "stream"),
           ("API Error: Response stalled mid-stream", "stream"))


def banner_kind(content: Any) -> str:
    """Which failure an error banner reports, from the words it opens with."""
    text = content if isinstance(content, str) else ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = str(content[0].get("text") or "")
    return next((kind for prefix, kind in BANNERS if text.startswith(prefix)), "other")


def _status(value: Any) -> Optional[int]:
    """An HTTP status as an integer; None when Claude Code logged none, or a number no
    status can be: NaN, Infinity or one past MAX_COUNT, which JSON parsing accepts and
    SQLite can't store. Comparing first keeps an integer too large for a float from
    being converted to one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if abs(value) <= MAX_COUNT else None


def _cost(value: Any) -> Optional[float]:
    """A cost in dollars, kept as the float Claude Code logged with no rounding; None
    when it is absent, a boolean, not a number, or not finite (NaN or Infinity, which
    JSON parsing accepts). A later fit compares this to a residual bound, so a non-finite
    value must become None rather than a literal NaN or Infinity that comparison lets through."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _record(obj: dict, key: str, rel: str) -> dict:
    """The fields a turn duration, hook run or compaction record keeps in common."""
    return {"key": key, "timestamp": parse_ts(field_get(obj, "timestamp")),
            "version": _text(field_get(obj, "version")), "entrypoint": _text(field_get(obj, "entrypoint")),
            "is_sidechain": bool(field_get(obj, "is_sidechain", default=False)), "source_file": rel}


def parse_file(fp: Path, rel: str) -> ParsedFile:
    """Parse one transcript. Claude Code writes each content block of a response on
    its own line with the response's usage repeated, so lines that share a
    message.id are merged. Raises OSError when the file can't be read."""
    parsed = ParsedFile(components=new_components(rel))
    # Set by lines between two responses: a prompt typed by the user opens
    # a new turn, and a compaction rewrites the conversation.
    prompt_pending = compact_pending = False
    main_thread_seen = False
    # Each tool call's response key and tool, and the calls each hook event ran on,
    # matched once the whole transcript is read: a hook's record follows its call.
    tool_calls: dict[str, tuple[str, str]] = {}
    hooked: dict[str, set[str]] = {event: set() for event in HOOK_EVENTS}
    with fp.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parsed.lines += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, RecursionError):  # RecursionError: nested past the decoder's limit
                parsed.bad_json += 1
                continue
            if not isinstance(obj, dict):
                continue
            if parsed.session_id is None:
                parsed.session_id = _text(field_get(obj, "session_id"))
            role = field_get(obj, "role")
            if role == "attachment":
                read_hook(hooked, obj)
                if not field_get(obj, "is_sidechain", default=False):
                    read_attachment(parsed.components, obj, main_thread_seen)
                continue
            if role == "user":
                content = field_get(obj, "content")
                if not field_get(obj, "is_meta") and not has_tool_result(content):
                    prompt_pending = True
                # Everything the user sent before the first response went with it.
                if not main_thread_seen and not field_get(obj, "is_sidechain", default=False):
                    parsed.components["message_chars"] += content_chars(content)[1]
                continue
            if role == "system":
                subtype = field_get(obj, "subtype")
                key = str(field_get(obj, "uuid") or f"{rel}:{line_no}")
                if subtype == "compact_boundary":
                    compact_pending = True
                    pre_tokens = field_get(obj, "compact_pre_tokens")
                    parsed.compactions.setdefault(key, {
                        **_record(obj, key, rel),
                        "trigger": _text(field_get(obj, "compact_trigger")),
                        "pre_tokens": None if pre_tokens is None else _num(pre_tokens),
                    })
                elif subtype == "turn_duration":
                    parsed.durations.setdefault(key, {
                        **_record(obj, key, rel),
                        "duration_ms": _num(field_get(obj, "duration_ms")),
                        "message_count": int(_num(field_get(obj, "message_count"))),
                    })
                elif subtype == "api_error":
                    # A request Claude Code retried by itself; its error message and
                    # connection details aren't kept, only that it happened. Claude Code
                    # writes one record per attempt ("retryAttempt of maxRetries"), so
                    # only the first is kept: one failure per request, not per attempt.
                    # Anything but a later attempt counts: no attempt number (older
                    # transcripts), one ccdrift can't read, or a version numbering them
                    # from 0. Dropping a real failure is worse than counting one.
                    attempt = field_get(obj, "retry_attempt")
                    if attempt is None or _num(attempt) <= 1:
                        parsed.failures.setdefault(key, {**_record(obj, key, rel), "kind": "retry", "status": None})
                elif subtype == "stop_hook_summary":
                    infos = field_get(obj, "hook_infos")
                    durations = [_num(i["durationMs"]) for i in infos if isinstance(i, dict)
                                 and i.get("durationMs") is not None] if isinstance(infos, list) else []
                    errors = field_get(obj, "hook_errors")
                    # Each hook's command is logged too; it may name private paths, so it isn't kept.
                    parsed.hook_runs.setdefault(key, {
                        **_record(obj, key, rel),
                        "hook_count": int(_num(field_get(obj, "hook_count"))),
                        "error_count": len(errors) if isinstance(errors, list) else 0,
                        "duration_ms": sum(durations) if durations else None,
                        "prevented": bool(field_get(obj, "prevented", default=False)),
                    })
                continue
            if role == "cost-state":
                # These records carry no timestamp, uuid, version, entrypoint or thread,
                # only a session id and an epoch-millisecond start, so they cannot use
                # _record() and must not claim a version they never logged.
                usage = field_get(obj, "cost_usage")
                if isinstance(usage, dict):
                    start = field_get(obj, "cost_start")
                    when = parse_ts(start)
                    session = _text(field_get(obj, "session_id"))
                    for model, counts in usage.items():
                        if not isinstance(counts, dict):
                            continue
                        name = _text(model) or "unknown"
                        # Session, start and model rather than a line position, so the
                        # same record copied into a resumed transcript is one row.
                        key = (f"{session}:{start}:{name}" if session and start is not None
                               else f"{rel}:{line_no}:{name}")
                        parsed.model_usage.setdefault(key, {
                            "key": key, "timestamp": when, "model": name,
                            **{column: _num(counts.get(raw)) for column, raw in zip(USAGE_COUNTS, USAGE_KEYS)},
                            "cost_usd": _cost(counts.get("costUSD")),
                            "source_file": rel,
                        })
                continue
            if role != "assistant":
                continue
            parsed.assistant_lines += 1
            if field_get(obj, "is_api_error") is True:
                # Claude Code writes the banner as a response of its own, with no API
                # call behind it: it is a failure, not a response.
                key = str(field_get(obj, "uuid") or f"{rel}:{line_no}")
                parsed.failures.setdefault(key, {
                    **_record(obj, key, rel), "kind": banner_kind(field_get(obj, "content")),
                    "status": _status(field_get(obj, "api_error_status")),
                })
                continue
            model = _text(field_get(obj, "model")) or "unknown"
            if model == "<synthetic>":  # Claude Code placeholder, no API call
                continue
            key = str(field_get(obj, "message_id") or field_get(obj, "request_id")
                      or f"{rel}:{line_no}")
            row = parsed.responses.get(key)
            if row is None:
                is_sidechain = bool(field_get(obj, "is_sidechain", default=False))
                row = parsed.responses[key] = {
                    "key":              key,
                    "timestamp":        parse_ts(field_get(obj, "timestamp")),
                    "model":            model,
                    "stop_reason":      None,
                    "miss_reason":      None,
                    **{name: None for name in ATTRIBUTION_FIELDS},
                    **{name: None for name in SETTING_FIELDS},
                    **{name: 0.0 for name in TOKEN_FIELDS},
                    "thinking_logged":  None,
                    "signature_chars":  0,
                    "visible_chars":    0,
                    "n_mcp_calls":      0,
                    "is_sidechain":     is_sidechain,
                    "new_prompt":       prompt_pending,
                    "after_compaction": compact_pending,
                    # The transcript's first main-thread response, a copy or not: a
                    # resumed session's transcript opens with copies of the responses
                    # before it, which the transcript they came from owns.
                    "opens_transcript": not is_sidechain and not main_thread_seen,
                    "source_file":      rel,
                    # Folded into field_census below and removed, so the row stays the
                    # shape `frame` and the store expect.
                    "census_paths":     set(),
                }
                main_thread_seen = main_thread_seen or not is_sidechain
            prompt_pending = compact_pending = False
            for name in ("stop_reason", "miss_reason", *ATTRIBUTION_FIELDS, *SETTING_FIELDS):
                if row[name] is None:
                    row[name] = _text(field_get(obj, name))
            # output_tokens grows while streaming, so the largest is the
            # latest. About a third of real responses never log the final
            # count (stop_reason stays null); no metric relies on it.
            for name in TOKEN_FIELDS:
                row[name] = max(row[name], _num(field_get(obj, name)))
            logged = field_get(obj, "thinking_logged")
            if logged is not None:
                row["thinking_logged"] = max(row["thinking_logged"] or 0.0, _num(logged))
            content = field_get(obj, "content")
            for tool_id, name in tool_uses(content):
                tool_calls.setdefault(tool_id, (key, name))
            signature, visible = content_chars(content)
            row["signature_chars"] += signature
            row["visible_chars"] += visible
            row["n_mcp_calls"] += len(mcp_tool_names(content))
            row["census_paths"].update(record_paths(obj))
    # The census counts only the responses the daily check judges on the thread test,
    # so a share worked out from it is a share of the population the rule judges. The
    # day is not yet known to be complete, which the rule applies instead.
    for row in parsed.responses.values():
        paths = row.pop("census_paths", None)
        if not paths or row["timestamp"] is None or row["is_sidechain"]:
            continue
        if str(row["entrypoint"] or "").startswith(SDK_ENTRYPOINT_PREFIX):
            continue
        day = row["timestamp"].astimezone(timezone.utc).date().isoformat()
        version = row["version"] or "unknown"
        parsed.field_days[(day, version)] = parsed.field_days.get((day, version), 0) + 1
        for path in paths:
            clean = _text(path)
            if clean is None:
                continue
            key = (day, version, clean)
            parsed.field_census[key] = parsed.field_census.get(key, 0) + 1
    # Every tool call counts once per hook event, hooked or not, under the day, version,
    # entrypoint and thread of the response that made it.
    for tool_id, (key, name) in tool_calls.items():
        row, tool = parsed.responses.get(key), hook_tool(name)
        if row is None or row["timestamp"] is None or tool is None:
            continue
        day = row["timestamp"].astimezone(timezone.utc).date().isoformat()
        for event in HOOK_EVENTS:
            counts = parsed.hook_coverage.setdefault(
                (day, row["version"], row["entrypoint"], row["is_sidechain"], event, tool), [0, 0])
            counts[0] += 1
            counts[1] += tool_id in hooked[event]
    # A transcript is one session, even when a resumed session's lines carry
    # another id. Every record kind is backfilled, including the cost records, so a table
    # read straight from the transcripts has the columns the store's own read of it does.
    session = parsed.session_id or fp.stem
    for row in (*parsed.responses.values(), *parsed.durations.values(), *parsed.hook_runs.values(),
                *parsed.compactions.values(), *parsed.failures.values(), *parsed.model_usage.values(),
                parsed.components):
        row["session_id"] = session
    return parsed


def parse_source(source: Path, verbose: bool = False) -> pd.DataFrame:
    """One row per API response in the transcripts under `source`. A response copied
    into a second transcript (e.g. when a session is resumed) counts once, from the
    transcript whose path sorts first."""
    rows: dict[str, dict] = {}
    files = jsonl_files(source)
    lines = bad_json = assistant_lines = 0
    for fp, rel in files:
        try:
            parsed = parse_file(fp, rel)
        except OSError as e:
            if verbose:
                print(LOG_LINES["unreadable"].format(path=fp, error=e), file=sys.stderr)
            continue
        lines += parsed.lines
        bad_json += parsed.bad_json
        assistant_lines += parsed.assistant_lines
        for key, row in parsed.responses.items():
            rows.setdefault(key, row)
    if verbose:
        print(LOG_LINES["counts"].format(files=len(files), lines=lines, bad_json=bad_json,
                                         assistant_lines=assistant_lines, responses=len(rows)), file=sys.stderr)
    return frame(list(rows.values()))


FLOAT_COLUMNS = TOKEN_FIELDS + ("thinking_logged",)


def _rows_with_parsed_timestamp(rows) -> pd.DataFrame:
    """The DataFrame both `frame` and `duration_frame` start from: raw rows (as
    parse_file makes them or the history store loads them), minus the internal
    `key`, with `timestamp` parsed to UTC. Empty when `rows` is empty."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop(columns=["key"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _with_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add the UTC day bucket used for binning. Called after sorting by timestamp,
    since callers may rely on `day` reflecting the final row order."""
    df["day"] = df["timestamp"].dt.tz_convert("UTC").dt.date.astype("string")
    return df


def frame(rows) -> pd.DataFrame:
    """The response table the detector reads, from raw response rows as parse_file
    makes them or the history store loads them: estimated token counts, the
    main-thread flag, idle gaps, the UTC day and the metric columns."""
    df = _rows_with_parsed_timestamp(rows)
    if df.empty:
        return df
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["thinking_tokens"] = (df["signature_chars"] * TOKENS_PER_SIGNATURE_CHAR).round()
    df["visible_tokens"] = (df["visible_chars"] / 4.0).round()
    # The history store loads these as 0/1 integers, and add_ratios combines them
    # with & and ~, which is silently wrong on raw ints.
    df["is_sidechain"] = df["is_sidechain"].fillna(False).astype(bool)
    df["new_prompt"] = df["new_prompt"].fillna(False).astype(bool)
    df["after_compaction"] = df["after_compaction"].fillna(False).astype(bool)
    df["main_thread"] = ~df["is_sidechain"]

    # Gaps are measured within one transcript: subagents share the parent's
    # sessionId but write their own file and keep their own cache prefix.
    transcript = ["source_file", "is_sidechain"]
    df = df.sort_values(transcript + ["timestamp"], kind="stable").reset_index(drop=True)
    grouped = df.groupby(transcript)
    df["gap_seconds"] = grouped["timestamp"].diff().dt.total_seconds()
    # What the response before had cached, which a tool-loop turn should read back.
    df["prev_cached"] = grouped["cache_read"].shift() + grouped["cache_creation"].shift()
    df = _with_day(df)
    return add_ratios(df)


def _record_frame(rows, floats: tuple[str, ...], flags: tuple[str, ...]) -> pd.DataFrame:
    """A table of non-response records (turn durations, hook runs, compactions), from
    raw rows, sorted by transcript and time, with its UTC day."""
    df = _rows_with_parsed_timestamp(rows)
    if df.empty:
        return df
    for col in floats:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in flags:
        df[col] = df[col].fillna(False).astype(bool)
    df = df.sort_values(["source_file", "timestamp"], kind="stable").reset_index(drop=True)
    return _with_day(df)


def duration_frame(rows) -> pd.DataFrame:
    """Turn durations, from raw rows, with their UTC day."""
    return _record_frame(rows, ("duration_ms",), ("is_sidechain",))


def hook_frame(rows) -> pd.DataFrame:
    """Stop-hook runs, from raw rows, with their UTC day."""
    return _record_frame(rows, ("duration_ms",), ("is_sidechain", "prevented"))


def compaction_frame(rows) -> pd.DataFrame:
    """Compactions, from raw rows, with their UTC day."""
    return _record_frame(rows, ("pre_tokens",), ("is_sidechain",))


def failure_frame(rows) -> pd.DataFrame:
    """Failed requests, from raw rows, with their UTC day."""
    return _record_frame(rows, ("status",), ("is_sidechain",))


def usage_frame(rows) -> pd.DataFrame:
    """Per-model usage and cost from Claude Code's own cost records, with its UTC day.
    Unlike every other record table this one carries no version, entrypoint or thread,
    because the records do not, so it declares no flag columns."""
    return _record_frame(rows, (*USAGE_COUNTS, "cost_usd"), ())


CENSUS_COLUMNS = ("day", "version", "path", "responses", "day_responses")


def census_frame(census: Mapping[tuple[str, str, str], int],
                 days: Mapping[tuple[str, str], int]) -> pd.DataFrame:
    """The key census: one row per UTC day, Claude Code version and key path, with the
    responses that carried the path and the responses of that day and version. Sorted,
    so two runs over one history read the same."""
    rows = [{"day": day, "version": version, "path": path, "responses": count,
             "day_responses": days.get((day, version), 0)}
            for (day, version, path), count in census.items()]
    df = pd.DataFrame(rows, columns=list(CENSUS_COLUMNS))
    if df.empty:
        return df
    return df.sort_values(["day", "version", "path"], kind="stable").reset_index(drop=True)


def coverage_frame(rows) -> pd.DataFrame:
    """Hook coverage, one row per transcript, day, version, entrypoint, thread, hook event
    and tool, sorted so two runs over one history read the same."""
    df = pd.DataFrame(list(rows), columns=list(COVERAGE_COLUMNS))
    for col in ("calls", "hooked"):
        df[col] = df[col].astype(int)
    df["is_sidechain"] = df["is_sidechain"].astype(bool)
    return df.sort_values(["source_file", "day", "is_sidechain", "event", "tool"], kind="stable").reset_index(drop=True)


def coverage_rows(parsed: ParsedFile, rel: str) -> list[dict]:
    """A parsed transcript's hook coverage as rows for coverage_frame."""
    session = parsed.session_id or Path(rel).stem
    return [{"source_file": rel, "session_id": session, "day": day, "version": version, "entrypoint": entrypoint,
             "is_sidechain": bool(sidechain), "event": event, "tool": tool, "calls": calls, "hooked": hooked}
            for (day, version, entrypoint, sidechain, event, tool), (calls, hooked) in parsed.hook_coverage.items()]


def components_frame(rows) -> pd.DataFrame:
    """What each session started with, one row per transcript, sorted by its path. A size
    Claude Code didn't log is NaN, not 0."""
    df = pd.DataFrame(list(rows), columns=list(COMPONENT_COLUMNS))
    for col in COMPONENT_SIZES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in COMPONENT_SETS:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df.sort_values("source_file", kind="stable").reset_index(drop=True)


@dataclass
class Tables:
    """Everything ccdrift reads from transcripts, one table per record kind, plus the
    key census (field_census), per-model usage and cost (model_usage) and what each
    session started with (components) and hook coverage (hook_coverage)."""
    responses: pd.DataFrame
    durations: pd.DataFrame
    hook_runs: pd.DataFrame
    compactions: pd.DataFrame
    failures: pd.DataFrame
    field_census: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_usage: pd.DataFrame = field(default_factory=pd.DataFrame)
    components: pd.DataFrame = field(default_factory=pd.DataFrame)
    hook_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_all(source: Path) -> Tables:
    """Every transcript under `source`, read once. A record copied into a second
    transcript counts from the one whose path sorts first."""
    kinds = {"responses": {}, "durations": {}, "hook_runs": {}, "compactions": {}, "failures": {}, "model_usage": {}}
    census: dict[tuple[str, str, str], int] = {}
    days: dict[tuple[str, str], int] = {}
    components, coverage = [], []
    for fp, rel in jsonl_files(source):
        try:
            parsed = parse_file(fp, rel)
        except OSError:
            continue
        for kind, rows in kinds.items():
            for key, row in getattr(parsed, kind).items():
                rows.setdefault(key, row)
        # Summed rather than deduplicated: a response copied into a resumed session's
        # transcript lands in the numerator and the denominator alike, so every share
        # worked out from the census is exact either way.
        for key, count in parsed.field_census.items():
            census[key] = census.get(key, 0) + count
        for day_key, count in parsed.field_days.items():
            days[day_key] = days.get(day_key, 0) + count
        components.append(parsed.components)
        coverage += coverage_rows(parsed, rel)
    return Tables(frame(list(kinds["responses"].values())), duration_frame(list(kinds["durations"].values())),
                  hook_frame(list(kinds["hook_runs"].values())),
                  compaction_frame(list(kinds["compactions"].values())),
                  failure_frame(list(kinds["failures"].values())),
                  census_frame(census, days),
                  usage_frame(list(kinds["model_usage"].values())), components_frame(components),
                  coverage_frame(coverage))


def parse_durations(source: Path) -> pd.DataFrame:
    """Every `turn_duration` record under `source`: how long each turn took, and over
    how many messages."""
    return parse_all(source).durations


# Agent SDK sessions log an entrypoint starting with this: "sdk-py", "sdk-ts".
SDK_ENTRYPOINT_PREFIX = "sdk-"


def outside_sdk(df: pd.DataFrame) -> pd.Series:
    """Whether each row comes from outside Agent SDK sessions, which are the user's own
    scripts. Transcripts from before Claude Code logged an entrypoint count as the CLI."""
    if "entrypoint" not in df:
        return pd.Series(True, index=df.index)
    return ~df["entrypoint"].fillna("").astype(str).str.startswith(SDK_ENTRYPOINT_PREFIX)


def judged_turns(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """The turns the daily check, the report and setting changes judge: main-thread
    turns of complete UTC days, without Agent SDK sessions. Subagent Haiku comes in
    bursts that flag on their own, a day still in progress holds only part of its
    turns, and SDK sessions are the user's own scripts, on the 5-minute cache."""
    keep = (df["day"].astype(str) < today.isoformat()) & df["main_thread"].astype(bool) & outside_sdk(df)
    return df[keep]


def first_days_by_version(turns: pd.DataFrame) -> pd.Series:
    """The UTC day each Claude Code version first appears among `turns`, or in the
    whole history store when the table carries `version_first_day`: the check reads
    only recent transcripts."""
    known = turns.dropna(subset=["version"])
    days = known["day"].astype(str)
    if "version_first_day" in known:
        days = known["version_first_day"].fillna(days).astype(str)
    return days.groupby(known["version"]).min()


def judged_subagent_turns(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Subagent turns of complete UTC days whose agent type Claude Code logged,
    without Agent SDK sessions."""
    if df.empty or "agent_type" not in df:
        return df.iloc[0:0]
    keep = ((df["day"].astype(str) < today.isoformat()) & df["is_sidechain"].astype(bool) & df["agent_type"].notna()
            & outside_sdk(df))
    return df[keep]


# Turns the cache metric uses. In real logs a caching regression showed up on
# main-thread turns that open with a new user prompt: on Claude Code
# 2.1.233-2.1.258 they missed 4.3% of the time (0.5% before and after), however
# long the pause, while tool-loop continuations almost never missed. A turn
# more than an hour after the previous one misses anyway (1 h TTL), as does the
# turn right after a compaction, so both are left out. Subagents are left out
# too: they keep a 5 min cache, and 88% of their turns after a longer pause miss.
CACHE_TTL_SECONDS = 3600
SUBAGENT_CACHE_TTL_SECONDS = 300

# Tool-loop turns: those that don't open with a prompt and don't follow a
# compaction. Each should read back everything the response before it in its
# transcript had cached, so one that reads under half of that rewrote the
# conversation. In real logs (Aug 6 - Sep 16, 2026) such a miss read only the
# system prompt and tools, 4-12% of what was cached; the rule agreed with the
# new-prompt rule below on every main-thread loop turn, while 7 subagent turns
# that added more input than they read counted as misses by that rule alone.
# Subagents keep a 5-minute cache (62 of 79 turns 5-10 minutes apart missed);
# no main-thread turn 5-60 minutes apart missed (147), so one gap serves both.
LOOP_GAP_SECONDS = 300
LOOP_MISS_SHARE = 0.5

# A prompt turn misses the cache when it reads under half its input from it.
# Misses are all-or-nothing: in real logs cutoffs of 0.3, 0.5 and 0.8 selected
# 2.82%, 2.82% and 2.88% of prompt turns.
MISS_RATIO = 0.5


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-turn metric columns from the raw ones. Runs after parsing
    and again after every injection."""
    df["thinking_fraction"] = df["thinking_tokens"] / (
        df["thinking_tokens"] + df["visible_tokens"]).clip(lower=1)
    denom = (df["cache_read"] + df["cache_creation"]).clip(lower=1)
    df["cache_read_ratio"] = df["cache_read"] / denom
    df["is_haiku"] = df["model"].str.lower().str.contains("haiku").astype(float)
    df["prompt_within_ttl"] = (df["main_thread"] & df["new_prompt"] & ~df["after_compaction"]
                               & (df["gap_seconds"] <= CACHE_TTL_SECONDS))
    df["prompt_cache_read_ratio"] = df["cache_read_ratio"].where(df["prompt_within_ttl"])
    df["is_miss"] = df["prompt_within_ttl"] & (df["cache_read_ratio"] < MISS_RATIO)
    if "prev_cached" in df:
        tokens = df["cache_read"] + df["cache_creation"]
        df["loop_turn"] = (~df["new_prompt"] & ~df["after_compaction"] & (df["gap_seconds"] <= LOOP_GAP_SECONDS)
                           & (tokens > 0) & (df["prev_cached"] > 0))
        df["is_loop_miss"] = df["loop_turn"] & (df["cache_read"] < LOOP_MISS_SHARE * df["prev_cached"])
    if "cache_1h" in df:
        writes = df["cache_1h"] + df["cache_5m"]
        tier = pd.Series([None] * len(df), index=df.index, dtype=object)
        tier.loc[writes > 0] = "5m"
        tier.loc[(writes > 0) & (df["cache_1h"] >= df["cache_5m"])] = "1h"
        df["cache_tier"] = tier
    return df


# No token count or duration in milliseconds comes near this; SQLite's integers stop
# at 2**63, and sums of a few thousand counts must stay below it.
MAX_COUNT = 1e12


def _num(v: Any) -> float:
    """`v` as a number; 0.0 when it isn't one, including Infinity and NaN, which JSON
    parsing accepts, and numbers beyond MAX_COUNT."""
    try:
        if v is None:
            return 0.0
        number = float(v)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and abs(number) <= MAX_COUNT else 0.0


# ---------------------------------------------------------------------------
# Schema peek (Step 0 helper)
# ---------------------------------------------------------------------------

# Values peek shows as logged: how Claude Code logs, not what was said, where or in
# which session. Any other text shows as its length, so the output can go into an issue.
PEEK_SHOWN = frozenset({"type", "role", "model", "version", "entrypoint", "effort", "speed", "service_tier",
                        "subtype", "stop_reason", "miss_reason", "timestamp"})


def _peek_value(value: Any, key: Optional[str] = None) -> Any:
    """`value` with every string not under a PEEK_SHOWN key replaced by its length. The
    blocks of a `content` list keep their type and show the rest by size alone: a tool
    call's input is whatever the conversation put there, keys included, even under a
    name shown as logged elsewhere, such as an MCP tool's "type" or "model"."""
    if isinstance(value, dict):
        return {k: _peek_value(v, k) for k, v in value.items()}
    if isinstance(value, list) and key == "content":
        return [{k: v if k == "type" else _peek_size(v) for k, v in block.items()} if isinstance(block, dict)
                else _peek_size(block) for block in value]
    if isinstance(value, list):
        return [_peek_value(v) for v in value]
    if isinstance(value, str) and key not in PEEK_SHOWN:
        return f"<{len(value)} chars>"
    return value


def _peek_size(value: Any) -> Any:
    """Text, an object or a list as its size; a number, boolean or null as logged."""
    if isinstance(value, dict):
        return f"<{len(value)} keys>"
    if isinstance(value, list):
        return f"<{len(value)} items>"
    if isinstance(value, str):
        return f"<{len(value)} chars>"
    return value


def peek(source: Path) -> bool:
    """Print the first assistant line in `source` and the fields resolved from it,
    with text shown as its length (see PEEK_SHOWN), for checking the parser against
    a new Claude Code version. False when there is no assistant line to show."""
    for fp in iter_jsonl_files(source):
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, RecursionError):  # as in parse_file
                        continue
                    if not (isinstance(obj, dict) and is_assistant(obj)):
                        continue
                    try:
                        shown = json.dumps(_peek_value(obj), indent=2)[:4000]
                        resolved = [f"  {logical:16s} -> {_peek_value(field_get(obj, logical), logical)!r}"[:120]
                                    for logical in CANDIDATES]
                    except RecursionError:  # decoded, but nested deeper than a walk over it can go
                        continue
                    # The transcript's path names the project folder, so it isn't shown.
                    print(LOG_LINES["peek_line"])
                    print(shown)
                    print(LOG_LINES["peek_fields"])
                    print("\n".join(resolved))
                    return True
        except OSError:
            continue
    print(no_transcripts_message(source), file=sys.stderr)
    return False
