"""Find and parse Claude Code session transcripts into one row per API response."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Defensive field access
# ---------------------------------------------------------------------------
# Claude Code's JSONL schema has shifted across versions. Rather than hard-code
# one path, every field is resolved by trying a list of candidate dotted paths
# and taking the first that exists. If you discover a new variant, add it here —
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
    "compact_trigger":   ["compactMetadata.trigger"],
    "compact_pre_tokens": ["compactMetadata.preTokens"],
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


def parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # epoch seconds or millis
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        return datetime.fromtimestamp(val, tz=timezone.utc)
    if isinstance(raw, str):
        s = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Parsing → turn-level feature table
# ---------------------------------------------------------------------------

# Response fields that keep the first value logged across a response's lines,
# and token counts that keep the largest.
SETTING_FIELDS = ("version", "entrypoint", "effort", "speed", "service_tier")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation", "cache_read", "cache_1h", "cache_5m")


@dataclass
class ParsedFile:
    """One transcript's responses, turn durations, hook runs and compactions by key,
    the transcript's first session id, and line counts for --verbose."""
    responses: dict[str, dict] = field(default_factory=dict)
    durations: dict[str, dict] = field(default_factory=dict)
    hook_runs: dict[str, dict] = field(default_factory=dict)
    compactions: dict[str, dict] = field(default_factory=dict)
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


def no_transcripts_message(source: Path) -> str:
    return (f"No Claude Code transcripts found in {source}. Pass --source DIR, or set "
            "CLAUDE_CONFIG_DIR if Claude Code keeps its files somewhere else.")


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _record(obj: dict, key: str, rel: str) -> dict:
    """The fields a turn duration, hook run or compaction record keeps in common."""
    return {"key": key, "timestamp": parse_ts(field_get(obj, "timestamp")),
            "version": _text(field_get(obj, "version")), "entrypoint": _text(field_get(obj, "entrypoint")),
            "is_sidechain": bool(field_get(obj, "is_sidechain", default=False)), "source_file": rel}


def parse_file(fp: Path, rel: str) -> ParsedFile:
    """Parse one transcript. Claude Code writes each content block of a response on
    its own line with the response's usage repeated, so lines that share a
    message.id are merged. Raises OSError when the file can't be read."""
    parsed = ParsedFile()
    # Set by lines between two responses: a prompt typed by the user opens
    # a new turn, and a compaction rewrites the conversation.
    prompt_pending = compact_pending = False
    with fp.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parsed.lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parsed.bad_json += 1
                continue
            if not isinstance(obj, dict):
                continue
            if parsed.session_id is None:
                parsed.session_id = _text(field_get(obj, "session_id"))
            role = field_get(obj, "role")
            if role == "user":
                if not field_get(obj, "is_meta") and not has_tool_result(field_get(obj, "content")):
                    prompt_pending = True
                continue
            if role == "system":
                subtype = field_get(obj, "subtype")
                key = field_get(obj, "uuid") or f"{rel}:{line_no}"
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
            if role != "assistant":
                continue
            parsed.assistant_lines += 1
            model = field_get(obj, "model") or "unknown"
            if model == "<synthetic>":  # Claude Code placeholder, no API call
                continue
            key = (field_get(obj, "message_id") or field_get(obj, "request_id")
                   or f"{rel}:{line_no}")
            row = parsed.responses.get(key)
            if row is None:
                row = parsed.responses[key] = {
                    "key":              key,
                    "timestamp":        parse_ts(field_get(obj, "timestamp")),
                    "model":            model,
                    **{name: None for name in SETTING_FIELDS},
                    **{name: 0.0 for name in TOKEN_FIELDS},
                    "thinking_logged":  None,
                    "signature_chars":  0,
                    "visible_chars":    0,
                    "n_mcp_calls":      0,
                    "is_sidechain":     bool(field_get(obj, "is_sidechain", default=False)),
                    "new_prompt":       prompt_pending,
                    "after_compaction": compact_pending,
                    "source_file":      rel,
                }
            prompt_pending = compact_pending = False
            for name in SETTING_FIELDS:
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
            signature, visible = content_chars(content)
            row["signature_chars"] += signature
            row["visible_chars"] += visible
            row["n_mcp_calls"] += len(mcp_tool_names(content))
    # A transcript is one session, even when a resumed session's lines carry
    # another id.
    session = parsed.session_id or fp.stem
    for row in (*parsed.responses.values(), *parsed.durations.values(),
                *parsed.hook_runs.values(), *parsed.compactions.values()):
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
                print(f"  ! could not read {fp}: {e}", file=sys.stderr)
            continue
        lines += parsed.lines
        bad_json += parsed.bad_json
        assistant_lines += parsed.assistant_lines
        for key, row in parsed.responses.items():
            rows.setdefault(key, row)
    if verbose:
        print(f"  files={len(files)} lines={lines} bad_json={bad_json} "
              f"assistant_lines={assistant_lines} responses={len(rows)}", file=sys.stderr)
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
    df["gap_seconds"] = df.groupby(transcript)["timestamp"].diff().dt.total_seconds()
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


@dataclass
class Tables:
    """Everything ccdrift reads from transcripts, one table per record kind."""
    responses: pd.DataFrame
    durations: pd.DataFrame
    hook_runs: pd.DataFrame
    compactions: pd.DataFrame


def parse_all(source: Path) -> Tables:
    """Every transcript under `source`, read once. A record copied into a second
    transcript counts from the one whose path sorts first."""
    kinds = {"responses": {}, "durations": {}, "hook_runs": {}, "compactions": {}}
    for fp, rel in jsonl_files(source):
        try:
            parsed = parse_file(fp, rel)
        except OSError:
            continue
        for kind, rows in kinds.items():
            for key, row in getattr(parsed, kind).items():
                rows.setdefault(key, row)
    return Tables(frame(list(kinds["responses"].values())), duration_frame(list(kinds["durations"].values())),
                  hook_frame(list(kinds["hook_runs"].values())),
                  compaction_frame(list(kinds["compactions"].values())))


def parse_durations(source: Path) -> pd.DataFrame:
    """Every `turn_duration` record under `source`: how long each turn took, and over
    how many messages."""
    return parse_all(source).durations


def judged_turns(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """The turns the daily check, the report and setting changes judge: main-thread
    turns of complete UTC days, without Agent SDK sessions. Subagent Haiku comes in
    bursts that flag on their own, a day still in progress holds only part of its
    turns, and SDK sessions are the user's own scripts, on the 5-minute cache.
    Transcripts from before Claude Code logged an entrypoint count as the CLI."""
    keep = (df["day"].astype(str) < today.isoformat()) & df["main_thread"].astype(bool)
    if "entrypoint" in df:
        keep &= ~df["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
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
    if "cache_1h" in df:
        writes = df["cache_1h"] + df["cache_5m"]
        tier = pd.Series([None] * len(df), index=df.index, dtype=object)
        tier.loc[writes > 0] = "5m"
        tier.loc[(writes > 0) & (df["cache_1h"] >= df["cache_5m"])] = "1h"
        df["cache_tier"] = tier
    return df


def _num(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Schema peek (Step 0 helper)
# ---------------------------------------------------------------------------

def peek(source: Path) -> bool:
    """Print the first assistant line in `source` and the fields resolved from it,
    for checking the parser against a new Claude Code version. False when there is
    no assistant line to show."""
    for fp in iter_jsonl_files(source):
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and is_assistant(obj):
                        print(f"# first assistant line from {fp}")
                        print(json.dumps(obj, indent=2)[:4000])
                        print("\n# resolved fields:")
                        for logical in CANDIDATES:
                            print(f"  {logical:16s} -> {field_get(obj, logical)!r}"[:120])
                        return True
        except OSError:
            continue
    print(no_transcripts_message(source), file=sys.stderr)
    return False
