"""Find and parse Claude Code session transcripts into one row per API response."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
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
    "timestamp":         ["timestamp", "message.timestamp", "createdAt"],
    "session_id":        ["sessionId", "session_id", "message.sessionId"],
    "content":           ["message.content", "content"],
    "is_sidechain":      ["isSidechain", "message.isSidechain", "is_sidechain"],
    "message_id":        ["message.id"],
    "request_id":        ["requestId", "request_id"],
    "is_meta":           ["isMeta", "is_meta"],
    "subtype":           ["subtype"],
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

def iter_jsonl_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    yield from sorted(source.rglob("*.jsonl"))


def default_source(environ: Mapping[str, str] = os.environ) -> Path:
    """Where Claude Code keeps session transcripts: $CLAUDE_CONFIG_DIR/projects when
    that variable is set, otherwise ~/.claude/projects."""
    config = environ.get("CLAUDE_CONFIG_DIR")
    return (Path(config).expanduser() if config else Path.home() / ".claude") / "projects"


def no_transcripts_message(source: Path) -> str:
    return (f"No Claude Code transcripts found in {source}. Pass --source DIR, or set "
            "CLAUDE_CONFIG_DIR if Claude Code keeps its files somewhere else.")


def parse_source(source: Path, verbose: bool = False) -> pd.DataFrame:
    # One row per API response. Claude Code writes each content block of a
    # response on its own line with the response's usage repeated, so lines that
    # share a message.id are merged. A response copied into a second file (e.g.
    # when a session is resumed) is counted from the first file only.
    turns: dict[str, dict] = {}
    n_files = 0
    n_lines = 0
    n_bad = 0
    n_assistant = 0
    for fp in iter_jsonl_files(source):
        n_files += 1
        rel = fp.name if source.is_file() else str(fp.relative_to(source))
        # Set by lines between two responses: a prompt typed by the user opens
        # a new turn, and a compaction rewrites the conversation.
        prompt_pending = compact_pending = False
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        n_bad += 1
                        continue
                    if not isinstance(obj, dict):
                        continue
                    role = field_get(obj, "role")
                    if role == "user":
                        if not field_get(obj, "is_meta") and not has_tool_result(field_get(obj, "content")):
                            prompt_pending = True
                        continue
                    if role == "system":
                        if field_get(obj, "subtype") == "compact_boundary":
                            compact_pending = True
                        continue
                    if role != "assistant":
                        continue
                    n_assistant += 1
                    model = field_get(obj, "model") or "unknown"
                    if model == "<synthetic>":  # Claude Code placeholder, no API call
                        continue
                    key = (field_get(obj, "message_id") or field_get(obj, "request_id")
                           or f"{rel}:{line_no}")
                    row = turns.get(key)
                    if row is None:
                        row = turns[key] = {
                            "session_id":      field_get(obj, "session_id", default=fp.stem),
                            "timestamp":       parse_ts(field_get(obj, "timestamp")),
                            "model":           model,
                            "input_tokens":    0.0,
                            "output_tokens":   0.0,
                            "cache_creation":  0.0,
                            "cache_read":      0.0,
                            "signature_chars": 0,
                            "visible_chars":   0,
                            "n_mcp_calls":     0,
                            "is_sidechain":    bool(field_get(obj, "is_sidechain", default=False)),
                            "new_prompt":      prompt_pending,
                            "after_compaction": compact_pending,
                            "source_file":     rel,
                        }
                    prompt_pending = compact_pending = False
                    if row["source_file"] != rel:
                        continue
                    # output_tokens grows while streaming, so the largest is the
                    # latest. About a third of real responses never log the final
                    # count (stop_reason stays null); no metric relies on it.
                    for col in ("input_tokens", "output_tokens", "cache_creation", "cache_read"):
                        row[col] = max(row[col], _num(field_get(obj, col)))
                    content = field_get(obj, "content")
                    signature, visible = content_chars(content)
                    row["signature_chars"] += signature
                    row["visible_chars"] += visible
                    row["n_mcp_calls"] += len(mcp_tool_names(content))
        except OSError as e:
            if verbose:
                print(f"  ! could not read {fp}: {e}", file=sys.stderr)

    if verbose:
        print(f"  files={n_files} lines={n_lines} bad_json={n_bad} "
              f"assistant_lines={n_assistant} responses={len(turns)}", file=sys.stderr)

    df = pd.DataFrame(list(turns.values()))
    if df.empty:
        return df

    df["thinking_tokens"] = (df["signature_chars"] * TOKENS_PER_SIGNATURE_CHAR).round()
    df["visible_tokens"] = (df["visible_chars"] / 4.0).round()
    df["is_sidechain"] = df["is_sidechain"].fillna(False).astype(bool)
    df["main_thread"] = ~df["is_sidechain"]

    # Gaps are measured within one transcript: subagents share the parent's
    # sessionId but write their own file and keep their own cache prefix.
    transcript = ["source_file", "is_sidechain"]
    df = df.sort_values(transcript + ["timestamp"], kind="stable").reset_index(drop=True)
    df["gap_seconds"] = df.groupby(transcript)["timestamp"].diff().dt.total_seconds()
    # day bucket (UTC) for binning
    df["day"] = df["timestamp"].dt.tz_convert("UTC").dt.date.astype("string")
    return add_ratios(df)


# Turns the cache metric uses. In real logs a caching regression showed up on
# main-thread turns that open with a new user prompt: on Claude Code
# 2.1.233-2.1.258 they missed 4.3% of the time (0.5% before and after), however
# long the pause, while tool-loop continuations almost never missed. A turn
# more than an hour after the previous one misses anyway (1 h TTL), as does the
# turn right after a compaction, so both are left out. Subagents are left out
# too: they keep a 5 min cache, and 88% of their turns after a longer pause miss.
CACHE_TTL_SECONDS = 3600
SUBAGENT_CACHE_TTL_SECONDS = 300


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
