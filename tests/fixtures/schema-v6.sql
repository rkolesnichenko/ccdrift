CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER, session_id TEXT,
    last_ts INTEGER);
CREATE TABLE responses (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    model TEXT, version TEXT, entrypoint TEXT, effort TEXT, speed TEXT, service_tier TEXT, agent_type TEXT,
    stop_reason TEXT, miss_reason TEXT,
    attribution_skill TEXT, attribution_plugin TEXT, attribution_mcp TEXT, git_branch TEXT,
    is_sidechain INTEGER, new_prompt INTEGER, after_compaction INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    cache_1h INTEGER, cache_5m INTEGER, thinking_logged INTEGER,
    signature_chars INTEGER, visible_chars INTEGER, n_mcp_calls INTEGER,
    opens_transcript INTEGER NOT NULL DEFAULT 0);
CREATE INDEX responses_file ON responses (file_id);
CREATE TABLE durations (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, duration_ms INTEGER, message_count INTEGER);
CREATE INDEX durations_file ON durations (file_id);
CREATE TABLE hook_runs (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, hook_count INTEGER, error_count INTEGER,
    duration_ms INTEGER, prevented INTEGER);
CREATE INDEX hook_runs_file ON hook_runs (file_id);
CREATE TABLE compactions (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, trigger TEXT, pre_tokens INTEGER);
CREATE INDEX compactions_file ON compactions (file_id);
CREATE TABLE failures (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, kind TEXT, status INTEGER);
CREATE INDEX failures_file ON failures (file_id);
CREATE TABLE model_usage (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    thinking_tokens INTEGER, web_searches INTEGER, cost_usd REAL);
CREATE INDEX model_usage_file ON model_usage (file_id);
CREATE TABLE field_census (
    file_id INTEGER NOT NULL, day TEXT NOT NULL, version TEXT NOT NULL, path TEXT NOT NULL,
    responses INTEGER NOT NULL, PRIMARY KEY (file_id, day, version, path));
CREATE TABLE field_days (
    file_id INTEGER NOT NULL, day TEXT NOT NULL, version TEXT NOT NULL,
    responses INTEGER NOT NULL, PRIMARY KEY (file_id, day, version));
INSERT INTO meta (key, value) VALUES ('schema_version', '6');
