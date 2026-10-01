//! Convert an agentic coding-session trace into a per-session replay trace.
//!
//! The input is the metadata-only coding-agent dataset published in
//! Azure/AzurePublicDataset (`GitHubCopilotCodingAgentDataset2026.md`): one
//! JSON object per session, gzipped JSONL shards, optionally bundled in a
//! per-day `.tar.gz`. It carries token counts, cache hits and timings but no
//! text, so the output describes the shape of each request and how much of
//! the previous request's prompt it reuses; a replayer supplies filler text.
//!
//! Facts about the source that the conversion depends on, measured on the
//! 2026-06-03 partition:
//!
//! - An LLM call's `timestamp` is when the call finished. Start is
//!   `timestamp - duration_ms`. Read as start times, 34% of consecutive calls
//!   within a turn overlap; read as end times, 0.1% do.
//! - In 99.9% of consecutive call pairs the next call starts after the
//!   previous one ends, so a session's calls are treated as one sequence.
//!   Overlapping pairs get `gap_ms` 0.
//! - Timestamps have millisecond resolution, except that 25% of turn-opening
//!   calls carry a whole-second timestamp. Their start and the adjacent
//!   `gap_ms` values may be off by up to 1 s.
//! - `message_metadata` segment lengths change on every call, including the
//!   system prompt, so they cannot locate the shared prefix. `tokens.cached`
//!   is used instead.
//! - Across turns, 59% of calls after an idle gap of 300 s or more report
//!   `cached == 0`, against 10% after shorter gaps. After shorter gaps the
//!   median of `cached / min(previous prompt, prompt)` is 0.977. After a gap
//!   of 300 s or more, 41% of calls report a nonzero `cached` whose median
//!   ratio is 0.34 across turns and 0.38 within a turn. A call after a gap of
//!   at least [`ConvertOptions::cache_ttl_ms`] whose `cached` is below the
//!   estimate from [`ConvertOptions::evicted_reuse_ratio`] gets the estimate
//!   as its `reuse`.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};

/// One session of the replay trace, written as one JSONL line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceSession {
    pub session_id: String,
    /// Start of the session's first call.
    pub start: DateTime<Utc>,
    /// Milliseconds from the earliest `start` in the output, rounded.
    pub start_ms: u64,
    /// Distinct model labels used by the session's calls.
    pub models: Vec<String>,
    pub calls: Vec<TraceCall>,
}

/// One LLM call of a replay session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceCall {
    /// Index of the user turn in the source record.
    pub turn: u32,
    /// The source's `initiator_type`, omitted when empty. On 2026-06-03, 93%
    /// of turn-opening calls are "user".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initiator: Option<String>,
    /// The source's anonymized model label, omitted when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Time from the end of the previous kept call to the start of this one.
    /// It is 0 for the first call and when the two overlap. It includes the
    /// duration of any dropped call between them (see
    /// [`ConvertStats::calls_dropped_no_prompt`]).
    pub gap_ms: u64,
    pub prompt: u64,
    pub completion: u64,
    /// Cached prompt tokens as reported by the source.
    pub cached: u64,
    /// `min(cached, previous prompt + previous completion, prompt)`, or an
    /// estimate when `reuse_inferred`. Tokens beyond the previous prompt come
    /// from the previous completion, so a replayer must cap `reuse` at the
    /// previous prompt plus the response it actually received. 0 for the
    /// first call; a prefix shared across sessions shows only in `cached`.
    pub reuse: u64,
    /// `reuse` was estimated because, after an idle gap of at least
    /// [`ConvertOptions::cache_ttl_ms`], the source's `cached` was below the
    /// estimate.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reuse_inferred: bool,
    /// The source's `duration_ms`, rounded to milliseconds.
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Keep only sessions whose calls all use one of these model labels.
    /// Empty keeps every session.
    pub models: Vec<String>,
    /// Drop sessions with any call whose prompt + completion exceeds this.
    pub max_context: Option<u64>,
    /// After an idle gap at least this long, a `cached` below the estimate
    /// from `evicted_reuse_ratio` is treated as eviction and the estimate is
    /// used as `reuse`.
    pub cache_ttl_ms: u64,
    /// Fraction of `min(previous prompt, prompt)` used as the estimated
    /// reuse. The default, 0.98, is the median of
    /// `cached / min(previous prompt, prompt)` across turns after idle gaps
    /// under 300 s on the 2026-06-03 partition (0.977).
    pub evicted_reuse_ratio: f64,
    /// Keep only sessions whose first call starts at or after this time.
    pub from: Option<DateTime<Utc>>,
    /// Keep only sessions whose first call starts before this time.
    pub to: Option<DateTime<Utc>>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            max_context: None,
            cache_ttl_ms: 300_000,
            evicted_reuse_ratio: 0.98,
            from: None,
            to: None,
        }
    }
}

/// Counts reported after a conversion.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct ConvertStats {
    pub sessions_read: u64,
    pub sessions_kept: u64,
    pub sessions_dropped_empty: u64,
    pub sessions_dropped_model: u64,
    pub sessions_dropped_context: u64,
    pub sessions_dropped_window: u64,
    /// Sessions whose computed start is outside chrono's representable range.
    pub sessions_dropped_bad_time: u64,
    pub calls_read: u64,
    pub calls_kept: u64,
    /// Calls whose source `tokens.prompt` is missing or 0. These are
    /// cancelled or failed calls, and successful calls that report no token
    /// counts.
    pub calls_dropped_no_prompt: u64,
    pub calls_reuse_inferred: u64,
    /// Calls whose `cached` exceeded `min(previous prompt + previous
    /// completion, prompt)` and was clamped to it.
    pub calls_reuse_clamped: u64,
}

impl ConvertStats {
    fn add(&mut self, o: &ConvertStats) {
        self.sessions_read += o.sessions_read;
        self.sessions_kept += o.sessions_kept;
        self.sessions_dropped_empty += o.sessions_dropped_empty;
        self.sessions_dropped_model += o.sessions_dropped_model;
        self.sessions_dropped_context += o.sessions_dropped_context;
        self.sessions_dropped_window += o.sessions_dropped_window;
        self.sessions_dropped_bad_time += o.sessions_dropped_bad_time;
        self.calls_read += o.calls_read;
        self.calls_kept += o.calls_kept;
        self.calls_dropped_no_prompt += o.calls_dropped_no_prompt;
        self.calls_reuse_inferred += o.calls_reuse_inferred;
        self.calls_reuse_clamped += o.calls_reuse_clamped;
    }
}

// Source records. Only the fields the conversion reads are declared; serde
// skips the rest (`message_metadata`, `tool_batches`, ids, and
// `result`/`status_code`, which no filter uses).

#[derive(Debug, Deserialize)]
struct RawSession {
    session_id: String,
    turns: Vec<RawTurn>,
}

#[derive(Debug, Deserialize)]
struct RawTurn {
    #[serde(default)]
    llm_calls: Vec<RawCall>,
}

#[derive(Debug, Deserialize)]
struct RawCall {
    timestamp: DateTime<Utc>,
    #[serde(default)]
    duration_ms: Option<f64>,
    #[serde(default)]
    initiator_type: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    tokens: Option<RawTokens>,
}

#[derive(Debug, Default, Deserialize)]
struct RawTokens {
    #[serde(default)]
    prompt: Option<u64>,
    #[serde(default)]
    completion: Option<u64>,
    #[serde(default)]
    cached: Option<u64>,
}

/// A source call with its start time resolved, before reuse is derived.
#[derive(Debug, Clone)]
struct FlatCall {
    turn: u32,
    start_us: i64,
    duration_us: i64,
    initiator: Option<String>,
    model: Option<String>,
    prompt: u64,
    completion: u64,
    cached: u64,
}

impl FlatCall {
    fn end_us(&self) -> i64 {
        self.start_us + self.duration_us
    }
}

fn flatten(raw: RawSession, stats: &mut ConvertStats) -> (String, Vec<FlatCall>) {
    let mut calls = Vec::new();
    for (turn, t) in raw.turns.into_iter().enumerate() {
        for c in t.llm_calls {
            stats.calls_read += 1;
            let tokens = c.tokens.unwrap_or_default();
            let prompt = tokens.prompt.unwrap_or(0);
            if prompt == 0 {
                stats.calls_dropped_no_prompt += 1;
                continue;
            }
            // `timestamp` is the end of the call (see module docs).
            let duration_us = (c.duration_ms.unwrap_or(0.0).max(0.0) * 1000.0).round() as i64;
            calls.push(FlatCall {
                turn: turn as u32,
                start_us: c.timestamp.timestamp_micros().saturating_sub(duration_us),
                duration_us,
                initiator: c.initiator_type.filter(|s| !s.is_empty()),
                model: c.model,
                prompt,
                completion: tokens.completion.unwrap_or(0),
                cached: tokens.cached.unwrap_or(0),
            });
        }
    }
    calls.sort_by_key(|c| c.start_us);
    (raw.session_id, calls)
}

fn derive_calls(
    calls: &[FlatCall],
    opts: &ConvertOptions,
    stats: &mut ConvertStats,
) -> Vec<TraceCall> {
    let mut out = Vec::with_capacity(calls.len());
    for (i, c) in calls.iter().enumerate() {
        let (gap_ms, reuse, inferred) = match i.checked_sub(1).map(|p| &calls[p]) {
            None => (0, 0, false),
            Some(prev) => {
                let gap_ms = ((c.start_us - prev.end_us()).max(0) as f64 / 1000.0).round() as u64;
                let bound = (prev.prompt + prev.completion).min(c.prompt);
                let base = prev.prompt.min(c.prompt);
                let est = (base as f64 * opts.evicted_reuse_ratio).round() as u64;
                if gap_ms >= opts.cache_ttl_ms && c.cached < est {
                    (gap_ms, est.min(bound), true)
                } else {
                    if c.cached > bound {
                        stats.calls_reuse_clamped += 1;
                    }
                    (gap_ms, c.cached.min(bound), false)
                }
            }
        };
        stats.calls_reuse_inferred += inferred as u64;
        out.push(TraceCall {
            turn: c.turn,
            initiator: c.initiator.clone(),
            model: c.model.clone(),
            gap_ms,
            prompt: c.prompt,
            completion: c.completion,
            cached: c.cached,
            reuse,
            reuse_inferred: inferred,
            duration_ms: (c.duration_us as f64 / 1000.0).round() as u64,
        });
    }
    out
}

/// Convert one source record. Returns `None` when a filter drops it; the
/// reason is counted in `stats`. `start_ms` is left at 0.
fn convert_session(
    raw: RawSession,
    opts: &ConvertOptions,
    stats: &mut ConvertStats,
) -> Option<TraceSession> {
    stats.sessions_read += 1;
    let (session_id, calls) = flatten(raw, stats);
    let Some(first) = calls.first() else {
        stats.sessions_dropped_empty += 1;
        return None;
    };
    let Some(start) = DateTime::from_timestamp_micros(first.start_us) else {
        stats.sessions_dropped_bad_time += 1;
        return None;
    };
    if opts.from.is_some_and(|f| start < f) || opts.to.is_some_and(|t| start >= t) {
        stats.sessions_dropped_window += 1;
        return None;
    }
    let models: BTreeSet<String> = calls.iter().filter_map(|c| c.model.clone()).collect();
    if !opts.models.is_empty()
        && !calls
            .iter()
            .all(|c| c.model.as_ref().is_some_and(|m| opts.models.contains(m)))
    {
        stats.sessions_dropped_model += 1;
        return None;
    }
    if let Some(max) = opts.max_context
        && calls.iter().any(|c| c.prompt + c.completion > max)
    {
        stats.sessions_dropped_context += 1;
        return None;
    }
    let calls = derive_calls(&calls, opts, stats);
    stats.sessions_kept += 1;
    stats.calls_kept += calls.len() as u64;
    Some(TraceSession {
        session_id,
        start,
        start_ms: 0,
        models: models.into_iter().collect(),
        calls,
    })
}

/// One JSONL shard as read from disk; `gzipped` says whether `bytes` needs
/// decompressing.
struct Shard {
    name: String,
    bytes: Vec<u8>,
    gzipped: bool,
}

fn is_tarball(p: &Path) -> bool {
    let s = p.to_string_lossy();
    s.ends_with(".tar.gz") || s.ends_with(".tgz")
}

fn is_shard_name(s: &str) -> bool {
    s.ends_with(".jsonl.gz") || s.ends_with(".jsonl")
}

/// Expand directories into the shard and tarball files under them, sorted.
fn expand_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut stack = vec![p.clone()];
            let mut found = Vec::new();
            while let Some(d) = stack.pop() {
                for e in
                    std::fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))?
                {
                    let path = e?.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if is_tarball(&path) || is_shard_name(&path.to_string_lossy()) {
                        found.push(path);
                    }
                }
            }
            found.sort();
            out.extend(found);
        } else {
            out.push(p.clone());
        }
    }
    if out.is_empty() {
        bail!("no .jsonl, .jsonl.gz or .tar.gz inputs found");
    }
    Ok(out)
}

fn read_shards(paths: &[PathBuf], mut send: impl FnMut(Shard) -> Result<()>) -> Result<()> {
    for p in paths {
        if is_tarball(p) {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            let mut ar = tar::Archive::new(GzDecoder::new(BufReader::new(f)));
            for entry in ar.entries()? {
                let mut entry = entry?;
                let name = entry.path()?.to_string_lossy().into_owned();
                if !is_shard_name(&name) {
                    continue;
                }
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                send(Shard {
                    gzipped: name.ends_with(".gz"),
                    name: format!("{}:{}", p.display(), name),
                    bytes,
                })?;
            }
        } else {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            send(Shard {
                gzipped: p.to_string_lossy().ends_with(".gz"),
                name: p.display().to_string(),
                bytes,
            })?;
        }
    }
    Ok(())
}

/// Converts every record in `shard`, appending kept sessions to `out` and
/// `(session_id, shard name)` for every record read to `seen`.
fn convert_shard(
    shard: &Shard,
    opts: &ConvertOptions,
    out: &mut Vec<TraceSession>,
    seen: &mut Vec<(String, String)>,
    stats: &mut ConvertStats,
) -> Result<()> {
    let reader: Box<dyn BufRead> = if shard.gzipped {
        Box::new(BufReader::new(GzDecoder::new(&shard.bytes[..])))
    } else {
        Box::new(&shard.bytes[..])
    };
    for (n, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("{}: line {}", shard.name, n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawSession = serde_json::from_str(&line)
            .with_context(|| format!("{}: line {}", shard.name, n + 1))?;
        seen.push((raw.session_id.clone(), shard.name.clone()));
        if let Some(s) = convert_session(raw, opts, stats) {
            out.push(s);
        }
    }
    Ok(())
}

/// Read every input, convert in parallel, and return the kept sessions
/// sorted by start with `start_ms` filled in.
pub fn convert(
    inputs: &[PathBuf],
    opts: &ConvertOptions,
    threads: usize,
) -> Result<(Vec<TraceSession>, ConvertStats)> {
    let paths = expand_inputs(inputs)?;
    let threads = threads.max(1);
    let (tx, rx) = sync_channel::<Shard>(threads * 2);

    let (mut sessions, stats) = std::thread::scope(|s| -> Result<_> {
        let rx: Arc<Mutex<Receiver<Shard>>> = Arc::new(Mutex::new(rx));
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let rx = Arc::clone(&rx);
                s.spawn(move || -> Result<_> {
                    let mut out = Vec::new();
                    let mut seen = Vec::new();
                    let mut stats = ConvertStats::default();
                    loop {
                        let shard = match rx.lock().unwrap().recv() {
                            Ok(shard) => shard,
                            Err(_) => break,
                        };
                        convert_shard(&shard, opts, &mut out, &mut seen, &mut stats)?;
                    }
                    Ok((out, seen, stats))
                })
            })
            .collect();
        drop(rx);

        // The workers hold the only references to the receiver. A worker
        // returns on its first error, so once every worker has returned,
        // `send` fails and the reader stops. The joins below report a
        // worker's error ahead of the read error.
        let read = read_shards(&paths, |shard| {
            tx.send(shard)
                .map_err(|_| anyhow::anyhow!("all workers exited"))
        });
        drop(tx);

        let mut sessions = Vec::new();
        let mut seen = Vec::new();
        let mut stats = ConvertStats::default();
        for w in workers {
            let (out, sn, st) = w.join().expect("convert worker panicked")?;
            sessions.extend(out);
            seen.extend(sn);
            stats.add(&st);
        }
        read?;

        // A repeated id means an input was given twice (for example a
        // tarball and its extracted directory) or a session is split across
        // shards; either would replay it twice.
        seen.sort_unstable();
        if let Some(w) = seen.windows(2).find(|w| w[0].0 == w[1].0) {
            bail!(
                "session_id {} appears in both {} and {}",
                w[0].0,
                w[0].1,
                w[1].1
            );
        }
        Ok((sessions, stats))
    })?;

    sessions.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| a.session_id.cmp(&b.session_id))
    });
    if let Some(t0) = sessions.first().map(|s| s.start) {
        for s in &mut sessions {
            let us = (s.start - t0).num_microseconds().unwrap_or(i64::MAX);
            s.start_ms = (us as f64 / 1000.0).round() as u64;
        }
    }
    Ok((sessions, stats))
}

/// Write sessions as JSONL to `output`, or stdout when `None`.
pub fn write_trace(sessions: &[TraceSession], output: Option<&Path>) -> Result<()> {
    let w: Box<dyn Write> = match output {
        Some(p) => Box::new(File::create(p).with_context(|| format!("creating {}", p.display()))?),
        None => Box::new(std::io::stdout().lock()),
    };
    let mut w = BufWriter::new(w);
    for s in sessions {
        serde_json::to_writer(&mut w, s)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// Entry point for the `convert-trace` subcommand.
pub fn run_convert_trace(
    inputs: &[PathBuf],
    output: Option<&Path>,
    opts: &ConvertOptions,
    threads: usize,
) -> Result<()> {
    let (sessions, stats) = convert(inputs, opts, threads)?;
    write_trace(&sessions, output)?;

    let span_s = sessions
        .last()
        .map(|s| s.start_ms as f64 / 1000.0)
        .unwrap_or(0.0);
    eprintln!(
        "sessions: read {} kept {} (dropped: empty {}, model {}, context {}, window {}, bad time {})",
        stats.sessions_read,
        stats.sessions_kept,
        stats.sessions_dropped_empty,
        stats.sessions_dropped_model,
        stats.sessions_dropped_context,
        stats.sessions_dropped_window,
        stats.sessions_dropped_bad_time,
    );
    eprintln!(
        "calls: read {} kept {} (no prompt {}); reuse inferred {}, clamped {}",
        stats.calls_read,
        stats.calls_kept,
        stats.calls_dropped_no_prompt,
        stats.calls_reuse_inferred,
        stats.calls_reuse_clamped,
    );
    eprintln!("session starts span {span_s:.0} s");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // One real session from the 2026-06-03 partition, trimmed to two turns of
    // three calls each.
    const FIXTURE: &str = include_str!("../tests/fixtures/coding_agent_session.json");

    fn fixture() -> RawSession {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn call(start_ms: i64, dur_ms: i64, prompt: u64, cached: u64) -> FlatCall {
        FlatCall {
            turn: 0,
            start_us: start_ms * 1000,
            duration_us: dur_ms * 1000,
            initiator: None,
            model: Some("Model A".into()),
            prompt,
            completion: 100,
            cached,
        }
    }

    #[test]
    fn fixture_start_is_timestamp_minus_duration() {
        let mut stats = ConvertStats::default();
        let s = convert_session(fixture(), &ConvertOptions::default(), &mut stats).unwrap();
        // First call: timestamp 06:23:30.000, duration 6498.1752 ms.
        assert_eq!(s.start.to_rfc3339(), "2026-06-03T06:23:23.501825+00:00");
        // Second call ends 06:23:35.531 after 2558.3419 ms, so it starts
        // 2972.658 ms after the first ended.
        assert_eq!(s.calls[1].gap_ms, 2973);
        assert_eq!(s.calls.len(), 6);
        assert_eq!(s.models, vec!["Model E".to_string()]);
    }

    #[test]
    fn fixture_reuse_follows_cached_and_infers_after_idle() {
        let mut stats = ConvertStats::default();
        let s = convert_session(fixture(), &ConvertOptions::default(), &mut stats).unwrap();
        let reuse: Vec<u64> = s.calls.iter().map(|c| c.reuse).collect();
        // Call 4 opens turn 2 almost five hours later with cached = 0, so its
        // reuse is 0.98 * min(20480, 39312) and marked inferred. Call 3
        // reports 18816 cached, more than call 2's 18324 + 120, and is clamped.
        assert_eq!(reuse, vec![0, 14779, 18444, 20070, 14209, 40744]);
        assert_eq!(stats.calls_reuse_clamped, 1);
        assert!(s.calls[3].reuse_inferred);
        assert_eq!(stats.calls_reuse_inferred, 1);
        assert_eq!(s.calls[3].initiator.as_deref(), Some("user"));
    }

    #[test]
    fn zero_cache_inside_ttl_is_a_real_miss() {
        let calls = [call(0, 1000, 5000, 0), call(2000, 1000, 6000, 0)];
        let mut stats = ConvertStats::default();
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].gap_ms, 1000);
        assert_eq!(out[1].reuse, 0);
        assert!(!out[1].reuse_inferred);
    }

    #[test]
    fn cached_beyond_previous_prompt_is_clamped() {
        // Previous context is 5000 prompt + 100 completion.
        let calls = [call(0, 1000, 5000, 0), call(1000, 1000, 6000, 5400)];
        let mut stats = ConvertStats::default();
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].reuse, 5100);
        assert_eq!(stats.calls_reuse_clamped, 1);
    }

    #[test]
    fn partial_hit_after_ttl_below_estimate_is_inferred() {
        // 0.98 * 5000 = 4900; the reported 1700 is treated as eviction.
        let calls = [call(0, 1000, 5000, 0), call(400_000, 1000, 6000, 1700)];
        let mut stats = ConvertStats::default();
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].reuse, 4900);
        assert!(out[1].reuse_inferred);

        // A hit at or above the estimate is used as reported.
        let calls = [call(0, 1000, 5000, 0), call(400_000, 1000, 6000, 5050)];
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].reuse, 5050);
        assert!(!out[1].reuse_inferred);

        // The same partial hit inside the TTL is used as reported.
        let calls = [call(0, 1000, 5000, 0), call(2000, 1000, 6000, 1700)];
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].reuse, 1700);
        assert!(!out[1].reuse_inferred);
    }

    #[test]
    fn overlapping_calls_get_zero_gap() {
        let calls = [call(0, 3000, 5000, 0), call(2000, 1000, 6000, 4000)];
        let mut stats = ConvertStats::default();
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[1].gap_ms, 0);
        assert_eq!(out[1].reuse, 4000);
    }

    #[test]
    fn first_call_reuse_is_zero_even_when_cached() {
        let calls = [call(0, 1000, 5000, 4000)];
        let mut stats = ConvertStats::default();
        let out = derive_calls(&calls, &ConvertOptions::default(), &mut stats);
        assert_eq!(out[0].cached, 4000);
        assert_eq!(out[0].reuse, 0);
        assert!(!out[0].reuse_inferred);
    }

    #[test]
    fn dropped_call_duration_folds_into_next_gap() {
        // Null the second call's prompt; call 3 then follows call 1.
        let mut raw: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        raw["turns"][0]["llm_calls"][1]["tokens"]["prompt"] = serde_json::Value::Null;
        let raw: RawSession = serde_json::from_value(raw).unwrap();
        let mut stats = ConvertStats::default();
        let s = convert_session(raw, &ConvertOptions::default(), &mut stats).unwrap();
        assert_eq!(stats.calls_dropped_no_prompt, 1);
        assert_eq!(s.calls.len(), 5);
        // Call 3 ends 06:23:38.065 after 2394.2392 ms: start 06:23:35.670761,
        // 5670.761 ms after call 1 ended at 06:23:30.000.
        assert_eq!(s.calls[1].gap_ms, 5671);
    }

    #[test]
    fn cli_parses_convert_trace_options() {
        use crate::cli::{Cli, Command};
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "llm-perf",
            "convert-trace",
            "day.tar.gz",
            "--model",
            "Model E",
            "--model",
            "Model A",
            "--from",
            "2026-06-03T00:00:00Z",
            "--max-context",
            "131072",
        ])
        .unwrap();
        let Command::ConvertTrace {
            inputs,
            models,
            from,
            max_context,
            cache_ttl_secs,
            ..
        } = cli.command
        else {
            panic!("wrong subcommand");
        };
        assert_eq!(inputs, vec![PathBuf::from("day.tar.gz")]);
        assert_eq!(models, vec!["Model E".to_string(), "Model A".to_string()]);
        assert_eq!(from.unwrap().to_rfc3339(), "2026-06-03T00:00:00+00:00");
        assert_eq!(max_context, Some(131_072));
        assert_eq!(cache_ttl_secs, 300);
    }

    #[test]
    fn filters_drop_sessions() {
        let opts = ConvertOptions {
            models: vec!["Model A".into()],
            ..Default::default()
        };
        let mut stats = ConvertStats::default();
        assert!(convert_session(fixture(), &opts, &mut stats).is_none());
        assert_eq!(stats.sessions_dropped_model, 1);

        // Largest call in the fixture is 43817 + 1655 = 45472 tokens.
        let opts = ConvertOptions {
            max_context: Some(45_471),
            ..Default::default()
        };
        assert!(convert_session(fixture(), &opts, &mut stats).is_none());
        assert_eq!(stats.sessions_dropped_context, 1);
        let opts = ConvertOptions {
            max_context: Some(45_472),
            ..Default::default()
        };
        assert!(convert_session(fixture(), &opts, &mut stats).is_some());

        let opts = ConvertOptions {
            from: Some("2026-06-03T07:00:00Z".parse().unwrap()),
            ..Default::default()
        };
        assert!(convert_session(fixture(), &opts, &mut stats).is_none());
        assert_eq!(stats.sessions_dropped_window, 1);

        // `to` is exclusive: a session starting exactly at `to` is dropped.
        let opts = ConvertOptions {
            to: Some("2026-06-03T06:23:23.501825Z".parse().unwrap()),
            ..Default::default()
        };
        assert!(convert_session(fixture(), &opts, &mut stats).is_none());
        assert_eq!(stats.sessions_dropped_window, 2);
        let opts = ConvertOptions {
            to: Some("2026-06-03T06:23:23.501826Z".parse().unwrap()),
            ..Default::default()
        };
        assert!(convert_session(fixture(), &opts, &mut stats).is_some());
    }

    #[test]
    fn convert_reads_shards_and_tarballs() {
        let dir = tempfile::tempdir().unwrap();
        // Each file gets one session at the fixture's time and one with its
        // first turn moved an hour later, under ids distinct across files.
        let gz = |ids: [&str; 2]| {
            let line = FIXTURE.replace('\n', "");
            let mut gz = Vec::new();
            let mut e = flate2::write::GzEncoder::new(&mut gz, flate2::Compression::fast());
            writeln!(e, "{}", line.replace("439cf5fc", ids[0])).unwrap();
            writeln!(
                e,
                "{}",
                line.replace("439cf5fc", ids[1])
                    .replace("2026-06-03T06:", "2026-06-03T07:")
            )
            .unwrap();
            e.finish().unwrap();
            gz
        };
        std::fs::write(
            dir.path().join("shard-0000.jsonl.gz"),
            gz(["439cf5fc", "00000000"]),
        )
        .unwrap();

        let tarball = dir.path().join("day.tar.gz");
        {
            let inner = gz(["11111111", "22222222"]);
            let f = File::create(&tarball).unwrap();
            let mut b = tar::Builder::new(flate2::write::GzEncoder::new(
                f,
                flate2::Compression::fast(),
            ));
            let mut h = tar::Header::new_gnu();
            h.set_size(inner.len() as u64);
            h.set_cksum();
            b.append_data(&mut h, "date=2026-06-03/shard-0001.jsonl.gz", &inner[..])
                .unwrap();
            b.into_inner().unwrap().finish().unwrap();
        }

        let (sessions, stats) =
            convert(&[dir.path().to_path_buf()], &ConvertOptions::default(), 2).unwrap();
        assert_eq!(stats.sessions_read, 4);
        let order: Vec<(u64, &str)> = sessions
            .iter()
            .map(|s| (s.start_ms, &s.session_id[..8]))
            .collect();
        assert_eq!(
            order,
            vec![
                (0, "11111111"),
                (0, "439cf5fc"),
                (3_600_000, "00000000"),
                (3_600_000, "22222222"),
            ]
        );
        assert_eq!(sessions[0].calls[0].model.as_deref(), Some("Model E"));

        let out = dir.path().join("trace.jsonl");
        write_trace(&sessions, Some(&out)).unwrap();
        let back: Vec<TraceSession> = std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(back, sessions);
    }

    #[test]
    fn a_failing_worker_does_not_block_the_reader() {
        // One worker fails on the first shard; the remaining shards exceed
        // the channel's capacity of 2 * threads.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a0.jsonl"), "{\"turns\": []}\n").unwrap();
        for i in 1..=6 {
            std::fs::write(
                dir.path().join(format!("a{i}.jsonl")),
                format!("{{\"session_id\":\"x{i}\",\"turns\":[]}}\n"),
            )
            .unwrap();
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let p = dir.path().to_path_buf();
        std::thread::spawn(move || {
            let _ = tx.send(convert(&[p], &ConvertOptions::default(), 1).is_err());
        });
        assert_eq!(
            rx.recv_timeout(std::time::Duration::from_secs(10)),
            Ok(true),
            "convert hung or succeeded"
        );
    }

    #[test]
    fn repeated_session_id_names_both_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let line = format!("{}\n", FIXTURE.replace('\n', ""));
        std::fs::write(dir.path().join("a.jsonl"), &line).unwrap();
        std::fs::write(dir.path().join("b.jsonl"), &line).unwrap();
        let err = convert(&[dir.path().to_path_buf()], &ConvertOptions::default(), 2).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("439cf5fc") && msg.contains("a.jsonl") && msg.contains("b.jsonl"),
            "{msg}"
        );
    }

    #[test]
    fn own_output_is_rejected_as_input() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.jsonl");
        std::fs::write(&src, format!("{}\n", FIXTURE.replace('\n', ""))).unwrap();
        let (sessions, _) = convert(&[src], &ConvertOptions::default(), 1).unwrap();
        let out = dir.path().join("trace.jsonl");
        write_trace(&sessions, Some(&out)).unwrap();
        let err = convert(&[out], &ConvertOptions::default(), 1).unwrap_err();
        assert!(
            format!("{err:#}").contains("missing field `turns`"),
            "{err:#}"
        );
    }

    #[test]
    fn malformed_line_names_shard_and_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.jsonl");
        std::fs::write(
            &p,
            format!("{}\n{{\"turns\": []}}\n", FIXTURE.replace('\n', "")),
        )
        .unwrap();
        let err = convert(&[p], &ConvertOptions::default(), 1).unwrap_err();
        assert!(format!("{err:#}").contains("bad.jsonl: line 2"), "{err:#}");
    }
}
