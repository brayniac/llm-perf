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
//!   `timestamp - duration_ms`; read as a start time, a third of consecutive
//!   calls within a turn would overlap, read as an end time 0.1% do.
//! - A session has at most one call in flight, so its calls form one sequence.
//! - `message_metadata` segment lengths change on every call, including the
//!   system prompt, so they cannot locate the shared prefix. `tokens.cached`
//!   is used instead.
//! - After the session has been idle for more than about five minutes the
//!   source's cache usually reports zero cached tokens, while shorter idle
//!   gaps keep ~98% of the previous prompt. Zero hits after a long gap are
//!   treated as the source's eviction, not as the prompt having changed; see
//!   [`ConvertOptions::cache_ttl_ms`].

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
    /// `start` relative to the earliest session start in the output.
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
    /// "user" for the call that opens a turn, "agent" for the agent loop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initiator: Option<String>,
    /// Time from the end of the previous call to the start of this one; 0 for
    /// the first call of a session.
    pub gap_ms: u64,
    pub prompt: u64,
    pub completion: u64,
    /// Cached prompt tokens as reported by the source.
    pub cached: u64,
    /// Leading tokens this prompt repeats from the previous call's prompt
    /// followed by its completion. 0 for the first call; prefix shared across
    /// sessions shows only in `cached`.
    pub reuse: u64,
    /// `reuse` was estimated because the source reported no cache hit after
    /// an idle gap longer than the cache TTL.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reuse_inferred: bool,
    /// The source's own latency for this call.
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Keep only sessions whose calls all use one of these model labels.
    /// Empty keeps every session.
    pub models: Vec<String>,
    /// Drop sessions with any call whose prompt + completion exceeds this.
    pub max_context: Option<u64>,
    /// A zero cache hit after an idle gap at least this long is treated as
    /// eviction, and `reuse` is estimated with `evicted_reuse_ratio`.
    pub cache_ttl_ms: u64,
    /// Fraction of `min(previous prompt, prompt)` assumed reused after
    /// eviction. 0.98 is the median observed for idle gaps under 300 s.
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
    pub calls_read: u64,
    pub calls_kept: u64,
    /// Calls with no prompt tokens (failed before reaching the model).
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
        self.calls_read += o.calls_read;
        self.calls_kept += o.calls_kept;
        self.calls_dropped_no_prompt += o.calls_dropped_no_prompt;
        self.calls_reuse_inferred += o.calls_reuse_inferred;
        self.calls_reuse_clamped += o.calls_reuse_clamped;
    }
}

// Source records. Only the fields the conversion reads are declared; serde
// skips the rest (`message_metadata`, `tool_batches`, ids).

#[derive(Debug, Deserialize)]
struct RawSession {
    session_id: String,
    #[serde(default)]
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
                start_us: c.timestamp.timestamp_micros() - duration_us,
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
                if c.cached == 0 && gap_ms >= opts.cache_ttl_ms {
                    let base = prev.prompt.min(c.prompt);
                    let est = (base as f64 * opts.evicted_reuse_ratio).round() as u64;
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
    let start = DateTime::from_timestamp_micros(first.start_us)?;
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

/// A unit of input handed to a worker: one JSONL shard, still compressed.
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

fn convert_shard(
    shard: &Shard,
    opts: &ConvertOptions,
    out: &mut Vec<TraceSession>,
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
    let rx: Arc<Mutex<Receiver<Shard>>> = Arc::new(Mutex::new(rx));

    let (mut sessions, stats) = std::thread::scope(|s| -> Result<_> {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let rx = Arc::clone(&rx);
                s.spawn(move || -> Result<(Vec<TraceSession>, ConvertStats)> {
                    let mut out = Vec::new();
                    let mut stats = ConvertStats::default();
                    loop {
                        let shard = match rx.lock().unwrap().recv() {
                            Ok(shard) => shard,
                            Err(_) => break,
                        };
                        convert_shard(&shard, opts, &mut out, &mut stats)?;
                    }
                    Ok((out, stats))
                })
            })
            .collect();

        // Each worker returns on its first error, so a send fails only once
        // every worker has exited; the joins below report the worker's error
        // ahead of the read error.
        let read = read_shards(&paths, |shard| {
            tx.send(shard)
                .map_err(|_| anyhow::anyhow!("all workers exited"))
        });
        drop(tx);

        let mut sessions = Vec::new();
        let mut stats = ConvertStats::default();
        for w in workers {
            let (out, st) = w.join().expect("convert worker panicked")?;
            sessions.extend(out);
            stats.add(&st);
        }
        read?;
        Ok((sessions, stats))
    })?;

    sessions.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| a.session_id.cmp(&b.session_id))
    });
    if let Some(t0) = sessions.first().map(|s| s.start) {
        for s in &mut sessions {
            s.start_ms = (s.start - t0).num_milliseconds() as u64;
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
        "sessions: read {} kept {} (dropped: empty {}, model {}, context {}, window {})",
        stats.sessions_read,
        stats.sessions_kept,
        stats.sessions_dropped_empty,
        stats.sessions_dropped_model,
        stats.sessions_dropped_context,
        stats.sessions_dropped_window,
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
    fn fixture_reuse_follows_cached_within_ttl() {
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
    }

    #[test]
    fn convert_reads_shards_and_tarballs() {
        let dir = tempfile::tempdir().unwrap();
        let line = FIXTURE.replace('\n', "");
        let mut gz = Vec::new();
        {
            let mut e = flate2::write::GzEncoder::new(&mut gz, flate2::Compression::fast());
            writeln!(e, "{line}").unwrap();
            writeln!(e, "{}", line.replace("439cf5fc", "00000000")).unwrap();
        }
        let shard = dir.path().join("shard-0000.jsonl.gz");
        std::fs::write(&shard, &gz).unwrap();

        let tarball = dir.path().join("day.tar.gz");
        {
            let f = File::create(&tarball).unwrap();
            let mut b = tar::Builder::new(flate2::write::GzEncoder::new(
                f,
                flate2::Compression::fast(),
            ));
            let mut h = tar::Header::new_gnu();
            h.set_size(gz.len() as u64);
            h.set_cksum();
            b.append_data(&mut h, "date=2026-06-03/shard-0001.jsonl.gz", &gz[..])
                .unwrap();
            b.into_inner().unwrap().finish().unwrap();
        }

        let (sessions, stats) =
            convert(&[dir.path().to_path_buf()], &ConvertOptions::default(), 2).unwrap();
        assert_eq!(stats.sessions_read, 4);
        assert_eq!(sessions.len(), 4);
        assert!(sessions.iter().all(|s| s.start_ms == 0));

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
