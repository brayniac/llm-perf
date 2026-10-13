//! Records the run into one `.dendro` archive in process, through
//! metriken-recorder: llm-perf's own registry, and the server under test
//! when `server_metrics_url` is set. Both are on one timeline, and the
//! archive is readable while the run goes.
//!
//! llm-perf is a local source: each tick takes a pass of the registry, so
//! histograms are recorded with their buckets. The server under test is
//! detected by metriken-recorder: a base URL (path `/`) or a
//! `/metrics/stream` URL is asked for its stream first and scraped at
//! `/metrics` if it has none; any other URL is scraped as Prometheus text.
//!
//! Each recording carries a `role` label, `loadgen` for llm-perf and
//! `service` for the server under test, so the two stay distinct when the
//! server names itself `llm-perf` too (another llm-perf's stream does).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use metriken_exposition::group_builder::{DefaultRouter, GroupBuilder};
use metriken_recorder::{
    ActiveTransport, CancellationToken, DetectConfig, Detected, EndpointSpec, Event, NoProbe,
    Recorder, RecorderConfig, RegistrySource, SourceId, SourceState, Stop, Summary,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::admin::STREAM_NAMESPACE;
use crate::config::MetricsConfig;

/// The `source` label llm-perf's recording carries, the same as the stream
/// handshake's.
pub const LLM_PERF_SOURCE: &str = STREAM_NAMESPACE;

/// The label that tells llm-perf's recording from the server under test's.
pub const ROLE: &str = "role";
/// llm-perf's `role`.
pub const ROLE_LOADGEN: &str = "loadgen";
/// The server under test's `role`.
pub const ROLE_SERVICE: &str = "service";

/// What the event logger keeps for [`RunRecorder::finish`].
#[derive(Default)]
struct Seen {
    /// The last reason each source was pending or refused.
    reasons: HashMap<SourceId, String>,
    /// A source is streamed, so finishing waits for its last frame.
    streamed: bool,
}

/// The recorder and its event logger, taken by whichever of
/// [`RunRecorder::finish`] and the signal watcher gets there first.
struct Live {
    recorder: Recorder,
    events: JoinHandle<()>,
}

/// What [`RunRecorder::finish`] and the signal watcher share.
struct Shared {
    live: Mutex<Option<Live>>,
    /// Cancelled by a signal while `finish` waits for a streamed server's
    /// last frame: the wait ends and the archive is finalized.
    cancel: CancellationToken,
    /// `finish` has returned; a signal now exits at once.
    finished: AtomicBool,
    seen: Arc<Mutex<Seen>>,
    path: PathBuf,
}

/// A recording in progress.
///
/// From [`RunRecorder::start`] on, SIGINT and SIGTERM (Ctrl-C elsewhere)
/// finalize the archive and exit with 130 or 143, without waiting for the
/// benchmark: a pass of llm-perf and of a scraped server, no wait for a
/// streamed one. A second signal during that exits at once. A signal while
/// `finish` waits for a streamed server's last frame ends the wait, and one
/// after `finish` returned exits at once. The handlers stay installed for the
/// rest of the process (tokio never removes them), which is why the watcher
/// keeps running after `finish`.
pub struct RunRecorder {
    shared: Arc<Shared>,
    bound: Duration,
}

impl RunRecorder {
    /// Start recording into `mc.recording_path()`, or return `None` when the
    /// config names no archive. An error (an archive that already exists, a
    /// directory that does not) is returned before the benchmark starts.
    pub async fn start(mc: &MetricsConfig) -> anyhow::Result<Option<Self>> {
        let Some(path) = mc.recording_path() else {
            return Ok(None);
        };
        if mc.rezolus_bin.is_some() {
            log::warn!(
                "[metrics] rezolus_bin is ignored: llm-perf records the server under test in \
                 process"
            );
        }
        let interval = mc.interval()?;
        let mut cfg = RecorderConfig::new(&path, interval);
        let timeout = cfg.timeout;
        cfg.events = 256;

        // reqwest 0.13 is built without a crypto provider; ring, as llm-perf's
        // own reqwest 0.12 client uses. Installing twice is harmless.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = recorder_reqwest::Client::builder()
            .build()
            .context("building the recorder's HTTP client")?;

        let mut specs = Vec::new();
        if let Some(url) = &mc.server_metrics_url {
            let url = url::Url::parse(url)
                .with_context(|| format!("[metrics] server_metrics_url {url:?}"))?;
            specs.push(EndpointSpec::new(url).with_label(ROLE, ROLE_SERVICE));
        }
        let detected = if specs.is_empty() {
            Vec::new()
        } else {
            metriken_recorder::detect(
                &client,
                &DetectConfig::new(interval, timeout),
                specs,
                &NoProbe,
            )
            .await
        };
        // Detection's reasons, keyed by position: endpoints are numbered first.
        let mut seen = Seen::default();
        for (i, d) in detected.iter().enumerate() {
            match d {
                Detected::Refused { spec, reason } => {
                    log::warn!(
                        "the server under test at {} cannot be recorded: {reason}; {path:?} \
                         records llm-perf only",
                        spec.url
                    );
                    seen.reasons.insert(SourceId(i as u32), reason.clone());
                }
                // The recorder reports a pending endpoint as an event too,
                // which `log_events` logs.
                Detected::Pending { reason, .. } => {
                    seen.reasons.insert(SourceId(i as u32), reason.clone());
                }
                Detected::Streamed { .. } | Detected::Scraped { .. } => {}
            }
        }

        let llm_perf = RegistrySource::new(
            GroupBuilder::new(DefaultRouter::new(STREAM_NAMESPACE)),
            BTreeMap::from([
                ("source".to_string(), LLM_PERF_SOURCE.to_string()),
                (ROLE.to_string(), ROLE_LOADGEN.to_string()),
            ]),
            BTreeMap::from([("version".to_string(), env!("CARGO_PKG_VERSION").to_string())]),
        );
        let (recorder, _ids, rx) = Recorder::start(
            cfg,
            client,
            Arc::new(NoProbe),
            detected,
            vec![Box::new(llm_perf)],
        )
        .await
        .with_context(|| format!("recording {path:?}"))?;
        if mc.server_metrics_url.is_some() {
            log::info!("recording llm-perf and the server under test into {path:?}");
        } else {
            log::info!("recording llm-perf into {path:?}");
        }
        let seen = Arc::new(Mutex::new(seen));
        let events = tokio::spawn(log_events(rx, Arc::clone(&seen)));
        let shared = Arc::new(Shared {
            live: Mutex::new(Some(Live { recorder, events })),
            cancel: CancellationToken::new(),
            finished: AtomicBool::new(false),
            seen,
            path,
        });
        match Signals::new() {
            Ok(signals) => {
                tokio::spawn(watch_signals(signals, Arc::clone(&shared)));
            }
            Err(e) => log::warn!(
                "cannot watch for SIGINT/SIGTERM ({e}); an interrupted run leaves {:?} \
                 unfinalized",
                shared.path
            ),
        }
        Ok(Some(Self {
            shared,
            bound: interval + timeout,
        }))
    }

    /// End the recording. After a run that `completed`: one more pass of
    /// llm-perf and a scraped server, a wait of up to one interval plus the
    /// timeout for a streamed server's next frame, then finalize. After a run
    /// that failed, no wait, since the benchmark's workers may still be
    /// sending load. Returns what went wrong, for the benchmark summary; it
    /// is also logged.
    pub async fn finish(self, completed: bool) -> Option<String> {
        let live = self
            .shared
            .live
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let Some(live) = live else {
            // A signal took the recorder; the watcher finalizes and exits.
            return std::future::pending().await;
        };
        let stop = if completed {
            if self.shared.seen.lock().map(|s| s.streamed).unwrap_or(false) {
                log::info!(
                    "finalizing {:?}: waiting up to {:?} for the server under test's last \
                     frame (Ctrl-C ends the wait)",
                    self.shared.path,
                    self.bound
                );
            }
            Stop::AfterExit {
                wall_ns: wall_now_ns(),
                bound: self.bound,
            }
        } else {
            Stop::Now
        };
        let failure = finalize(live, stop, self.shared.cancel.clone(), &self.shared).await;
        self.shared.finished.store(true, Ordering::SeqCst);
        failure
    }
}

fn wall_now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default()
}

/// Finish the recorder, wait for its event logger, and say what went wrong.
async fn finalize(
    live: Live,
    stop: Stop,
    cancel: CancellationToken,
    shared: &Shared,
) -> Option<String> {
    let path = &shared.path;
    let result = live.recorder.finish(stop, cancel).await;
    let _ = live.events.await;
    let failure = match result {
        Ok(summary) => {
            let seen = shared.seen.lock().unwrap_or_else(|e| e.into_inner());
            summary_failure(&summary, &seen.reasons, path)
        }
        Err(e) => Some(format!(
            "recording {path:?} failed: {e}; it holds the ticks committed before the failure"
        )),
    };
    match &failure {
        Some(failure) => log::error!("{failure}"),
        None => log::info!("finished recording {path:?}"),
    }
    failure
}

/// SIGINT and SIGTERM, registered before the watcher starts so none is
/// missed in between.
struct Signals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
}

impl Signals {
    #[cfg(unix)]
    fn new() -> std::io::Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};
        Ok(Self {
            interrupt: signal(SignalKind::interrupt())?,
            terminate: signal(SignalKind::terminate())?,
        })
    }

    #[cfg(not(unix))]
    fn new() -> std::io::Result<Self> {
        Ok(Self {})
    }

    /// The next signal, as the exit status a shell gives it.
    #[cfg(unix)]
    async fn next(&mut self) -> i32 {
        tokio::select! {
            _ = self.interrupt.recv() => 130,
            _ = self.terminate.recv() => 143,
        }
    }

    #[cfg(not(unix))]
    async fn next(&mut self) -> i32 {
        let _ = tokio::signal::ctrl_c().await;
        130
    }
}

async fn watch_signals(mut signals: Signals, shared: Arc<Shared>) {
    loop {
        let code = signals.next().await;
        let live = shared.live.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(live) = live {
            log::warn!(
                "interrupted; finalizing {:?} (again to exit now)",
                shared.path
            );
            let cancel = CancellationToken::new();
            // The log writer is not flushed by `exit`, so the outcome also
            // goes to stderr.
            tokio::select! {
                failure = finalize(live, Stop::Now, cancel, &shared) => match failure {
                    Some(failure) => eprintln!("llm-perf: interrupted; {failure}"),
                    None => eprintln!("llm-perf: interrupted; finalized {:?}", shared.path),
                },
                _ = signals.next() => {
                    eprintln!("llm-perf: interrupted again; {:?} is not finalized", shared.path)
                }
            }
            std::process::exit(code);
        }
        if shared.finished.load(Ordering::SeqCst) || shared.cancel.is_cancelled() {
            std::process::exit(code);
        }
        log::warn!("interrupted; ending the wait and finalizing (again to exit now)");
        shared.cancel.cancel();
    }
}

/// What the summary says went wrong, if anything: no data at all, or a server
/// under test that was never recorded, with the last reason it gave.
fn summary_failure(
    summary: &Summary,
    reasons: &HashMap<SourceId, String>,
    path: &Path,
) -> Option<String> {
    if summary.discarded {
        return Some(format!(
            "nothing was recorded, so {path:?} was removed rather than finalized"
        ));
    }
    let missing: Vec<String> = summary
        .sources
        .iter()
        .filter(|s| s.passes_total == 0)
        .filter_map(|s| {
            let endpoint = s.endpoint.as_deref()?;
            let state = match s.state {
                SourceState::Refused => "refused",
                SourceState::Pending | SourceState::Activating => "not detected",
                _ => "no frame or scrape arrived",
            };
            Some(match reasons.get(&s.id) {
                Some(reason) => format!("{endpoint} ({state}: {reason})"),
                None => format!("{endpoint} ({state})"),
            })
        })
        .collect();
    (!missing.is_empty()).then(|| {
        format!(
            "{path:?} has no rows for the server under test at {}",
            missing.join(", ")
        )
    })
}

/// Log what the recorder reports, and keep what `finish` needs. Per-frame
/// and per-commit events are left to debug.
async fn log_events(mut rx: mpsc::Receiver<Event>, seen: Arc<Mutex<Seen>>) {
    let note = |f: &mut dyn FnMut(&mut Seen)| {
        if let Ok(mut s) = seen.lock() {
            f(&mut s);
        }
    };
    while let Some(event) = rx.recv().await {
        match event {
            Event::Active {
                source,
                name,
                transport,
                inferred,
            } => {
                note(&mut |s| {
                    s.reasons.remove(&source);
                    s.streamed |= transport == ActiveTransport::Stream;
                });
                if inferred {
                    log::info!(
                        "recording source={name} ({transport:?}); the name was made from its URL"
                    );
                } else {
                    log::info!("recording source={name} ({transport:?})");
                }
            }
            Event::Pending { source, reason } => {
                log::warn!("{source} is not recorded yet: {reason}");
                note(&mut |s| {
                    s.reasons.insert(source, reason.clone());
                });
            }
            Event::Dropped { source, reason } => {
                log::warn!("{source}'s stream dropped ({reason}); reconnecting")
            }
            Event::Reconnected {
                source, restarted, ..
            } => log::info!("{source} reconnected (restarted: {restarted})"),
            Event::Refused { source, reason } => {
                log::warn!("{source} is refused: {reason}");
                note(&mut |s| {
                    s.reasons.insert(source, reason.clone());
                });
            }
            Event::Restarted {
                source, version, ..
            } => log::warn!(
                "{source} restarted (version {})",
                version.as_deref().unwrap_or("unknown")
            ),
            Event::Failed { error } => log::error!("the recording failed: {error}"),
            other => log::debug!("recorder: {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metriken_storage::{Catalog, DendroCatalog};
    use std::path::{Path, PathBuf};

    fn mc(output: &Path, url: Option<String>) -> MetricsConfig {
        MetricsConfig {
            output: output.to_path_buf(),
            interval: "100ms".to_string(),
            batch_size: None,
            server_metrics_url: url,
            server_metrics_output: None,
            rezolus_bin: None,
        }
    }

    /// Each source's `source` label and its tables.
    fn sources(path: &Path) -> BTreeMap<String, Vec<String>> {
        let catalog = DendroCatalog::open(path).unwrap();
        catalog
            .sources()
            .unwrap()
            .into_iter()
            .map(|s| {
                let name = s.labels.get("source").cloned().unwrap_or_default();
                (name, catalog.tables(s.id).unwrap())
            })
            .collect()
    }

    /// Each recording's labels.
    fn label_sets(path: &Path) -> Vec<BTreeMap<String, String>> {
        let catalog = DendroCatalog::open(path).unwrap();
        let mut sets: Vec<_> = catalog
            .sources()
            .unwrap()
            .into_iter()
            .map(|s| s.labels)
            .collect();
        sets.sort();
        sets
    }

    fn labels(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    async fn record_for(mc: &MetricsConfig, run: Duration) -> Option<String> {
        let recorder = RunRecorder::start(mc).await.unwrap().unwrap();
        crate::metrics::REQUEST_LATENCY
            .increment(1_000_000)
            .unwrap();
        tokio::time::sleep(run).await;
        recorder.finish(true).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn records_llm_perf_and_a_prometheus_server_into_one_archive() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/metrics")
            .with_header("content-type", "text/plain; version=0.0.4")
            .with_body("# TYPE vllm_requests_total counter\nvllm_requests_total 5\n")
            .create_async()
            .await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.dendro");
        let mc = mc(&path, Some(format!("{}/metrics", server.url())));

        assert_eq!(record_for(&mc, Duration::from_millis(600)).await, None);

        let sources = sources(&path);
        assert_eq!(sources.len(), 2, "{sources:?}");
        // `llm-perf/main/occupants` is there too once any test in this
        // process has written a member of a histogram or counter group.
        let main = format!("{STREAM_NAMESPACE}/main");
        assert!(
            sources
                .get(LLM_PERF_SOURCE)
                .is_some_and(|tables| tables.contains(&main)),
            "{sources:?}"
        );
        let server = sources
            .iter()
            .find(|(name, _)| *name != LLM_PERF_SOURCE)
            .unwrap();
        assert_eq!(
            *server.1,
            [metriken_recorder::PROMETHEUS_GROUP],
            "{sources:?}"
        );
        let sets = label_sets(&path);
        assert!(
            sets.iter()
                .any(|l| l.get("role").map(String::as_str) == Some(ROLE_SERVICE)
                    && l.get("source").map(String::as_str) != Some(LLM_PERF_SOURCE)),
            "{sets:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn without_a_server_url_the_archive_holds_llm_perf_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.dendro");

        assert_eq!(
            record_for(&mc(&path, None), Duration::from_millis(300)).await,
            None
        );

        let sources = sources(&path);
        assert_eq!(sources.keys().collect::<Vec<_>>(), [LLM_PERF_SOURCE]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_that_never_answers_is_reported_and_llm_perf_is_kept() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.dendro");
        let url = format!("http://127.0.0.1:{port}/metrics");

        let failure = record_for(&mc(&path, Some(url.clone())), Duration::from_millis(300))
            .await
            .expect("a missing server is reported");
        assert!(failure.contains(&url), "{failure}");
        assert!(failure.contains("not detected"), "{failure}");

        let sources = sources(&path);
        assert_eq!(sources.keys().collect::<Vec<_>>(), [LLM_PERF_SOURCE]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_existing_archive_is_an_error_before_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let path: PathBuf = dir.path().join("run.dendro");
        std::fs::write(&path, b"keep me").unwrap();

        let err = RunRecorder::start(&mc(&path, None)).await.err().unwrap();
        assert!(format!("{err:#}").contains("run.dendro"), "{err:#}");
        assert_eq!(std::fs::read(&path).unwrap(), b"keep me");
    }

    #[tokio::test]
    async fn no_archive_without_a_dendro_output_or_server() {
        let dir = tempfile::tempdir().unwrap();
        let mc = mc(&dir.path().join("run.parquet"), None);
        assert!(RunRecorder::start(&mc).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn another_llm_perf_streamed_as_the_server_stays_a_separate_recording() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
        tokio::spawn(crate::admin::start_server(addr));
        for _ in 0..50 {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.dendro");

        let failure = record_for(
            &mc(&path, Some(format!("http://{addr}/"))),
            Duration::from_millis(600),
        )
        .await;
        assert_eq!(failure, None);

        // Both name themselves llm-perf; the role tells them apart.
        let sets = label_sets(&path);
        let ours = labels(&[("role", ROLE_LOADGEN), ("source", LLM_PERF_SOURCE)]);
        assert!(sets.contains(&ours), "{sets:?}");
        assert_eq!(sets.len(), 2, "{sets:?}");
        let server = sets.iter().find(|l| **l != ours).unwrap();
        assert_eq!(
            server.get("source").map(String::as_str),
            Some(LLM_PERF_SOURCE)
        );
        assert_eq!(server.get("role").map(String::as_str), Some(ROLE_SERVICE));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_answering_404_is_reported_with_the_status() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/wrong")
            .with_status(404)
            .create_async()
            .await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.dendro");
        let url = format!("{}/wrong", server.url());

        let failure = record_for(&mc(&path, Some(url.clone())), Duration::from_millis(300))
            .await
            .expect("a 404 is reported");
        assert!(failure.contains(&url), "{failure}");
        assert!(failure.contains("404"), "{failure}");
    }
}
