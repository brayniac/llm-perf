//! Records the run into one `.dendro` archive in process, through
//! metriken-recorder: llm-perf's own registry, and the server under test
//! when `server_metrics_url` is set. Both are on one timeline, and the
//! archive is readable while the run goes and after a crash.
//!
//! llm-perf is a local source: each tick takes a pass of the registry, so
//! histograms are recorded with their buckets. The server under test is
//! detected by metriken-recorder: a base URL (path `/`) or a
//! `/metrics/stream` URL is asked for its stream first and scraped at
//! `/metrics` if it has none; any other URL is scraped as Prometheus text.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use metriken_exposition::group_builder::{DefaultRouter, GroupBuilder};
use metriken_recorder::{
    CancellationToken, DetectConfig, Detected, EndpointSpec, Event, NoProbe, Recorder,
    RecorderConfig, RegistrySource, SourceState, Stop, Summary,
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::admin::STREAM_NAMESPACE;
use crate::config::MetricsConfig;

/// The `source` label llm-perf's recording carries, the same as the stream
/// handshake's.
pub const LLM_PERF_SOURCE: &str = STREAM_NAMESPACE;

/// A recording in progress.
pub struct RunRecorder {
    recorder: Recorder,
    events: JoinHandle<()>,
    path: std::path::PathBuf,
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
            specs.push(EndpointSpec::new(url));
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
        for d in &detected {
            match d {
                Detected::Refused { spec, reason } => log::warn!(
                    "the server under test at {} cannot be recorded: {reason}; {path:?} \
                     records llm-perf only",
                    spec.url
                ),
                Detected::Pending { spec, reason } => log::warn!(
                    "the server under test at {} did not answer ({reason}); it is recorded \
                     once it does",
                    spec.url
                ),
                Detected::Streamed { .. } | Detected::Scraped { .. } => {}
            }
        }

        let llm_perf = RegistrySource::new(
            GroupBuilder::new(DefaultRouter::new(STREAM_NAMESPACE)),
            [("source".to_string(), LLM_PERF_SOURCE.to_string())].into(),
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
        Ok(Some(Self {
            recorder,
            events: tokio::spawn(log_events(rx)),
            path,
            bound: interval + timeout,
        }))
    }

    /// End the recording now: one more pass of llm-perf and a scraped
    /// server, a wait of up to one interval plus the timeout for a streamed
    /// server's next frame, then finalize. Returns what went wrong, for the
    /// benchmark summary; it is also logged.
    pub async fn finish(self) -> Option<String> {
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or_default();
        let stop = Stop::AfterExit {
            wall_ns,
            bound: self.bound,
        };
        let result = self.recorder.finish(stop, CancellationToken::new()).await;
        let _ = self.events.await;
        let failure = match result {
            Ok(summary) => summary_failure(&summary, &self.path),
            Err(e) => Some(format!(
                "recording {:?} failed: {e}; it holds the ticks committed before the failure",
                self.path
            )),
        };
        match &failure {
            Some(failure) => log::error!("{failure}"),
            None => log::info!("finished recording {:?}", self.path),
        }
        failure
    }
}

/// What the summary says went wrong, if anything: no data at all, or a server
/// under test that was never recorded.
fn summary_failure(summary: &Summary, path: &std::path::Path) -> Option<String> {
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
            let why = match s.state {
                SourceState::Refused => "it cannot be recorded",
                SourceState::Pending | SourceState::Activating => "it never answered",
                _ => "no frame or scrape arrived",
            };
            Some(format!("{endpoint} ({why})"))
        })
        .collect();
    (!missing.is_empty()).then(|| {
        format!(
            "{path:?} has no rows for the server under test at {}",
            missing.join(", ")
        )
    })
}

/// Log what the recorder reports. Per-frame and per-commit events are left
/// to debug.
async fn log_events(mut rx: mpsc::Receiver<Event>) {
    while let Some(event) = rx.recv().await {
        match event {
            Event::Active {
                name,
                transport,
                inferred,
                ..
            } => {
                if inferred {
                    log::info!(
                        "recording source={name} ({transport:?}); the name was made from its URL"
                    );
                } else {
                    log::info!("recording source={name} ({transport:?})");
                }
            }
            Event::Pending { source, reason } => {
                log::warn!("{source} is not recorded yet: {reason}")
            }
            Event::Dropped { source, reason } => {
                log::warn!("{source}'s stream dropped ({reason}); reconnecting")
            }
            Event::Reconnected {
                source, restarted, ..
            } => log::info!("{source} reconnected (restarted: {restarted})"),
            Event::Refused { source, reason } => log::warn!("{source} is refused: {reason}"),
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

    async fn record_for(mc: &MetricsConfig, run: Duration) -> Option<String> {
        let recorder = RunRecorder::start(mc).await.unwrap().unwrap();
        crate::metrics::REQUEST_LATENCY
            .increment(1_000_000)
            .unwrap();
        tokio::time::sleep(run).await;
        recorder.finish().await
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
        assert!(failure.contains("never answered"), "{failure}");

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
}
