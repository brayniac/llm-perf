//! The admin HTTP server: `/metrics` (Prometheus text), `/metrics.json`,
//! `/vars`, `/vars.json`, `/metrics/binary` (a msgpack snapshot) and
//! `/metrics/stream` (the registry as dendro replication frames).

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use metriken_exposition::group_builder::{DefaultRouter, GroupBuilder};
use metriken_exposition::stream::{Refusal, RegistryPasses, StreamRequest, StreamSubscription};
use metriken_exposition::{Snapshot, Snapshotter, SnapshotterBuilder};
use warp::Filter;
use warp::http::{HeaderValue, StatusCode};
use warp::reply::{Reply, Response};

/// The namespace of the one group the stream carries: `llm-perf/main`.
pub const STREAM_NAMESPACE: &str = "llm-perf";

/// How long a built stream pass is reused, and the shortest interval
/// served. Well below the intervals a recorder asks for
/// (one second by default): at an interval equal to the floor, wake-time
/// jitter can hand a subscriber the previous boundary's pass and that
/// interval's frame is empty (`RegistryPasses::new`). rezolus and
/// cachecannon use the same value.
pub const UPDATE_FLOOR: Duration = Duration::from_millis(10);

/// The passes every stream subscription on this server shares: one registry
/// read and one encode per pass, however many recorders are subscribed.
type Passes = RegistryPasses<DefaultRouter>;

/// Start the HTTP admin server for metrics
pub async fn start_server(addr: SocketAddr) {
    info!("Starting metrics server on {}", addr);
    warp::serve(routes()).run(addr).await;
}

/// Every route the admin server serves.
fn routes() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let passes = Arc::new(RegistryPasses::new(
        GroupBuilder::new(DefaultRouter::new(STREAM_NAMESPACE)),
        UPDATE_FLOOR,
    ));
    let snapshotter = Arc::new(
        SnapshotterBuilder::new()
            .metadata("source".to_string(), env!("CARGO_PKG_NAME").to_string())
            .metadata("version".to_string(), env!("CARGO_PKG_VERSION").to_string())
            .build(),
    );

    metrics_endpoint()
        .or(metrics_stream_endpoint(passes))
        .or(metrics_binary_endpoint(snapshotter))
        .or(metrics_json_endpoint())
        .or(vars_endpoint())
        .or(vars_json_endpoint())
}

/// GET /metrics - Prometheus/OpenMetrics format
fn metrics_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone
{
    warp::path!("metrics")
        .and(warp::get())
        .and_then(prometheus_metrics)
}

/// GET /metrics/binary - one msgpack `Snapshot` of the registry.
///
/// `rezolus record` probes `<base>/metrics/binary` before `<base>/metrics`
/// when given a base URL, and an endpoint that answers with a snapshot is
/// classified as a metriken source. Recording to a `.dendro`, rezolus at or
/// after 6.0.0-alpha.45 (until 6.0.0 is released, a build of rezolus main at
/// or after ab24b072) subscribes to `/metrics/stream`; rezolus 5.23 to 5.26
/// scrape this route into the archive; 6.0 alphas before alpha.45 refuse the
/// stream. Recording to anything else, rezolus scrapes this route. Without
/// it, llm-perf is scraped as a Prometheus endpoint.
fn metrics_binary_endpoint(
    snapshotter: Arc<Snapshotter>,
) -> impl Filter<Extract = (Response,), Error = warp::Rejection> + Clone {
    warp::path!("metrics" / "binary")
        .and(warp::get())
        .map(move || binary_response(&snapshotter))
}

fn binary_response(snapshotter: &Snapshotter) -> Response {
    match Snapshot::to_msgpack(&snapshotter.snapshot()) {
        Ok(bytes) => {
            warp::reply::with_header(bytes, "content-type", "application/msgpack").into_response()
        }
        Err(e) => {
            log::warn!("failed to serialize snapshot as msgpack: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// GET /metrics/stream - the registry as dendro replication frames, through
/// metriken-exposition's stream route.
///
/// Query parameters: `interval` (humantime, default `1s`) and `layout`
/// (`wide`, the default, or `long`). The body is dendro's preamble and a
/// handshake naming `source=llm-perf`, then one frame per interval until the
/// client disconnects. A recorder that accepts the
/// `application/vnd.metriken.replication.v1+dendro` content type puts
/// llm-perf into a `.dendro` from this: rows stamped here at read time and
/// histograms as histograms. Every metric goes through [`DefaultRouter`], so
/// they land in one windowless group, `llm-perf/main`.
///
/// A malformed request (a zero or unparseable interval, an unknown layout, a
/// parameter given twice) is refused with 400 and the reason as the body.
/// The request's interval is an offer: one below [`UPDATE_FLOOR`] is served
/// at the floor, and the frame-interval header names the interval served,
/// which a recorder takes as the stream's. Every pass inside the floor is
/// shared, so a subscription faster than it would send mostly empty frames
/// on the host generating load.
fn metrics_stream_endpoint(
    passes: Arc<Passes>,
) -> impl Filter<Extract = (Response,), Error = warp::Rejection> + Clone {
    warp::path!("metrics" / "stream")
        .and(warp::get())
        .and(warp::query::raw().or(warp::any().map(String::new)).unify())
        .map(move |query: String| stream_response(&passes, &query))
}

fn stream_response(passes: &Arc<Passes>, query: &str) -> Response {
    let mut request = match StreamRequest::parse(query) {
        Ok(request) => request,
        Err(refusal) => return refusal_response(refusal),
    };
    request.interval = request.interval.max(passes.floor());
    let layout = request.layout;
    let subscription =
        StreamSubscription::new(request, passes.floor(), stream_labels(), stream_metadata());
    let headers = subscription.headers();
    let passes = Arc::clone(passes);
    let body = subscription.into_stream(
        move |now| {
            let passes = Arc::clone(&passes);
            async move { passes.pass(layout, now) }
        },
        (),
        SystemTime::now,
    );
    let mut response = warp::reply::stream(body).into_response();
    for (name, value) in headers {
        match HeaderValue::try_from(value) {
            Ok(value) => {
                response.headers_mut().insert(name, value);
            }
            Err(e) => log::warn!("stream header {name} is not a header value: {e}"),
        }
    }
    response
}

/// The handshake's labels: what a recorder names this source by.
fn stream_labels() -> BTreeMap<String, String> {
    [("source".to_string(), STREAM_NAMESPACE.to_string())].into()
}

/// The handshake's metadata. The version is here, not in the labels, so two
/// builds recorded side by side are the same source kind.
fn stream_metadata() -> BTreeMap<String, String> {
    [("version".to_string(), env!("CARGO_PKG_VERSION").to_string())].into()
}

/// A refused stream request: its status and its message as a text body.
fn refusal_response(refusal: Refusal) -> Response {
    let status = StatusCode::from_u16(refusal.status).unwrap_or(StatusCode::BAD_REQUEST);
    warp::reply::with_status(refusal.message, status).into_response()
}

/// GET /metrics.json - JSON format (Finagle compatible)
fn metrics_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("metrics.json")
        .and(warp::get())
        .and_then(json_metrics)
}

/// GET /vars - Human readable format
fn vars_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars").and(warp::get()).and_then(human_metrics)
}

/// GET /vars.json - JSON format (alias)
fn vars_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars.json")
        .and(warp::get())
        .and_then(json_metrics)
}

async fn prometheus_metrics() -> Result<impl warp::Reply, Infallible> {
    let options = metriken_exposition::PrometheusOptions::default()
        .with_percentiles(vec![0.5, 0.9, 0.95, 0.99, 0.999]);
    let content = metriken_exposition::prometheus_text(&options);
    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; version=0.0.4; charset=utf-8",
    ))
}

async fn json_metrics() -> Result<impl warp::Reply, Infallible> {
    use metriken::Value;
    use serde_json::json;

    let mut metrics = serde_json::Map::new();
    let quantile_values = [0.5, 0.9, 0.95, 0.99, 0.999];

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Gauge(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Histogram(histogram)) => {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                {
                    for (quantile, bucket) in result.entries() {
                        let key = format!("{}/p{}", name, (quantile.as_f64() * 1000.0) as u32);
                        metrics.insert(key, json!(bucket.end()));
                    }
                }
            }
            Some(Value::HistogramGroup(group)) => {
                let snapshot = group.metadata_snapshot();
                for (idx, meta) in &snapshot {
                    if let Some(loaded) = group.load_histogram(*idx)
                        && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                    {
                        let suffix = format_metadata_suffix(meta);
                        for (quantile, bucket) in result.entries() {
                            let key = format!(
                                "{}{}/p{}",
                                name,
                                suffix,
                                (quantile.as_f64() * 1000.0) as u32
                            );
                            metrics.insert(key, json!(bucket.end()));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    Ok(warp::reply::json(&metrics))
}

async fn human_metrics() -> Result<impl warp::Reply, Infallible> {
    use metriken::Value;

    let quantile_values = [0.5, 0.9, 0.95, 0.99, 0.999];

    // Collect all metric lines first, then sort
    let mut lines = Vec::new();

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Gauge(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Histogram(histogram)) => {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                {
                    for (quantile, bucket) in result.entries() {
                        lines.push(format!(
                            "{}/p{}: {}",
                            name,
                            (quantile.as_f64() * 1000.0) as u32,
                            bucket.end()
                        ));
                    }
                }
            }
            Some(Value::HistogramGroup(group)) => {
                let snapshot = group.metadata_snapshot();
                for (idx, meta) in &snapshot {
                    if let Some(loaded) = group.load_histogram(*idx)
                        && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                    {
                        let suffix = format_metadata_suffix(meta);
                        for (quantile, bucket) in result.entries() {
                            lines.push(format!(
                                "{}{}/p{}: {}",
                                name,
                                suffix,
                                (quantile.as_f64() * 1000.0) as u32,
                                bucket.end()
                            ));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    lines.sort();
    // Pre-allocate output string based on line count
    let mut content = String::with_capacity(lines.len() * 40);
    for line in lines {
        content.push_str(&line);
        content.push('\n');
    }

    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; charset=utf-8",
    ))
}

/// Format metadata as a path suffix (e.g., "/reasoning/small").
fn format_metadata_suffix(meta: &std::collections::HashMap<String, String>) -> String {
    if meta.is_empty() {
        return String::new();
    }
    let mut parts: Vec<&str> = meta.values().map(|v| v.as_str()).collect();
    parts.sort();
    format!("/{}", parts.join("/"))
}

use log::info;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metrics;
    use dendro::replicate::Frame;
    use dendro::replicate::wire::FrameReader;
    use std::time::Duration;

    /// The admin routes on an ephemeral port of the test's runtime.
    async fn start() -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(warp::serve(routes()).incoming(listener).run());
        addr
    }

    async fn get(addr: SocketAddr, target: &str) -> reqwest::Response {
        reqwest::get(format!("http://{addr}{target}"))
            .await
            .unwrap()
    }

    fn header<'a>(response: &'a reqwest::Response, name: &str) -> Option<&'a str> {
        response.headers().get(name).and_then(|v| v.to_str().ok())
    }

    /// Read the body until it holds at least `want` complete frames. The
    /// response is chunked, and a frame can arrive in several chunks, so the
    /// bytes so far are decoded again after each one and a trailing partial
    /// frame is left for the next.
    async fn frames(response: &mut reqwest::Response, want: usize) -> Vec<Frame> {
        let mut body = Vec::new();
        loop {
            let chunk = tokio::time::timeout(Duration::from_secs(5), response.chunk())
                .await
                .expect("no data within 5 s")
                .expect("body read failed")
                .expect("body ended");
            body.extend_from_slice(&chunk);
            let Ok(mut reader) = FrameReader::new(std::io::Cursor::new(&body)) else {
                continue;
            };
            let mut decoded = Vec::new();
            while let Ok(Some(frame)) = reader.next_frame() {
                decoded.push(frame);
            }
            if decoded.len() >= want {
                return decoded;
            }
        }
    }

    /// The route end to end: headers, then dendro's preamble, the handshake
    /// naming llm-perf, and rows frames carrying `llm-perf/main` with the
    /// TTFT histogram for each context size in it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_stream_serves_handshake_and_ttft_rows() {
        Metrics::init();
        Metrics::record_ttft(Duration::from_millis(25), 100);
        let addr = start().await;

        let mut response = get(addr, "/metrics/stream?interval=100ms").await;
        assert_eq!(response.status(), 200);
        assert_eq!(
            header(&response, "content-type"),
            Some("application/vnd.metriken.replication.v1+dendro")
        );
        assert_eq!(
            header(&response, "x-metriken-frame-interval"),
            Some("100ms")
        );
        assert_eq!(header(&response, "x-metriken-update-floor"), Some("10ms"));
        assert_eq!(header(&response, "x-metriken-layout"), Some("wide"));

        let frames = frames(&mut response, 4).await;
        match &frames[0] {
            Frame::Handshake {
                labels, metadata, ..
            } => {
                assert_eq!(labels.get("source").map(String::as_str), Some("llm-perf"));
                assert_eq!(
                    metadata.get("version").map(String::as_str),
                    Some(env!("CARGO_PKG_VERSION"))
                );
            }
            other => panic!("first frame is not a handshake: {other:?}"),
        }

        // The first rows frame carries the group with its schema; each
        // member's labels are in it. Three rows frames, so the stream is
        // shown to keep going.
        let group = format!("{STREAM_NAMESPACE}/main");
        let mut seqs = Vec::new();
        for (i, frame) in frames[1..4].iter().enumerate() {
            let Frame::Rows { seq, rows, .. } = frame else {
                panic!("frame {} is not a rows frame: {frame:?}", i + 1);
            };
            seqs.push(*seq);
            if i > 0 {
                continue;
            }
            let row = rows
                .iter()
                .find(|r| r.stream == group)
                .unwrap_or_else(|| panic!("first rows frame has no {group} row"));
            let row = metriken_model::wal::decode_wal_group_row(&row.row).unwrap();
            let schema = row.schema.expect("the first row carries the schema");
            let member = schema
                .histograms
                .iter()
                .position(|m| {
                    m.metadata.get("metric").map(String::as_str) == Some("ttft")
                        && m.metadata.get("context_size").map(String::as_str) == Some("small")
                })
                .unwrap_or_else(|| panic!("no ttft member with context_size=small: {schema:?}"));
            let (_, _, buckets) = row.histograms[member]
                .as_ref()
                .expect("the ttft member has a value");
            // At least: the registry is process-global and other tests
            // record TTFT too.
            assert!(buckets.iter().sum::<u64>() >= 1, "ttft count is 0");
        }
        assert!(
            seqs.windows(2).all(|w| w[1] > w[0]),
            "seq does not advance: {seqs:?}"
        );
    }

    /// `layout=long`, which rezolus's recorder asks for. The header says
    /// long, and `llm-perf/main` still arrives as a wide row with the TTFT
    /// histogram in it: the long layout makes a group long only when every
    /// metric in it is a counter, gauge or histogram group, and llm-perf's
    /// one group also holds plain counters, gauges and histograms.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_stream_long_layout_serves_ttft_rows() {
        Metrics::init();
        Metrics::record_ttft(Duration::from_millis(25), 100);
        let addr = start().await;

        let mut response = get(addr, "/metrics/stream?interval=100ms&layout=long").await;
        assert_eq!(response.status(), 200);
        assert_eq!(header(&response, "x-metriken-layout"), Some("long"));

        let frames = frames(&mut response, 2).await;
        assert!(
            matches!(frames[0], Frame::Handshake { .. }),
            "{:?}",
            frames[0]
        );
        let Frame::Rows { rows, .. } = &frames[1] else {
            panic!("frame 1 is not a rows frame: {:?}", frames[1]);
        };
        let group = format!("{STREAM_NAMESPACE}/main");
        let streams: Vec<&str> = rows.iter().map(|r| r.stream.as_str()).collect();
        assert_eq!(streams, [group.as_str()], "streams in the first rows frame");
        let row = metriken_model::wal::decode_wal_group_row(&rows[0].row)
            .expect("llm-perf/main is a wide row under layout=long");
        let schema = row.schema.expect("the first row carries the schema");
        let member = schema
            .histograms
            .iter()
            .position(|m| m.metadata.get("metric").map(String::as_str) == Some("ttft"))
            .unwrap_or_else(|| panic!("no ttft member: {schema:?}"));
        let (_, _, buckets) = row.histograms[member]
            .as_ref()
            .expect("the ttft member has a value");
        assert!(buckets.iter().sum::<u64>() >= 1, "ttft count is 0");
    }

    #[tokio::test]
    async fn metrics_stream_without_a_query_uses_the_defaults() {
        let addr = start().await;
        let response = get(addr, "/metrics/stream").await;
        assert_eq!(response.status(), 200);
        assert_eq!(header(&response, "x-metriken-frame-interval"), Some("1s"));
        assert_eq!(header(&response, "x-metriken-layout"), Some("wide"));
    }

    #[tokio::test]
    async fn metrics_stream_refuses_a_zero_interval() {
        let addr = start().await;
        let response = get(addr, "/metrics/stream?interval=0s").await;
        assert_eq!(response.status(), 400);
        let body = response.text().await.unwrap();
        assert!(body.contains("greater than zero"), "{body}");
    }

    /// An offer below the floor is served at the floor, and the
    /// frame-interval header says so.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_stream_serves_an_interval_below_the_floor_at_the_floor() {
        let addr = start().await;
        let mut response = get(addr, "/metrics/stream?interval=1ms").await;
        assert_eq!(response.status(), 200);
        assert_eq!(header(&response, "x-metriken-frame-interval"), Some("10ms"));
        let frames = frames(&mut response, 1).await;
        assert!(
            matches!(frames[0], Frame::Handshake { .. }),
            "{:?}",
            frames[0]
        );
    }

    /// An interval equal to the floor is accepted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_stream_accepts_an_interval_at_the_floor() {
        let addr = start().await;
        let mut response = get(addr, "/metrics/stream?interval=10ms").await;
        assert_eq!(response.status(), 200);
        let frames = frames(&mut response, 1).await;
        assert!(
            matches!(frames[0], Frame::Handshake { .. }),
            "{:?}",
            frames[0]
        );
    }

    /// `/metrics/binary` is what `rezolus record` probes to classify an
    /// endpoint as a metriken source: a msgpack snapshot it can decode.
    #[tokio::test]
    async fn metrics_binary_serves_a_msgpack_snapshot() {
        Metrics::init();
        Metrics::record_ttft(Duration::from_millis(25), 100);
        let addr = start().await;
        let response = get(addr, "/metrics/binary").await;
        assert_eq!(response.status(), 200);
        assert_eq!(
            header(&response, "content-type"),
            Some("application/msgpack")
        );
        let body = response.bytes().await.unwrap();
        let mut snapshot = Snapshot::from_msgpack(&body).expect("body is not a snapshot");
        assert!(
            snapshot.histograms().iter().any(|h| {
                h.metadata.get("metric").map(String::as_str) == Some("ttft")
                    && h.metadata.get("context_size").map(String::as_str) == Some("small")
            }),
            "no ttft histogram for context_size=small"
        );
    }

    /// The routes that were there before are still served beside the new ones.
    #[tokio::test]
    async fn existing_routes_still_serve() {
        let addr = start().await;
        for target in ["/metrics", "/metrics.json", "/vars", "/vars.json"] {
            assert_eq!(get(addr, target).await.status(), 200, "{target}");
        }
        assert_eq!(get(addr, "/metrics/other").await.status(), 404);
    }
}
