use metriken::{AtomicHistogram, Counter, Gauge, LazyCounter, LazyGauge, metric};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum RequestStatus {
    Success,
    Failed(ErrorType),
    Timeout,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Connection,
    Http4xx(u16),
    Http5xx(u16),
    Parse,
    Timeout,
    Other,
}

/// Generation phase for reasoning models.
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    /// Reasoning/thinking tokens (e.g., Qwen3 `reasoning_content`, DeepSeek-R1)
    Reasoning,
    /// Visible content tokens
    Content,
}

// Global running flag for background tasks
pub static RUNNING: AtomicBool = AtomicBool::new(false);

// Request metrics
#[metric(
    name = "requests",
    description = "Total number of requests",
    metadata = { status = "sent" }
)]
pub static REQUESTS_SENT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Successful requests",
    metadata = { status = "success" }
)]
pub static REQUESTS_SUCCESS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Failed requests",
    metadata = { status = "failed" }
)]
pub static REQUESTS_FAILED: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Timed out requests",
    metadata = { status = "timeout" }
)]
pub static REQUESTS_TIMEOUT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "requests",
    description = "Request retries",
    metadata = { status = "retried" }
)]
pub static REQUESTS_RETRIED: LazyCounter = LazyCounter::new(Counter::default);

// Error category metrics
#[metric(
    name = "errors",
    description = "Connection errors",
    metadata = { "type" = "connection" }
)]
pub static ERRORS_CONNECTION: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "HTTP 4xx errors",
    metadata = { "type" = "http_4xx" }
)]
pub static ERRORS_HTTP_4XX: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "HTTP 5xx errors",
    metadata = { "type" = "http_5xx" }
)]
pub static ERRORS_HTTP_5XX: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "Parse errors",
    metadata = { "type" = "parse" }
)]
pub static ERRORS_PARSE: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "errors",
    description = "Other errors",
    metadata = { "type" = "other" }
)]
pub static ERRORS_OTHER: LazyCounter = LazyCounter::new(Counter::default);

// Token metrics
#[metric(
    name = "tokens",
    description = "Input tokens processed",
    metadata = { direction = "input" }
)]
pub static TOKENS_INPUT: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "tokens",
    description = "Output tokens generated",
    metadata = { direction = "output" }
)]
pub static TOKENS_OUTPUT: LazyCounter = LazyCounter::new(Counter::default);

// Concurrency metrics
#[metric(
    name = "requests_inflight",
    description = "Current number of requests in flight"
)]
pub static REQUESTS_INFLIGHT: LazyGauge = LazyGauge::new(Gauge::default);

// Latency metrics (in nanoseconds)
// Histogram parameters: (grouping_power=7, max_value_power=64)
// This gives 128 buckets per power of 2 (~0.54% relative precision), covering the full 64-bit range

// TTFT histograms — reasoning phase (context-size bucketed)
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "small", phase = "reasoning" })]
pub static TTFT_REASONING_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "medium", phase = "reasoning" })]
pub static TTFT_REASONING_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "large", phase = "reasoning" })]
pub static TTFT_REASONING_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "reasoning" })]
pub static TTFT_REASONING_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "reasoning" })]
pub static TTFT_REASONING_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

// TTFT histograms — content phase (context-size bucketed)
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "small", phase = "content" })]
pub static TTFT_CONTENT_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "medium", phase = "content" })]
pub static TTFT_CONTENT_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "large", phase = "content" })]
pub static TTFT_CONTENT_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "content" })]
pub static TTFT_CONTENT_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "content" })]
pub static TTFT_CONTENT_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

#[metric(
    name = "request_latency",
    description = "Total request latency in nanoseconds",
    metadata = { unit = "nanoseconds" }
)]
pub static REQUEST_LATENCY: AtomicHistogram = AtomicHistogram::new(7, 64);

// TPOT — per phase
#[metric(name = "tpot", metadata = { unit = "nanoseconds", phase = "reasoning" })]
pub static TPOT_REASONING: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "tpot", metadata = { unit = "nanoseconds", phase = "content" })]
pub static TPOT_CONTENT: AtomicHistogram = AtomicHistogram::new(7, 64);

// Think duration — time from first reasoning token to first content token
#[metric(
    name = "think_duration",
    description = "Reasoning duration in nanoseconds",
    metadata = { unit = "nanoseconds" }
)]
pub static THINK_DURATION: AtomicHistogram = AtomicHistogram::new(7, 64);

// ITL histograms — reasoning phase (context-size bucketed)
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "small", phase = "reasoning" })]
pub static ITL_REASONING_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "medium", phase = "reasoning" })]
pub static ITL_REASONING_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "large", phase = "reasoning" })]
pub static ITL_REASONING_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "reasoning" })]
pub static ITL_REASONING_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "reasoning" })]
pub static ITL_REASONING_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

// ITL histograms — content phase (context-size bucketed)
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "small", phase = "content" })]
pub static ITL_CONTENT_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "medium", phase = "content" })]
pub static ITL_CONTENT_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "large", phase = "content" })]
pub static ITL_CONTENT_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "content" })]
pub static ITL_CONTENT_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "content" })]
pub static ITL_CONTENT_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

// All TTFT histograms for aggregation
pub static ALL_TTFT: [&AtomicHistogram; 10] = [
    &TTFT_REASONING_SMALL,
    &TTFT_REASONING_MEDIUM,
    &TTFT_REASONING_LARGE,
    &TTFT_REASONING_XLARGE,
    &TTFT_REASONING_XXLARGE,
    &TTFT_CONTENT_SMALL,
    &TTFT_CONTENT_MEDIUM,
    &TTFT_CONTENT_LARGE,
    &TTFT_CONTENT_XLARGE,
    &TTFT_CONTENT_XXLARGE,
];

// All ITL histograms for aggregation
pub static ALL_ITL: [&AtomicHistogram; 10] = [
    &ITL_REASONING_SMALL,
    &ITL_REASONING_MEDIUM,
    &ITL_REASONING_LARGE,
    &ITL_REASONING_XLARGE,
    &ITL_REASONING_XXLARGE,
    &ITL_CONTENT_SMALL,
    &ITL_CONTENT_MEDIUM,
    &ITL_CONTENT_LARGE,
    &ITL_CONTENT_XLARGE,
    &ITL_CONTENT_XXLARGE,
];

// All TPOT histograms for aggregation
pub static ALL_TPOT: [&AtomicHistogram; 2] = [&TPOT_REASONING, &TPOT_CONTENT];

pub struct Metrics;

impl Metrics {
    pub fn init() {
        // Metriken metrics are automatically registered via the #[metric] attribute
    }

    pub fn record_request_sent() {
        REQUESTS_SENT.increment();
        REQUESTS_INFLIGHT.increment();
    }

    pub fn record_request_complete(status: RequestStatus) {
        REQUESTS_INFLIGHT.decrement();
        match status {
            RequestStatus::Success => {
                REQUESTS_SUCCESS.increment();
            }
            RequestStatus::Failed(error_type) => {
                REQUESTS_FAILED.increment();
                match error_type {
                    ErrorType::Connection => ERRORS_CONNECTION.increment(),
                    ErrorType::Http4xx(_) => ERRORS_HTTP_4XX.increment(),
                    ErrorType::Http5xx(_) => ERRORS_HTTP_5XX.increment(),
                    ErrorType::Parse => ERRORS_PARSE.increment(),
                    ErrorType::Timeout => REQUESTS_TIMEOUT.increment(),
                    ErrorType::Other => ERRORS_OTHER.increment(),
                };
            }
            RequestStatus::Timeout => {
                REQUESTS_TIMEOUT.increment();
                ERRORS_OTHER.increment();
            }
        }
    }

    pub fn record_tokens(input: u64, output: u64) {
        TOKENS_INPUT.add(input);
        TOKENS_OUTPUT.add(output);
    }

    pub fn record_ttft(duration: Duration, input_tokens: u64, phase: Phase) {
        let nanos = duration.as_nanos() as u64;
        let histogram = match (phase, input_tokens) {
            (Phase::Reasoning, 0..=200) => &TTFT_REASONING_SMALL,
            (Phase::Reasoning, 201..=500) => &TTFT_REASONING_MEDIUM,
            (Phase::Reasoning, 501..=2000) => &TTFT_REASONING_LARGE,
            (Phase::Reasoning, 2001..=8000) => &TTFT_REASONING_XLARGE,
            (Phase::Reasoning, _) => &TTFT_REASONING_XXLARGE,
            (Phase::Content, 0..=200) => &TTFT_CONTENT_SMALL,
            (Phase::Content, 201..=500) => &TTFT_CONTENT_MEDIUM,
            (Phase::Content, 501..=2000) => &TTFT_CONTENT_LARGE,
            (Phase::Content, 2001..=8000) => &TTFT_CONTENT_XLARGE,
            (Phase::Content, _) => &TTFT_CONTENT_XXLARGE,
        };
        let _ = histogram.increment(nanos);
    }

    pub fn record_tpot(duration: Duration, phase: Phase) {
        let histogram = match phase {
            Phase::Reasoning => &TPOT_REASONING,
            Phase::Content => &TPOT_CONTENT,
        };
        let _ = histogram.increment(duration.as_nanos() as u64);
    }

    pub fn record_think_duration(duration: Duration) {
        let _ = THINK_DURATION.increment(duration.as_nanos() as u64);
    }

    pub fn record_itl(duration: Duration, input_tokens: u64, phase: Phase) {
        let nanos = duration.as_nanos() as u64;
        let histogram = match (phase, input_tokens) {
            (Phase::Reasoning, 0..=200) => &ITL_REASONING_SMALL,
            (Phase::Reasoning, 201..=500) => &ITL_REASONING_MEDIUM,
            (Phase::Reasoning, 501..=1000) => &ITL_REASONING_LARGE,
            (Phase::Reasoning, 1001..=2000) => &ITL_REASONING_XLARGE,
            (Phase::Reasoning, _) => &ITL_REASONING_XXLARGE,
            (Phase::Content, 0..=200) => &ITL_CONTENT_SMALL,
            (Phase::Content, 201..=500) => &ITL_CONTENT_MEDIUM,
            (Phase::Content, 501..=1000) => &ITL_CONTENT_LARGE,
            (Phase::Content, 1001..=2000) => &ITL_CONTENT_XLARGE,
            (Phase::Content, _) => &ITL_CONTENT_XXLARGE,
        };
        let _ = histogram.increment(nanos);
    }

    pub fn record_latency(duration: Duration) {
        let _ = REQUEST_LATENCY.increment(duration.as_nanos() as u64);
    }

    pub fn record_retry() {
        REQUESTS_RETRIED.increment();
    }
}
