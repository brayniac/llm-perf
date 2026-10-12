# Recording llm-perf and the server under test into one `.dendro`

**Status:** OPEN. Plan opened 2026-10-08; steps 3 and 4 built 2026-10-11
(branches `deps/metriken-0.11` and `feat/metrics-stream`, see "As built").

Related entries opened the same day: rezolus
`docs/journal/2026-10-08-6-0-release-readiness.md`, metriken
`docs/journal/2026-10-08-one-recording-stack.md` (one recording, exposition
and viewing stack for every metriken producer), systemslab
`docs/journal/2026-10-08-dendro-artifacts.md`, cachecannon
`docs/journal/2026-10-08-dendro-recording.md`.

## Today

- `metriken` 0.8.0 and `metriken-exposition` 0.14.0 from a git dependency
  (rev `98db358` in `Cargo.lock`).
- `[metrics] output`: `src/snapshot.rs` appends msgpack snapshots to a
  temporary file and converts them to parquet at exit through
  `MsgpackToParquet`.
- The optional admin server's `/metrics` (`src/admin.rs`) renders histograms
  as percentile gauges (`PrometheusOptions::with_percentiles`).
- `src/server_metrics.rs` runs `rezolus record <url> <output> --interval <i>`
  to record the server under test's Prometheus `/metrics` into a second
  parquet, `<output-stem>.server.parquet` by default (`src/config.rs`). Its
  comment explains why it avoids `--endpoint url,source=...`: the annotated
  form collided with the `[OUTPUT]` positional.
- Downstream, rezolus ships an `llm-perf.json` KPI template and recognizes
  `source=llm-perf`; the client and server recordings are separate files,
  joined with `rezolus recording combine`.

## Plan

1. **One recorder for both sides.** Change `record_args` to
   `rezolus record --endpoint <server>/metrics,source=<service> --endpoint <llm-perf admin>/metrics,source=llm-perf -o <run>.dendro --interval <i>`.
   `-o` removes the positional collision noted in `src/server_metrics.rs`.
   The run then leaves one archive holding the server and llm-perf on one
   timeline, readable while the run goes and after a crash. This needs rezolus
   6.0's `record`. v5.25.1's `record` help text says a Prometheus endpoint
   records into a `.dendro` too; that is not run.
2. **Histograms as buckets on `/metrics`.** Expose histograms as Prometheus
   `_bucket` series rather than percentile gauges, so a scrape keeps a
   distribution that rezolus's Prometheus conversion turns back into a
   histogram, approximately. The percentile gauges are kept if anything reads
   them. `src/admin.rs` builds its output with
   `PrometheusOptions::with_percentiles`; whether metriken-exposition 0.14 can
   emit `_bucket` series without a metriken change is not checked.
3. **Leave the git dependency.** Move to published metriken 0.11 and
   metriken-exposition 0.21. `metriken-core` declares `links`, so a build holds
   one metriken-core version: llm-perf's metriken 0.8 is on core 0.1, and the
   stream route and recording its own registry (both through
   `metriken-exposition`) need metriken 0.11, on core 0.3.
4. **Serve `/metrics/stream`** through `metriken-exposition`'s stream route
   for a registry (metriken entry, piece 1), and pass llm-perf's base URL in
   `record_args` rather than its `/metrics` path, so the recorder can find the
   stream. Once rezolus detects a source by its stream (rezolus entry,
   "Endpoint detection"), the recorder records llm-perf as it records a
   Rezolus agent: rows stamped by llm-perf at read time and native histograms,
   with no Prometheus conversion. Step 2's buckets then only matter to other
   Prometheus consumers.
5. **Record in process with `metriken-recorder`** (metriken entry, piece 2):
   llm-perf records the server under test and its own registry into one
   archive itself, in place of running `rezolus record`, and no longer needs a
   rezolus binary. This replaces step 1's launcher.
6. **The template moves here.** llm-perf's dashboard template
   (`llm-perf.json`, today in rezolus's `crates/dashboard/templates/`) is
   owned by llm-perf and written into the archive, in the stream's handshake
   and in its own source's metadata. The viewer loads it from the archive
   (metriken entry, piece 3).
7. **Retire the parquet-at-exit path** once step 5 is in use, keeping
   `[metrics] output` as an option for one release.

Step 1 needs no metriken change and can ship first; step 2 may need one (see
above). Step 4 waits for the metriken entry's path step 3 (the stream route),
step 5 for path step 4 (`metriken-recorder`), and step 6 for path step 5
(templates).

## GO criteria

- One llm-perf run leaves one `.dendro` with the server and llm-perf as two
  sources, and `rezolus view <run>.dendro` shows the llm-perf KPI dashboard for
  the llm-perf recording without `recording combine`.
- After step 4, TTFT and inter-token latency in that archive are histograms,
  with the same percentiles as the parquet path for the same run within the
  histogram's bucket error.

## As built

### Step 3: published metriken (branch `deps/metriken-0.11`)

- `Cargo.toml`: `metriken = "0.11.3"`, `metriken-exposition = "0.22.0"`
  (features `json`, `msgpack`) and `metriken-storage = "0.2.0"` (no default
  features, `convert`). The plan named metriken-exposition 0.21; 0.22 is the
  first release with the stream route (`stream::RegistryPasses`) that step 4
  uses. The msgpack-to-parquet conversion (`MsgpackToParquet`,
  `ParquetOptions`, `ParquetHistogramType`) moved to `metriken_storage::convert`;
  `src/snapshot.rs` imports it from there.
- histogram resolves to 1.6.0 (was 1.1.0); `quantiles` is inherent there, so
  five `use histogram::SampleQuantiles` imports became unused and were removed,
  and the requirement became `"1.6"`.
- metriken-storage adds arrow/parquet 58 beside llm-perf's own 56, and lru
  0.12.5, which `cargo audit` reports as unsound (RUSTSEC-2026-0002,
  RUSTSEC-2026-0253); audit warns and exits 0.
- metriken 0.11 reads a never-written `CounterGroup` slot as absent. Measured
  against a local mock server (8 s, 60 QPS, `max_outstanding_requests = 4`,
  `[metrics] interval = "1s"`) before `Metrics::init` wrote any slot: metriken
  0.8's parquet had 42 columns, 0.11's 39, the missing three being `requests`
  with status `error`, `retried` and `timeout`; `/metrics` lost the same
  series plus `canceled` while no request had been canceled yet. Downstream,
  rezolus's `llm-perf.json` error-rate panel would show no data for a clean
  run. `Metrics::init` now writes every slot of `requests` and `tokens` (the
  two counter groups that template queries) with `add(idx, 0)`, which marks a
  slot written in metriken 0.11.3 without changing a count already there.
  Rerun the same way against main (e626b9e): the parquet has 42 columns on
  main and 43 here, and `/metrics` 136 series on main and 137 here; in both
  the only addition is `requests{status="dropped"}` (below), and `requests`
  error, timeout, canceled and retried read 0 as on main. The other counter
  groups (`errors`, `conversations`, `cache`, `replay_reuse`,
  `replay_sessions`) keep metriken 0.11's absent-until-written behaviour;
  `errors`, `cache` and `conversations` had no slot written on that run and
  were absent under 0.8 too. `/vars` keys and the JSON report's keys are
  unchanged.
- Separately fixed: `Metrics::init` labeled 6 of the `requests` group's 7
  slots, so `dropped` (slot 6) was never exposed; the snapshot and exposition
  walk labeled slots only. With the label, the same run adds a
  `requests{status="dropped"}` series and column (124 on `/metrics` at 7 s). A
  test now checks every slot of every group has labels.

### Step 4: `/metrics/stream` (branch `feat/metrics-stream`)

- `src/admin.rs` serves `GET /metrics/stream` through metriken-exposition's
  stream route (features `stream`, `stream-tokio`): one shared
  `RegistryPasses<DefaultRouter>` over `GroupBuilder::new(DefaultRouter::new("llm-perf"))`
  with a 10 ms floor, `StreamRequest::parse` on the raw query, a 400 for an
  interval below the floor (cachecannon's refusal message), and
  `StreamSubscription::into_stream` as the warp body with the subscription's
  headers set on the response. Handshake labels `source=llm-perf`, metadata
  `version`. Every metric lands in one windowless group, `llm-perf/main`.
- It also serves `GET /metrics/binary`, a msgpack `Snapshot` from a
  `Snapshotter` (metadata `source`, `version`). rezolus's recorder classifies a
  base URL by probing `/metrics/binary` first (`probe_endpoint`, rezolus
  `src/recorder/mod.rs`), and only a source classified as msgpack is streamed
  into a `.dendro`; without the route llm-perf is scraped as Prometheus.
- `record_args` (`src/server_metrics.rs`): when `server_metrics_output` ends in
  `.dendro`, it builds `rezolus record --endpoint <server_metrics_url>
  [--endpoint http://<admin>,source=llm-perf] -o <output> --interval <i>`. The
  admin endpoint is added only when `[admin]` is enabled; a listen address on
  `0.0.0.0`/`::` becomes loopback. Any other extension keeps the positional
  form, so step 1's two-endpoint archive is opt-in by extension and the
  default output stays `<stem>.server.parquet`. The server endpoint gets no
  `source=`: there is no config key naming the service, and rezolus infers one
  from host and port (with a warning, as on the positional path). The
  extension match is exact (`dendro`), as rezolus's `Format` match is.
- Version gate (`admin_endpoint_support`, `ServerMetricsRecorder::spawn`):
  for a `.dendro` with `[admin]` enabled, llm-perf runs `<rezolus_bin>
  --version` and adds its own endpoint only for 6.0.0-alpha.45 or later
  (alpha.45 is the version ab24b072 set, the commit that made `record` accept
  the metriken stream's content type), or for 5.23.0 or a later 5.x. 5.23.0
  is the first 5.x tag whose recorder has a `Format::Dendro`; before it a
  `.dendro` output is written as parquet (the reviewer saw this with
  5.21.1-alpha.1). `src/recorder/` differs by one line between v5.23.0 and
  v5.26.0, so 5.24 and 5.25 are taken to scrape `/metrics/binary` into the
  archive as 5.23.0 and 5.26.0 do; 5.23.0 and 5.26.0 were run end to end (a
  reviewer ran v5.23.0: the archive held the server scrape and
  `source=llm-perf` as an `unattributed` table, exit 0). Anything else (a
  6.0 alpha before alpha.45, older than 5.23.0, unparseable, or a
  `--version` still running after 5 s, which is then killed) logs a
  warning with the version or the timeout and the reason, and the server
  under test is recorded alone. No
  `semver` dependency: the parser handles `X.Y.Z[-pre][+build]`, with `beta`
  and `rc` pre-releases of 6.0.0 accepted.
- `ServerMetricsRecorder::finish` returns the failure when rezolus exits
  non-zero, is killed 30 s after SIGINT, or cannot be waited on; it is logged
  at error and printed as a `Warning:` line in the console report and the
  brief summary (not in the JSON report).
- `[log]`: at `error`, `warn` and `info`, `LogConfig::env_filter` adds
  `warp::server=off` before the configured `filter` directives, which can
  replace it. warp's server module logs at debug and error only, and its
  error is `server connection error: hyper::Error(IncompleteMessage)` when
  rezolus closes `/metrics/stream` at the end of a run. The filter also
  hides warp's other connection errors and its accept errors (`accept
  error`, for example on running out of file descriptors; warp 0.4.3
  `server.rs:477`). A bind failure still shows, since warp panics with
  `failed to bind to address` rather than logging it. An EnvFilter cannot
  lower an event's level, so "debug or more verbose" is approximated by
  showing these events only when `level` is `debug` or `trace`.
- Tests (`admin::tests`): the stream over a socket (content type, the three
  `x-metriken-*` headers, handshake `source=llm-perf`, three rows frames with
  advancing `seq`, the first carrying a `ttft` member labeled
  `context_size=small` with count >= 1, decoded with dendro 0.4's
  `FrameReader` and metriken-model's `decode_wal_group_row`); refusal of `0s`
  and `1ms`; acceptance at `10ms`; defaults with no query string;
  `/metrics/binary` decoding as a `Snapshot` with that member; the old routes
  still answering. The stream test was seen red with the pass fetch returning
  `None` and with `record_ttft` not recording.
- `metrics_stream_long_layout_serves_ttft_rows`: `layout=long`, as rezolus's
  recorder asks; the header says `long`, and the first rows frame carries
  exactly one stream, `llm-perf/main`, which decodes as a wide
  `WalGroupRow` with a `ttft` member. The mutation `passes.pass(layout, now)`
  to `passes.pass(Layout::Wide, now)` does not turn it red, and cannot: a
  group is long only when every metric in it is a counter, gauge or histogram
  group (`GroupBuilder::build_stream`), and `llm-perf/main` also holds plain
  counters, gauges and histograms, so both layouts send the same row. What
  catches the mutation is the unused `layout` binding under clippy
  `-D warnings`. The `llm-perf/main/occupants` table in an archive recorded
  by rezolus ab24b072 is written by rezolus's dendro writer from that wide
  row, not sent by llm-perf.
- Server-metrics tests: the version decision table (alpha.44 no, alpha.45
  yes, 6.0.0 yes, 5.26.0 yes, 5.22.1 and 5.23.0-alpha.1 no, unparseable no),
  a stub rezolus that prints a version and records its arguments (alpha.45
  gets the admin endpoint, alpha.44 does not), and a non-zero exit reported
  by `finish`. Each was seen red: the probe result ignored, the alpha floor at
  44, and the success check removed. The existing SIGINT stub test raced (a
  SIGINT before the stub's `trap` killed the shell, seen once in the full
  suite after `finish` began reporting the exit); the stubs write a ready file
  after installing the trap and the test waits for it.
- Frame size on the real registry, measured during a 20 s run against a local
  mock (8 workers, all five TTFT context sizes written): opening 190 bytes,
  then 193,318-196,301 bytes per frame at 1 s, identical for `wide` and `long`
  (`DefaultRouter` makes no slotted groups, so the long layout has nothing to
  send long). A frame carries every histogram member's full bucket array
  (grouping power 7, 64-bit range: 7,424 buckets), so this is about 190 KB/s on
  the wire per subscriber, independent of load.
- GO check against rezolus `upstream/main` (ab24b072), on macOS: a 60 s run
  against the mock with `[admin]` and `[metrics] output = run.parquet`,
  recorded by `rezolus record --endpoint http://127.0.0.1:19091,source=llm-perf
  -o run.dendro --interval 1s --duration 66s`. `rezolus mcp describe-metrics
  run.dendro` lists `ttft`, `ttft_content` (`context_size`), `itl`
  (`context_size`, `phase`), `tpot` (`phase`), `request_latency`,
  `schedule_slip` and `ttft_by_cache` as histograms. For every member of
  `ttft`, `ttft_content`, `itl`, `tpot` and `request_latency`, the count, p50
  bucket and p99 bucket of the last sample are identical in `run.dendro` and
  `run.parquet` (both last samples at 01:40:28.001, 0.15 ms apart, after the
  last of the 4,000 requests completed): for example `ttft{context_size=small}`
  count 3,184, p50 [22806528, 22937599] ns, p99 [38273024, 38535167] ns;
  `request_latency` count 4,000, p99 [176160768, 177209343] ns. The p99 bucket
  ends match the JSON report's `context_latency.small.ttft_p99_ms` (38.535167)
  and `request_p99_ms` (177.209343). `run.dendro` was 458,752 bytes for 60 rows
  (about 450 KB per minute), beside a 57,588-byte parquet.
- A 15 s run with `server_metrics_url` pointing at the mock's `/metrics`,
  `server_metrics_output = "l2.dendro"` and `[admin]` enabled left one archive
  with two recordings, `source=127.0.0.1-18080` (`prometheus/scrape`, 15 rows)
  and `source=llm-perf` (`llm-perf/main`, 16 rows).
- Gate end to end (8 s runs against the mock, `[admin]` enabled,
  `server_metrics_output = "<n>.dendro"`): with rezolus ab24b072
  (6.0.0-alpha.45) the archive has `source=127.0.0.1-38080` and
  `source=llm-perf` (`llm-perf/main` with `ttft`, `itl`, `tpot`,
  `request_latency`); with 6.0.0-alpha.44 (9f59f9e8) llm-perf logs the
  warning and the archive has the server recording only; with 5.26.0 both
  recordings, llm-perf's as an `unattributed` table. A reviewer ran v5.23.0
  the same way: the archive held the server scrape and `source=llm-perf` as
  an `unattributed` table, and rezolus exited 0. None of the three logged
  the `warp::server` error. With the output in a missing directory, rezolus
  exited 1 and the summary printed `Warning: rezolus record exited with exit
  status: 1; ...`.
- Not checked: `rezolus view` showing the llm-perf dashboard for the llm-perf
  recording of that archive (the first GO criterion's second half).
