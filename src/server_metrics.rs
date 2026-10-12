//! Launches `rezolus record` as a child process to capture the server-under-test's
//! Prometheus /metrics during a benchmark, writing a parquet time-aligned with the
//! client metrics. A thin launcher — rezolus does the scraping/parsing/parquet.
//!
//! When the output is a `.dendro` and the admin server is enabled, the same
//! `rezolus record` also records llm-perf itself from the admin server's
//! `/metrics/stream`, so one archive holds both sources on one timeline.

use crate::config::MetricsConfig;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::process::{Child, Command};

/// The `source` label llm-perf's own recording carries, the same as the
/// stream handshake's (`admin::STREAM_NAMESPACE`).
const LLM_PERF_SOURCE: &str = "llm-perf";

/// Build the `rezolus record` argument vector (after the binary name).
/// `mc.server_metrics_url` MUST be Some (callers gate on it). `admin` is the
/// admin server's listen address when it is enabled.
///
/// Output not ending in `.dendro`: `rezolus record <URL> <OUTPUT> --interval
/// <i>`, the server only. A plain positional URL: rezolus auto-detects the
/// Prometheus protocol via its startup probe (verified against a live ferallm
/// /metrics). The positional `[URL]` is taken literally, so it cannot carry
/// `,source=`.
///
/// Output ending in `.dendro`: `rezolus record --endpoint <URL> [--endpoint
/// http://<admin>,source=llm-perf] -o <OUTPUT> --interval <i>`. `-o` names
/// the output, so the annotated `--endpoint` form no longer collides with the
/// `[OUTPUT]` positional. The admin endpoint is the base URL, not its
/// `/metrics`: given a base URL rezolus probes `/metrics/binary`, classifies
/// llm-perf as a metriken source and subscribes to `/metrics/stream`, where a
/// `/metrics` URL would be scraped as Prometheus text. rezolus streams it at
/// or after 6.0.0-alpha.45 (until 6.0.0 is released, a build of rezolus main
/// at or after ab24b072); rezolus 5.23 to 5.26 scrape `/metrics/binary` into
/// the `.dendro` instead; 6.0 alphas before alpha.45 refuse the stream.
/// `ServerMetricsRecorder::spawn` checks `rezolus --version` and passes
/// `admin` only for a version that records it ([`admin_endpoint_support`]).
/// Without the admin server only the server under test is recorded.
pub fn record_args(mc: &MetricsConfig, admin: Option<SocketAddr>) -> Vec<String> {
    let url = mc.server_metrics_url.clone().unwrap_or_default();
    let output = mc.resolved_server_output();
    let interval = mc.interval.clone();
    let output = output.to_string_lossy().into_owned();
    if !is_dendro(&output) {
        return vec![
            "record".to_string(),
            url,
            output,
            "--interval".to_string(),
            interval,
        ];
    }
    let mut args = vec!["record".to_string(), "--endpoint".to_string(), url];
    if let Some(admin) = admin {
        args.push("--endpoint".to_string());
        args.push(format!(
            "{},source={LLM_PERF_SOURCE}",
            admin_base_url(admin)
        ));
    }
    args.extend(["-o".to_string(), output, "--interval".to_string(), interval]);
    args
}

/// rezolus picks the format from the extension by exact match (`dendro`), so
/// `run.DENDRO` is not a dendro output there and is not one here.
fn is_dendro(output: &str) -> bool {
    std::path::Path::new(output)
        .extension()
        .is_some_and(|ext| ext == "dendro")
}

/// The first 6.0 pre-release whose `record` accepts the metriken stream's
/// content type (rezolus ab24b072).
const STREAM_ALPHA: u64 = 45;

/// The first 5.x release whose `record` writes a dendro archive for a
/// `.dendro` output. Before it, `record -o run.dendro` writes parquet under
/// that name.
const DENDRO_5X: (u64, u64) = (23, 0);

/// Whether the rezolus that printed `version_output` (`rezolus --version`)
/// records llm-perf's admin endpoint into a `.dendro`. `Err` carries the
/// reason it does not, for the warning.
///
/// Yes for 6.0.0-alpha.45 and later pre-releases of 6.0.0 (`beta`, `rc`),
/// 6.0.0 and anything after it, which stream `/metrics/stream`, and for
/// 5.23.0 and later 5.x, which scrape `/metrics/binary` into the archive.
/// No for 6.0 alphas before alpha.45, which refuse the stream's content type,
/// for anything older than 5.23.0, and for output that does not parse.
pub fn admin_endpoint_support(version_output: &str) -> Result<(), String> {
    let unparseable = || format!("cannot parse a version from {:?}", version_output.trim());
    let token = version_output
        .split_whitespace()
        .find(|t| t.starts_with(|c: char| c.is_ascii_digit()))
        .ok_or_else(unparseable)?;
    let token = token.split('+').next().unwrap_or(token);
    let (core, pre) = match token.split_once('-') {
        Some((core, pre)) => (core, Some(pre)),
        None => (token, None),
    };
    let parts: Vec<u64> = core
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| unparseable())?;
    let [major, minor, patch] = parts[..] else {
        return Err(unparseable());
    };
    if (major, minor, patch) > (6, 0, 0) {
        return Ok(());
    }
    if (major, minor, patch) == (6, 0, 0) {
        let Some(pre) = pre else { return Ok(()) };
        if pre.starts_with("beta") || pre.starts_with("rc") {
            return Ok(());
        }
        let n: u64 = pre
            .strip_prefix("alpha.")
            .and_then(|n| n.parse().ok())
            .ok_or_else(unparseable)?;
        return if n >= STREAM_ALPHA {
            Ok(())
        } else {
            Err(format!(
                "6.0 alphas before alpha.{STREAM_ALPHA} refuse the metriken stream's content type"
            ))
        };
    }
    if major == 5 && ((minor, patch) > DENDRO_5X || ((minor, patch) == DENDRO_5X && pre.is_none()))
    {
        return Ok(());
    }
    Err(format!(
        "rezolus before {}.{}.{} writes parquet to a path ending in .dendro",
        5, DENDRO_5X.0, DENDRO_5X.1
    ))
}

/// Run `<bin> --version` and decide whether it records the admin endpoint.
fn probe_admin_support(bin: &str) -> Result<(), String> {
    let output = std::process::Command::new(bin)
        .arg("--version")
        .output()
        .map_err(|e| format!("could not run `{bin} --version`: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    admin_endpoint_support(&stdout)
        .map_err(|why| format!("`{bin} --version` printed {:?}: {why}", stdout.trim()))
}

/// The URL a local recorder reaches the admin server at. A listen address
/// on all interfaces (`0.0.0.0`, `::`) is reached on loopback.
fn admin_base_url(admin: SocketAddr) -> String {
    let ip = match admin.ip() {
        std::net::IpAddr::V4(ip) if ip.is_unspecified() => std::net::Ipv4Addr::LOCALHOST.into(),
        std::net::IpAddr::V6(ip) if ip.is_unspecified() => std::net::Ipv6Addr::LOCALHOST.into(),
        ip => ip,
    };
    format!("http://{}", SocketAddr::new(ip, admin.port()))
}

/// A running `rezolus record` child.
pub struct ServerMetricsRecorder {
    child: Child,
}

impl ServerMetricsRecorder {
    /// Spawn `rezolus record` for the configured server endpoint. Returns `None`
    /// (with a warning) if the binary can't be started, so the benchmark proceeds
    /// without server metrics.
    pub fn spawn(mc: &MetricsConfig, admin: Option<SocketAddr>) -> Option<Self> {
        let bin = mc
            .rezolus_bin
            .clone()
            .unwrap_or_else(|| "rezolus".to_string());
        let output = mc.resolved_server_output();
        let dendro = is_dendro(&output.to_string_lossy());
        let admin = match admin {
            Some(admin) if dendro => match probe_admin_support(&bin) {
                Ok(()) => Some(admin),
                Err(why) => {
                    log::warn!(
                        "{why}; {output:?} records the server under test only, not llm-perf"
                    );
                    None
                }
            },
            admin => admin,
        };
        let args = record_args(mc, admin);
        if admin.is_none() && dendro {
            log::info!("{output:?} records the server under test only");
        }
        match Command::new(&bin).args(&args).spawn() {
            Ok(child) => {
                log::info!(
                    "started `{bin} record` for server metrics -> {:?}",
                    mc.resolved_server_output()
                );
                Some(Self { child })
            }
            Err(e) => {
                log::warn!(
                    "could not start `{bin} record` for server metrics: {e} \
                     (continuing without server metrics)"
                );
                None
            }
        }
    }

    /// Stop the recorder cleanly: one SIGINT (rezolus finalizes its output on
    /// the first SIGINT), then wait up to 30s; SIGKILL as a backstop.
    ///
    /// Returns a description of the failure when rezolus did not exit 0 (or
    /// had to be killed), for the benchmark summary; it is also logged.
    pub async fn finish(mut self) -> Option<String> {
        let failure = match self.child.id() {
            Some(pid) => {
                // SAFETY: pid is a live child PID from tokio; SIGINT is well-defined.
                unsafe {
                    libc::kill(pid as libc::pid_t, libc::SIGINT);
                }
                match tokio::time::timeout(Duration::from_secs(30), self.child.wait()).await {
                    Ok(Ok(status)) if status.success() => {
                        log::info!("rezolus record finished ({status})");
                        None
                    }
                    Ok(Ok(status)) => Some(format!(
                        "rezolus record exited with {status}; the server metrics \
                         recording may be missing or incomplete"
                    )),
                    Ok(Err(e)) => Some(format!("rezolus record wait failed: {e}")),
                    Err(_) => {
                        let _ = self.child.kill().await;
                        Some(
                            "rezolus record didn't exit 30s after SIGINT and was killed; \
                             the server metrics recording may be incomplete"
                                .to_string(),
                        )
                    }
                }
            }
            None => Some("rezolus record already exited before finish()".to_string()),
        };
        if let Some(failure) = &failure {
            log::error!("{failure}");
        }
        failure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mc(url: &str, out: Option<&str>) -> MetricsConfig {
        MetricsConfig {
            output: PathBuf::from("run.parquet"),
            interval: "1s".to_string(),
            batch_size: None,
            server_metrics_url: Some(url.to_string()),
            server_metrics_output: out.map(PathBuf::from),
            rezolus_bin: None,
        }
    }

    #[test]
    fn record_args_builds_positional_url_output_then_interval() {
        // rezolus record <URL> <OUTPUT> --interval <i> (plain URL; protocol auto-detected)
        let args = record_args(&mc("http://localhost:4242/metrics", None), None);
        assert_eq!(args[0], "record");
        assert_eq!(args[1], "http://localhost:4242/metrics");
        assert_eq!(args[2], "run.server.parquet"); // derived default output (positional)
        assert_eq!(args[3], "--interval");
        assert_eq!(args[4], "1s");
    }

    #[test]
    fn record_args_uses_explicit_output_as_second_positional() {
        let args = record_args(&mc("http://h/metrics", Some("srv.parquet")), None);
        assert_eq!(args[2], "srv.parquet");
    }

    #[test]
    fn record_args_for_parquet_ignore_the_admin_server() {
        let admin = "127.0.0.1:9090".parse().unwrap();
        let args = record_args(&mc("http://h/metrics", None), Some(admin));
        assert_eq!(
            args,
            [
                "record",
                "http://h/metrics",
                "run.server.parquet",
                "--interval",
                "1s"
            ]
        );
    }

    #[test]
    fn record_args_for_dendro_add_llm_perf_from_the_admin_server() {
        let admin = "127.0.0.1:9090".parse().unwrap();
        let args = record_args(&mc("http://h/metrics", Some("run.dendro")), Some(admin));
        assert_eq!(
            args,
            [
                "record",
                "--endpoint",
                "http://h/metrics",
                "--endpoint",
                "http://127.0.0.1:9090,source=llm-perf",
                "-o",
                "run.dendro",
                "--interval",
                "1s",
            ]
        );
    }

    #[test]
    fn record_args_for_dendro_without_admin_record_the_server_only() {
        let args = record_args(&mc("http://h/metrics", Some("run.dendro")), None);
        assert_eq!(
            args,
            [
                "record",
                "--endpoint",
                "http://h/metrics",
                "-o",
                "run.dendro",
                "--interval",
                "1s"
            ]
        );
    }

    #[test]
    fn admin_on_all_interfaces_is_reached_on_loopback() {
        assert_eq!(
            admin_base_url("0.0.0.0:9090".parse().unwrap()),
            "http://127.0.0.1:9090"
        );
        assert_eq!(
            admin_base_url("[::]:9090".parse().unwrap()),
            "http://[::1]:9090"
        );
        assert_eq!(
            admin_base_url("10.0.0.5:9090".parse().unwrap()),
            "http://10.0.0.5:9090"
        );
    }

    #[tokio::test]
    async fn spawn_and_finish_signals_child_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let stub = dir.path().join("fake_rezolus.sh");
        // Stub invoked as `record <endpoint> <output> --interval <i>`: the OUTPUT
        // positional is $3. Trap SIGINT, write it (mimics rezolus finalizing), exit.
        // `ready` is written after the trap is installed: a SIGINT before that
        // kills the shell, which under a loaded test run took over 200 ms.
        let ready = dir.path().join("ready");
        std::fs::write(
            &stub,
            format!(
                "#!/bin/sh\nout=\"$3\"\n\
                 trap 'echo done > \"$out\"; exit 0' INT\n\
                 touch '{}'\n\
                 while true; do sleep 0.05; done\n",
                ready.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let out = dir.path().join("run.server.parquet");
        let mut config = mc("http://127.0.0.1:9/metrics", Some(out.to_str().unwrap()));
        config.rezolus_bin = Some(stub.to_string_lossy().into_owned());

        let rec = ServerMetricsRecorder::spawn(&config, None).expect("spawn stub");
        wait_for(&ready).await;
        assert_eq!(rec.finish().await, None);

        assert!(
            out.exists(),
            "stub should have written its output on SIGINT"
        );
        assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), "done");
    }

    #[test]
    fn admin_endpoint_support_decision_table() {
        let yes = [
            "rezolus 6.0.0-alpha.45",
            "rezolus 6.0.0-alpha.46\n",
            "rezolus 6.0.0-beta.1",
            "rezolus 6.0.0-rc.2",
            "rezolus 6.0.0",
            "rezolus 6.0.1",
            "rezolus 6.1.0-alpha.1",
            "rezolus 7.0.0",
            "rezolus 5.23.0",
            "rezolus 5.26.0",
            "rezolus 5.26.0+mmap.2",
            "rezolus 5.27.0-alpha.1",
        ];
        for v in yes {
            assert_eq!(admin_endpoint_support(v), Ok(()), "{v:?}");
        }
        let no = [
            ("rezolus 6.0.0-alpha.44", "alpha.45"),
            ("rezolus 6.0.0-alpha.1", "alpha.45"),
            ("rezolus 5.22.1", "5.23.0"),
            ("rezolus 5.23.0-alpha.1", "5.23.0"),
            ("rezolus 5.21.1-alpha.1", "5.23.0"),
            ("rezolus 4.1.2", "5.23.0"),
            ("", "cannot parse"),
            ("rezolus", "cannot parse"),
            ("rezolus six", "cannot parse"),
            ("rezolus 6.0", "cannot parse"),
            ("rezolus 6.0.0-dev", "cannot parse"),
        ];
        for (v, why) in no {
            let err = admin_endpoint_support(v).expect_err(v);
            assert!(err.contains(why), "{v:?}: {err}");
        }
    }

    #[test]
    fn is_dendro_matches_the_extension_exactly() {
        assert!(is_dendro("run.dendro"));
        assert!(is_dendro("/a/b/run.dendro"));
        assert!(!is_dendro("run.DENDRO"));
        assert!(!is_dendro("run.Dendro"));
        assert!(!is_dendro("run.dendro.parquet"));
        assert!(!is_dendro("dendro"));
    }

    async fn wait_for(path: &std::path::Path) {
        for _ in 0..500 {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("{} never appeared", path.display());
    }

    /// A stub `rezolus` that prints `version` for `--version` and otherwise
    /// installs a SIGINT trap that exits `code`, then writes its arguments to
    /// `<dir>/args`, one per line.
    fn version_stub(dir: &std::path::Path, version: &str, code: u8) -> String {
        let stub = dir.join("fake_rezolus.sh");
        let args = dir.join("args");
        std::fs::write(
            &stub,
            format!(
                "#!/bin/sh\n\
                 if [ \"$1\" = --version ]; then echo 'rezolus {version}'; exit 0; fi\n\
                 trap 'exit {code}' INT\n\
                 printf '%s\\n' \"$@\" > '{}.tmp' && mv '{0}.tmp' '{0}'\n\
                 while true; do sleep 0.05; done\n",
                args.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        stub.to_string_lossy().into_owned()
    }

    async fn run_stub(version: &str, code: u8) -> (Vec<String>, Option<String>) {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("run.dendro");
        let mut config = mc("http://127.0.0.1:9/metrics", Some(out.to_str().unwrap()));
        config.rezolus_bin = Some(version_stub(dir.path(), version, code));
        let admin = "127.0.0.1:9090".parse().unwrap();
        let rec = ServerMetricsRecorder::spawn(&config, Some(admin)).expect("spawn stub");
        let args_file = dir.path().join("args");
        wait_for(&args_file).await;
        let failure = rec.finish().await;
        let args = std::fs::read_to_string(&args_file)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        (args, failure)
    }

    #[tokio::test]
    async fn spawn_adds_the_admin_endpoint_for_a_rezolus_that_streams_it() {
        let (args, failure) = run_stub("6.0.0-alpha.45", 0).await;
        assert!(
            args.iter()
                .any(|a| a == "http://127.0.0.1:9090,source=llm-perf"),
            "{args:?}"
        );
        assert_eq!(failure, None);
    }

    #[tokio::test]
    async fn spawn_records_the_server_only_for_a_rezolus_that_refuses_the_stream() {
        let (args, _) = run_stub("6.0.0-alpha.44", 0).await;
        assert!(
            !args.iter().any(|a| a.contains("source=llm-perf")),
            "{args:?}"
        );
        assert_eq!(
            args[..3],
            ["record", "--endpoint", "http://127.0.0.1:9/metrics"]
        );
    }

    #[tokio::test]
    async fn finish_reports_a_non_zero_exit() {
        let (_, failure) = run_stub("6.0.0-alpha.45", 3).await;
        let failure = failure.expect("a non-zero exit is reported");
        assert!(failure.contains("exited with"), "{failure}");
    }
}
