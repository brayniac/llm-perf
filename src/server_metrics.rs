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
/// `/metrics` URL would be scraped as Prometheus text. It needs rezolus 6.0,
/// whose recorder streams metriken sources into a `.dendro`. Without the admin
/// server only the server under test is recorded.
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

fn is_dendro(output: &str) -> bool {
    std::path::Path::new(output)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("dendro"))
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
        let args = record_args(mc, admin);
        if admin.is_none() && is_dendro(&mc.resolved_server_output().to_string_lossy()) {
            log::info!(
                "[admin] is not enabled, so {:?} records the server under test only",
                mc.resolved_server_output()
            );
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

    /// Stop the recorder cleanly: one SIGINT (rezolus finalizes its parquet on the
    /// first SIGINT), then wait up to 30s; SIGKILL as a backstop.
    pub async fn finish(mut self) {
        match self.child.id() {
            Some(pid) => {
                // SAFETY: pid is a live child PID from tokio; SIGINT is well-defined.
                unsafe {
                    libc::kill(pid as libc::pid_t, libc::SIGINT);
                }
                match tokio::time::timeout(Duration::from_secs(30), self.child.wait()).await {
                    Ok(Ok(status)) => log::info!("rezolus record finished ({status})"),
                    Ok(Err(e)) => log::warn!("rezolus record wait failed: {e}"),
                    Err(_) => {
                        log::warn!(
                            "rezolus record didn't exit 30s after SIGINT; killing \
                             (server parquet may be incomplete)"
                        );
                        let _ = self.child.kill().await;
                    }
                }
            }
            None => log::warn!("rezolus record already exited before finish()"),
        }
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
        std::fs::write(
            &stub,
            "#!/bin/sh\nout=\"$3\"\n\
             trap 'echo done > \"$out\"; exit 0' INT\n\
             while true; do sleep 0.05; done\n",
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
        tokio::time::sleep(Duration::from_millis(200)).await;
        rec.finish().await;

        assert!(
            out.exists(),
            "stub should have written its output on SIGINT"
        );
        assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), "done");
    }
}
