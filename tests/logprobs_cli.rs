//! `llm-perf logprobs` run as a process against a mock server: the exit status
//! and the records written in each `[logprobs]` mode.

use std::path::Path;
use std::process::Command;

const FIXTURE: &str =
    include_str!("fixtures/llama-server/qwen3.5-0.8b-q2_k/completions-logprobs.jsonl");

fn captured_response(case: &str) -> String {
    FIXTURE
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .find(|v| v["case"] == case)
        .unwrap()["response"]
        .to_string()
}

/// Write a config for `mode` against `url` with a two-prompt corpus and
/// return its path.
fn write_config(dir: &Path, url: &str, mode: &str) -> std::path::PathBuf {
    std::fs::write(
        dir.join("corpus.jsonl"),
        "{\"prompt\":\"The capital of France is\"}\n{\"prompt\":\"Water boils at\"}\n",
    )
    .unwrap();
    let config = format!(
        r#"
[endpoint]
base_url = "{url}/v1"
model = "local"
timeout = 60
max_retries = 0
max_tokens = 1

[input]
file = "{corpus}"

[load]
concurrent_requests = 1
total_requests = 2

[logprobs]
enabled = true
top_logprobs = 5
output = "{out}"
mode = "{mode}"

[output]
format = "json"
quiet = true
"#,
        corpus = dir.join("corpus.jsonl").display(),
        out = dir.join("out.jsonl").display(),
    );
    let path = dir.join("lp.toml");
    std::fs::write(&path, config).unwrap();
    path
}

fn run(config: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llm-perf"))
        .arg("logprobs")
        .arg(config)
        .output()
        .unwrap()
}

#[test]
fn raw_mode_writes_raw_records() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/completions")
        .with_body(captured_response("legacy_logprobs_int"))
        .expect(2)
        .create();
    let dir = tempfile::tempdir().unwrap();
    let out = run(&write_config(dir.path(), &server.url(), "raw"));
    assert!(out.status.success(), "{out:?}");
    mock.assert();
    let records = std::fs::read_to_string(dir.path().join("out.jsonl")).unwrap();
    let modes: Vec<String> = records
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["mode"].to_string())
        .collect();
    assert_eq!(modes, ["\"raw\"", "\"raw\""]);
}

#[test]
fn raw_mode_with_no_records_fails() {
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/completions")
        .with_status(400)
        .with_body(r#"{"error":"bad request"}"#)
        .expect(2)
        .create();
    let dir = tempfile::tempdir().unwrap();
    let out = run(&write_config(dir.path(), &server.url(), "raw"));
    mock.assert();
    assert!(!out.status.success(), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no logprobs were captured for any of 2 prompts"),
        "{stderr}"
    );
}

#[test]
fn chat_mode_with_no_records_still_exits_zero() {
    // The chat mode's exit status is unchanged.
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(400)
        .with_body(r#"{"error":"bad request"}"#)
        .expect(2)
        .create();
    let dir = tempfile::tempdir().unwrap();
    let out = run(&write_config(dir.path(), &server.url(), "chat"));
    mock.assert();
    assert!(out.status.success(), "{out:?}");
}

#[test]
fn raw_mode_with_no_prompts_fails() {
    // Every corpus line lacks `prompt`, so nothing is loaded or sent.
    let server = mockito::Server::new();
    let dir = tempfile::tempdir().unwrap();
    let config = write_config(dir.path(), &server.url(), "raw");
    std::fs::write(dir.path().join("corpus.jsonl"), "{\"text\":\"a\"}\n").unwrap();
    let out = run(&config);
    assert!(!out.status.success(), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no prompts to send from"), "{stderr}");
}

#[test]
fn raw_mode_with_sample_size_zero_keeps_existing_output() {
    let server = mockito::Server::new();
    let dir = tempfile::tempdir().unwrap();
    let config = write_config(dir.path(), &server.url(), "raw");
    let toml = std::fs::read_to_string(&config)
        .unwrap()
        .replace("[input]\n", "[input]\nsample_size = 0\n");
    std::fs::write(&config, toml).unwrap();
    std::fs::write(dir.path().join("out.jsonl"), "earlier capture\n").unwrap();
    let out = run(&config);
    assert!(!out.status.success(), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("(sample_size = 0)"), "{stderr}");
    let kept = std::fs::read_to_string(dir.path().join("out.jsonl")).unwrap();
    assert_eq!(kept, "earlier capture\n");
}

#[test]
fn raw_mode_with_unwritable_output_fails() {
    // The writer task fails at `File::create` on the blocking pool, normally
    // before the first response arrives; the next record send then fails and
    // the run stops. Of 20 prompts, at most 2 are sent: the bound allows the
    // writer to finish after the first record was buffered.
    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/v1/completions")
        .with_body(captured_response("legacy_logprobs_int"))
        .expect_at_least(1)
        .expect_at_most(2)
        .create();
    let dir = tempfile::tempdir().unwrap();
    let config = write_config(dir.path(), &server.url(), "raw");
    let corpus: String = (0..20)
        .map(|i| format!("{{\"prompt\":\"p{i}\"}}\n"))
        .collect();
    std::fs::write(dir.path().join("corpus.jsonl"), corpus).unwrap();
    let toml = std::fs::read_to_string(&config).unwrap().replace(
        &dir.path().join("out.jsonl").display().to_string(),
        &dir.path().join("missing/out.jsonl").display().to_string(),
    );
    std::fs::write(&config, toml).unwrap();
    let out = run(&config);
    assert!(!out.status.success(), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("writing "), "{stderr}");
    mock.assert();
}
