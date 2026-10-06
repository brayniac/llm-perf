//! What replay reads from the server before a run: its build and the longest
//! call it accepts.

use crate::client::OpenAIClient;
use crate::config::ReplayServer;
use anyhow::{Context, Result};

/// Server properties a replay run depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerInfo {
    /// llama-server's `build_info` or vLLM's version, for the report.
    pub build: Option<String>,
    /// Longest prompt plus completion one request may have, in tokens.
    pub n_ctx: u64,
}

/// Read [`ServerInfo`] from the server.
///
/// llama-server: `/props` (`build_info`, `default_generation_settings.n_ctx`).
///
/// vLLM: `/version`, and the served model's `max_model_len` from `/v1/models`.
pub async fn probe(client: &OpenAIClient, server: ReplayServer, model: &str) -> Result<ServerInfo> {
    match server {
        ReplayServer::LlamaServer => {
            let props = client
                .props()
                .await
                .context("reading the server's /props")?;
            llama_server_info(&props)
        }
        ReplayServer::Vllm => {
            let version = client
                .get_root_text("/version")
                .await
                .context("reading vLLM's /version")?;
            let models = client
                .get_root_text("/v1/models")
                .await
                .context("reading vLLM's /v1/models")?;
            vllm_info(&version, &models, model)
        }
    }
}

fn llama_server_info(props: &serde_json::Value) -> Result<ServerInfo> {
    let n_ctx = props
        .pointer("/default_generation_settings/n_ctx")
        .and_then(|v| v.as_u64())
        .context("/props has no default_generation_settings.n_ctx")?;
    Ok(ServerInfo {
        build: props
            .get("build_info")
            .and_then(|b| b.as_str())
            .map(str::to_string),
        n_ctx,
    })
}

fn vllm_info(version: &str, models: &str, model: &str) -> Result<ServerInfo> {
    let version: serde_json::Value =
        serde_json::from_str(version).context("parsing vLLM's /version")?;
    let models: serde_json::Value =
        serde_json::from_str(models).context("parsing vLLM's /v1/models")?;
    let entry = models
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|d| {
            d.iter()
                .find(|m| m.get("id").and_then(|i| i.as_str()) == Some(model))
        })
        .with_context(|| format!("/v1/models does not list {model}"))?;
    let n_ctx = entry
        .get("max_model_len")
        .and_then(|v| v.as_u64())
        .with_context(|| format!("/v1/models has no max_model_len for {model}"))?;
    Ok(ServerInfo {
        build: version
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        n_ctx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_server_info_reads_captured_props() {
        let props: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/llama-server/llama-3.1-8b-instruct-q8_0/props.json"
        ))
        .unwrap();
        let info = llama_server_info(&props).unwrap();
        assert_eq!(info.build.as_deref(), Some("b1-4ebdf2c"));
        assert!(info.n_ctx > 0);
        assert!(llama_server_info(&serde_json::json!({})).is_err());
    }

    #[test]
    fn vllm_info_reads_captured_endpoints() {
        let version = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/version.json");
        let models = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/models.json");
        let info = vllm_info(version, models, "qwen3.5-9b").unwrap();
        assert_eq!(
            info,
            ServerInfo {
                build: Some("0.31.0".to_string()),
                n_ctx: 131072,
            }
        );
        let err = vllm_info(version, models, "other").unwrap_err();
        assert!(err.to_string().contains("does not list other"), "{err}");
    }
}
