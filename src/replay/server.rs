//! What replay reads from the server before a run: its build, the longest
//! call it accepts, and the size of the blocks its prefix cache reuses.

use crate::client::OpenAIClient;
use crate::config::ReplayServer;
use anyhow::{Context, Result, bail};

/// Server properties a replay run depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerInfo {
    /// llama-server's `build_info` or vLLM's version, for the report.
    pub build: Option<String>,
    /// Longest prompt plus completion one request may have, in tokens.
    pub n_ctx: u64,
    /// The prefix cache reuses whole blocks of this many tokens; 1 when it
    /// reuses any prefix length.
    pub cache_block: u64,
}

/// Read [`ServerInfo`] from the server.
///
/// llama-server: `/props` (`build_info`, `default_generation_settings.n_ctx`);
/// the cache reuses any prefix length.
///
/// vLLM: `/version`, the served model's `max_model_len` from `/v1/models`,
/// and `block_size` from the `vllm:cache_config_info` metric. Fails if that
/// metric shows prefix caching disabled.
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
            let metrics = client
                .get_root_text("/metrics")
                .await
                .context("reading vLLM's /metrics")?;
            vllm_info(&version, &models, &metrics, model)
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
        cache_block: 1,
    })
}

fn vllm_info(version: &str, models: &str, metrics: &str, model: &str) -> Result<ServerInfo> {
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
    let label = |name: &str| metric_label(metrics, "vllm:cache_config_info", name);
    match label("enable_prefix_caching").as_deref() {
        Some("True") => {}
        Some(_) => {
            bail!("vLLM is running without prefix caching; start it with --enable-prefix-caching")
        }
        None => bail!("/metrics has no vllm:cache_config_info enable_prefix_caching label"),
    }
    let cache_block = label("block_size")
        .context("/metrics has no vllm:cache_config_info block_size")?
        .parse::<u64>()
        .context("parsing vllm:cache_config_info block_size")?;
    if cache_block == 0 {
        bail!("vllm:cache_config_info reports block_size 0");
    }
    Ok(ServerInfo {
        build: version
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        n_ctx,
        cache_block,
    })
}

/// The value of label `name` on the first sample of `metric` in a Prometheus
/// text exposition. Escaped quotes (`\"`) inside label values are not
/// handled.
fn metric_label(text: &str, metric: &str, name: &str) -> Option<String> {
    let mut rest = text
        .lines()
        .find_map(|l| l.strip_prefix(metric)?.strip_prefix('{'))?;
    // Each label is `key="value"` followed by `,` or the closing `}`.
    loop {
        let (key, after) = rest.split_once("=\"")?;
        let (value, after) = after.split_once('"')?;
        if key.trim() == name {
            return Some(value.to_string());
        }
        rest = after.strip_prefix(',')?;
    }
}

/// The longest prefix of a prompt of `prompt_tokens` tokens that a cache
/// reusing whole `cache_block`-token blocks can serve, given `reuse` tokens
/// match a cached prompt. The server computes at least the prompt's last
/// token, so at most `prompt_tokens - 1` tokens are reusable.
pub fn cacheable(reuse: u64, prompt_tokens: u64, cache_block: u64) -> u64 {
    let n = reuse.min(prompt_tokens.saturating_sub(1));
    n - n % cache_block.max(1)
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
        assert_eq!(info.cache_block, 1);
        assert!(info.n_ctx > 0);
        assert!(llama_server_info(&serde_json::json!({})).is_err());
    }

    #[test]
    fn vllm_info_reads_captured_endpoints() {
        let version = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/version.json");
        let models = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/models.json");
        let metrics = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/metrics.txt");
        let info = vllm_info(version, models, metrics, "qwen3.5-9b").unwrap();
        assert_eq!(
            info,
            ServerInfo {
                build: Some("0.31.0".to_string()),
                n_ctx: 131072,
                cache_block: 528,
            }
        );
        let err = vllm_info(version, models, metrics, "other").unwrap_err();
        assert!(err.to_string().contains("does not list other"), "{err}");
        let off = metrics.replace(
            "enable_prefix_caching=\"True\"",
            "enable_prefix_caching=\"False\"",
        );
        let err = vllm_info(version, models, &off, "qwen3.5-9b").unwrap_err();
        assert!(err.to_string().contains("--enable-prefix-caching"), "{err}");
        let err = vllm_info(version, models, "", "qwen3.5-9b").unwrap_err();
        assert!(
            err.to_string().contains("no vllm:cache_config_info"),
            "{err}"
        );
    }

    #[test]
    fn metric_label_reads_values_with_commas_and_braces() {
        let text = "# HELP m x\nm_other{block_size=\"1\"} 1\nm{a=\"[1, 2]\",b=\"{x}\",block_size=\"528\"} 1.0\n";
        assert_eq!(metric_label(text, "m", "a").as_deref(), Some("[1, 2]"));
        assert_eq!(metric_label(text, "m", "b").as_deref(), Some("{x}"));
        assert_eq!(
            metric_label(text, "m", "block_size").as_deref(),
            Some("528")
        );
        assert_eq!(metric_label(text, "m", "missing"), None);
        assert_eq!(metric_label(text, "absent", "a"), None);
    }

    #[test]
    fn cacheable_rounds_down_to_whole_blocks() {
        // The captured vLLM calls: a 7023-token prompt sent twice cached 6864
        // tokens (13 blocks of 528) the second time.
        assert_eq!(cacheable(7023, 7023, 528), 6864);
        assert_eq!(cacheable(527, 7023, 528), 0);
        assert_eq!(cacheable(528, 7023, 528), 528);
        // A prompt of exactly whole blocks keeps its last token uncached.
        assert_eq!(cacheable(1056, 1056, 528), 528);
        assert_eq!(cacheable(900, 1000, 1), 900);
        assert_eq!(cacheable(1000, 1000, 1), 999);
        assert_eq!(cacheable(0, 0, 1), 0);
    }

    fn tokens(json: &str) -> Vec<u64> {
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        v["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap())
            .collect()
    }

    #[test]
    fn cacheable_matches_captured_vllm_next_turn() {
        // call3 is call1's messages, call1's reply and a new user message.
        // Qwen3.5's template renders the past assistant turn without the empty
        // think block that ends call1's generation prompt, so the rendered
        // prompts share 7019 tokens, and vLLM reported 6864 cached.
        let call1 = tokens(include_str!(
            "../../tests/fixtures/vllm/qwen3.5-9b-fp8/tokenize-chat.response.json"
        ));
        let call3 = tokens(include_str!(
            "../../tests/fixtures/vllm/qwen3.5-9b-fp8/tokenize-chat3.response.json"
        ));
        let shared = call1.iter().zip(&call3).take_while(|(a, b)| a == b).count() as u64;
        assert_eq!(shared, 7019);
        assert_eq!(call3.len(), 7100);
        let sse = include_str!("../../tests/fixtures/vllm/qwen3.5-9b-fp8/call3.response.sse");
        assert!(sse.contains("\"cached_tokens\":6864"));
        assert_eq!(cacheable(shared, call3.len() as u64, 528), 6864);
    }
}
