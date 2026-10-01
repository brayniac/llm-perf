use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub comment: String,
    pub endpoint: EndpointConfig,
    pub inference: InferenceConfig,
    pub load: LoadConfig,
    #[serde(default)]
    pub log: LogConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EndpointConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    600
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default = "default_num_shots")]
    pub num_shots: usize,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// How the prompt is sent to the server. See [`PromptMode`].
    #[serde(default)]
    pub mode: PromptMode,
    /// Context length available to one request, in tokens. For llama-server
    /// this is `default_generation_settings.n_ctx` from `GET /props`, which is
    /// less than `-c` when `-np` is greater than 1.
    ///
    /// When set, each question's prompt drops shots (last first) until
    /// `prompt_tokens + max_tokens` fits, counting tokens with llama-server's
    /// `/tokenize` (and `/apply-template` in chat mode). A question that does
    /// not fit at 0 shots is skipped. Unset: always use `num_shots`.
    #[serde(default)]
    pub max_context_tokens: Option<u32>,
}

/// How the few-shot prompt is sent to the server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    /// System prompt plus one user/assistant turn pair per shot, sent to
    /// `/chat/completions`. The server applies the model's chat template.
    #[default]
    Chat,
    /// One plain-text prompt in the layout of TIGER-Lab's
    /// `evaluate_from_local.py`, sent to `/completions` with no chat template.
    /// Use this for base (non-chat) models.
    Completion,
}

impl PromptMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PromptMode::Chat => "chat",
            PromptMode::Completion => "completion",
        }
    }
}

impl std::fmt::Display for PromptMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn default_top_p() -> f32 {
    1.0
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_num_shots() -> usize {
    5
}

fn default_system_prompt() -> String {
    "The following are multiple choice questions (with answers) about {subject}. Think step by \
	 step and then finish your answer with \"the answer is (X)\" where X is the correct letter \
	 choice."
        .to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadConfig {
    #[serde(default = "default_categories")]
    pub categories: Vec<String>,
    #[serde(default = "default_subset")]
    pub subset: f64,
    #[serde(default = "default_concurrent_requests")]
    pub concurrent_requests: usize,
}

fn default_categories() -> Vec<String> {
    vec!["all".to_string()]
}

fn default_subset() -> f64 {
    1.0
}

fn default_concurrent_requests() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LogConfig {
    #[serde(default)]
    pub verbosity: u8,
    #[serde(default)]
    pub log_prompt: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
[endpoint]
base_url = "http://localhost:8080/v1"

[inference]

[load]
"#;

    #[test]
    fn mode_defaults_to_chat() {
        let config: Config = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.inference.mode, PromptMode::Chat);
    }

    #[test]
    fn mode_parses_from_toml() {
        let toml = MINIMAL.replace("[inference]", "[inference]\nmode = \"completion\"");
        let config: Config = toml::from_str(&toml).unwrap();
        assert_eq!(config.inference.mode, PromptMode::Completion);

        let toml = MINIMAL.replace("[inference]", "[inference]\nmode = \"chat\"");
        let config: Config = toml::from_str(&toml).unwrap();
        assert_eq!(config.inference.mode, PromptMode::Chat);
    }

    #[test]
    fn max_context_tokens_is_optional() {
        let config: Config = toml::from_str(MINIMAL).unwrap();
        assert_eq!(config.inference.max_context_tokens, None);

        let toml = MINIMAL.replace("[inference]", "[inference]\nmax_context_tokens = 2048");
        let config: Config = toml::from_str(&toml).unwrap();
        assert_eq!(config.inference.max_context_tokens, Some(2048));
    }

    #[test]
    fn unknown_mode_is_rejected() {
        let toml = MINIMAL.replace("[inference]", "[inference]\nmode = \"raw\"");
        assert!(toml::from_str::<Config>(&toml).is_err());
    }

    #[test]
    fn mode_display_matches_config_spelling() {
        assert_eq!(PromptMode::Chat.to_string(), "chat");
        assert_eq!(PromptMode::Completion.to_string(), "completion");
    }
}
