use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;
use tokio::sync::Semaphore;

use crate::client::{
    ChatCompletionRequest, ClientConfig, ClientError, CompletionRequest, OpenAIClient, Usage,
};

use super::config::{Config, PromptMode};
use super::dataset::Question;
use super::extract::extract_answer;
use super::fit::{ShotFit, fit_shots};
use super::prompt::{build_completion_prompt, build_messages};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: i64,
    pub question: String,
    pub category: String,
    pub options: Vec<String>,
    pub answer: String,
    pub answer_index: i64,
    pub response: String,
    pub pred: Option<String>,
    /// The exact prompt sent, when `log_prompt` is set. In chat mode this is
    /// one entry per chat message; in completion mode it is a single entry with
    /// role `"prompt"` holding the whole prompt string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Vec<PromptMessage>>,
    /// Number of few-shot examples in the prompt that was sent. `None` for a
    /// skipped question and in result files written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shots_used: Option<usize>,
    /// The prompt did not fit `max_context_tokens` even with 0 shots, so no
    /// request was sent. Counted in the accuracy denominator.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub skipped: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    pub correct: u32,
    pub wrong: u32,
    pub extraction_failures: u32,
    pub errors: u32,
    /// Questions not sent because they did not fit `max_context_tokens`.
    pub skipped: u32,
    /// Number of answered questions at each shot count.
    pub shots_used: BTreeMap<usize, u32>,
}

impl CategoryStats {
    /// Accuracy denominator: every question attempted, including request
    /// errors and questions skipped for length.
    pub fn total(&self) -> u32 {
        self.correct + self.wrong + self.errors + self.skipped
    }

    /// Count a saved result (used when resuming from a result file).
    fn record_saved(&mut self, r: &QuestionResult) {
        if r.skipped {
            self.skipped += 1;
            return;
        }
        match &r.pred {
            Some(pred) if pred == &r.answer => self.correct += 1,
            Some(_) => self.wrong += 1,
            None => {
                self.wrong += 1;
                self.extraction_failures += 1;
            }
        }
        if let Some(k) = r.shots_used {
            *self.shots_used.entry(k).or_default() += 1;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    pub prompt_tokens: Vec<u32>,
    pub completion_tokens: Vec<u32>,
}

pub struct EvaluationResult {
    pub category_stats: HashMap<String, CategoryStats>,
    pub token_stats: TokenStats,
}

/// Shared atomic counters for lock-free progress reporting.
struct ProgressCounters {
    completed: AtomicU32,
    correct: AtomicU32,
    wrong: AtomicU32,
    extraction_failures: AtomicU32,
    errors: AtomicU32,
    skipped: AtomicU32,
    prompt_tokens: AtomicU32,
    completion_tokens: AtomicU32,
    total: u32,
}

/// Load existing results from a category result file for resume support.
fn load_existing_results(path: &Path) -> Vec<QuestionResult> {
    if !path.exists() {
        return Vec::new();
    }
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Atomically write `bytes` to `path` by writing a sibling temp file and
/// renaming it into place. A crash or error mid-write leaves the existing
/// file untouched, rather than truncating it (which would make resume discard
/// all prior results for the category).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Save results to a category result file.
fn save_results(results: &[QuestionResult], path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    write_atomic(path, json.as_bytes())
}

/// Save category summary to a JSON file.
fn save_summary(stats: &HashMap<String, CategoryStats>, path: &Path) -> Result<()> {
    let mut summary: HashMap<String, serde_json::Value> = HashMap::new();
    let mut total_corr = 0u32;
    let mut total_wrong = 0u32;
    let mut total_errors = 0u32;
    let mut total_skipped = 0u32;

    for (category, s) in stats {
        let total = s.total();
        let acc = if total > 0 {
            s.correct as f64 / total as f64
        } else {
            0.0
        };
        summary.insert(
            category.clone(),
            serde_json::json!({
                "corr": s.correct,
                "wrong": s.wrong,
                "errors": s.errors,
                "skipped": s.skipped,
                "extraction_failures": s.extraction_failures,
                "acc": acc,
            }),
        );
        total_corr += s.correct;
        total_wrong += s.wrong;
        total_errors += s.errors;
        total_skipped += s.skipped;
    }

    let total = total_corr + total_wrong + total_errors + total_skipped;
    let acc = if total > 0 {
        total_corr as f64 / total as f64
    } else {
        0.0
    };
    summary.insert(
        "total".to_string(),
        serde_json::json!({
            "corr": total_corr,
            "wrong": total_wrong,
            "errors": total_errors,
            "skipped": total_skipped,
            "acc": acc,
        }),
    );

    let json = serde_json::to_string_pretty(&summary)?;
    write_atomic(path, json.as_bytes())
}

fn format_eta(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{}m{}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else {
        format!("{}m{}s", secs / 60, secs % 60)
    }
}

fn print_status(
    category: &str,
    counters: &ProgressCounters,
    start: Instant,
    overall: &ProgressCounters,
    overall_start: Instant,
) {
    let completed = counters.completed.load(Ordering::Relaxed);
    let correct = counters.correct.load(Ordering::Relaxed);
    let wrong = counters.wrong.load(Ordering::Relaxed);
    let failures = counters.extraction_failures.load(Ordering::Relaxed);
    let errors = counters.errors.load(Ordering::Relaxed);
    let skipped = counters.skipped.load(Ordering::Relaxed);
    let prompt_tokens = overall.prompt_tokens.load(Ordering::Relaxed);
    let completion_tokens = overall.completion_tokens.load(Ordering::Relaxed);
    let total = correct + wrong + errors + skipped;
    let acc = if total > 0 {
        correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let elapsed_secs = start.elapsed().as_secs_f64();
    let elapsed = elapsed_secs as u64;

    let mut msg = format!(
        "  {}: {}/{} completed, {}/{} correct ({:.2}%)",
        category, completed, counters.total, correct, total, acc
    );
    if failures > 0 {
        msg.push_str(&format!(", {} failed extractions", failures));
    }
    if errors > 0 {
        msg.push_str(&format!(", {} errors", errors));
    }
    if skipped > 0 {
        msg.push_str(&format!(", {} skipped (too long)", skipped));
    }

    // Token throughput (use overall elapsed for accurate rates)
    let overall_elapsed = overall_start.elapsed().as_secs_f64();
    if overall_elapsed > 0.0 && completion_tokens > 0 {
        let prompt_tps = prompt_tokens as f64 / overall_elapsed;
        let completion_tps = completion_tokens as f64 / overall_elapsed;
        msg.push_str(&format!(
            ", {:.0} prompt tk/s, {:.0} completion tk/s",
            prompt_tps, completion_tps
        ));
    }

    // Overall ETA
    let overall_completed = overall.completed.load(Ordering::Relaxed);
    if overall_completed > 0 && overall_completed < overall.total {
        let overall_remaining = overall.total - overall_completed;
        let secs_per_item = overall_elapsed / overall_completed as f64;
        let eta_secs = (overall_remaining as f64 * secs_per_item) as u64;
        msg.push_str(&format!(", ETA {}", format_eta(eta_secs)));
    }

    msg.push_str(&format!(", {} elapsed", format_eta(elapsed)));
    eprintln!("{}", msg);
}

/// Append `result` to the category's result file and rewrite its summary.
async fn persist(
    results: &tokio::sync::Mutex<Vec<QuestionResult>>,
    stats: &tokio::sync::Mutex<CategoryStats>,
    result: QuestionResult,
    category: &str,
    result_path: &Path,
    summary_path: &Path,
) {
    {
        let mut res = results.lock().await;
        res.push(result);

        // Deduplicate by question_id
        let mut seen = std::collections::HashSet::new();
        res.retain(|r| seen.insert(r.question_id));

        let _ = save_results(&res, result_path);
    }
    let s = stats.lock().await;
    let mut summary_stats: HashMap<String, CategoryStats> = HashMap::new();
    summary_stats.insert(category.to_string(), s.clone());
    let _ = save_summary(&summary_stats, summary_path);
}

/// The text whose token count stands for the prompt's length when fitting
/// shots into `max_context_tokens`.
///
/// Completion mode: the exact prompt. Chat mode: the message contents joined
/// by blank lines. That omits the chat template's role markers and special
/// tokens, so in chat mode the count is low by a few tokens per message.
/// In both modes llama-server's `/tokenize` does not add BOS by default, so
/// the count is also one token below what generation sees.
fn prompt_text_for_counting(
    mode: PromptMode,
    system_prompt: &str,
    shots: &[Question],
    question: &Question,
) -> String {
    match mode {
        PromptMode::Chat => {
            build_messages(system_prompt, shots, &question.question, &question.options)
                .into_iter()
                .map(|m| m.content)
                .collect::<Vec<_>>()
                .join("\n\n")
        }
        PromptMode::Completion => {
            build_completion_prompt(system_prompt, shots, &question.question, &question.options)
        }
    }
}

/// Fail early when `max_context_tokens` is set but cannot be applied: the
/// generation budget alone fills the window, or the server has no usable
/// `/tokenize` endpoint (it is a llama.cpp server extension).
async fn check_tokenize_available(
    client: &OpenAIClient,
    max_context_tokens: u32,
    max_tokens: u32,
) -> Result<()> {
    if max_tokens >= max_context_tokens {
        anyhow::bail!(
            "max_tokens ({max_tokens}) must be less than max_context_tokens \
             ({max_context_tokens}); otherwise no prompt fits"
        );
    }
    match client.tokenize("The answer is (A).").await {
        Ok(n) if n > 0 => Ok(()),
        Ok(_) => anyhow::bail!(
            "max_context_tokens is set, but the server's /tokenize endpoint returned \
             no tokens for a non-empty string. It needs llama-server's \
             POST /tokenize ({{\"content\": ...}} -> {{\"tokens\": [...]}})."
        ),
        Err(e) => anyhow::bail!(
            "max_context_tokens is set, which counts prompt tokens with the server's \
             POST /tokenize endpoint (served by llama-server at the server root, not \
             under /v1), but that request failed: {e}. Unset max_context_tokens for \
             servers without /tokenize."
        ),
    }
}

/// Run evaluation across all specified categories.
pub async fn run_evaluation(
    config: &Config,
    model: &str,
    test_data: &HashMap<String, Vec<Question>>,
    val_data: &HashMap<String, Vec<Question>>,
    output_dir: &Path,
) -> Result<EvaluationResult> {
    let client_config = ClientConfig {
        base_url: config.endpoint.base_url.clone(),
        api_key: config.endpoint.api_key.clone(),
        model: model.to_string(),
        timeout: std::time::Duration::from_secs(config.endpoint.timeout),
        max_retries: 3,
        retry_initial_delay_ms: 1000,
        retry_max_delay_ms: 30000,
        pool_size: config.load.concurrent_requests,
        // Non-streaming offline eval: no streaming idle timeout, and retrying
        // timeouts is desirable here (no coordinated-omission concern), matching
        // the prior retry-all-transient behavior.
        stream_idle_timeout: None,
        retry_on_timeout: true,
        chat_template_kwargs: None,
        ignore_eos: None,
    };

    let client = Arc::new(OpenAIClient::new(client_config)?);
    let semaphore = Arc::new(Semaphore::new(config.load.concurrent_requests));

    let mut all_stats: HashMap<String, CategoryStats> = HashMap::new();
    let mut all_token_stats = TokenStats::default();

    // Determine which categories to evaluate
    let categories: Vec<String> = if config.load.categories.contains(&"all".to_string()) {
        let mut cats: Vec<String> = test_data.keys().cloned().collect();
        cats.sort();
        cats
    } else {
        config.load.categories.clone()
    };

    let system_prompt_template = &config.inference.system_prompt;
    let max_context_tokens = config.inference.max_context_tokens;

    if let Some(ctx) = max_context_tokens {
        check_tokenize_available(&client, ctx, config.inference.max_tokens).await?;
    }

    // Compute overall question count for ETA across all categories
    let overall_total: u32 = categories
        .iter()
        .filter_map(|cat| test_data.get(cat))
        .map(|qs| qs.len() as u32)
        .sum();

    let overall_start = Instant::now();
    let overall_counters = Arc::new(ProgressCounters {
        completed: AtomicU32::new(0),
        correct: AtomicU32::new(0),
        wrong: AtomicU32::new(0),
        extraction_failures: AtomicU32::new(0),
        errors: AtomicU32::new(0),
        skipped: AtomicU32::new(0),
        prompt_tokens: AtomicU32::new(0),
        completion_tokens: AtomicU32::new(0),
        total: overall_total,
    });

    for category in &categories {
        let test_questions = match test_data.get(category) {
            Some(q) => q,
            None => {
                eprintln!(
                    "Warning: category '{}' not found in test data, skipping.",
                    category
                );
                continue;
            }
        };

        let cot_examples: Vec<Question> = val_data
            .get(category)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .take(config.inference.num_shots)
            .collect();

        let system_prompt = system_prompt_template.replace("{subject}", category);

        let result_path = output_dir.join(format!("{}_result.json", category));
        let summary_path = output_dir.join(format!("{}_summary.json", category));

        // Load existing results for resume
        let existing_results = load_existing_results(&result_path);
        let existing_ids: std::collections::HashSet<i64> =
            existing_results.iter().map(|r| r.question_id).collect();

        // Count stats from existing results
        let mut cat_stats = CategoryStats::default();
        for r in &existing_results {
            cat_stats.record_saved(r);
        }

        // Filter to only new questions
        let new_questions: Vec<&Question> = test_questions
            .iter()
            .filter(|q| !existing_ids.contains(&q.question_id))
            .collect();

        let total = test_questions.len();
        let already_done = existing_ids.len();

        // Account for already-completed questions in overall counters
        overall_counters
            .completed
            .fetch_add(already_done as u32, Ordering::Relaxed);
        overall_counters
            .correct
            .fetch_add(cat_stats.correct, Ordering::Relaxed);
        overall_counters
            .wrong
            .fetch_add(cat_stats.wrong, Ordering::Relaxed);
        overall_counters
            .extraction_failures
            .fetch_add(cat_stats.extraction_failures, Ordering::Relaxed);
        overall_counters
            .skipped
            .fetch_add(cat_stats.skipped, Ordering::Relaxed);

        if new_questions.is_empty() {
            eprintln!(
                "{}: all {}/{} questions already completed, skipping.",
                category, already_done, total
            );
            all_stats.insert(category.clone(), cat_stats);
            continue;
        }

        eprintln!(
            "{}: {}/{} already done, {} remaining.",
            category,
            already_done,
            total,
            new_questions.len()
        );

        let cat_start = Instant::now();

        // Atomic counters for lock-free progress reporting
        let counters = Arc::new(ProgressCounters {
            completed: AtomicU32::new(0),
            correct: AtomicU32::new(cat_stats.correct),
            wrong: AtomicU32::new(cat_stats.wrong),
            extraction_failures: AtomicU32::new(cat_stats.extraction_failures),
            errors: AtomicU32::new(cat_stats.errors),
            skipped: AtomicU32::new(cat_stats.skipped),
            prompt_tokens: AtomicU32::new(0),
            completion_tokens: AtomicU32::new(0),
            total: total as u32,
        });

        // Spawn periodic status printer (every 60s)
        let status_counters = Arc::clone(&counters);
        let status_overall = Arc::clone(&overall_counters);
        let status_category = category.clone();
        let status_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                print_status(
                    &status_category,
                    &status_counters,
                    cat_start,
                    &status_overall,
                    overall_start,
                );
            }
        });

        // Shared state for collecting results
        let results = Arc::new(tokio::sync::Mutex::new(existing_results));
        let stats = Arc::new(tokio::sync::Mutex::new(cat_stats));
        let token_stats = Arc::new(tokio::sync::Mutex::new(TokenStats::default()));

        let mut handles = Vec::new();

        for question in new_questions {
            let client: Arc<OpenAIClient> = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            let results = Arc::clone(&results);
            let stats = Arc::clone(&stats);
            let token_stats = Arc::clone(&token_stats);
            let counters = Arc::clone(&counters);
            let overall_counters = Arc::clone(&overall_counters);
            let result_path = result_path.clone();
            let summary_path = summary_path.clone();
            let system_prompt = system_prompt.clone();
            let cot_examples = cot_examples.clone();
            let question = question.clone();
            let log_prompt = config.log.log_prompt;
            let temperature = config.inference.temperature;
            let top_p = config.inference.top_p;
            let max_tokens = config.inference.max_tokens;
            let frequency_penalty = config.inference.frequency_penalty;
            let presence_penalty = config.inference.presence_penalty;
            let model = model.to_string();
            let verbosity = config.log.verbosity;
            let mode = config.inference.mode;
            let category = category.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Choose how many shots to send. Without max_context_tokens this
                // is always every available shot (num_shots, or fewer if the
                // validation split has fewer for this category).
                let shots = match max_context_tokens {
                    None => cot_examples.len(),
                    Some(ctx) => {
                        let fit = fit_shots(cot_examples.len(), max_tokens, ctx, |k| {
                            let text = prompt_text_for_counting(
                                mode,
                                &system_prompt,
                                &cot_examples[..k],
                                &question,
                            );
                            let client = Arc::clone(&client);
                            async move { client.tokenize(&text).await }
                        })
                        .await;
                        match fit {
                            Ok(ShotFit::Fits { shots, .. }) => shots,
                            Ok(ShotFit::TooLong { zero_shot_tokens }) => {
                                if verbosity >= 1 {
                                    eprintln!(
                                        "Skipping question {}: {} prompt tokens at 0 shots + \
                                         {} max_tokens exceeds max_context_tokens {}",
                                        question.question_id, zero_shot_tokens, max_tokens, ctx
                                    );
                                }
                                stats.lock().await.skipped += 1;
                                counters.skipped.fetch_add(1, Ordering::Relaxed);
                                overall_counters.skipped.fetch_add(1, Ordering::Relaxed);
                                let result = QuestionResult {
                                    question_id: question.question_id,
                                    question: question.question.clone(),
                                    category: question.category.clone(),
                                    options: question.options.clone(),
                                    answer: question.answer.clone(),
                                    answer_index: question.answer_index,
                                    response: String::new(),
                                    pred: None,
                                    prompt: None,
                                    shots_used: None,
                                    skipped: true,
                                };
                                persist(
                                    &results,
                                    &stats,
                                    result,
                                    &category,
                                    &result_path,
                                    &summary_path,
                                )
                                .await;
                                counters.completed.fetch_add(1, Ordering::Relaxed);
                                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            Err(e) => {
                                eprintln!(
                                    "Error for question {} (tokenize): {}",
                                    question.question_id, e
                                );
                                stats.lock().await.errors += 1;
                                counters.errors.fetch_add(1, Ordering::Relaxed);
                                counters.completed.fetch_add(1, Ordering::Relaxed);
                                overall_counters.errors.fetch_add(1, Ordering::Relaxed);
                                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                };
                let cot_examples = &cot_examples[..shots];

                let stop = Some(vec!["Question:".to_string()]);

                // Send the request in the configured mode. Both arms yield the
                // generated text, the server's token usage, and the prompt as it
                // would be logged.
                let (outcome, prompt_messages) = match mode {
                    PromptMode::Chat => {
                        let messages = build_messages(
                            &system_prompt,
                            cot_examples,
                            &question.question,
                            &question.options,
                        );
                        let request = ChatCompletionRequest {
                            model,
                            messages: messages.clone(),
                            max_tokens: Some(max_tokens),
                            temperature: Some(temperature),
                            top_p: Some(top_p),
                            frequency_penalty: Some(frequency_penalty),
                            presence_penalty: Some(presence_penalty),
                            stop,
                            stream: Some(false),
                            stream_options: None,
                            logprobs: None,
                            top_logprobs: None,
                            chat_template_kwargs: None,
                            ignore_eos: None,
                        };
                        let outcome = client.chat_completion(request).await.map(|r| {
                            let text = r
                                .choices
                                .first()
                                .map(|c| c.message.content.clone())
                                .unwrap_or_default();
                            (text, r.usage)
                        });
                        let logged = messages
                            .into_iter()
                            .map(|m| PromptMessage {
                                role: m.role,
                                content: m.content,
                            })
                            .collect::<Vec<_>>();
                        (outcome, logged)
                    }
                    PromptMode::Completion => {
                        let prompt = build_completion_prompt(
                            &system_prompt,
                            cot_examples,
                            &question.question,
                            &question.options,
                        );
                        let request = CompletionRequest {
                            model,
                            prompt: prompt.clone(),
                            max_tokens: Some(max_tokens),
                            temperature: Some(temperature),
                            top_p: Some(top_p),
                            frequency_penalty: Some(frequency_penalty),
                            presence_penalty: Some(presence_penalty),
                            stop,
                            stream: Some(false),
                        };
                        let outcome = client.completion(request).await.map(|r| {
                            let text = r
                                .choices
                                .first()
                                .map(|c| c.text.clone())
                                .unwrap_or_default();
                            (text, r.usage)
                        });
                        let logged = vec![PromptMessage {
                            role: "prompt".to_string(),
                            content: prompt,
                        }];
                        (outcome, logged)
                    }
                };

                let (response_text, usage): (String, Usage) = match outcome {
                    Ok(out) => out,
                    Err(e) => {
                        let error_kind = match e.downcast_ref::<ClientError>() {
                            Some(ClientError::Connection(_)) => "connection",
                            Some(ClientError::Timeout(_)) => "timeout",
                            Some(ClientError::Http4xx { status, .. }) => {
                                // Leak a short label; there are only a few distinct status codes
                                Box::leak(format!("http {status}").into_boxed_str())
                            }
                            Some(ClientError::Http5xx { status, .. }) => {
                                Box::leak(format!("http {status}").into_boxed_str())
                            }
                            Some(ClientError::Parse(_)) => "parse",
                            Some(ClientError::StreamError { .. }) => "stream",
                            Some(ClientError::Other(_)) | None => "unknown",
                        };
                        eprintln!(
                            "Error for question {} ({}): {}",
                            question.question_id, error_kind, e
                        );
                        {
                            let mut s = stats.lock().await;
                            s.errors += 1;
                        }
                        counters.errors.fetch_add(1, Ordering::Relaxed);
                        counters.completed.fetch_add(1, Ordering::Relaxed);
                        overall_counters.errors.fetch_add(1, Ordering::Relaxed);
                        overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                };

                // Track token usage
                {
                    let mut ts = token_stats.lock().await;
                    ts.prompt_tokens.push(usage.prompt_tokens);
                    ts.completion_tokens.push(usage.completion_tokens);
                }
                overall_counters
                    .prompt_tokens
                    .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
                overall_counters
                    .completion_tokens
                    .fetch_add(usage.completion_tokens, Ordering::Relaxed);

                let response_text = response_text.trim().to_string();

                let pred = extract_answer(&response_text);
                let pred_str = pred.map(|c| c.to_string());

                if verbosity >= 2 {
                    eprintln!(
                        "Q{}: pred={:?} answer={} | {}",
                        question.question_id,
                        pred_str,
                        question.answer,
                        &response_text[..response_text.len().min(100)]
                    );
                }

                let prompt_log = log_prompt.then_some(prompt_messages);

                let result = QuestionResult {
                    question_id: question.question_id,
                    question: question.question.clone(),
                    category: question.category.clone(),
                    options: question.options.clone(),
                    answer: question.answer.clone(),
                    answer_index: question.answer_index,
                    response: response_text,
                    pred: pred_str.clone(),
                    prompt: prompt_log,
                    shots_used: Some(shots),
                    skipped: false,
                };

                // Update stats
                {
                    let mut s = stats.lock().await;
                    *s.shots_used.entry(shots).or_default() += 1;
                    match &pred_str {
                        Some(p) if p == &question.answer => {
                            s.correct += 1;
                            counters.correct.fetch_add(1, Ordering::Relaxed);
                            overall_counters.correct.fetch_add(1, Ordering::Relaxed);
                        }
                        Some(_) => {
                            s.wrong += 1;
                            counters.wrong.fetch_add(1, Ordering::Relaxed);
                            overall_counters.wrong.fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            s.wrong += 1;
                            s.extraction_failures += 1;
                            counters.wrong.fetch_add(1, Ordering::Relaxed);
                            counters.extraction_failures.fetch_add(1, Ordering::Relaxed);
                            overall_counters.wrong.fetch_add(1, Ordering::Relaxed);
                            overall_counters
                                .extraction_failures
                                .fetch_add(1, Ordering::Relaxed);
                            if verbosity >= 2 {
                                // Show the tail of the response where the answer should be
                                let tail = if result.response.len() > 300 {
                                    format!(
                                        "...{}",
                                        &result.response[result.response.len() - 300..]
                                    )
                                } else {
                                    result.response.clone()
                                };
                                eprintln!(
                                    "Extraction failed for Q{}: «{}»",
                                    question.question_id, tail
                                );
                            }
                        }
                    }
                }

                persist(
                    &results,
                    &stats,
                    result,
                    &category,
                    &result_path,
                    &summary_path,
                )
                .await;

                counters.completed.fetch_add(1, Ordering::Relaxed);
                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Stop the status printer
        status_handle.abort();

        // Print final status for this category
        print_status(
            category,
            &counters,
            cat_start,
            &overall_counters,
            overall_start,
        );

        // Collect final stats
        let final_stats = stats.lock().await.clone();
        let final_token_stats = token_stats.lock().await.clone();

        // Final save
        {
            let final_results = results.lock().await;
            save_results(&final_results, &result_path)?;
        }

        // Save final summary
        {
            let mut summary_stats: HashMap<String, CategoryStats> = HashMap::new();
            summary_stats.insert(category.clone(), final_stats.clone());
            save_summary(&summary_stats, &summary_path)?;
        }

        all_token_stats
            .prompt_tokens
            .extend(&final_token_stats.prompt_tokens);
        all_token_stats
            .completion_tokens
            .extend(&final_token_stats.completion_tokens);
        all_stats.insert(category.clone(), final_stats);
    }

    Ok(EvaluationResult {
        category_stats: all_stats,
        token_stats: all_token_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(id: i64) -> QuestionResult {
        QuestionResult {
            question_id: id,
            question: "q".to_string(),
            category: "cat".to_string(),
            options: vec!["A".to_string(), "B".to_string()],
            answer: "A".to_string(),
            answer_index: 0,
            response: "the answer is (A)".to_string(),
            pred: Some("A".to_string()),
            prompt: None,
            shots_used: Some(5),
            skipped: false,
        }
    }

    #[test]
    fn resume_counts_skipped_and_shots_used() {
        let mut stats = CategoryStats::default();
        let mut wrong = sample_result(2);
        wrong.pred = Some("B".to_string());
        wrong.shots_used = Some(3);
        let mut skipped = sample_result(3);
        skipped.pred = None;
        skipped.shots_used = None;
        skipped.skipped = true;
        let mut legacy = sample_result(4);
        legacy.shots_used = None;

        for r in [sample_result(1), wrong, skipped, legacy] {
            stats.record_saved(&r);
        }
        assert_eq!(stats.correct, 2);
        assert_eq!(stats.wrong, 1);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.extraction_failures, 0);
        assert_eq!(stats.total(), 4);
        assert_eq!(stats.shots_used, BTreeMap::from([(3, 1), (5, 1)]));
    }

    #[test]
    fn result_without_new_fields_still_loads() {
        let json = r#"[{"question_id": 1, "question": "q", "category": "c",
            "options": ["a"], "answer": "A", "answer_index": 0,
            "response": "the answer is (A)", "pred": "A"}]"#;
        let r: Vec<QuestionResult> = serde_json::from_str(json).unwrap();
        assert_eq!(r[0].shots_used, None);
        assert!(!r[0].skipped);
    }

    #[test]
    fn completion_counting_text_is_the_exact_prompt() {
        let q = Question {
            question_id: 1,
            question: "Q?".to_string(),
            options: vec!["a".to_string()],
            answer: "A".to_string(),
            answer_index: 0,
            cot_content: String::new(),
            category: "math".to_string(),
        };
        assert_eq!(
            prompt_text_for_counting(PromptMode::Completion, "H", &[], &q),
            build_completion_prompt("H", &[], "Q?", &q.options)
        );
        let chat = prompt_text_for_counting(PromptMode::Chat, "H", &[], &q);
        assert!(chat.starts_with("H\n\nQuestion: Q?"));
    }

    #[test]
    fn save_results_roundtrips_through_load() {
        let dir = std::env::temp_dir().join(format!("mmlu_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cat_result.json");

        let results = vec![sample_result(1), sample_result(2)];
        save_results(&results, &path).unwrap();
        let loaded = load_existing_results(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].question_id, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_atomic_preserves_existing_file_when_write_fails() {
        // A crash/failure mid-write must not destroy already-saved results.
        let dir = std::env::temp_dir().join(format!("mmlu_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("results.json");
        std::fs::write(&dest, b"GOOD").unwrap();

        // Block the temp path by occupying it with a directory, so the temp
        // write fails before any rename can touch the destination.
        let tmp = dest.with_extension("tmp");
        std::fs::create_dir_all(&tmp).unwrap();

        let result = write_atomic(&dest, b"NEW");
        assert!(
            result.is_err(),
            "write should fail when temp path is blocked"
        );
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            b"GOOD",
            "destination must retain its prior contents on a failed write"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
