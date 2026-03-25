use anyhow::{Context, Result};
use arrow::array::StringArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Known dataset names that can be auto-downloaded from HuggingFace.
pub const KNOWN_DATASETS: &[(&str, &str)] = &[
    (
        "openorca",
        "Open-Orca/OpenOrca (~1 GB download, 1M GPT-4 prompts)",
    ),
    (
        "sharegpt",
        "ShareGPT Vicuna (~670 MB download, ~90K multi-turn conversations)",
    ),
];

/// Resolve an input path: if it exists on disk, use it directly.
/// If it matches a known dataset name, download from HuggingFace and convert to JSONL.
pub async fn resolve_input(path: &Path) -> Result<PathBuf> {
    // If the path exists on disk, use it directly
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // Extract the name (strip directory components and extension)
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match name.as_str() {
        "openorca" => download_openorca().await,
        "sharegpt" => download_sharegpt().await,
        _ => {
            let known = KNOWN_DATASETS
                .iter()
                .map(|(name, desc)| format!("  {name:12} — {desc}"))
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "Input file '{}' not found.\n\nKnown datasets (auto-downloaded from HuggingFace):\n{}",
                path.display(),
                known
            );
        }
    }
}

fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(home)
        .join(".cache")
        .join("llm-perf")
        .join("datasets");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

async fn download_sharegpt() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("sharegpt.jsonl");

    if cached_path.exists() {
        log::info!("Using cached ShareGPT dataset: {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading ShareGPT dataset from HuggingFace (~670 MB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("anon8231489123/ShareGPT_Vicuna_unfiltered".to_string());
    let path = repo
        .get("ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
        .await
        .context("Failed to download ShareGPT dataset")?;

    log::info!("Converting ShareGPT dataset to JSONL...");

    // Parse JSON array and convert to our JSONL format
    let data = std::fs::read_to_string(&path).context("Failed to read downloaded ShareGPT JSON")?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&data).context("Failed to parse ShareGPT JSON")?;

    let file = std::fs::File::create(&cached_path)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut count = 0;

    for entry in &entries {
        if let Some(conversations) = entry.get("conversations").and_then(|c| c.as_array()) {
            if conversations.is_empty() {
                continue;
            }
            let line = serde_json::json!({ "conversations": conversations });
            serde_json::to_writer(&mut writer, &line)?;
            writeln!(&mut writer)?;
            count += 1;
        }
    }

    writer.flush()?;
    log::info!(
        "Cached {} ShareGPT conversations to {}",
        count,
        cached_path.display()
    );

    Ok(cached_path)
}

async fn download_openorca() -> Result<PathBuf> {
    let cache = cache_dir()?;
    let cached_path = cache.join("openorca.jsonl");

    if cached_path.exists() {
        log::info!("Using cached OpenOrca dataset: {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading OpenOrca dataset from HuggingFace (GPT-4 subset, ~1 GB)...");

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("Open-Orca/OpenOrca".to_string());
    let path = repo
        .get("1M-GPT4-Augmented.parquet")
        .await
        .context("Failed to download OpenOrca dataset")?;

    log::info!("Converting OpenOrca dataset to JSONL...");

    let file = std::fs::File::open(&path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let out_file = std::fs::File::create(&cached_path)?;
    let mut writer = std::io::BufWriter::new(out_file);
    let mut count = 0;

    for batch in reader {
        let batch = batch?;

        let question_col = batch
            .column_by_name("question")
            .context("missing 'question' column in OpenOrca parquet")?;
        let questions = question_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("'question' column is not a string array")?;

        // system_prompt column is optional
        let system_col = batch.column_by_name("system_prompt");
        let systems = system_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

        for i in 0..batch.num_rows() {
            let question = questions.value(i);

            // Prepend system prompt if present and non-empty
            let prompt = if let Some(systems) = systems {
                let system = systems.value(i);
                if system.is_empty() {
                    question.to_string()
                } else {
                    format!("{}\n\n{}", system, question)
                }
            } else {
                question.to_string()
            };

            let line = serde_json::json!({ "prompt": prompt });
            serde_json::to_writer(&mut writer, &line)?;
            writeln!(&mut writer)?;
            count += 1;
        }
    }

    writer.flush()?;
    log::info!(
        "Cached {} OpenOrca prompts to {}",
        count,
        cached_path.display()
    );

    Ok(cached_path)
}
