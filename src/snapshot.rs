use anyhow::Result;
use chrono::{Timelike, Utc};
use metriken_exposition::{Snapshot, SnapshotterBuilder};
use metriken_storage::convert::{MsgpackToParquet, ParquetHistogramType, ParquetOptions};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::time::{Instant, interval_at, timeout};

use crate::config::Config;
use crate::metrics::RUNNING;

/// Captures metric snapshots at regular intervals and writes them to a parquet file
pub async fn capture_snapshots(config: Config) -> Result<()> {
    let metrics_config = match config.metrics.as_ref() {
        Some(cfg) => cfg,
        None => return Ok(()), // No metrics configured
    };

    // Parse interval
    let interval_duration = humantime::parse_duration(&metrics_config.interval)?;

    // Create a temporary file for msgpack data (will be converted to parquet)
    let temp_file = NamedTempFile::new()?;
    let file = File::from_std(temp_file.reopen()?);
    let mut writer = BufWriter::new(file);

    // Get an aligned start time (aligned to the second)
    let start = Instant::now() - Duration::from_nanos(Utc::now().nanosecond() as u64)
        + Duration::from_secs(1);

    // Create interval timer
    let mut interval = interval_at(start, interval_duration);

    // Build snapshotter with metadata
    let snapshotter = SnapshotterBuilder::new()
        .metadata("source".to_string(), env!("CARGO_PKG_NAME").to_string())
        .metadata("version".to_string(), env!("CARGO_PKG_VERSION").to_string())
        .metadata(
            "sampling_interval_ms".to_string(),
            interval_duration.as_millis().to_string(),
        )
        .build();

    log::info!(
        "Starting metrics capture to {:?} with interval {:?}",
        metrics_config.output,
        interval_duration
    );

    // Main snapshot loop
    while RUNNING.load(Ordering::Relaxed) {
        // Use timeout to check RUNNING flag periodically
        if timeout(Duration::from_secs(1), interval.tick())
            .await
            .is_err()
        {
            continue;
        }

        let snapshot = snapshotter.snapshot();

        // Serialize to msgpack using the Snapshot::to_msgpack API
        let buf = Snapshot::to_msgpack(&snapshot).expect("failed to serialize snapshot");

        // Write to temporary file
        if let Err(e) = writer.write_all(&buf).await {
            log::error!("Error writing metrics snapshot: {}", e);
            break;
        }
    }

    // Flush writer
    writer.flush().await?;
    drop(writer); // Close the writer

    // Convert msgpack to parquet
    log::info!("Converting msgpack to parquet...");

    parquet_converter(metrics_config.batch_size)
        .convert_file_path(temp_file.path(), &metrics_config.output)?;

    log::info!(
        "Successfully wrote metrics snapshots to {:?} (parquet format)",
        metrics_config.output
    );

    Ok(())
}

/// The msgpack-to-parquet converter for the file written at exit: standard
/// histograms, `batch_size` rows per batch when given, and llm-perf's KPI
/// template ([`crate::template`]) as the file's `service_queries` metadata,
/// which the Rezolus viewer renders as llm-perf's KPI section.
fn parquet_converter(batch_size: Option<usize>) -> MsgpackToParquet {
    let mut opts = ParquetOptions::new().histogram_type(ParquetHistogramType::Standard);
    if let Some(batch_size) = batch_size {
        opts = opts.max_batch_size(batch_size);
    }
    MsgpackToParquet::with_options(opts).metadata(
        crate::template::KEY_SERVICE_QUERIES.to_string(),
        crate::template::SERVICE_QUERIES.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parquet written at exit carries the KPI template in its file
    /// metadata.
    #[test]
    fn parquet_carries_the_kpi_template() {
        let dir = tempfile::tempdir().unwrap();
        let staged = dir.path().join("run.msgpack");
        let parquet = dir.path().join("run.parquet");
        let snapshotter = SnapshotterBuilder::new()
            .metadata("source".to_string(), env!("CARGO_PKG_NAME").to_string())
            .build();
        let mut bytes = Vec::new();
        for _ in 0..2 {
            bytes.extend(Snapshot::to_msgpack(&snapshotter.snapshot()).unwrap());
        }
        std::fs::write(&staged, bytes).unwrap();

        parquet_converter(None)
            .convert_file_path(&staged, &parquet)
            .unwrap();

        let file = std::fs::File::open(&parquet).unwrap();
        let reader =
            parquet::file::reader::SerializedFileReader::new(file).expect("a parquet file");
        let kv = parquet::file::reader::FileReader::metadata(&reader)
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default();
        let template = kv
            .iter()
            .find(|e| e.key == crate::template::KEY_SERVICE_QUERIES)
            .and_then(|e| e.value.as_deref());
        assert_eq!(template, Some(crate::template::SERVICE_QUERIES));
    }
}
