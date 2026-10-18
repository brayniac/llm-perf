//! llm-perf's KPI template: the `service_queries` metadata value that the
//! Rezolus viewer renders as llm-perf's KPI section.
//!
//! The template travels in the recording. `/metrics/stream` sends it in its
//! handshake metadata, which a stream recorder (`rezolus record` to
//! `.dendro`) copies into the source's metadata; the in-process recording
//! ([`crate::recording`]) writes it into llm-perf's own source's metadata;
//! and the parquet written at exit ([`crate::snapshot`]) carries it in the
//! file's metadata. The JSON is a
//! Rezolus `ServiceExtension` (`crates/dashboard/src/service_extension.rs`
//! in rezolus): a `service_name` and a list of KPIs, each with a `role`,
//! `title`, `query`, `type` and optional `unit_system`.
//!
//! This file is the source of truth for llm-perf's template. Rezolus keeps
//! a copy in its built-in registry (`crates/dashboard/templates/
//! llm-perf.json`) only as the fallback for recordings that carry no
//! template, such as those made before this one was sent, so the two do not
//! have to match.

/// The key the template is stored under, in both the handshake and the
/// parquet metadata.
pub const KEY_SERVICE_QUERIES: &str = "service_queries";

/// The template, verbatim from `templates/llm-perf.json`.
pub const SERVICE_QUERIES: &str = include_str!("../templates/llm-perf.json");

#[cfg(test)]
mod tests {
    use super::*;

    /// The unit systems the Rezolus viewer can scale: rezolus's
    /// `dashboard::plot::Unit` (`crates/dashboard/src/plot.rs`). The viewer
    /// scales a KPI only by one of these; with any other value it warns and
    /// renders the KPI as a plain count, and `rezolus recording annotate`
    /// refuses the template.
    const UNIT_SYSTEMS: [&str; 9] = [
        "time",
        "bytes",
        "bitrate",
        "datarate",
        "percentage",
        "frequency",
        "power",
        "count",
        "rate",
    ];

    fn template() -> serde_json::Value {
        serde_json::from_str(SERVICE_QUERIES).expect("the template is JSON")
    }

    /// The fields rezolus's `ServiceExtension` requires, with the types it
    /// reads them as.
    #[test]
    fn template_has_the_service_extension_shape() {
        let doc = template();
        assert_eq!(doc["service_name"], "llm-perf");
        let kpis = doc["kpis"].as_array().expect("kpis is an array");
        assert!(!kpis.is_empty());
        for kpi in kpis {
            for key in ["role", "title", "query", "type"] {
                assert!(kpi[key].is_string(), "KPI {kpi} has no string {key}");
            }
            if let Some(unit) = kpi.get("unit_system") {
                let unit = unit.as_str().expect("unit_system is a string");
                assert!(
                    UNIT_SYSTEMS.contains(&unit),
                    "KPI {} has unit_system {unit:?}, which the viewer does not know",
                    kpi["title"]
                );
            }
        }
    }

    /// Every metric a KPI query selects (a name followed by `{`) is one
    /// llm-perf registers, so a renamed metric fails here rather than leaving
    /// an empty chart.
    #[test]
    fn template_queries_name_registered_metrics() {
        let registered: std::collections::HashSet<String> = metriken::metrics()
            .iter()
            .map(|m| m.name().to_string())
            .collect();
        let doc = template();
        let mut checked = 0;
        for kpi in doc["kpis"].as_array().unwrap() {
            let query = kpi["query"].as_str().unwrap();
            for name in selected_metrics(query) {
                assert!(
                    registered.contains(name),
                    "KPI {} queries {name}, which llm-perf does not register",
                    kpi["title"]
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no metric names found in the queries");
    }

    /// Every selector names llm-perf's source: in a recording holding
    /// llm-perf and the server under test, a selector without it would also
    /// match the server's series.
    #[test]
    fn every_selector_names_llm_perfs_source() {
        let wanted = format!("source=\"{}\"", crate::recording::LLM_PERF_SOURCE);
        let doc = template();
        for kpi in doc["kpis"].as_array().unwrap() {
            let query = kpi["query"].as_str().unwrap();
            for (name, matchers) in selectors(query) {
                assert!(
                    matchers.split(',').any(|m| m.trim() == wanted),
                    "KPI {}: {name}{{{matchers}}} does not select {wanted}",
                    kpi["title"]
                );
            }
        }
    }

    /// Every label a selector matches on, other than `source`, is one the
    /// metric's slots carry after `Metrics::init`, with that value, so a
    /// renamed status or direction fails here rather than leaving an empty
    /// chart. A matcher on any metric not listed here fails too, so the list
    /// grows with the template.
    #[test]
    fn selector_labels_are_ones_the_metrics_carry() {
        use crate::metrics::{Metrics, REQUESTS, TOKENS};
        Metrics::init();
        let groups: [(&str, &metriken::CounterGroup); 2] =
            [("requests", &REQUESTS), ("tokens", &TOKENS)];
        let doc = template();
        let mut checked = 0;
        for kpi in doc["kpis"].as_array().unwrap() {
            let query = kpi["query"].as_str().unwrap();
            for (name, matchers) in selectors(query) {
                for matcher in matchers.split(',').map(str::trim).filter(|m| !m.is_empty()) {
                    let (key, value) = matcher
                        .split_once('=')
                        .unwrap_or_else(|| panic!("{matcher} is not key=\"value\""));
                    assert!(
                        !key.ends_with('!') && !value.starts_with('~'),
                        "{matcher}: only equality matchers are checked"
                    );
                    if key == "source" {
                        continue;
                    }
                    let value = value.trim_matches('"');
                    let (_, group) = groups
                        .iter()
                        .find(|(n, _)| *n == name)
                        .unwrap_or_else(|| panic!("{name}{{{matchers}}}: {name} is not checked"));
                    assert!(
                        (0..group.entries()).any(|idx| group
                            .load_metadata(idx)
                            .is_some_and(|m| m.get(key).map(String::as_str) == Some(value))),
                        "KPI {}: no {name} slot carries {key}={value:?}",
                        kpi["title"]
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no label matchers found in the queries");
    }

    /// The selectors in `query`: each metric name directly followed by `{`,
    /// with the text between the braces.
    fn selectors(query: &str) -> Vec<(&str, &str)> {
        let bytes = query.as_bytes();
        let mut out = Vec::new();
        for (i, b) in bytes.iter().enumerate() {
            if *b != b'{' {
                continue;
            }
            let start = bytes[..i]
                .iter()
                .rposition(|c| !(c.is_ascii_alphanumeric() || *c == b'_' || *c == b':'))
                .map_or(0, |p| p + 1);
            let end = query[i..].find('}').map_or(query.len(), |e| i + e);
            if start < i {
                out.push((&query[start..i], &query[i + 1..end]));
            }
        }
        out
    }

    /// The metric names in `query` that are directly followed by `{`.
    fn selected_metrics(query: &str) -> Vec<&str> {
        selectors(query).into_iter().map(|(name, _)| name).collect()
    }

    #[test]
    fn selectors_read_names_and_matchers() {
        assert_eq!(
            selectors("sum(irate(tokens{direction=\"input\",source=\"x\"}[5s])) / up{}"),
            [("tokens", "direction=\"input\",source=\"x\""), ("up", "")]
        );
    }

    #[test]
    fn selected_metrics_reads_names_before_a_selector() {
        assert_eq!(
            selected_metrics("sum(irate(requests_sent{source=\"x\"}[5s])) / get_count{}"),
            ["requests_sent", "get_count"]
        );
    }
}
