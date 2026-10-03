//! Scaled source timeline for a session's calls.

use crate::trace::TraceCall;
use std::time::Duration;

/// When each call of a session would start on the source timeline, scaled.
#[derive(Debug, Clone, PartialEq)]
pub struct CallSlot {
    /// Offset from the session's first call.
    pub offset: Duration,
    /// The gap before this call was shortened by `max_gap`.
    pub gap_capped: bool,
}

/// Offsets of `calls` from the session's first call: each call starts after
/// the previous call's `duration_ms` and its own `gap_ms`, both divided by
/// `speedup`, with the scaled gap capped at `max_gap`.
pub fn session_slots(
    calls: &[TraceCall],
    speedup: f64,
    max_gap: Option<Duration>,
) -> Vec<CallSlot> {
    let scale = |ms: u64| Duration::from_secs_f64(ms as f64 / 1000.0 / speedup);
    let mut slots = Vec::with_capacity(calls.len());
    let mut offset = Duration::ZERO;
    for (i, c) in calls.iter().enumerate() {
        let mut gap_capped = false;
        if i > 0 {
            let mut gap = scale(c.gap_ms);
            if let Some(max) = max_gap
                && gap > max
            {
                gap = max;
                gap_capped = true;
            }
            offset += scale(calls[i - 1].duration_ms) + gap;
        }
        slots.push(CallSlot { offset, gap_capped });
    }
    slots
}

/// Offset of a session's first call from the run start: `start_ms - t0`
/// divided by `speedup`.
pub fn session_start(start_ms: u64, t0_ms: u64, speedup: f64) -> Duration {
    Duration::from_secs_f64(start_ms.saturating_sub(t0_ms) as f64 / 1000.0 / speedup)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(gap_ms: u64, duration_ms: u64) -> TraceCall {
        TraceCall {
            turn: 0,
            initiator: None,
            model: None,
            gap_ms,
            prompt: 100,
            completion: 10,
            cached: 0,
            reuse: 0,
            reuse_inferred: false,
            duration_ms,
        }
    }

    #[test]
    fn offsets_accumulate_scaled_durations_and_gaps() {
        let calls = [call(0, 4000), call(2000, 1000), call(600_000, 500)];
        let slots = session_slots(&calls, 2.0, None);
        let ms: Vec<u128> = slots.iter().map(|s| s.offset.as_millis()).collect();
        // (4000 + 2000) / 2 = 3000; then + (1000 + 600000) / 2 = 300500.
        assert_eq!(ms, vec![0, 3000, 303_500]);
        assert!(slots.iter().all(|s| !s.gap_capped));
    }

    #[test]
    fn max_gap_caps_the_scaled_gap_only() {
        let calls = [call(0, 4000), call(600_000, 1000)];
        let slots = session_slots(&calls, 2.0, Some(Duration::from_secs(60)));
        // 4000 / 2 + min(600000 / 2, 60000) = 62000.
        assert_eq!(slots[1].offset, Duration::from_millis(62_000));
        assert!(slots[1].gap_capped);
        assert!(!slots[0].gap_capped);
    }

    #[test]
    fn session_start_divides_by_speedup() {
        assert_eq!(session_start(3_600_000, 0, 24.0), Duration::from_secs(150));
        assert_eq!(session_start(10, 20, 1.0), Duration::ZERO);
    }
}
