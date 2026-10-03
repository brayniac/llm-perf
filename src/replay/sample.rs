//! Stable, nested session sampling.

use anyhow::{Context, Result, bail};

/// Whether to replay the session with id `session_id` at fraction `sample`.
///
/// The first 16 hex digits of the id, XORed with `sample_seed`, are compared
/// with `sample * 2^64`. For a fixed seed the choice is the same on every run,
/// and every session selected at one fraction is also selected at any larger
/// fraction. `sample = 1` selects every session.
pub fn selected(session_id: &str, sample: f64, sample_seed: u64) -> Result<bool> {
    let Some(prefix) = session_id.get(..16) else {
        bail!("session_id {session_id:?} is shorter than 16 hex digits");
    };
    let v = u64::from_str_radix(prefix, 16)
        .with_context(|| format!("session_id {session_id:?} does not start with 16 hex digits"))?;
    if sample >= 1.0 {
        return Ok(true);
    }
    // `as` saturates, so fractions just below 1 map to u64::MAX - ε, not 0.
    let threshold = (sample * 2f64.powi(64)) as u64;
    Ok((v ^ sample_seed) <= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u64) -> Vec<String> {
        // Spread ids across the u64 range with a fixed odd multiplier.
        (0..n)
            .map(|i| format!("{:016x}{:016x}", i.wrapping_mul(0x9E37_79B9_7F4A_7C15), i))
            .collect()
    }

    #[test]
    fn sample_one_selects_everything() {
        assert!(selected("ffffffffffffffff0000000000000000", 1.0, 0).unwrap());
        assert!(selected("ffffffffffffffff0000000000000000", 1.0, 7).unwrap());
    }

    #[test]
    fn smaller_samples_are_subsets_of_larger_ones() {
        let ids = ids(20_000);
        let at = |f: f64| -> Vec<bool> { ids.iter().map(|i| selected(i, f, 3).unwrap()).collect() };
        let small = at(0.05);
        let large = at(0.10);
        assert!(small.iter().zip(&large).all(|(s, l)| !s || *l));
        let n_small = small.iter().filter(|s| **s).count();
        let n_large = large.iter().filter(|s| **s).count();
        // Binomial sd at n = 20,000: ~31 for 5%, ~42 for 10%.
        assert!((n_small as i64 - 1000).abs() < 150, "{n_small}");
        assert!((n_large as i64 - 2000).abs() < 200, "{n_large}");
    }

    #[test]
    fn seed_changes_the_sample_not_its_size() {
        let ids = ids(20_000);
        let a: Vec<bool> = ids.iter().map(|i| selected(i, 0.1, 0).unwrap()).collect();
        let b: Vec<bool> = ids
            .iter()
            .map(|i| selected(i, 0.1, 0xdead_beef_0000_0000).unwrap())
            .collect();
        assert_ne!(a, b);
        let (na, nb) = (
            a.iter().filter(|x| **x).count(),
            b.iter().filter(|x| **x).count(),
        );
        assert!((na as i64 - nb as i64).abs() < 300, "{na} {nb}");
    }

    #[test]
    fn non_hex_or_short_ids_are_errors() {
        assert!(selected("not-hex-at-all-xxxxxxxx", 0.5, 0).is_err());
        assert!(selected("abc", 0.5, 0).is_err());
    }
}
