//! Stable, nested session sampling.

use anyhow::{Context, Result, bail};

/// Whether to replay the session with id `session_id` at fraction `sample`.
///
/// The first 16 hex digits of the id, XORed with `sample_seed` multiplied by
/// an odd constant, are compared with `sample * 2^64`. The multiplication
/// spreads a small seed across all 64 bits, so seeds 1 and 2 select different
/// samples, usually disjoint at small fractions; seed 0 leaves the id
/// unchanged. For a fixed seed the choice is the
/// same on every run, and every session selected at one fraction is also
/// selected at any larger fraction. `sample = 1` selects every session.
pub fn selected(session_id: &str, sample: f64, sample_seed: u64) -> Result<bool> {
    let Some(prefix) = session_id.get(..16) else {
        bail!("session_id {session_id:?} is shorter than 16 hex digits");
    };
    let v = u64::from_str_radix(prefix, 16)
        .with_context(|| format!("session_id {session_id:?} does not start with 16 hex digits"))?;
    if sample >= 1.0 {
        return Ok(true);
    }
    // sample < 1 here, so sample * 2^64 < 2^64 and the cast does not saturate.
    let threshold = (sample * 2f64.powi(64)) as u64;
    let key = sample_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Ok((v ^ key) <= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u64) -> Vec<String> {
        // splitmix64 of the index: uniform ids unrelated to the seed constant.
        (0..n)
            .map(|i| {
                let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                format!("{:016x}{:016x}", z ^ (z >> 31), i)
            })
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
    fn small_seeds_select_mostly_different_samples_of_the_same_size() {
        let ids = ids(20_000);
        let at = |seed: u64| -> Vec<bool> {
            ids.iter()
                .map(|i| selected(i, 0.1, seed).unwrap())
                .collect()
        };
        let (a, b) = (at(1), at(2));
        let (na, nb) = (
            a.iter().filter(|x| **x).count(),
            b.iter().filter(|x| **x).count(),
        );
        let both = a.iter().zip(&b).filter(|(x, y)| **x && **y).count();
        // XOR selects a block of ids around the key, so samples for keys that
        // differ in their high bits are disjoint. Seeds 1 and 2 must at least
        // not select the same sessions.
        assert!(both < na / 2, "overlap {both} of {na}");
        assert!((na as i64 - nb as i64).abs() < 300, "{na} {nb}");
    }

    #[test]
    fn non_hex_or_short_ids_are_errors() {
        assert!(selected("not-hex-at-all-xxxxxxxx", 0.5, 0).is_err());
        assert!(selected("abc", 0.5, 0).is_err());
    }
}
