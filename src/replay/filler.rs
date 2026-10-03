//! Filler text whose length in tokens is known.

use super::render::Renderer;
use anyhow::{Result, bail};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Candidate words: lowercase dictionary words that tokenized as one
/// space-prefixed token under both Llama 3.1 8B and Qwen3.5 9B. The pool for
/// a run is re-checked against the server's tokenizer.
const CANDIDATES: &str = include_str!("words.txt");

/// Fewest single-token words a pool needs.
const MIN_POOL: usize = 256;

/// Words checked as a block at startup.
const CHECK_WORDS: usize = 10_000;

/// Space-prefixed words that are each one token under the server's tokenizer.
#[derive(Debug, Clone)]
pub struct FillerPool {
    words: Vec<String>,
}

impl FillerPool {
    /// Build the pool from the embedded candidates: keep each word that
    /// tokenizes as a single space-prefixed token in a block of all
    /// candidates, then check that a block of 10,000 drawn words is 10,000
    /// tokens.
    pub async fn build<R: Renderer>(renderer: &R) -> Result<Self> {
        let candidates: Vec<&str> = CANDIDATES.lines().filter(|w| !w.is_empty()).collect();
        let text: String = candidates.iter().map(|w| format!(" {w}")).collect();
        let pieces = renderer.tokenize_pieces(&text).await?;

        // Group pieces into words: a word starts at each piece beginning with
        // a space.
        let mut groups: Vec<Vec<&str>> = Vec::new();
        for p in &pieces {
            let piece = p.piece.as_deref().unwrap_or("");
            if piece.starts_with(' ') || groups.is_empty() {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(piece);
        }
        let words: Vec<String> = groups
            .into_iter()
            .filter(|g| g.len() == 1 && g[0].len() > 1)
            .map(|g| g[0].to_string())
            .collect();
        if words.len() < MIN_POOL {
            bail!(
                "only {} of {} filler candidates are single tokens under the server's tokenizer; need {MIN_POOL}",
                words.len(),
                candidates.len()
            );
        }

        let pool = Self { words };
        let mut rng = StdRng::seed_from_u64(0);
        let sample = pool.text(&mut rng, CHECK_WORDS);
        let n = renderer.tokenize(&sample).await?.len();
        if n != CHECK_WORDS {
            bail!(
                "{CHECK_WORDS} filler words tokenized to {n} tokens; filler sizes would be wrong"
            );
        }
        Ok(pool)
    }

    #[cfg(test)]
    pub fn from_words(words: &[&str]) -> Self {
        Self {
            words: words.iter().map(|w| format!(" {w}")).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// `n` words drawn from the pool, each with its leading space.
    pub fn text(&self, rng: &mut StdRng, n: usize) -> String {
        let mut s = String::with_capacity(n * 8);
        for _ in 0..n {
            s.push_str(&self.words[rng.gen_range(0..self.words.len())]);
        }
        s
    }
}

/// Generator for one call's filler, fixed by the run seed, the session and the
/// call index. Hashing with FNV-1a rather than `DefaultHasher` gives the same
/// seed on every Rust release. `StdRng`'s algorithm can change with a `rand`
/// upgrade, which would change the filler.
pub fn call_rng(seed: u64, session_id: &str, call: usize) -> StdRng {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let bytes = seed
        .to_le_bytes()
        .into_iter()
        .chain(session_id.bytes())
        .chain((call as u64).to_le_bytes());
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    StdRng::seed_from_u64(h)
}

#[cfg(test)]
mod tests {
    use super::super::render::fake::FakeRenderer;
    use super::*;

    #[tokio::test]
    async fn pool_from_embedded_candidates_passes_its_check() {
        let r = FakeRenderer::default();
        let pool = FillerPool::build(&r).await.unwrap();
        assert_eq!(pool.len(), CANDIDATES.lines().count());
    }

    #[test]
    fn call_rng_is_deterministic_and_varies_by_call() {
        let pool = FillerPool::from_words(&["a", "b", "c", "d", "e", "f", "g", "h"]);
        let t = |call| pool.text(&mut call_rng(1, "s", call), 32);
        assert_eq!(t(0), t(0));
        assert_ne!(t(0), t(1));
    }
}
