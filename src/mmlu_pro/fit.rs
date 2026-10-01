//! Per-question few-shot reduction so a prompt fits a context window.
//!
//! This follows `eval_cot` in TIGER-Lab's `evaluate_from_local.py`: start at
//! `num_shots`, and while the prompt plus the generation budget does not fit,
//! drop one shot and rebuild. Shots are the first `k` validation examples for
//! the category, so lowering `k` drops the last shot first.
//!
//! The caller's token count is exact: it is llama-server's own tokenization of
//! the prompt the generation request will carry, including BOS and, in chat
//! mode, the rendered chat template. That is why a prompt that fills the
//! window exactly (`<=`) is accepted.
//!
//! Differences from upstream:
//! - Upstream keeps a prompt when `prompt_tokens < context - max_tokens`
//!   (strict). Here it is kept when `prompt_tokens + max_tokens <= context`.
//! - Upstream has no lower bound on `k`; once it reaches 0 the next step slices
//!   `val_df[:-1]`. Here the search stops at 0 shots and reports
//!   [`ShotFit::TooLong`] if even that does not fit.
//! - Upstream counts with the Hugging Face tokenizer in-process; here the
//!   server under test does the counting, so the count uses the model's GGUF
//!   tokenizer, which can differ from the Hugging Face one.

use anyhow::Result;
use std::future::Future;

/// Outcome of fitting a question's prompt into the context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotFit {
    /// The prompt with `shots` examples fits.
    Fits { shots: usize, prompt_tokens: usize },
    /// The prompt does not fit even with 0 shots.
    TooLong { zero_shot_tokens: usize },
}

/// Find the largest shot count `k <= num_shots` for which
/// `count_tokens(k) + max_tokens <= max_context_tokens`.
///
/// `count_tokens(k)` must build the prompt with the first `k` shots and return
/// its token count. It is called once per attempt, from `num_shots` down.
pub async fn fit_shots<F, Fut>(
    num_shots: usize,
    max_tokens: u32,
    max_context_tokens: u32,
    mut count_tokens: F,
) -> Result<ShotFit>
where
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = Result<usize>>,
{
    let budget = max_context_tokens as usize;
    let max_tokens = max_tokens as usize;
    let mut k = num_shots;
    loop {
        let prompt_tokens = count_tokens(k).await?;
        if prompt_tokens + max_tokens <= budget {
            return Ok(ShotFit::Fits {
                shots: k,
                prompt_tokens,
            });
        }
        if k == 0 {
            return Ok(ShotFit::TooLong {
                zero_shot_tokens: prompt_tokens,
            });
        }
        k -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Fake tokenizer: `base` tokens for the header and test question plus
    /// `per_shot` tokens per example. Records the shot counts it was asked for.
    fn fake(
        base: usize,
        per_shot: usize,
        calls: &RefCell<Vec<usize>>,
    ) -> impl FnMut(usize) -> std::future::Ready<Result<usize>> + '_ {
        move |k| {
            calls.borrow_mut().push(k);
            std::future::ready(Ok(base + per_shot * k))
        }
    }

    #[tokio::test]
    async fn keeps_all_shots_when_they_fit() {
        let calls = RefCell::new(Vec::new());
        let fit = fit_shots(5, 100, 2048, fake(200, 300, &calls))
            .await
            .unwrap();
        assert_eq!(
            fit,
            ShotFit::Fits {
                shots: 5,
                prompt_tokens: 1700
            }
        );
        assert_eq!(*calls.borrow(), vec![5]);
    }

    #[tokio::test]
    async fn drops_one_shot_at_a_time_until_it_fits() {
        // With max_tokens 512 and a 2048 window the prompt must be <= 1536
        // tokens: 200 + 500k <= 1536 first holds at k = 2.
        let calls = RefCell::new(Vec::new());
        let fit = fit_shots(5, 512, 2048, fake(200, 500, &calls))
            .await
            .unwrap();
        assert_eq!(
            fit,
            ShotFit::Fits {
                shots: 2,
                prompt_tokens: 1200
            }
        );
        assert_eq!(*calls.borrow(), vec![5, 4, 3, 2]);
    }

    #[tokio::test]
    async fn exact_fill_is_accepted() {
        let calls = RefCell::new(Vec::new());
        // 100 + 100*3 = 400, plus 100 generation = 500 = window.
        let fit = fit_shots(3, 100, 500, fake(100, 100, &calls))
            .await
            .unwrap();
        assert_eq!(
            fit,
            ShotFit::Fits {
                shots: 3,
                prompt_tokens: 400
            }
        );
    }

    #[tokio::test]
    async fn falls_back_to_zero_shot() {
        let calls = RefCell::new(Vec::new());
        let fit = fit_shots(5, 1000, 2048, fake(1000, 2000, &calls))
            .await
            .unwrap();
        assert_eq!(
            fit,
            ShotFit::Fits {
                shots: 0,
                prompt_tokens: 1000
            }
        );
        assert_eq!(*calls.borrow(), vec![5, 4, 3, 2, 1, 0]);
    }

    #[tokio::test]
    async fn reports_too_long_when_zero_shots_do_not_fit() {
        let calls = RefCell::new(Vec::new());
        let fit = fit_shots(2, 1000, 2048, fake(1500, 10, &calls))
            .await
            .unwrap();
        assert_eq!(
            fit,
            ShotFit::TooLong {
                zero_shot_tokens: 1500
            }
        );
        // Stops at 0; never asks for a negative shot count.
        assert_eq!(*calls.borrow(), vec![2, 1, 0]);
    }

    #[tokio::test]
    async fn tokenizer_error_is_returned() {
        let fit = fit_shots(3, 10, 100, |_| {
            std::future::ready(Err::<usize, _>(anyhow::anyhow!("no /tokenize")))
        })
        .await;
        assert!(fit.is_err());
    }
}
