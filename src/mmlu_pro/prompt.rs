use crate::client::Message;

use super::dataset::Question;

const CHOICE_MAP: &[char] = &['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];

/// Format a question and its options into the expected text format.
/// Returns (formatted_question, cot_content).
fn format_example(question: &str, options: &[String], cot_content: &str) -> (String, String) {
    let cot = if cot_content.is_empty() {
        "Let's think step by step.".to_string()
    } else if let Some(stripped) = cot_content.strip_prefix("A: ") {
        stripped.trim().to_string()
    } else {
        cot_content.trim().to_string()
    };

    let mut example = format!("Question: {}\nOptions: ", question);
    for (i, opt) in options.iter().enumerate() {
        if i < CHOICE_MAP.len() {
            example.push_str(&format!("{}. {}\n", CHOICE_MAP[i], opt));
        }
    }

    (example.trim().to_string(), cot)
}

/// Build multi-chat prompt messages for a question.
///
/// Structure:
/// - System message with {subject} substituted
/// - For each CoT example: user message (question + options) → assistant message (Answer: cot)
/// - Final user message with the test question
pub fn build_messages(
    system_prompt: &str,
    cot_examples: &[Question],
    question: &str,
    options: &[String],
) -> Vec<Message> {
    let mut messages = Vec::new();

    // System message
    messages.push(Message {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    });

    // CoT examples as multi-turn conversation
    for example in cot_examples {
        let (formatted, cot_content) =
            format_example(&example.question, &example.options, &example.cot_content);
        messages.push(Message {
            role: "user".to_string(),
            content: formatted,
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: format!("Answer: {}", cot_content),
        });
    }

    // Test question
    let (formatted, _) = format_example(question, options, "");
    messages.push(Message {
        role: "user".to_string(),
        content: formatted,
    });

    messages
}

/// Separator between the header and the first question in completion mode.
///
/// Upstream reads the header from `cot_prompt_lib/initial_prompt.txt`, which
/// ends in three newlines, and then appends one more.
const COMPLETION_HEADER_SEPARATOR: &str = "\n\n\n\n";

/// Prefix of every `cot_content` in the MMLU-Pro validation split.
const COT_PREFIX: &str = "A: Let's think step by step.";

/// What the test question ends with; the model continues from here.
const COMPLETION_ANSWER_CUE: &str = "Answer: Let's think step by step.";

/// Format one question in the completion-mode layout, without an answer:
/// `Question:\n{question}\nOptions:\n{A. opt\n...}`.
fn format_completion_question(question: &str, options: &[String]) -> String {
    let mut out = format!("Question:\n{}\nOptions:\n", question);
    for (letter, opt) in CHOICE_MAP.iter().zip(options) {
        out.push_str(&format!("{}. {}\n", letter, opt));
    }
    out
}

/// Build the single plain-text prompt used in completion mode.
///
/// This reproduces `generate_cot_prompt` / `format_cot_example` from
/// TIGER-Lab's `evaluate_from_local.py`, which is the script used for
/// published base-model MMLU-Pro numbers:
///
/// - `header` (the system prompt with `{subject}` already substituted),
///   followed by four newlines
/// - for each shot: the question block, then the shot's `cot_content` with
///   `"A: Let's think step by step."` replaced by
///   `"Answer: Let's think step by step."`, then a blank line
/// - the test question block, ending with `"Answer: Let's think step by step."`
///   and no trailing newline
///
/// Unlike [`build_messages`], the `cot_content` is not trimmed, because
/// upstream does not trim it.
pub fn build_completion_prompt(
    header: &str,
    cot_examples: &[Question],
    question: &str,
    options: &[String],
) -> String {
    let mut prompt = String::new();
    prompt.push_str(header);
    prompt.push_str(COMPLETION_HEADER_SEPARATOR);

    for example in cot_examples {
        prompt.push_str(&format_completion_question(
            &example.question,
            &example.options,
        ));
        prompt.push_str(
            &example
                .cot_content
                .replace(COT_PREFIX, COMPLETION_ANSWER_CUE),
        );
        prompt.push_str("\n\n");
    }

    prompt.push_str(&format_completion_question(question, options));
    prompt.push_str(COMPLETION_ANSWER_CUE);
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_example_basic() {
        let options = vec!["Yes".to_string(), "No".to_string()];
        let (example, cot) =
            format_example("Is the sky blue?", &options, "Let's think step by step.");
        assert!(example.contains("Question: Is the sky blue?"));
        assert!(example.contains("A. Yes"));
        assert!(example.contains("B. No"));
        assert_eq!(cot, "Let's think step by step.");
    }

    #[test]
    fn test_format_example_strips_a_prefix() {
        let options = vec!["opt1".to_string()];
        let (_, cot) = format_example("Q?", &options, "A: The answer is...");
        assert_eq!(cot, "The answer is...");
    }

    #[test]
    fn test_build_messages_structure() {
        let examples = vec![Question {
            question_id: 1,
            question: "Example Q?".to_string(),
            options: vec!["A".to_string(), "B".to_string()],
            answer: "A".to_string(),
            answer_index: 0,
            cot_content: "Let's think...".to_string(),
            category: "math".to_string(),
        }];

        let messages = build_messages(
            "System prompt about {subject}",
            &examples,
            "Test Q?",
            &["X".to_string(), "Y".to_string()],
        );

        assert_eq!(messages.len(), 4); // system + user + assistant + user
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
        assert!(messages[2].content.starts_with("Answer: "));
        assert_eq!(messages[3].role, "user");
    }

    fn shot(id: i64, question: &str, options: &[&str], cot: &str) -> Question {
        Question {
            question_id: id,
            question: question.to_string(),
            options: options.iter().map(|o| o.to_string()).collect(),
            answer: "A".to_string(),
            answer_index: 0,
            cot_content: cot.to_string(),
            category: "math".to_string(),
        }
    }

    const HEADER: &str = "The following are multiple choice questions (with answers) about \
                          math. Think step by step and then finish your answer with \
                          \"the answer is (X)\" where X is the correct letter choice.";

    #[test]
    fn completion_prompt_matches_reference_layout_exactly() {
        let shots = vec![shot(
            1,
            "What is 1+1?",
            &["2", "3"],
            "A: Let's think step by step. 1+1 is 2. The answer is (A).",
        )];
        let prompt = build_completion_prompt(
            HEADER,
            &shots,
            "What is 2+2?",
            &["3".to_string(), "4".to_string(), "5".to_string()],
        );
        let expected = format!(
            "{HEADER}\n\n\n\n\
             Question:\nWhat is 1+1?\nOptions:\nA. 2\nB. 3\n\
             Answer: Let's think step by step. 1+1 is 2. The answer is (A).\n\n\
             Question:\nWhat is 2+2?\nOptions:\nA. 3\nB. 4\nC. 5\n\
             Answer: Let's think step by step."
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn completion_prompt_starts_with_header() {
        let prompt = build_completion_prompt(HEADER, &[], "Q?", &["x".to_string()]);
        assert!(prompt.starts_with(&format!("{HEADER}\n\n\n\nQuestion:\n")));
    }

    #[test]
    fn completion_prompt_has_one_block_per_shot_plus_test_question() {
        let shots: Vec<Question> = (0..5)
            .map(|i| {
                shot(
                    i,
                    &format!("Shot {i}?"),
                    &["a", "b"],
                    "A: Let's think step by step. The answer is (A).",
                )
            })
            .collect();
        let prompt = build_completion_prompt(HEADER, &shots, "Test?", &["a".to_string()]);

        assert_eq!(prompt.matches("Question:\n").count(), 6);
        assert_eq!(prompt.matches("Options:\n").count(), 6);
        assert_eq!(
            prompt.matches("Answer: Let's think step by step.").count(),
            6
        );
        // The dataset's "A: " prefix is replaced, never left in place.
        assert!(!prompt.contains("A: Let's think"));
        // Shots appear in the given order, before the test question.
        let first = prompt.find("Shot 0?").unwrap();
        let last = prompt.find("Shot 4?").unwrap();
        let test = prompt.find("Test?").unwrap();
        assert!(first < last && last < test);
    }

    #[test]
    fn completion_prompt_zero_shot_is_header_then_test_question() {
        let prompt = build_completion_prompt(HEADER, &[], "Q?", &["x".to_string()]);
        assert_eq!(
            prompt,
            format!(
                "{HEADER}\n\n\n\nQuestion:\nQ?\nOptions:\nA. x\n\
                 Answer: Let's think step by step."
            )
        );
    }

    #[test]
    fn completion_prompt_ends_with_answer_cue() {
        let shots = vec![shot(1, "Q1?", &["a"], "A: Let's think step by step. X.")];
        let prompt = build_completion_prompt(HEADER, &shots, "Q2?", &["b".to_string()]);
        assert!(prompt.ends_with("\nAnswer: Let's think step by step."));
        assert!(!prompt.ends_with('\n'));
    }

    #[test]
    fn completion_prompt_keeps_cot_without_the_standard_prefix_unchanged() {
        // Upstream only replaces the exact "A: Let's think step by step." text.
        let shots = vec![shot(1, "Q1?", &["a"], "Some other reasoning.")];
        let prompt = build_completion_prompt(HEADER, &shots, "Q2?", &["b".to_string()]);
        assert!(prompt.contains("A. a\nSome other reasoning.\n\nQuestion:\nQ2?"));
    }
}
