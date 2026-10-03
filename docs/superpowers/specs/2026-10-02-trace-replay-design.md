# Trace Replay Design

**Date:** 2026-10-02
**Status:** Draft

## Goal

Replay the per-session traces written by `llm-perf convert-trace` against an
OpenAI-compatible server, with the trace's prompt lengths, reuse, completion
lengths, gaps and session arrival times. The replay measures how many of each
call's reusable prompt tokens the server serves from its prefix cache, by
comparing the reuse each call is built to have with the cached tokens the
server reports.

Trace field meanings are in README.md under `convert-trace` and in the
`TraceCall` docs in `src/trace.rs`. Trace token counts are used as
target-model token counts; the source's models and tokenizers are unknown.

First target: llama-server (llama.cpp), with Llama 3.1 8B as the
dense-attention control and Qwen3.5 9B as the realistic target.

Out of scope for this spec: sending token IDs instead of text, vLLM and SGLang
specifics, tool-call message shapes, and multimodal content.

---

## Capture findings this design depends on

Captured 2026-10-02 against llama-server built from upstream `4ebdf2c`, with
`-c 65536 -np 2`. Fixtures and the capture script are in
`tests/fixtures/llama-server/`. Shared prefixes were measured by rendering both
requests with `/apply-template` and tokenizing them.

| Model (GGUF q8_0) | Call 2: previous prompt + reply + new message | Call 3: as call 2, new message cut in half |
|---|---|---|
| Llama 3.1 8B Instruct | prompt 5,980, cached 4,888 (= 4,841 prompt + 47 of 48 reply tokens) | prompt 5,440, shared prefix 5,435, cached 5,435 |
| Qwen3.5 9B | prompt 6,266, shared prefix 5,111, cached 5,111 | prompt 5,705, shared prefix 5,696, cached 5,161 |

- llama-server fills `usage.prompt_tokens_details.cached_tokens` in
  non-streamed responses and in the final streamed chunk when
  `stream_options.include_usage` is set. The client already parses it.
  `timings.cache_n` carries the same value.
- On Llama 3.1 8B, a reply sent back as an assistant message is reused except
  for its last generated token, whose KV is computed by the next request.
  Keeping 0, 20 or all 64 tokens of a reply gave cached = previous prompt + 0,
  + 20 and + 63.
- Qwen3.5's template ends the generation prompt with an empty think block
  (`assistant\n<think>\n\n</think>\n\n`) but renders a past assistant turn as
  `assistant\n` followed by the content. The previous prompt and the next one
  diverge 4 tokens before the previous prompt's end, so no reply token is ever
  in the shared prefix.
- `qwen35` is a hybrid recurrent architecture in llama.cpp. The server resumes
  a cached prompt only from a context checkpoint: one at the start of each
  user message (spaced by at least `--checkpoint-min-step`, default 8192,
  except the last), and ones 4 + n_ubatch and 4 tokens before the end of each
  prompt (`--ctx-checkpoints`, default 32 per slot). In call 3, 5,161 is the
  start of call 2's last user message.
- With thinking enabled, Qwen3.5 returned `content: ""` and all text in
  `reasoning_content` at `max_tokens` 1, 16 and 48. Its template drops
  `reasoning_content` from assistant turns before the last user message.
- Both templates trim leading whitespace from message content, so a message's
  first word tokenizes differently from the same word mid-message.
- Llama 3.1's template inserts the current date (`Today Date: ...`) unless
  `chat_template_kwargs.date_string` is set.
- `/tokenize` (with `with_pieces`) and `/detokenize` are served at the server
  root. Single-token, round-trip-stable filler words exist in quantity: 3,311
  for Qwen3.5 and 3,071 for Llama 3.1 from a dictionary word list, and 10,000
  random pool words tokenized to exactly 10,000 tokens for every seed tried.
- From the server log: with `-np 2` given explicitly, `kv_unified = 'false'`
  and each slot holds `-c / -np` tokens. The default `--cache-ram` is 8192 MiB,
  which holds about 65,536 tokens of Llama 3.1 8B KV (128 KiB per token).

## Trace facts this design depends on

From the 2026-06-03 partition converted with `--from`/`--to` for that day
(54,008 sessions, 1,736,534 calls):

| Per-slot context | Sessions that fit | Calls in those sessions |
|---|---|---|
| 32,768 | 32.3% | 10.1% |
| 65,536 | 53.4% | 27.4% |
| 131,072 | 83.9% | 64.1% |
| 262,144 | 98.8% | 97.7% |

- 211,167 calls (12% of non-first calls) have `prompt == reuse`: no new tokens.
- 553,126 calls (32%) have `reuse` extending into the previous completion.
- 77.8% of gaps are shorter than the previous call's source duration; divided
  by 24, 95.2% are.
- 3,777 sessions (7%) use more than one model label.

---

## Config

A top-level `[replay]` section selects replay. With it, `[input]` is rejected
and `[load]` is optional; of `[load]`, only `duration_seconds` and
`warmup_duration` are accepted. `Config.input` is an `Option<InputConfig>`, and
`Config.load` takes its defaults when the section is absent. Because `[load]`'s
fields have serde defaults, `Config::from_toml` checks the keys present in the
raw `[load]` table rather than the deserialized values. `Config::validate`
requires `total_requests` or `duration_seconds` only when `[replay]` is absent.
`[saturation]`, `[logprobs]`, `[conversation]` and `[metrics]` are rejected
with `[replay]`.

```toml
[replay]
trace = "trace.jsonl"     # output of `llm-perf convert-trace`
sample = 0.01             # fraction of sessions to replay, in (0, 1]
sample_seed = 0           # selects a different sample of the same size
speedup = 24.0            # divides session start offsets, call durations and gaps; > 0
max_gap_ms = 300000       # optional cap on a gap after speedup; unset = no cap
seed = 1                  # filler text seed
system_prompt_tokens = 0  # optional system prompt shared by every session
log = "replay.jsonl"      # optional per-call log
```

Replay rejects `endpoint.max_tokens` and `endpoint.max_retries > 0` (a retry
resends a prompt the server may already have cached). Generation requests use
a client that neither retries nor reuses an idle connection: llama-server can
close a kept-alive connection just as the client reuses it, which fails the
request with "connection closed before message completed". Render, tokenize
and detokenize requests are idempotent and use a pooled client that retries.
Each call sends
`max_tokens = max(completion, 1)` and `ignore_eos = true` and logs
`finish_reason`; a call whose `completion_tokens < max_tokens` is counted as
EOS not ignored.

Replay requires reasoning output to be off
(`endpoint.chat_template_kwargs = { enable_thinking = false }` for Qwen3.5)
and sends back `content` only. For Llama 3.1, set
`chat_template_kwargs = { date_string = "26 Jul 2024" }` so the rendered
prompt does not change at midnight.

`system_prompt_tokens` defaults to 0. On 2026-06-03, 38.7% of sessions' first
calls report cached tokens at the source, with a median of 6,690 (p10 2,329,
p90 32,000). That spread suggests prefixes shared within a user or repository
rather than one global system prompt, which a single shared system prompt does
not reproduce.

### Server context

Runs use 131,072 tokens per slot for both models (Llama 3.1's maximum), with
the trace converted using `convert-trace --max-context 131072`, so the Llama
and Qwen runs replay the same sessions: 83.9% of sessions and 64.1% of calls on
2026-06-03. At startup the replay reads the per-slot context from `/props`
(`default_generation_settings.n_ctx`) and fails if any selected call's
`prompt + completion` exceeds it.

Sessions with several model labels are replayed against the one configured
model; the labels are anonymised.

### Session sampling

Session ids in the dataset are 32 hex digits. A session is replayed when
`(u64::from_str_radix(&session_id[..16], 16) ^ sample_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)) <= (sample * 2^64) as u64`,
with `sample = 1` selecting every session. For a fixed `sample_seed`, selection
is stable across runs and nested: every session in a 0.05 sample is also in
the 0.10 sample. An id that is not hex fails the run. The report gives the
selected count and fraction.

### Time

Let `t0` be the smallest `start_ms` among selected sessions. A session's first
call is scheduled at run time `(start_ms - t0) / speedup`. Call `n` is
scheduled at the session's start plus the sum, over the calls before it, of
`(duration_ms + gap_ms) / speedup` with each gap capped by `max_gap_ms`, and
also no earlier than the previous call's completion. `lag_ms` is the send time
minus that scaled source time.

When the server completes each call within the source's `duration_ms /
speedup`, a session follows the scaled source timeline, and the traffic between
two of its calls does not depend on `speedup`. When the server is slower, the
session falls behind, `lag_ms` grows, and cache pressure rises with `speedup`.
On 2026-06-03, 95.2% of gaps divided by 24 are shorter than the previous
call's source duration, so at 24x the server's latency decides which case a
run is in. Report `lag_ms` with every result.

`max_gap_ms` caps a gap after scaling. Capping shortens the time, and so the
traffic, between those two calls; calls after a capped gap are flagged
`gap_capped` in the log.

---

## Request construction

Requests go to `/v1/chat/completions`, streamed with
`stream_options.include_usage`.

Each session keeps the message list and rendered token sequence of its
previous call, and the tokenized reply. The rendered token sequence of a
message list is the `/apply-template` output for it (with
`endpoint.chat_template_kwargs`) tokenized with `add_special` and
`parse_special`.

For call `n`:

1. Start from call `n-1`'s message list followed by call `n-1`'s reply
   (`content`) as an assistant message.
2. Cut the list so its rendered sequence shares `reuse` tokens with call
   `n-1`'s rendered prompt followed by the tokenized reply. Whole messages past
   the cut are dropped. A message that straddles the cut is cut at a token
   boundary of its content. A cut that falls in template tokens
   between two messages keeps the earlier message whole and drops the later
   one.
3. Add filler so the rendered prompt reaches `prompt` tokens: on the end of
   the last kept message when it is a user message, otherwise in a new user
   message. The size is measured by rendering, and corrected once. A new
   message needs at least one filler word, so when the kept part plus one
   message already exceeds `prompt` (12% of non-first calls have
   `prompt == reuse`, and most of those end on the previous reply), the prompt
   overshoots and the call is flagged `overshoot` in the log.
4. Render and tokenize the result. `expected_reuse` is the length of its
   common token prefix with call `n-1`'s rendered prompt followed by the
   tokenized reply.

If `reuse` exceeds call `n-1`'s rendered prompt plus the tokenized reply, all
of it is kept and the difference is logged as `shortfall`. Re-tokenizing a
reply's text can give fewer tokens than were generated (91 for a 92-token
reply in one Llama 3.1 call), so a call that keeps the whole reply can show a
shortfall of a token or two.

After a session's first call, `expected_reuse` never falls below the
template's fixed preamble. On Llama 3.1
without a system message that is 30 tokens (BOS, the system header and its
date lines), so a call whose traced `reuse` is 0 still shares and has cached
those 30 tokens.

The first call of a session is the optional shared system prompt plus one user
message of filler. The system prompt is kept whole in every call.

Filler is deterministic for a given `seed`, so a second run against a server
that still holds the first run's prompts gets cache hits on first calls.
Restart the server or change `seed` between runs.

### Filler

At startup the replay builds a pool of filler words from an embedded list:
a candidate is kept when tokenizing the block of all candidates, each with a
leading space, gives it one token. A filler message of `k` tokens is `k`
words drawn from the pool with a generator seeded by `(seed, session_id, call
index)`. Startup checks that a 10,000-word sample tokenizes to 10,000 tokens and
fails otherwise. Because templates trim a message's leading whitespace, sizes
and cut positions come from rendering and tokenizing (above), not from word
counts.

### Prompt size check

Each call records `target_prompt` and the server's `prompt_tokens`. Each time
another 100 calls have reported `prompt_tokens`, warmup included, the run fails
if the median of `|prompt_tokens - target_prompt| / target_prompt` over those
calls exceeds 0.01.

---

## Metrics and output

Per call, appended to `replay.log` when set:

`session_id`, `call`, `model` (trace label), `scheduled_ms`, `sent_ms`,
`lag_ms`, `gap_capped`, `warmup`, `target_prompt`, `prompt_tokens`,
`overshoot`, `reuse` (trace), `reuse_inferred`, `expected_reuse`, `shortfall`,
`cached_tokens`, `max_tokens`, `completion_tokens`, `finish_reason`,
`ttft_ms`, `e2e_ms`, `error`.

Aggregates, named like the existing metrics in `src/metrics.rs`:

- `replay_reuse` counter group, `kind` = `expected` or `cached`;
- `replay_reuse_permille` histogram of `1000 * cached / expected` for calls
  with `expected_reuse > 0`;
- `replay_sessions` counter group, `status` = `started`, `completed`,
  `failed` or `truncated`;
- `schedule_slip` records `lag_ms` for successful non-warmup calls;
- the existing request, token, TTFT, ITL, TPOT and latency metrics.

The run ends with a summary (console or JSON per `output.format`): session
outcomes, call counts, expected and cached reuse totals, prefill tokens
computed (`prompt_tokens - cached_tokens`), percentiles of cached/expected,
`lag_ms` and TTFT, and the median prompt size error.

The log is the primary output. The aggregates are for reading a run without
post-processing.

---

## Stopping

The run ends when every selected session has finished, or at
`duration_seconds`. At the duration limit no new call is sent, sessions waiting
for a start time or a gap stop at once, in-flight calls finish within the
endpoint timeout, and sessions cut short are counted in the report.

A failed call ends its session: the session is counted as failed and its
remaining calls are not sent.

`warmup_duration` excludes calls sent in the first N seconds from the
aggregates; the log still records them.

---

## Validation

1. Unit tests with no server: sampling, time scaling, cutting message lists at
   token boundaries, filler determinism.
2. Fixture tests: parse the captured responses in `tests/fixtures/llama-server/`.
3. Llama 3.1 8B, sessions replayed one at a time: `expected_reuse - cached` is
   0, or 1 when the whole previous reply is kept. Any other value is a replay
   bug. With several sessions, the RAM prompt cache evicts and larger
   differences are expected.
4. Qwen3.5 9B, same sample: `cached` is the largest checkpoint position at or
   below `min(expected_reuse, previous prompt - 4)`. This prediction depends on
   the replay sending each call's new content as one user message.
