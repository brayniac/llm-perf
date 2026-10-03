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
`warmup_duration` are accepted. This makes `Config.input` an
`Option<InputConfig>` (32 call sites in `src/benchmark.rs`, `src/main.rs` and
`src/report.rs`) and `Config.load` optional, and turns the `[load]` fields with
serde defaults (`concurrent_requests`, `arrival_distribution`) into `Option` so
an explicit value can be rejected. `Config::validate` requires
`total_requests` or `duration_seconds` only when `[replay]` is absent.

```toml
[replay]
trace = "trace.jsonl"     # output of `llm-perf convert-trace`
sample = 0.01             # fraction of sessions to replay, in (0, 1]
sample_seed = 0           # selects a different sample of the same size
speedup = 24.0            # divides session start offsets and gaps; > 0
max_gap_ms = 300000       # optional cap on a gap after speedup; unset = no cap
seed = 1                  # filler text seed
system_prompt_tokens = 0  # optional system prompt shared by every session
```

Replay rejects `endpoint.max_tokens` and `endpoint.max_retries > 0` (a retry
resends a prompt the server may already have cached). Each call sends
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
`(u64::from_str_radix(&session_id[..16], 16) ^ sample_seed) <= (sample * 2^64) as u64`,
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
   boundary of its content. A cut that falls in a message's template tokens
   keeps that message whole without content.
3. Append one user message of new filler so the rendered prompt reaches
   `prompt` tokens. When `prompt - reuse` is smaller than one message's
   template tokens plus one (12% of non-first calls have `prompt == reuse`),
   the new message has one filler token, the prompt overshoots `prompt`, and
   the call is flagged `overshoot` in the log.
4. Render and tokenize the result. `expected_reuse` is the length of its
   common token prefix with call `n-1`'s rendered prompt followed by the
   tokenized reply.

If `reuse` exceeds call `n-1`'s rendered prompt plus the tokenized reply, all
of it is kept and the difference is logged as `shortfall`.

The first call of a session is the optional shared system prompt plus one user
message of filler.

### Filler

At startup the replay builds a pool of filler words: each word is a single
token with a leading space under the server's tokenizer, and tokenizing the
detokenized word gives the same token. A filler message of `k` tokens is `k`
words drawn from the pool with a generator seeded by `(seed, session_id, call
index)`. Startup checks that a 10,000-word sample tokenizes to 10,000 tokens and
fails otherwise. Because templates trim a message's leading whitespace, sizes
and cut positions come from rendering and tokenizing (above), not from word
counts.

### Prompt size check

Each call records `target_prompt` and the server's `prompt_tokens`. After call
100, and every 100 calls after that, the run fails if the median of
`|prompt_tokens - target_prompt| / target_prompt` over all calls so far
exceeds 0.01.

---

## Metrics and output

Per call, appended to a JSONL file:

`session_id`, `call`, `model` (trace label), `scheduled_ms`, `sent_ms`, `overshoot`,
`lag_ms`, `gap_capped`, `target_prompt`, `prompt_tokens`, `reuse` (trace),
`reuse_inferred`, `expected_reuse`, `shortfall`, `cached_tokens`,
`completion_tokens`, `finish_reason`, `ttft_ms`, `e2e_ms`, `error`.

Aggregates, named like the existing metrics in `src/metrics.rs`:

- `replay_reuse_expected` and `replay_reuse_cached` counters;
- `replay_reuse_permille` histogram of `1000 * cached / expected` for calls
  with `expected_reuse > 0`;
- prefill tokens computed (`prompt_tokens - cached_tokens`);
- the existing `schedule_slip`, `cache`, `ttft_by_cache`, TTFT, ITL and
  end-to-end metrics.

The log is the primary output. The aggregates are for reading a run without
post-processing.

---

## Stopping

The run ends when every selected session has finished, or at
`duration_seconds`. At the duration limit no new call is sent, in-flight calls
finish within the endpoint timeout, and sessions cut short are counted in the
report.

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
