"""Capture llama-server /v1/completions logprobs responses as JSON lines.

    python3 -I capture-completions-logprobs.py [base-url] > out.jsonl

base-url defaults to http://127.0.0.1:8080. Each line holds the UTC send time,
the case name, the request path, the request body, the HTTP status, and the
response (parsed JSON, or the body text when it is not JSON, as for a streamed
response). The `chat_same_prompt` case sends the prompt to
/v1/chat/completions for comparison.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PROMPT = "The capital of France is"


def completion(prompt, **extra):
    return {"model": "local", "prompt": prompt, "max_tokens": 1, **extra}


CASES = [
    ("legacy_logprobs_int", "/v1/completions", completion(PROMPT, logprobs=5)),
    ("n_probs", "/v1/completions", completion(PROMPT, n_probs=5)),
    ("logprobs_bool_true", "/v1/completions", completion(PROMPT, logprobs=True)),
    (
        "logprobs_bool_true_top_logprobs",
        "/v1/completions",
        completion(PROMPT, logprobs=True, top_logprobs=5),
    ),
    ("top20", "/v1/completions", completion("def fibonacci(n):\n    ", logprobs=20)),
    ("cjk", "/v1/completions", completion("日本の首都は", logprobs=5)),
    ("empty_prompt", "/v1/completions", completion("", logprobs=5)),
    ("no_logprobs", "/v1/completions", completion(PROMPT)),
    ("max_tokens_2", "/v1/completions", completion(PROMPT, logprobs=3, max_tokens=2)),
    ("stream", "/v1/completions", completion(PROMPT, logprobs=5, stream=True)),
    # Greedy, so the next token is a fragment of a UTF-8 character: llama-server
    # stores no probs for a token that leaves the text incomplete.
    (
        "sampled_partial_utf8",
        "/v1/completions",
        completion(
            "Emoji list: 😀 😃 😄 😁 😆 😅 🤣 😂 🙂 🙃 😉 😊 😇 🥰 😍 🤩 😘",
            logprobs=3,
            temperature=0,
        ),
    ),
    # A UTF-8 fragment among the alternatives has token "" and its bytes.
    (
        "top_partial_utf8",
        "/v1/completions",
        completion("Mathematical: 𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃", logprobs=3, temperature=0),
    ),
    ("greedy", "/v1/completions", completion(PROMPT, logprobs=5, temperature=0)),
    (
        "chat_same_prompt",
        "/v1/chat/completions",
        {
            "model": "local",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
        },
    ),
]

for name, path, body in CASES:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req) as r:
            status, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode()
    try:
        resp = json.loads(raw)
    except ValueError:
        resp = raw
    ms = int(t * 1000)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ms // 1000)) + ".%03dZ" % (ms % 1000)
    line = {"ts": ts, "case": name, "path": path, "request": body, "status": status, "response": resp}
    print(json.dumps(line, ensure_ascii=False))
