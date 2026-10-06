#!/bin/sh
# Capture vLLM responses for a three-call session into a directory.
#
#   capture.sh <base-url> <model> <out-dir>
#
# The server must run with --enable-prefix-caching and
# --enable-prompt-tokens-details, freshly started so call1 finds an empty
# cache. Calls are streamed with usage, 64 tokens, thinking disabled.
#
# call1: system + long user message.
# call2: call1 again.
# call3: call1's messages + call1's reply as an assistant message + a new user
#        message.
set -eu

U=$1
M=$2
F=$3
mkdir -p "$F"

LONG=$(yes 'The quick brown fox jumps over the lazy dog near the river bank.' | head -500 | tr '\n' ' ')
KW='{"enable_thinking": false}'

post() {  # $1 = path, $2 = request file, $3 = response file
	curl -sf "$U$1" -H 'content-type: application/json' -d @"$2" > "$3"
}

curl -sf "$U/v1/models" > "$F/models.json"
curl -sf "$U/version" > "$F/version.json"

jq -n --arg m "$M" --arg u "$LONG" \
	'{model: $m, messages: [{role: "system", content: "You are a helpful assistant."}, {role: "user", content: $u}]}' \
	> "$F/messages1.json"

jq --argjson kw "$KW" '. + {add_generation_prompt: true, return_token_strs: true, chat_template_kwargs: $kw}' \
	"$F/messages1.json" > "$F/tokenize-chat.request.json"
post /tokenize "$F/tokenize-chat.request.json" "$F/tokenize-chat.response.json"

jq -n --arg m "$M" '{model: $m, prompt: "Hello world, this is a test.", add_special_tokens: false, return_token_strs: true}' \
	> "$F/tokenize-text.request.json"
post /tokenize "$F/tokenize-text.request.json" "$F/tokenize-text.response.json"

jq --arg m "$M" '{model: $m, tokens: .tokens}' "$F/tokenize-text.response.json" > "$F/detokenize.request.json"
post /detokenize "$F/detokenize.request.json" "$F/detokenize.response.json"

chat() {  # $1 = call name, $2 = messages file
	jq --argjson kw "$KW" \
		'. + {stream: true, stream_options: {include_usage: true}, max_tokens: 64, ignore_eos: true, chat_template_kwargs: $kw}' \
		"$2" > "$F/$1.request.json"
	post /v1/chat/completions "$F/$1.request.json" "$F/$1.response.sse"
}

chat call1 "$F/messages1.json"
REPLY=$(grep '^data: {' "$F/call1.response.sse" | sed 's/^data: //' | jq -r '.choices[0].delta.content // empty' | tr -d '\n')
[ -n "$REPLY" ] || { echo "call1 returned empty content" >&2; exit 1; }
chat call2 "$F/messages1.json"

jq --arg r "$REPLY" '.messages += [{role: "assistant", content: $r}, {role: "user", content: "Now say it backwards."}]' \
	"$F/messages1.json" > "$F/messages3.json"
chat call3 "$F/messages3.json"
jq --argjson kw "$KW" '. + {add_generation_prompt: true, chat_template_kwargs: $kw}' \
	"$F/messages3.json" > "$F/tokenize-chat3.request.json"
post /tokenize "$F/tokenize-chat3.request.json" "$F/tokenize-chat3.response.json"

curl -sf "$U/metrics" > "$F/metrics.txt"
rm "$F/messages1.json" "$F/messages3.json"
