#!/bin/sh
# Capture llama-server responses for a three-call session into a directory.
#
#   capture.sh <base-url> <out-dir> [extra JSON merged into every request]
#
# call1: system + long user message.
# call2: call1's messages + call1's reply as an assistant message + a new user
#        message; streamed with usage.
# call3: as call2, but the last user message cut to half its length.
# Slots are erased first so call1 starts from an empty cache.
set -eu

U=$1
F=$2
EXTRA=${3:-'{}'}
mkdir -p "$F"

FILL1=$(i=1; while [ $i -le 400 ]; do printf "alpha%d bravo charlie delta echo foxtrot. " $i; i=$((i+1)); done)
FILL2=$(i=1; while [ $i -le 120 ]; do printf "golf%d hotel india juliet kilo. " $i; i=$((i+1)); done)
FILL2CUT=$(i=1; while [ $i -le 60 ]; do printf "golf%d hotel india juliet kilo. " $i; i=$((i+1)); done)

req() {
	jq -n --arg u1 "$1" --arg a "$2" --arg u2 "$3" --argjson stream "$4" --argjson extra "$EXTRA" '
	  {model: "replay", max_tokens: 48, ignore_eos: true, temperature: 0, stream: $stream}
	  + (if $stream then {stream_options: {include_usage: true}} else {} end)
	  + {messages: ([{role: "system", content: "You are a coding agent."}, {role: "user", content: $u1}]
	      + (if $a == "" then [] else [{role: "assistant", content: $a}, {role: "user", content: $u2}] end))}
	  + $extra'
}

for s in $(curl -s "$U/slots" | jq -r '.[].id'); do
	curl -s -X POST "$U/slots/$s?action=erase" > /dev/null
done

req "$FILL1" "" "" false > "$F/call1.request.json"
curl -s "$U/v1/chat/completions" -H 'content-type: application/json' -d @"$F/call1.request.json" > "$F/call1.response.json"
A=$(jq -r '.choices[0].message.content' "$F/call1.response.json")
[ -n "$A" ] || { echo "call1 returned empty content" >&2; exit 1; }

req "$FILL1" "$A" "$FILL2" true > "$F/call2.request.json"
curl -sN "$U/v1/chat/completions" -H 'content-type: application/json' -d @"$F/call2.request.json" > "$F/call2.response.sse"

req "$FILL1" "$A" "$FILL2CUT" false > "$F/call3.request.json"
curl -s "$U/v1/chat/completions" -H 'content-type: application/json' -d @"$F/call3.request.json" > "$F/call3.response.json"

curl -s "$U/tokenize" -H 'content-type: application/json' \
	-d '{"content": " golf1 hotel india", "add_special": false, "with_pieces": true}' > "$F/tokenize.json"
curl -s "$U/detokenize" -H 'content-type: application/json' \
	-d "{\"tokens\": $(jq -c '[.tokens[].id]' "$F/tokenize.json")}" > "$F/detokenize.json"
curl -s "$U/slots" > "$F/slots.json"
curl -s "$U/metrics" > "$F/metrics.txt"
curl -s "$U/props" | jq '{build_info, default_generation_settings: {n_ctx: .default_generation_settings.n_ctx}}' > "$F/props.json"

summ() { jq -c '{prompt: .usage.prompt_tokens, cached: .usage.prompt_tokens_details.cached_tokens, cache_n: .timings.cache_n, prompt_n: .timings.prompt_n}'; }
echo "call1 $(summ < "$F/call1.response.json")"
echo "call2 $(grep '^data: {' "$F/call2.response.sse" | tail -1 | sed 's/^data: //' | summ)"
echo "call3 $(summ < "$F/call3.response.json")"
