#!/usr/bin/env bash
set -euo pipefail

# Run inside dsv4-pr72-final.  The service must already be running.
# Usage: run_ttft_compare.sh HOST PORT OUTPUT_DIR
host=${1:-127.0.0.1}
port=${2:-18990}
out=${3:-/workspace/dsv4/logs/ttft}
mkdir -p "$out"

warmup=10
samples=100
input_len=8
output_len=1

vllm bench serve --host "$host" --port "$port" --model dsv4 \
  --tokenizer /workspace/models/DeepSeek-V4-Flash-w8a8-mtp \
  --trust-remote-code --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len "$input_len" \
  --random-output-len "$output_len" --ignore-eos --num-prompts "$warmup" \
  --max-concurrency 1 >"$out/warmup.log" 2>&1

vllm bench serve --host "$host" --port "$port" --model dsv4 \
  --tokenizer /workspace/models/DeepSeek-V4-Flash-w8a8-mtp \
  --trust-remote-code --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len "$input_len" \
  --random-output-len "$output_len" --ignore-eos --num-prompts "$samples" \
  --max-concurrency 1 >"$out/samples.log" 2>&1

python - "$out/samples.log" "$out/result.txt" <<'PY'
import re, statistics, sys
text = open(sys.argv[1], errors="replace").read()
patterns = [r"Mean TTFT.*?([0-9]+(?:\.[0-9]+)?)", r"Median TTFT.*?([0-9]+(?:\.[0-9]+)?)"]
vals = []
for p in patterns:
    vals += [float(x) for x in re.findall(p, text, re.I)]
if not vals:
    # Keep the complete benchmark output available when the installed vLLM
    # changes its summary wording.
    raise SystemExit("No TTFT summary found; inspect samples.log")
mean = statistics.mean(vals)
std = statistics.stdev(vals) if len(vals) > 1 else 0.0
open(sys.argv[2], "w").write(f"summary_values_ms={vals}\nmean_ms={mean:.6f}\nstddev_ms={std:.6f}\n")
print(open(sys.argv[2]).read(), end="")
PY
