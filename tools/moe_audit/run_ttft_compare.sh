#!/usr/bin/env bash
set -euo pipefail

# Run inside dsv4-pr72-final.  The service must already be running.
# Usage: run_ttft_compare.sh HOST PORT OUTPUT_DIR
host=${1:-127.0.0.1}
port=${2:-18990}
out=${3:-/workspace/dsv4/logs/ttft}
mkdir -p "$out"

export VLLM_VERSION=0.23.0

vllm bench serve --host "$host" --port "$port" --model dsv4 \
  --tokenizer "${P_MODEL:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}" \
  --trust-remote-code --backend vllm --endpoint /v1/completions \
  --dataset-name random --random-input-len 8 --random-range-ratio 0 \
  --random-output-len 1 --ignore-eos --num-prompts 100 --num-warmups 10 \
  --seed 1024 --max-concurrency 1 --save-result --save-detailed \
  --result-dir "$out" --result-filename samples.json >"$out/samples.log" 2>&1

python - "$out/samples.json" "$out/result.txt" <<'PY'
import json, statistics, sys
with open(sys.argv[1]) as stream:
    data = json.load(stream)
assert data['completed'] == 100, data['completed']
assert data['input_lens'] == [8] * 100, data['input_lens']
assert not any(data['errors']), data['errors']
vals = [x * 1000 for x in data['ttfts']]
assert len(vals) == 100 and all(x > 0 for x in vals)
mean = statistics.mean(vals)
std = statistics.pstdev(vals)
open(sys.argv[2], "w").write(f"requests=100\ninput_tokens=8\nmean_ms={mean:.6f}\npopulation_stddev_ms={std:.6f}\n")
print(open(sys.argv[2]).read(), end="")
PY
