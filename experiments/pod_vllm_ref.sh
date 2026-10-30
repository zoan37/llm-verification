#!/bin/bash
# Serve a model with vLLM (BF16 release weights) on this machine and collect honest reference replies through the
# same request path as OpenRouter endpoints.
#   experiments/pod_vllm_ref.sh <hf-dir> <openrouter-model-id> <tag> <label> [extra vllm args...]
set -u
HF_DIR=$1; MODEL=$2; TAG=$3; LABEL=$4; shift 4
VLLM=${VLLM:-/root/vllm-venv/bin/vllm}
PY=${PY:-/root/lvenv/bin/python}
mkdir -p logs
$VLLM serve "$HF_DIR" --served-model-name ref --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --max-logprobs 20 --port 8000 "$@" > "logs/vllm-$(basename "$HF_DIR").log" 2>&1 &
VPID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "vllm exited"; tail -20 "logs/vllm-$(basename "$HF_DIR").log"; exit 1; }
  sleep 5
done
LLMVERIFY_API_BASE=http://127.0.0.1:8000/v1 $PY experiments/collect_local_api.py --model "$MODEL" --served-name ref \
    --tag "$TAG" --label "$LABEL"
kill $VPID; wait $VPID 2>/dev/null
echo "REF_DONE $TAG"
