#!/usr/bin/env bash
# Start one DJev-serve front (server_vllm.py) on this machine.
#
#   serving/serve.sh                          # fast engine on GPU 0, port 8765: the configuration of the release
#   GPUS=1 PORT=8766 serving/serve.sh         # a second front on GPU 1; put the fronts behind gateway.py
#   GPUS=0,1 serving/serve.sh                 # one front, one fast worker per GPU
#   ENGINE=inproc GPUS=0,1 serving/serve.sh   # vLLM's own engine in this process, one replica per GPU
#   ENGINE=http serving/serve.sh              # a separate `vllm serve` plus this front talking to it over HTTP
#
# Env: ENGINE (fast | inproc | http), GPUS, PORT, HOST, CANVAS, WIDTH, MOE, MEM, BATCH_MAX, BATCH_WAIT_MS,
#      PREFIX_SLOTS, KEYFILE, VPORT, FRONT_EXTRA, VLLM_EXTRA (see `python server_vllm.py --help`).
# Cold start is ~6 min (FlashInfer autotuning and CUDA-graph capture), ~2 min with warm caches; the front prints
# {"ready": true} when it accepts requests. KEYFILE=path requires that key on /api/* (Bearer token or basic auth).
set -euo pipefail
ENGINE=${ENGINE:-fast}
GPUS=${GPUS:-0}
PORT=${PORT:-8765}
HOST=${HOST:-127.0.0.1}
CANVAS=${CANVAS:-64}           # served canvas length; 64 is ~3 ms faster than 256 (longer scaffolds are split)
WIDTH=${WIDTH:-64}             # [inproc/http] minimum per-request canvas width
if [ "$ENGINE" = fast ]; then DEFAULT_MOE=flashinfer_cutlass; else DEFAULT_MOE=triton; fi
MOE=${MOE:-$DEFAULT_MOE}       # flashinfer_cutlass needs a one-time nvcc JIT (MAX_JOBS limits its parallelism)
MEM=${MEM:-0.85}
BATCH_MAX=${BATCH_MAX:-4}      # [fast] up to this many queued reads share one forward
BATCH_WAIT_MS=${BATCH_WAIT_MS:-4}
PREFIX_SLOTS=${PREFIX_SLOTS:-8}
VPORT=${VPORT:-8000}           # [http] port of `vllm serve`
KEYFILE=${KEYFILE:-}
FRONT_EXTRA=${FRONT_EXTRA:-}
VLLM_EXTRA=${VLLM_EXTRA:-}
KEYARG=${KEYFILE:+--api-key-file $KEYFILE}
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
# Some vLLM container images preset these; flashinfer_cutlass MoE fails under VLLM_BATCH_INVARIANT=1.
unset VLLM_BATCH_INVARIANT VLLM_USE_V2_MODEL_RUNNER

case "$ENGINE" in
fast)
  exec python server_vllm.py --engine fast --gpus "$GPUS" --ids main --served-canvas "$CANVAS" \
    --moe-backend "$MOE" --gpu-memory-utilization "$MEM" --batch-max "$BATCH_MAX" --batch-wait-ms "$BATCH_WAIT_MS" \
    --prefix-slots "$PREFIX_SLOTS" --host "$HOST" --port "$PORT" $KEYARG $FRONT_EXTRA ;;
inproc)
  exec python server_vllm.py --engine inproc --gpus "$GPUS" --ids main --served-canvas "$CANVAS" --width "$WIDTH" \
    --moe-backend "$MOE" --gpu-memory-utilization "$MEM" --host "$HOST" --port "$PORT" $KEYARG $FRONT_EXTRA ;;
http)
  DP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
  export CUDA_VISIBLE_DEVICES=$GPUS
  vllm serve google/diffusiongemma-26B-A4B-it --served-model-name djev --host 127.0.0.1 --port "$VPORT" \
    --diffusion-config "{\"canvas_length\": $CANVAS}" --max-logprobs 128 --async-scheduling --enable-prefix-caching \
    --moe-backend "$MOE" --gpu-memory-utilization "$MEM" --max-model-len 4096 --limit-mm-per-prompt '{"image": 8}' \
    -dp "$DP" $VLLM_EXTRA &
  VPID=$!
  trap 'kill $VPID 2>/dev/null; wait $VPID 2>/dev/null' EXIT
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$VPORT/health" >/dev/null && break
    kill -0 $VPID 2>/dev/null || { echo "vllm serve exited"; exit 1; }
    sleep 2
  done
  python server_vllm.py --engine http --upstream "http://127.0.0.1:$VPORT" --model djev --served-canvas "$CANVAS" \
    --width "$WIDTH" --thought prompt --host "$HOST" --port "$PORT" $KEYARG $FRONT_EXTRA ;;
*)
  echo "ENGINE must be fast, inproc or http" >&2; exit 2 ;;
esac
