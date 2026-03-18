#!/bin/bash
# =============================================================================
# Benchmark script for three scenarios with varying prefix lengths
#
# Scenarios:
#   1. Baseline (no CPU offloading)
#   2. With CPU offloading (--cpu-offload-gb)
#   3. With LMCache KV cache offloading (--kv-offloading-backend lmcache)
#
# Each scenario is tested with prefix lengths at 0%, 10%, 20%, 30%
# of the input token length (10000 tokens).
#
# Prerequisites:
#   - pip install lmcache  (for the LMCache scenario)
#   - A GPU with enough memory for the model
#
# Usage:
#   ./vllm_profile/run_benchmark_scenarios.sh [--model MODEL] [--num-prompts N]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
OUTPUT_LEN=100
REQUEST_RATE="${REQUEST_RATE:-inf}"
ENDPOINT="/v1/completions"
RESULT_DIR="$SCRIPT_DIR/benchmark_results"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
HOST="localhost"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"
PROMPTS_FILE="./prompts.jsonl"  # Path to a prompts JSONL file (--dataset-name custom)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --request-rate) REQUEST_RATE="$2"; shift 2 ;;
        --port) PORT="$2"; BASE_URL="http://${HOST}:${PORT}"; shift 2 ;;
        --cpu-offload-gb) CPU_OFFLOAD_GB="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model to benchmark (default: $MODEL)"
            echo "  --num-prompts N         Number of prompts (default: $NUM_PROMPTS)"
            echo "  --request-rate RATE     Request rate in req/s (default: $REQUEST_RATE)"
            echo "  --port PORT             Server port (default: $PORT)"
            echo "  --cpu-offload-gb GB     CPU offload size in GB (default: $CPU_OFFLOAD_GB)"
            echo "  --max-num-seqs N        Max number of sequences (default: $MAX_NUM_SEQS)"
            echo "  --max-num-batched-tokens N  Max batched tokens per iteration (default: $MAX_NUM_BATCHED_TOKENS)"
            echo "  --max-model-len N       Max model context length (default: $MAX_MODEL_LEN)"
            echo "  --prompts-file FILE     Path to JSONL prompts file (use --dataset-name custom)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="$RESULT_DIR/$TIMESTAMP"
mkdir -p "$RESULT_DIR"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
start_server() {
    local scenario_name="$1"
    shift
    local extra_args=("$@")

    echo "[$scenario_name] Starting vllm server..."
    echo "  Model: $MODEL"
    echo "  Extra args: ${extra_args[*]}"

    vllm serve "$MODEL" \
        --port "$PORT" \
        --enable-prefix-caching \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        "${extra_args[@]}" \
        > "$RESULT_DIR/${scenario_name}_server.log" 2>&1 &
    SERVER_PID=$!

    # Wait for the server to be ready
    echo "[$scenario_name] Waiting for server (PID $SERVER_PID) to be ready..."
    local max_wait=300
    local waited=0
    while ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[$scenario_name] ERROR: Server process died. Check $RESULT_DIR/${scenario_name}_server.log"
            return 1
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "[$scenario_name] ERROR: Server did not become ready within ${max_wait}s"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[$scenario_name] Server ready after ${waited}s"
}

stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

get_prefix_cache_hit_rate() {
    # Query Prometheus metrics from the vLLM server and compute hit rate.
    # Returns "N/A" if metrics are unavailable.
    local metrics
    metrics=$(curl -s "${BASE_URL}/metrics" 2>/dev/null) || { echo "N/A"; return; }

    local queries hits
    queries=$(echo "$metrics" | grep -E '^vllm:prefix_cache_queries_total\b' | awk '{s+=$2} END {print s+0}')
    hits=$(echo "$metrics" | grep -E '^vllm:prefix_cache_hits_total\b' | awk '{s+=$2} END {print s+0}')

    if [ -z "$queries" ] || [ "$queries" = "0" ]; then
        echo "0.00% (queries=$queries, hits=$hits)"
    else
        local rate
        rate=$(awk "BEGIN {printf \"%.2f\", ($hits / $queries) * 100}")
        echo "${rate}% (queries=$queries, hits=$hits)"
    fi
}

_collect_metrics_and_save() {
    # Usage: _collect_metrics_and_save <scenario_name> <result_filename> <pre_queries> <pre_hits>
    local scenario_name="$1" result_filename="$2" pre_queries="$3" pre_hits="$4"

    local post_metrics
    post_metrics=$(curl -s "${BASE_URL}/metrics" 2>/dev/null) || true
    local post_queries post_hits
    post_queries=$(echo "$post_metrics" | grep -E '^vllm:prefix_cache_queries_total\b' | awk '{s+=$2} END {print s+0}')
    post_hits=$(echo "$post_metrics" | grep -E '^vllm:prefix_cache_hits_total\b' | awk '{s+=$2} END {print s+0}')

    local kv_cache_usage
    kv_cache_usage=$(echo "$post_metrics" | grep -E '^vllm:kv_cache_usage\b' | awk '{print $2}')
    if [ -z "$kv_cache_usage" ]; then
        kv_cache_usage="N/A"
    fi
    local kv_cache_usage_pct
    if [ "$kv_cache_usage" != "N/A" ]; then
        kv_cache_usage_pct=$(awk "BEGIN {printf \"%.2f\", $kv_cache_usage * 100}")
    else
        kv_cache_usage_pct="N/A"
    fi

    local delta_queries delta_hits hit_rate
    delta_queries=$((post_queries - pre_queries))
    delta_hits=$((post_hits - pre_hits))
    if [ "$delta_queries" -gt 0 ] 2>/dev/null; then
        hit_rate=$(awk "BEGIN {printf \"%.2f\", ($delta_hits / $delta_queries) * 100}")
    else
        hit_rate="0.00"
    fi

    echo "  [${scenario_name}] Prefix cache hit rate: ${hit_rate}% (queries=${delta_queries}, hits=${delta_hits})"
    echo "  [${scenario_name}] KV cache usage: ${kv_cache_usage_pct}%"
    echo "  [${scenario_name}] Saved: ${RESULT_DIR}/${result_filename}.json"

    if command -v python3 &>/dev/null && [ -f "$RESULT_DIR/${result_filename}.json" ]; then
        python3 -c "
import json, sys
f = sys.argv[1]
with open(f) as fh: d = json.load(fh)
d['prefix_cache_hit_rate_pct'] = float(sys.argv[2])
d['prefix_cache_queries'] = int(sys.argv[3])
d['prefix_cache_hits'] = int(sys.argv[4])
kv_usage = sys.argv[5]
d['kv_cache_usage_pct'] = float(kv_usage) * 100 if kv_usage != 'N/A' else None
with open(f, 'w') as fh: json.dump(d, fh, indent=2)
" "$RESULT_DIR/${result_filename}.json" "$hit_rate" "$delta_queries" "$delta_hits" "$kv_cache_usage"
    fi
}

run_benchmarks_for_scenario() {
    local scenario_name="$1"
    local result_filename="${scenario_name}_custom"

    echo ""
    echo "  [${scenario_name}] Running benchmark with custom prompts: $PROMPTS_FILE"

    local pre_metrics
    pre_metrics=$(curl -s "${BASE_URL}/metrics" 2>/dev/null) || true
    local pre_queries pre_hits
    pre_queries=$(echo "$pre_metrics" | grep -E '^vllm:prefix_cache_queries_total\b' | awk '{s+=$2} END {print s+0}')
    pre_hits=$(echo "$pre_metrics" | grep -E '^vllm:prefix_cache_hits_total\b' | awk '{s+=$2} END {print s+0}')

    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --endpoint "$ENDPOINT" \
        --port "$PORT" \
        --dataset-name custom \
        --dataset-path "$PROMPTS_FILE" \
        --num-prompts "$NUM_PROMPTS" \
        --output-len "$OUTPUT_LEN" \
        --request-rate "$REQUEST_RATE" \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "${result_filename}.json"

    _collect_metrics_and_save "$scenario_name" "$result_filename" "$pre_queries" "$pre_hits"
}

# Cleanup on exit
trap stop_server EXIT

# ---------------------------------------------------------------------------
# Scenario 1: Baseline (no CPU offloading)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Scenario 1: Baseline (no offloading)"
echo "============================================================"

start_server "baseline"
run_benchmarks_for_scenario "baseline"
stop_server

# ---------------------------------------------------------------------------
# Scenario 2: With CPU Offloading
# ---------------------------------------------------------------------------
# echo ""
# echo "============================================================"
# echo "  Scenario 2: CPU Offloading (${CPU_OFFLOAD_GB} GB)"
# echo "============================================================"

# start_server "cpu_offload" --cpu-offload-gb "$CPU_OFFLOAD_GB"
# run_benchmarks_for_scenario "cpu_offload"
# stop_server

# ---------------------------------------------------------------------------
# Scenario 3: With LMCache
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Scenario 3: LMCache KV cache offloading"
echo "============================================================"

if ! python -c "import lmcache" 2>/dev/null; then
    echo "WARNING: lmcache is not installed. Installing..."
    pip install lmcache
fi

start_server "lmcache" --kv-offloading-backend lmcache --kv-offloading-size 100 --disable-hybrid-kv-cache-manager
run_benchmarks_for_scenario "lmcache"
stop_server

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All benchmarks complete!"
echo "============================================================"
echo "Results saved to: $RESULT_DIR"
echo ""
echo "Result files:"
ls -1 "$RESULT_DIR"/*.json 2>/dev/null || echo "  (no JSON results found)"
echo ""
echo "Server logs:"
ls -1 "$RESULT_DIR"/*_server.log 2>/dev/null || echo "  (no server logs found)"
