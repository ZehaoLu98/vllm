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
NUM_PROMPTS="${NUM_PROMPTS:-100}"
INPUT_LEN=10000
OUTPUT_LEN=100
ENDPOINT="/v1/completions"
RESULT_DIR="$SCRIPT_DIR/benchmark_results"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30}"
HOST="localhost"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

# Prefix lengths: 0%, 10%, 20%, 30% of INPUT_LEN
PREFIX_PERCENTS=(0 10 20 30)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --port) PORT="$2"; BASE_URL="http://${HOST}:${PORT}"; shift 2 ;;
        --cpu-offload-gb) CPU_OFFLOAD_GB="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model to benchmark (default: $MODEL)"
            echo "  --num-prompts N         Number of prompts (default: $NUM_PROMPTS)"
            echo "  --port PORT             Server port (default: $PORT)"
            echo "  --cpu-offload-gb GB     CPU offload size in GB (default: $CPU_OFFLOAD_GB)"
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

run_benchmarks_for_scenario() {
    local scenario_name="$1"

    for pct in "${PREFIX_PERCENTS[@]}"; do
        local prefix_len=$((INPUT_LEN * pct / 100))
        local result_filename="${scenario_name}_prefix_${pct}pct"

        echo ""
        echo "  [${scenario_name}] Running benchmark: prefix_len=${prefix_len} (${pct}% of ${INPUT_LEN})"

        vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --endpoint "$ENDPOINT" \
            --port "$PORT" \
            --dataset-name random \
            --num-prompts "$NUM_PROMPTS" \
            --input-len "$INPUT_LEN" \
            --output-len "$OUTPUT_LEN" \
            --random-prefix-len "$prefix_len" \
            --save-result \
            --result-dir "$RESULT_DIR" \
            --result-filename "${result_filename}.json"

        echo "  [${scenario_name}] Saved: ${RESULT_DIR}/${result_filename}.json"
    done
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
echo ""
echo "============================================================"
echo "  Scenario 2: CPU Offloading (${CPU_OFFLOAD_GB} GB)"
echo "============================================================"

start_server "cpu_offload" --cpu-offload-gb "$CPU_OFFLOAD_GB"
run_benchmarks_for_scenario "cpu_offload"
stop_server

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

start_server "lmcache" --kv-offloading-backend lmcache
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
