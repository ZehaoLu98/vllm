#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Bash script to run vllm_profile with multiple batch sizes

# Define batch sizes array (max_num_seqs values)
BATCH_SIZES=(4 8 16 32 48)

# Define max_num_batched_tokens configurations
# Values can be integers or suffixed strings like "32k"
MAX_NUM_BATCHED_TOKENS_CONFIGS=("100k")

# Define prompts file configurations
# Each entry is a path to a prompts file that is passed to vllm_profile.py
# through the VLLM_PROMPTS_FILE environment variable
PROMPTS_CONFIGS=("./prompt_generator/gpt_oss_120b/random_512_512.txt" "./prompt_generator/gpt_oss_120b/random_8192_512.txt" "./prompt_generator/gpt_oss_120b/random_512_8192.txt")

# Output directory for logs
OUTPUT_DIR="./experiment_results"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="$OUTPUT_DIR/experiment_$TIMESTAMP"
mkdir -p "$EXPERIMENT_DIR"

echo "=========================================="
echo "Starting batch size experiments"
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $EXPERIMENT_DIR"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "MAX_NUM_BATCHED_TOKENS configs: ${MAX_NUM_BATCHED_TOKENS_CONFIGS[@]}"
echo "Prompts configs: ${PROMPTS_CONFIGS[@]}"
echo "=========================================="
echo ""

# Optional: Additional scheduler config parameters
# Uncomment and modify as needed

# Maximum number of sequences that can be partially prefilled concurrently
# MAX_NUM_PARTIAL_PREFILLS=1

# Maximum number of long prompts that will be prefilled concurrently
# MAX_LONG_PARTIAL_PREFILLS=1

# Threshold for considering a request as long (in tokens)
# LONG_PREFILL_TOKEN_THRESHOLD=0

# Scheduling policy: "fcfs" or "priority"
# SCHEDULING_POLICY="fcfs"

# Enable chunked prefill: "true", "false", or "none"
ENABLE_CHUNKED_PREFILL="false"

# Disable chunked multimodal input (flag, set to "true" to enable)
# DISABLE_CHUNKED_MM_INPUT="true"

# Custom scheduler class path
# SCHEDULER_CLS="vllm.v1.core.sched.scheduler.Scheduler"

# Disable hybrid KV cache manager: "true", "false", or "none"
# DISABLE_HYBRID_KV_CACHE_MANAGER="none"

# Async scheduling: "true", "false", or "none"
# ASYNC_SCHEDULING="none"

# Stream interval (buffer size for streaming in terms of token length)
# STREAM_INTERVAL=1

# Loop through each batch size
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo "Running experiments with batch_size=$batch_size"
    echo "=========================================="
    echo ""

    # Loop through each MAX_NUM_BATCHED_TOKENS configuration
    for max_batched_tokens in "${MAX_NUM_BATCHED_TOKENS_CONFIGS[@]}"; do

      # Loop through each prompts file configuration
      for prompts_file in "${PROMPTS_CONFIGS[@]}"; do
        echo "------------------------------------------"
        echo "Running: batch_size=$batch_size, max_num_batched_tokens=$max_batched_tokens, prompts_file=$prompts_file"
        echo "------------------------------------------"

        # Create output file for this run (sanitize values for filename)
        SANITIZED_TOKENS=$(echo "$max_batched_tokens" | tr -d ' /')
        SANITIZED_PROMPTS=$(basename "$prompts_file" | tr -d ' /')
        OUTPUT_FILE="$EXPERIMENT_DIR/batch_${batch_size}_tokens_${SANITIZED_TOKENS}_prompts_${SANITIZED_PROMPTS}_output.log"

        # Build the command
        CMD="python vllm_profile.py --max_num_seqs $batch_size"

        # Add max_num_batched_tokens parameter
        CMD="$CMD --max_num_batched_tokens $max_batched_tokens"

        # Add optional parameters if defined
        if [ ! -z "$MAX_NUM_PARTIAL_PREFILLS" ]; then
            CMD="$CMD --max_num_partial_prefills $MAX_NUM_PARTIAL_PREFILLS"
        fi

        if [ ! -z "$MAX_LONG_PARTIAL_PREFILLS" ]; then
            CMD="$CMD --max_long_partial_prefills $MAX_LONG_PARTIAL_PREFILLS"
        fi

        if [ ! -z "$LONG_PREFILL_TOKEN_THRESHOLD" ]; then
            CMD="$CMD --long_prefill_token_threshold $LONG_PREFILL_TOKEN_THRESHOLD"
        fi

        if [ ! -z "$SCHEDULING_POLICY" ]; then
            CMD="$CMD --scheduling_policy $SCHEDULING_POLICY"
        fi

        if [ ! -z "$ENABLE_CHUNKED_PREFILL" ]; then
            CMD="$CMD --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL"
        fi

        if [ ! -z "$DISABLE_CHUNKED_MM_INPUT" ]; then
            if [ "$DISABLE_CHUNKED_MM_INPUT" = "true" ]; then
                CMD="$CMD --disable_chunked_mm_input"
            fi
        fi

        if [ ! -z "$SCHEDULER_CLS" ]; then
            CMD="$CMD --scheduler_cls $SCHEDULER_CLS"
        fi

        if [ ! -z "$DISABLE_HYBRID_KV_CACHE_MANAGER" ]; then
            CMD="$CMD --disable_hybrid_kv_cache_manager $DISABLE_HYBRID_KV_CACHE_MANAGER"
        fi

        if [ ! -z "$ASYNC_SCHEDULING" ]; then
            CMD="$CMD --async_scheduling $ASYNC_SCHEDULING"
        fi

        if [ ! -z "$STREAM_INTERVAL" ]; then
            CMD="$CMD --stream_interval $STREAM_INTERVAL"
        fi

        # Print command
        echo "Command: VLLM_PROMPTS_FILE=$prompts_file $CMD"
        echo "Output: $OUTPUT_FILE"
        echo ""

        # Run the command and save output
        VLLM_PROMPTS_FILE="$prompts_file" $CMD 2>&1 | tee "$OUTPUT_FILE"

        # Check exit status
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo ""
            echo "✓ Config (batch_size=$batch_size, max_num_batched_tokens=$max_batched_tokens, prompts_file=$prompts_file) completed successfully"
        else
            echo ""
            echo "✗ Config (batch_size=$batch_size, max_num_batched_tokens=$max_batched_tokens, prompts_file=$prompts_file) failed"
        fi

        echo ""
        echo "Waiting 5 seconds before next run..."
        sleep 5
        echo ""
      done
    done
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: $EXPERIMENT_DIR"
echo "=========================================="

# Generate summary file
SUMMARY_FILE="$EXPERIMENT_DIR/summary.txt"
echo "Experiment Summary" > "$SUMMARY_FILE"
echo "==================" >> "$SUMMARY_FILE"
echo "Timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Batch sizes tested: ${BATCH_SIZES[@]}" >> "$SUMMARY_FILE"
echo "MAX_NUM_BATCHED_TOKENS configs tested: ${MAX_NUM_BATCHED_TOKENS_CONFIGS[@]}" >> "$SUMMARY_FILE"
echo "Prompts configs tested: ${PROMPTS_CONFIGS[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Optional Parameters Used:" >> "$SUMMARY_FILE"
echo "-------------------------" >> "$SUMMARY_FILE"
[ ! -z "$MAX_NUM_PARTIAL_PREFILLS" ] && echo "  MAX_NUM_PARTIAL_PREFILLS: $MAX_NUM_PARTIAL_PREFILLS" >> "$SUMMARY_FILE"
[ ! -z "$MAX_LONG_PARTIAL_PREFILLS" ] && echo "  MAX_LONG_PARTIAL_PREFILLS: $MAX_LONG_PARTIAL_PREFILLS" >> "$SUMMARY_FILE"
[ ! -z "$LONG_PREFILL_TOKEN_THRESHOLD" ] && echo "  LONG_PREFILL_TOKEN_THRESHOLD: $LONG_PREFILL_TOKEN_THRESHOLD" >> "$SUMMARY_FILE"
[ ! -z "$SCHEDULING_POLICY" ] && echo "  SCHEDULING_POLICY: $SCHEDULING_POLICY" >> "$SUMMARY_FILE"
[ ! -z "$ENABLE_CHUNKED_PREFILL" ] && echo "  ENABLE_CHUNKED_PREFILL: $ENABLE_CHUNKED_PREFILL" >> "$SUMMARY_FILE"
[ ! -z "$DISABLE_CHUNKED_MM_INPUT" ] && echo "  DISABLE_CHUNKED_MM_INPUT: $DISABLE_CHUNKED_MM_INPUT" >> "$SUMMARY_FILE"
[ ! -z "$SCHEDULER_CLS" ] && echo "  SCHEDULER_CLS: $SCHEDULER_CLS" >> "$SUMMARY_FILE"
[ ! -z "$DISABLE_HYBRID_KV_CACHE_MANAGER" ] && echo "  DISABLE_HYBRID_KV_CACHE_MANAGER: $DISABLE_HYBRID_KV_CACHE_MANAGER" >> "$SUMMARY_FILE"
[ ! -z "$ASYNC_SCHEDULING" ] && echo "  ASYNC_SCHEDULING: $ASYNC_SCHEDULING" >> "$SUMMARY_FILE"
[ ! -z "$STREAM_INTERVAL" ] && echo "  STREAM_INTERVAL: $STREAM_INTERVAL" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Output files:" >> "$SUMMARY_FILE"
ls -lh "$EXPERIMENT_DIR"/*.log >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
