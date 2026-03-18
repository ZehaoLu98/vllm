#!/bin/bash
set -e

# =============================================================================
# Benchmark Scenarios
# =============================================================================
#
# After running this script, you can benchmark three scenarios:
#
# --- 1. Without CPU Offloading (baseline) ---
#   # Terminal 1: Start the server
#   vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#
#   # Terminal 2: Run the benchmark
#   vllm bench serve \
#     --backend vllm \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --endpoint /v1/completions \
#     --dataset-name random \
#     --num-prompts 1000 --save-result \
#     --result-dir "./vllm_profile/" \
#     --result-filename "without_cpu_offloading" \
#     --input-len 10000 --output-len 100
#
# --- 2. With CPU Offloading (model weights offloaded to CPU) ---
#   # Terminal 1: Start the server with --cpu-offload-gb
#   vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --cpu-offload-gb 10
#
#   # Terminal 2: Run the benchmark
#   vllm bench serve \
#     --backend vllm \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --endpoint /v1/completions \
#     --dataset-name random \
#     --num-prompts 1000 --save-result \
#     --result-dir "./vllm_profile/" \
#     --result-filename "with_cpu_offloading" \
#     --input-len 10000 --output-len 100
#
# --- 3. With LMCache (KV cache offloaded to CPU via LMCache) ---
#   # First install lmcache from source (see below).
#
#   # Terminal 1: Start the server with --kv-offloading-backend lmcache
#   # NOTE: --kv-offloading-size is required (GiB of CPU memory for KV cache).
#   #       --disable-hybrid-kv-cache-manager is needed because LMCache
#   #       does not yet support HMA (Hybrid Memory Allocation).
#   vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --kv-offloading-backend lmcache \
#     --kv-offloading-size 50 \
#     --disable-hybrid-kv-cache-manager
#
#   # Terminal 2: Run the benchmark
#   vllm bench serve \
#     --backend vllm \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --endpoint /v1/completions \
#     --dataset-name random \
#     --num-prompts 1000 --save-result \
#     --result-dir "./vllm_profile/" \
#     --result-filename "with_lmcache" \
#     --input-len 10000 --output-len 100
#
# =============================================================================

# Parse command line arguments
ENABLE_NSYS=false
FROM_SOURCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-nsys)
            ENABLE_NSYS=true
            shift
            ;;
        --from-source)
            FROM_SOURCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--from-source] [--enable-nsys]"
            echo "  --from-source    Install vllm in editable mode using precompiled wheels (no compilation)"
            echo "  --enable-nsys    Install Nsight Systems and enable non-admin profiling access"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
if [ "$FROM_SOURCE" = true ]; then
    # ---------------------------------------------------------------------------
    # Build from source without compiling (using precompiled wheels)
    # ---------------------------------------------------------------------------
    # Downloads precompiled C++/CUDA binaries from https://wheels.vllm.ai for the
    # merge-base commit of your branch with upstream main, then installs vllm in
    # editable mode so local Python changes take effect immediately.
    #
    # If the auto-detected commit has no wheel (CI build failure), override with
    # VLLM_PRECOMPILED_WHEEL_COMMIT. To check wheel availability:
    #   curl -s https://wheels.vllm.ai/<commit>/vllm/metadata.json
    #   curl -s https://wheels.vllm.ai/nightly/vllm/metadata.json
    #
    # Useful env vars:
    #   VLLM_PRECOMPILED_WHEEL_COMMIT   - full 40-char hash (must exist on upstream main)
    #   VLLM_PRECOMPILED_WHEEL_VARIANT  - e.g. cu129, cu130 (default: auto-detected)
    #   VLLM_PRECOMPILED_WHEEL_LOCATION - direct URL/path to a .whl file (skips detection)
    # ---------------------------------------------------------------------------
    echo "Installing vllm from source (editable) with precompiled wheels..."
    VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-04a9e064db4dcf57519f1333796ba7face46248b} \
        VLLM_USE_PRECOMPILED=1 uv pip install --editable .
else
    uv pip install vllm --torch-backend=auto
fi

# ---------------------------------------------------------------------------
# Install lmcache from source
# ---------------------------------------------------------------------------
# Pre-built lmcache wheels are compiled against a specific PyTorch version.
# If your PyTorch version differs (e.g. 2.10.0+cu128), the C extension will
# fail with "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...".
# Building from source ensures lmcache is compiled against your exact PyTorch.
# ---------------------------------------------------------------------------
echo "Installing lmcache from source (compiling against current PyTorch)..."
uv pip install "lmcache @ git+https://github.com/LMCache/LMCache.git" --no-build-isolation

if [ "$ENABLE_NSYS" = true ]; then
    echo "Installing Nsight Systems..."
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb
    sudo dpkg -i NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb

    # Reload NVIDIA drivers to enable profiling for all users
    echo "Enabling NVIDIA profiling for non-admin users..."

    # Stop window manager and unload all NVIDIA kernel modules
    sudo systemctl isolate multi-user.target
    sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia 2>/dev/null || true

    # Load nvidia module with non-admin access enabled
    sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
    sudo modprobe nvidia_uvm nvidia_peermem

    # Verify the setting took effect
    echo "Verifying profiling access..."
    cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly

    # Restart window manager
    sudo systemctl isolate graphical.target

    # Set environment variable for detailed NVTX labeling in nsys
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
    # install nvtx
    uv pip install nvtx

    echo "Nsight Systems installation and configuration complete!"
    echo "Example usage:"
    echo "  nsys profile --cuda-memory-usage --gpu-metrics-devices=help"
    echo "  nsys profile --cuda-memory-usage true --gpu-metrics-devices all --trace-fork-before-exec true python ./vllm_profile/vllm_profile.py"
else
    echo "Nsight Systems installation skipped. Use --enable-nsys to install."
fi