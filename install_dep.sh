#!/bin/bash
set -e

# Parse command line arguments
ENABLE_NSYS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-nsys)
            ENABLE_NSYS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--enable-nsys]"
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
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto

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