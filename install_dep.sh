#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb
sudo dpkg -i NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb


# Reload NVIDIA drivers to enable profiling for all users
sudo modprobe -rf nvidia_peermem
sudo modprobe -rf nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo systemctl isolate graphical

nsys profile --cuda-memory-usage --gpu-metrics-devices=help
nsys profile --cuda-memory-usage true  --gpu-metrics-devices all --trace-fork-before-exec true  python ./vllm_profile/vllm_profile.py