curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
