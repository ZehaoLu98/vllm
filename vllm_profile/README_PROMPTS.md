# vLLM Profiling - Prompt Loader

This directory contains the vLLM profiling script with prompts separated into an external file for easier management.

## Files

- **vllm_profile.py**: Main profiling script
- **prompt_loader.py**: Module for loading prompts from external files
- **prompts.jsonl**: Default prompt file containing all test prompts
- **run_batch_experiments.sh**: Batch experiment runner script

## Usage

### Basic Usage

Run the profiler with default prompts:

```bash
python vllm_profile.py
```

### Using Custom Prompts

You can use custom prompts by either:

1. **Setting an environment variable:**
   ```bash
  export VLLM_PROMPTS_FILE=/path/to/your/prompts.jsonl
   python vllm_profile.py
   ```

2. **Providing a JSONL prompt file:**

  **JSONL format (prompts.jsonl):**
  ```json
  {"prompt":"First prompt text here","output_tokens":100}
  {"prompt":"Second prompt text here","output_tokens":256}
  {"prompt":"Third prompt text here"}
  ```

  - `prompt` is required.
  - `output_tokens` is optional; when omitted, the profiler uses its default value.

### Prompt Loader API

The `prompt_loader` module provides the following functions:

```python
from prompt_loader import load_prompts, get_default_prompts_path

# Load prompts from default location
prompts = load_prompts(get_default_prompts_path())

# Load prompts from custom JSONL file
prompts = load_prompts("my_prompts.jsonl")

# Explicit JSONL type is also supported
prompts = load_prompts("my_prompts.jsonl", file_type="jsonl")
```

#### Function Reference

- **`load_prompts(file_path, file_type="auto")`**: Load prompts from a file
  - `file_path`: Path to the prompt file
  - `file_type`: "jsonl" or "auto" (default: "auto")
  - Returns: List of prompt strings

- **`load_prompts_from_jsonl(file_path)`**: Load prompts from JSONL file
  - `file_path`: Path to JSONL file containing one object per line
  - Returns: List of prompt strings

- **`load_prompts_with_output_tokens(file_path, default_output_tokens=512)`**:
  Load prompts and per-prompt output token counts from JSONL
  - Returns: Tuple `(prompts, output_tokens)`

- **`get_default_prompts_path()`**: Get path to default prompts.jsonl
  - Returns: Absolute path to prompts.jsonl in the module directory

## Benefits of Separated Prompts

1. **Easier Management**: Edit prompts without modifying the main script
2. **Version Control**: Track prompt changes separately from code changes
3. **Reusability**: Share prompt sets across different scripts or experiments
4. **Flexibility**: Switch between different prompt sets using environment variables
5. **Readability**: Cleaner main script without long embedded strings

## Example: Creating Custom Prompt Set

Create a file `my_custom_prompts.jsonl`:

```json
{"prompt":"Explain how transformers work in machine learning.","output_tokens":128}
{"prompt":"Describe the attention mechanism in neural networks.","output_tokens":128}
{"prompt":"What are the benefits of using GPU acceleration for deep learning?"}
```

Use it:

```bash
export VLLM_PROMPTS_FILE=my_custom_prompts.jsonl
python vllm_profile.py
```

## Command Line Arguments

The main script supports various SchedulerConfig parameters:

```bash
python vllm_profile.py \
    --max_num_seqs 256 \
    --max_num_batched_tokens 8192 \
    --enable_chunked_prefill true \
    --scheduling_policy fcfs
```

See `python vllm_profile.py --help` for all available options.

## Notes

- The default `prompts.jsonl` contains comprehensive prompts focused on GPU/CUDA optimization topics
- Prompts are loaded once at script startup
- Only `.jsonl` prompt files are supported; `.json` and `.txt` files are rejected with explicit errors
- For production use, consider creating prompt sets tailored to your specific use case
