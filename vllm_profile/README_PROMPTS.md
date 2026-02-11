# vLLM Profiling - Prompt Loader

This directory contains the vLLM profiling script with prompts separated into an external file for easier management.

## Files

- **vllm_profile.py**: Main profiling script
- **prompt_loader.py**: Module for loading prompts from external files
- **prompts.json**: Default prompt file containing all test prompts
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
   export VLLM_PROMPTS_FILE=/path/to/your/prompts.json
   python vllm_profile.py
   ```

2. **Creating prompts in different formats:**

   **JSON format (prompts.json):**
   ```json
   [
     "First prompt text here",
     "Second prompt text here",
     "Third prompt text here"
   ]
   ```

   **Text format (prompts.txt):**
   ```
   First prompt text here
   ---
   Second prompt text here
   ---
   Third prompt text here
   ```

### Prompt Loader API

The `prompt_loader` module provides the following functions:

```python
from prompt_loader import load_prompts, get_default_prompts_path

# Load prompts from default location
prompts = load_prompts(get_default_prompts_path())

# Load prompts from custom JSON file
prompts = load_prompts("my_prompts.json")

# Load prompts from text file
prompts = load_prompts("my_prompts.txt")

# Auto-detect file format
prompts = load_prompts("prompts_file", file_type="auto")
```

#### Function Reference

- **`load_prompts(file_path, file_type="auto")`**: Load prompts from a file
  - `file_path`: Path to the prompt file
  - `file_type`: "json", "text", or "auto" (default: "auto")
  - Returns: List of prompt strings

- **`load_prompts_from_json(file_path)`**: Load prompts from JSON file
  - `file_path`: Path to JSON file containing an array of prompts
  - Returns: List of prompt strings

- **`load_prompts_from_text(file_path, delimiter="\n---\n")`**: Load prompts from text file
  - `file_path`: Path to text file
  - `delimiter`: String separating prompts (default: "\n---\n")
  - Returns: List of prompt strings

- **`get_default_prompts_path()`**: Get path to default prompts.json
  - Returns: Absolute path to prompts.json in the module directory

## Benefits of Separated Prompts

1. **Easier Management**: Edit prompts without modifying the main script
2. **Version Control**: Track prompt changes separately from code changes
3. **Reusability**: Share prompt sets across different scripts or experiments
4. **Flexibility**: Switch between different prompt sets using environment variables
5. **Readability**: Cleaner main script without long embedded strings

## Example: Creating Custom Prompt Set

Create a file `my_custom_prompts.json`:

```json
[
  "Explain how transformers work in machine learning.",
  "Describe the attention mechanism in neural networks.",
  "What are the benefits of using GPU acceleration for deep learning?"
]
```

Use it:

```bash
export VLLM_PROMPTS_FILE=my_custom_prompts.json
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

- The default `prompts.json` contains 63 comprehensive prompts focused on GPU/CUDA optimization topics
- Prompts are loaded once at script startup
- Invalid JSON or missing files will raise appropriate errors with helpful messages
- For production use, consider creating prompt sets tailored to your specific use case
