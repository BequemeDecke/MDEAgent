# mdeagent
## Requirements
1. It is recommended to use `uv` for project management.
2. You need a LLM. It is developed with the OpenAI interface, but with adjusting the model it should work as well.

## Installation
1. Clone the repository: `git clone git@github.com:BequemeDecke/MDEAgent.git`
2. Init the submodules (mdagent-skills): `git submodule init`
3. Install the dependencies: `uv sync` or `pip install -r requirements.txt`

## Usage

### Basic Command
```bash
uv run main.py -s <source_model_path> -t <target_model_path> -g <group_id> -a <artifact_id> <task_specification>
```

### Example
```bash
uv run main.py \
    -s .mdeagent-tests/setup/metamodels/Families \
    -t .mdeagent-tests/setup/metamodels/Persons \
    -g de.hofuniversity \
    -a MDEAgentFamilyToPerson \
    "Transform the Families model to the Persons model, ensure that the transformation keeps the consistency of the data synchronized."
```

### Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--source-model-path` | `-s` | ✅ | Path to the source model (e.g., Families metamodel) |
| `--target-model-path` | `-t` | ✅ | Path to the target model (e.g., Persons metamodel) |
| `--group-id` | `-g` | ✅ | Maven group ID (e.g., de.hofuniversity) |
| `--artifact-id` | `-a` | ✅ | Maven artifact ID (e.g., MDEAgentFamilyToPerson) |
| `task_specification` | — | ✅ | Description of the transformation task |

| Optional Argument | Short | Default | Description |
|---|---|---|---|
| `--log-level` | `-l` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--use-langfuse` | — | `false` | Enable LangFuse for monitoring and tracing |

### Example with Optional Flags
```bash
uv run main.py \
    -s .mdeagent-tests/setup/metamodels/Families \
    -t .mdeagent-tests/setup/metamodels/Persons \
    -g de.hofuniversity \
    -a MDEAgentFamilyToPerson \
    --log-level DEBUG \
    --use-langfuse \
    "Transform the Families model to the Persons model."
```

## Benchmark

### Fetching Models

To fetch the latest available LLM models from the FAU server, you need to:

1. **Set your API key** as an environment variable:
   ```bash
   export FAU_API_KEY="your-api-key-here"
   ```

2. **Run the fetch script**:
   ```bash
   uv run fetch-models
   ```

   This will:
   - Fetch models from the FAU server and save them to `.mdeagent-benchmark/models.json`
   - Convert the JSON to CSV and save it to `.mdeagent-benchmark/models.csv`

### Running the Benchmark

The benchmark runs the MDEAgent against all fetched models and collects results via LangFuse.

```bash
uv run benchmark
```

**Benchmark arguments**:

| Argument | Default | Description |
|---|---|---|
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--num-iterations` | `5` | Number of times each model should be run |
| `--max-concurrency` | `3` | Maximum number of tasks to run in parallel |

**Example**:
```bash
uv run benchmark --log-level DEBUG --num-iterations 3 --max-concurrency 5
```

Results are saved to `.mdeagent-benchmark/results.json` and individual result files in `.mdeagent-benchmark/results/`.

## Scripts

### JSON to CSV Conversion

Convert the fetched models JSON file to CSV format:

```bash
uv run json-to-csv .mdeagent-benchmark/models.json .mdeagent-benchmark/models.csv
```

**Arguments**:
- `input`: Path to the input JSON file
- `output`: Path to the output CSV file

The `fetch_models.py` script calls this automatically after fetching models, so you usually don't need to run it manually.
