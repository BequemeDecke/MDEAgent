# mdeagent
## Requirements
1. It is recommended to use `uv` for project management.
2. You need a LLM. It is developed with the OpenAI interface, but with adjusting the model it should work as well.

## Installation
1. Clone the repository: `git clone git@github.com:BequemeDecke/MDEAgent.git`
2. Init the submodules (mdeagent-metamodels): `git submodule update --init --recursive`
3. Install the dependencies: `uv sync` or `pip install -r requirements.txt`

## Environment Variables

All required environment variables are loaded from a `.env` file in the project root. Create this file based on the template below.

### Model Configuration

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | ✅ | API key for the LLM provider |
| `BASE_URL` | ✅ | Base URL for the LLM API |
| `BASE_MODEL` | ✅ | The base model to use |
| `CODING_MODEL` | ✅ | The coding model to use |
| `REQUEST_TIMEOUT` | ❌ | Request timeout in seconds (default: `60`) |
| `MAX_RETRIES` | ❌ | Maximum number of retries (default: `0`) |

### LangFuse Configuration

| Variable | Required | Description |
|---|---|---|
| `LANGFUSE_SECRET_KEY` | ✅ | LangFuse secret key |
| `LANGFUSE_PUBLIC_KEY` | ✅ | LangFuse public key |
| `LANGFUSE_BASE_URL` | ✅ | LangFuse server URL |

### Variables Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `UPDATED_FILE_INDEX` | ❌ | `13` | Index where the file path starts in the tool message content for the write_file tool |
| `TRANSFORMATION_CLASS_NAME` | ❌ | `MDEAgentTransformation` | Name of the generated transformation class |

### Agent Control Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `WORKFLOW_MAX_ITERATIONS` | ❌ | `5` | Maximum number of iterations for the workflow transformation process |
| `SUBGRAPH_MAX_ITERATIONS` | ❌ | `3` | Maximum number of iterations for a subgraph process |
| `TRANSFORMATION_IMPLEMENTATION_STRATEGY` | ❌ | `deep_agent` | Strategy for transformation implementation. Options: `deep_agent`, `hybrid_agent`, `template_based`, `pi` |

### Example `.env` File

```bash
# Model Configuration
API_KEY="your-api-key-here"
BASE_URL="https://api.your-provider.com/v1"
BASE_MODEL="your-base-model"
CODING_MODEL="your-coding-model"

# LangFuse Configuration
LANGFUSE_SECRET_KEY="your-secret-key"
LANGFUSE_PUBLIC_KEY="your-public-key"
LANGFUSE_BASE_URL="http://localhost:3000"

# Optional: Agent Control
# WORKFLOW_MAX_ITERATIONS=5
# SUBGRAPH_MAX_ITERATIONS=3
# TRANSFORMATION_IMPLEMENTATION_STRATEGY=deep_agent
```

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

The benchmark runs the MDEAgent against all fetched models and collects results via LangFuse. Therefore, a LangFuse server needs to be running. You can set it up using Docker:

[LangFuse Self-Hosting with Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose)

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
