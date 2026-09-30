# mdeagent
## Requirements
1. It is recommended to use `uv` for project management.
2. You need a LLM. It is developed with the OpenAI interface, but with adjusting the model it should work as well.

## Installation
1. Clone the repository: `git clone git@github.com:BequemeDecke/mdagent.git`
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
