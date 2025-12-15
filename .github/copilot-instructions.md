# AI Coding Agent Instructions for kauto

## Project Overview
kauto is a framework for autonomous AI agents in Kaggle competitions. It separates generic framework code (`core/`) from competition-specific implementations (`competitions/[comp]/`). Agents act as "Experimenter" while humans are "Director", focusing on strategic decisions.

## Architecture
- **Core Framework** (`core/`): Reusable components (agent system, experiment management, utilities)
- **Competition-Specific** (`competitions/[comp]/`): Data processing, models, experiments
- **Agent System**: MCP server-based with tools for file ops, experiment execution, analysis
- **Data Flow**: Raw data → Processed (crops, features) → Models → Submissions

## Critical Workflows

### Environment Setup
- **Local Development**: Mac with CPU/MPS, small data subsets
- **Kaggle Execution**: GPU environment, full data
- **Dependency Management**: Use `uv sync` (not pip/conda)
- **Path Abstraction**: Always use `setup_directories(base_dir, data_dir)` to handle local vs Kaggle paths

### Experiment Execution
- **Local**: `uv run python competitions/[comp]/experiments/expXXX.py`
- **Kaggle**: Upload notebook or script, run on GPU
- **Debug Mode**: Set `DEBUG = not IS_KAGGLE` for small datasets (e.g., 100-200 samples)

### Competition Setup
- **New Competition**: `uv run python core/utils/competition_manager.py [comp_slug]`
- **Submission**: `uv run python core/utils/submitter.py [comp] submissions/file.csv "message"`

## Coding Conventions

### Device & Performance
```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = device.type == "cuda"
scaler = GradScaler(enabled=use_amp)
```

### Model Structure
- Use refactored `AtmaCupModel` with separate backbone/head
- Backbone: ResNet18 (frozen), Head: Classifier/ArcFace/Embedding
- Forward signature: `model(images, targets=labels)` for ArcFace training

### Validation & Metrics
- **StratifiedGroupKFold**: Group by `quarter`, stratify by `label_id`
- **Metrics**: Macro F1 (overall + player-only), Background recall/precision
- **Background Class**: Label -1 maps to class 11 in model

### Data Processing
- **Crops**: Generate player crops from bbox, background samples
- **Transforms**: Heavy aug for train (RandomResizedCrop, ColorJitter), resize-only for val/test
- **Datasets**: `MixedImageDataset` for train (players + BG), `ImageDataset` for test

### Experiment Structure
```python
# Standard pattern
exp_name = "expXXX_description"
dirs = setup_directories(base_dir, data_dir)
model = AtmaCupModel(num_classes=12, pretrained=True, freeze_backbone=True)
# Training loop with tqdm, validation, save best model
# Inference on test, create submission
save_results(results, output_dir, exp_name)
```

### Imports & Modules
- Import from `src/` for competition-specific utils
- Handle import errors gracefully (local vs Kaggle)
- Use relative paths in experiments, absolute in core

## Agent Integration
- **Tools**: run_experiment, read_file, write_code, analyze_results, submit_to_kaggle
- **State Management**: Track iterations, experiments, best scores in JSON
- **Prompts**: Use templates from `prompts/` for hypothesis generation, analysis
- **MCP Server**: Interface for external LLM orchestration

## Key Files
- `src/models.py`: AtmaCupModel with backbone/head separation
- `src/utils.py`: setup_directories, save_results, create_submission
- `core/agent/base.py`: Agent base class with state management
- `pyproject.toml`: Dependencies (uv-managed)
- `competitions/docs/AGENTS.md`: Environment-specific guidelines

## Common Patterns
- Experiments numbered sequentially (exp001_baseline, exp002_*, etc.)
- Outputs saved to `output/exp_name/`, submissions to `submissions/`
- Config via Hydra YAML in `configs/`
- Background augmentation: Generate synthetic BG samples to balance classes
- Cross-validation: Single hold-out fold for speed, full CV for final</content>
<parameter name="filePath">/Users/tatsuki/work/kaggle/atma/.github/copilot-instructions.md