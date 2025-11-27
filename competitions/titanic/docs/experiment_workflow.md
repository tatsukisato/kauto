# Experiment Workflow

This document describes the workflow for managing experiments in the `kauto` system.

## Directory Structure

We separate the **Library Code** (`src/`) from the **Experiment Definitions** (`experiments/`).

```text
competitions/[competition]/
├── src/                # Shared Library
│   ├── features/       # Feature engineering logic (Classes/Functions)
│   ├── models/         # Model wrappers (LGBM, XGB, etc.)
│   └── utils/          # Metrics, Logging, etc.
├── experiments/        # Experiment Scripts
│   ├── exp001_baseline.py
│   ├── exp002_add_feature.py
│   └── ...
└── output/             # Experiment Artifacts (Gitignored)
    ├── exp001_baseline/
    └── exp002_add_feature/
```

## Workflow

### 1. Define Features & Models (`src/`)
Reusable logic should be implemented in `src/`.
- **Features**: Inherit from `src.features.base.BaseFeature`.
- **Models**: Create wrapper classes in `src.models`.

### 2. Create an Experiment Script (`experiments/`)
Create a new python script (e.g., `exp002_new_idea.py`) in the `experiments/` folder.
This script should:
1.  **Import** necessary classes from `src`.
2.  **Define Configuration** (Hyperparameters, Feature sets).
3.  **Orchestrate** the pipeline:
    - Load Data
    - Generate Features (using `src.features`)
    - Train Model (using `src.models`)
    - Save Results (to `output/exp002_new_idea/`)
    - Generate Submission

### 3. Run the Experiment
Execute the script using `uv run`:

```bash
uv run python competitions/titanic/experiments/exp001_baseline.py
```

### 4. Review Results
Check the `output/[exp_name]/` directory for:
- Trained models
- Logs
- Predictions
- Feature importance plots (if implemented)

Check `submissions/` for the final submission file.
