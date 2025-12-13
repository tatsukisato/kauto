"""
Experiment logging utilities
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def log_experiment_to_markdown(
    exp_name: str,
    description: str,
    config: dict,
    results: dict,
    observations: list = None,
    next_steps: list = None,
    log_file: str = "docs/EXPERIMENTS.md"
):
    """Log experiment results to EXPERIMENTS.md
    
    Args:
        exp_name: Experiment name (e.g., "exp001_baseline")
        description: Brief description of the experiment
        config: Configuration dictionary
        results: Results dictionary with metrics
        observations: List of observations/learnings
        next_steps: List of next steps
        log_file: Path to the log file
    """
    log_path = Path(log_file)
    
    # Read existing content
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# 実験ログ\n\nこのファイルには全ての実験の記録を時系列で記載します。\n\n---\n\n"
    
    # Find the template section and remove it
    if "## テンプレート" in content:
        content = content.split("## テンプレート")[0].rstrip() + "\n\n---\n\n"
    
    # Create new experiment entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"## {exp_name}\n\n"
    entry += f"**実施日時**: {timestamp}\n\n"
    entry += f"### 概要\n{description}\n\n"
    
    # Configuration
    entry += "### 設定\n"
    if 'model' in config:
        entry += f"- **モデル**: {config['model']}\n"
    if 'features' in config:
        entry += f"- **特徴量**: {', '.join(config['features'])}\n"
    if 'validation' in config:
        entry += f"- **バリデーション**: {config['validation']}\n"
    if 'params' in config:
        entry += "- **パラメータ**:\n  ```yaml\n"
        for key, value in config['params'].items():
            entry += f"  {key}: {value}\n"
        entry += "  ```\n"
    entry += "\n"
    
    # Results
    entry += "### 結果\n"
    if 'macro_f1' in results:
        entry += f"- **Validation Macro F1**: {results['macro_f1']:.4f}\n"
    if 'best_iteration' in results:
        entry += f"- **Best Iteration**: {results['best_iteration']}\n"
    if 'train_samples' in results:
        entry += f"- **学習サンプル数**: {results['train_samples']:,}\n"
    if 'val_samples' in results:
        entry += f"- **バリデーションサンプル数**: {results['val_samples']:,}\n"
    entry += "\n"
    
    # Per-class F1 scores
    if 'per_class_f1' in results:
        entry += "### クラスごとのF1スコア\n\n"
        entry += "| Player | F1 Score | 備考 |\n"
        entry += "|--------|----------|------|\n"
        for label, f1 in sorted(results['per_class_f1'].items()):
            entry += f"| Player_{label} | {f1:.4f} | |\n"
        entry += "\n"
    
    # Observations
    if observations:
        entry += "### 観察・学び\n"
        for i, obs in enumerate(observations, 1):
            entry += f"{i}. {obs}\n"
        entry += "\n"
    
    # Next steps
    if next_steps:
        entry += "### 次のステップ\n"
        for step in next_steps:
            entry += f"- [ ] {step}\n"
        entry += "\n"
    
    entry += "---\n\n"
    
    # Add template at the end
    template = """## テンプレート（次回実験用）

```markdown
## exp00X_experiment_name

**実施日時**: YYYY-MM-DD HH:MM

### 概要
[実験の簡潔な説明]

### 設定
- **モデル**: 
- **特徴量**: 
- **バリデーション**: 
- **パラメータ**:

### 結果
- **Validation Macro F1**: 
- **Best Iteration**: 
- **学習サンプル数**: 
- **バリデーションサンプル数**: 

### 観察・学び
1. 
2. 

### 次のステップ
- [ ] 
```
"""
    
    # Insert new entry after the header
    lines = content.split('\n')
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == '---' and i > 0:
            header_end = i + 1
            break
    
    new_content = '\n'.join(lines[:header_end]) + '\n\n' + entry + template
    
    # Write back
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Experiment logged to {log_file}")


def log_experiment_to_json(
    exp_name: str,
    config: dict,
    results: dict,
    log_file: str = "output/experiment_history.json"
):
    """Log experiment to JSON file for programmatic access
    
    Args:
        exp_name: Experiment name
        config: Configuration dictionary
        results: Results dictionary
        log_file: Path to the JSON log file
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing logs
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Create new log entry
    entry = {
        'experiment': exp_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results
    }
    
    # Append and save
    logs.append(entry)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Experiment logged to {log_file}")


def get_experiment_summary(log_file: str = "output/experiment_history.json") -> pd.DataFrame:
    """Get summary of all experiments as DataFrame
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        DataFrame with experiment summary
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        return pd.DataFrame()
    
    with open(log_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    
    summary = []
    for log in logs:
        entry = {
            'experiment': log['experiment'],
            'timestamp': log['timestamp'],
            'macro_f1': log['results'].get('macro_f1', None),
            'best_iteration': log['results'].get('best_iteration', None),
        }
        summary.append(entry)
    
    return pd.DataFrame(summary)
