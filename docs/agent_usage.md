# エージェント使用ガイド

## 概要

このガイドでは、kautoの自律エージェントシステムの使用方法を説明します。

## 前提条件

- Phase 1, 2が完了していること
- コンペティションがセットアップされていること
- 必要な依存関係がインストールされていること

## 基本的な使い方

### 1. エージェントツールの使用

エージェントツールは個別に使用できます：

```python
from pathlib import Path
from core.agent.tools import AgentTools

# ツールの初期化
tools = AgentTools(
    base_dir=Path.cwd(),
    competition_name="titanic"
)

# 実験の実行
result = tools.run_experiment("experiments/exp001_baseline.py")
print(result)

# ファイルの読み込み
content = tools.read_file("docs/README.md")

# 結果の分析
analysis = tools.analyze_results("exp001_baseline")
```

### 2. 実験ランナーの使用

Hydra設定を使用して実験を実行：

```python
from pathlib import Path
from core.experiment.runner import ExperimentRunner

runner = ExperimentRunner(base_dir=Path.cwd())

# 基本的な実行
result = runner.run(
    competition_name="titanic",
    exp_script="experiments/exp001_baseline.py"
)

# 設定オーバーライド付き実行
result = runner.run(
    competition_name="titanic",
    exp_script="experiments/exp001_baseline.py",
    config_overrides={
        "model.learning_rate": 0.1,
        "experiment.seed": 123
    }
)
```

### 3. 実験トラッカーの使用

実験を記録・追跡：

```python
from pathlib import Path
from core.experiment.tracker import ExperimentTracker

tracker = ExperimentTracker(
    competition_dir=Path("competitions/titanic")
)

# 実験を記録
tracker.log_experiment(
    exp_name="exp001_baseline",
    config={"model": "lgbm", "features": "basic"},
    metrics={"cv_score": 0.8372},
    success=True,
    notes="Baseline experiment"
)

# ベスト実験を取得
best = tracker.get_best_experiment(metric_name="cv_score")
print(f"Best experiment: {best['exp_name']}, Score: {best['metrics']['cv_score']}")

# 最近の実験を取得
recent = tracker.get_recent_experiments(n=5)
```

## 自律エージェントの実行（今後実装予定）

将来的には、以下のようにエージェントを実行できるようになります：

```bash
# 基本的な実行
uv run python core/agent/orchestrator.py \
  --competition titanic \
  --max-iterations 10

# カスタム設定で実行
uv run python core/agent/orchestrator.py \
  --competition titanic \
  --max-iterations 5 \
  --auto-submit false
```

## 設定管理

### Hydra設定の構造

```yaml
# competitions/titanic/configs/config.yaml
defaults:
  - features: basic
  - model: lgbm_default

experiment:
  name: exp001_baseline
  seed: 42

data:
  train_path: data/raw/train.csv
  test_path: data/raw/test.csv
```

### 設定のオーバーライド

コマンドラインから設定をオーバーライド：

```bash
uv run python experiments/exp001_baseline.py \
  experiment.name=exp001_custom \
  model.learning_rate=0.1 \
  experiment.seed=123
```

## プロンプトのカスタマイズ

エージェントの動作は`prompts/`ディレクトリのプロンプトで制御できます：

- `agent_system.md`: エージェントの基本動作
- `hypothesis_generation.md`: 仮説生成のロジック
- `result_analysis.md`: 結果分析の方法

これらのファイルを編集することで、エージェントの動作をカスタマイズできます。

## トラブルシューティング

### 実験が失敗する

1. ログを確認: `output/[exp_name]/`
2. 設定を確認: `configs/`
3. エラーメッセージを確認

### 状態がおかしい

エージェントの状態をリセット：

```bash
rm core/agent/state/titanic_state.json
```

### 実験履歴を確認

```bash
cat competitions/titanic/output/experiment_history.json | jq
```

## ベストプラクティス

1. **段階的な変更**: 一度に1つの変更のみ
2. **設定の記録**: すべての実験設定を記録
3. **定期的な提出**: 改善が見られたら提出
4. **ドキュメント化**: 学びをドキュメントに記録
