# オーケストレーター使用ガイド

## 概要

`ExperimentOrchestrator`は、LLMを使用して自律的に実験を計画・実行・分析するメインコンポーネントです。

## セットアップ

### 1. LLMプロバイダーの設定

#### Gemini (推奨)
```bash
export GOOGLE_API_KEY="your-api-key"
uv add google-generativeai
```

#### OpenAI
```bash
export OPENAI_API_KEY="your-api-key"
uv add openai
```

#### Anthropic
```bash
export ANTHROPIC_API_KEY="your-api-key"
uv add anthropic
```

## 使用方法

### 基本的な実行

```bash
# Geminiを使用（デフォルト）
uv run python core/agent/orchestrator.py \
  --competition titanic \
  --max-iterations 5

# OpenAIを使用
uv run python core/agent/orchestrator.py \
  --competition titanic \
  --max-iterations 5 \
  --llm-provider openai

# 自動提出を有効化
uv run python core/agent/orchestrator.py \
  --competition titanic \
  --max-iterations 10 \
  --auto-submit
```

### Pythonスクリプトから使用

```python
from pathlib import Path
from core.agent.orchestrator import ExperimentOrchestrator

# オーケストレーターの初期化
orchestrator = ExperimentOrchestrator(
    competition_name="titanic",
    base_dir=Path.cwd(),
    llm_provider="gemini",  # or "openai", "anthropic"
    model_name="gemini-2.0-flash-exp"  # オプション
)

# 実験ループの実行
orchestrator.run(
    max_iterations=5,
    auto_submit=False
)
```

## 実験ループの流れ

各イテレーションで以下のステップが実行されます：

1. **仮説生成**: LLMがコンペドキュメントと実験履歴を分析し、改善の仮説を生成
2. **実装**: 仮説に基づいて実験スクリプトを作成
3. **実験実行**: 実験を実行し、ログとメトリクスをキャプチャ
4. **結果分析**: LLMが結果を分析し、改善の有無を判定
5. **意思決定**: 改善が見られた場合は提出を検討
6. **報告**: イテレーション結果をレポート

## 出力

### 実験スクリプト
`competitions/[competition]/experiments/exp00X_[name].py`

### イテレーションレポート
`competitions/[competition]/output/iteration_X_report.md`

### 実験履歴
`competitions/[competition]/output/experiment_history.json`

### エージェント状態
`core/agent/state/[competition]_state.json`

## カスタマイズ

### プロンプトの編集

エージェントの動作は`prompts/`ディレクトリのプロンプトで制御できます：

- `agent_system.md`: エージェントの役割とワークフロー
- `hypothesis_generation.md`: 仮説生成のロジック
- `result_analysis.md`: 結果分析の方法

### モデルの変更

```python
orchestrator = ExperimentOrchestrator(
    competition_name="titanic",
    base_dir=Path.cwd(),
    llm_provider="gemini",
    model_name="gemini-1.5-pro"  # より高性能なモデル
)
```

## トラブルシューティング

### LLM APIエラー

- APIキーが正しく設定されているか確認
- レート制限に達していないか確認
- ネットワーク接続を確認

### 実験が失敗する

- ログを確認: `output/[exp_name]/`
- エージェント状態を確認: `core/agent/state/[competition]_state.json`
- 手動で実験スクリプトを実行してデバッグ

### 状態のリセット

```bash
# エージェント状態をリセット
rm core/agent/state/titanic_state.json

# 実験履歴をリセット
rm competitions/titanic/output/experiment_history.json
```

## ベストプラクティス

1. **少ないイテレーションから開始**: 最初は`--max-iterations 3`程度で動作確認
2. **自動提出は慎重に**: 最初は`--auto-submit`なしで実行し、結果を確認
3. **定期的な状態確認**: 実験履歴とレポートを定期的にレビュー
4. **プロンプトの調整**: 結果が期待通りでない場合はプロンプトを調整
