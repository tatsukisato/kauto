# Antigravity エージェントガイド

## 概要

kAutoは、Antigravity（このAIアシスタント）を直接使用して自律的な実験ループを実行できます。外部LLM APIを呼び出す代わりに、Antigravityのエージェント機能を活用します。

## アーキテクチャ

```
Antigravity (あなた)
    ↓ 使用
MCP Server (core/agent/mcp_server.py)
    ↓ 呼び出し
Agent Tools (core/agent/tools.py)
    ↓ 実行
Experiments & Kaggle API
```

## セットアップ

MCPサーバーは既に実装されており、Antigravityから直接呼び出すことができます。

## 使用方法

### 1. コンペティションの設定

まず、作業するコンペティションを設定します：

```
コンペティションを"titanic"に設定してください。
```

### 2. 現状の確認

```
以下を確認してください：
1. コンペのドキュメント（docs/README.md）
2. 最近の実験履歴（直近5件）
3. 現在のベストスコア
```

### 3. 仮説の生成

```
以下の情報をもとに、次の実験の仮説を3つ提案してください：
- データの特性
- 既存の実験結果
- 改善の余地がある部分
```

### 4. 実験の実装

```
以下の仮説に基づいて、新しい実験スクリプトを作成してください：
[仮説の内容]

実験名: exp002_[description]
```

### 5. 実験の実行

```
experiments/exp002_[description].py を実行してください。
```

### 6. 結果の分析

```
exp002_[description] の結果を分析してください：
1. CVスコアは改善しましたか？
2. 何が成功/失敗の要因でしたか？
3. 次に試すべきことは何ですか？
```

### 7. 提出（改善があった場合）

```
改善が見られたので、exp002_[description] の結果をKaggleに提出してください。
```

## 利用可能なツール

Antigravityは以下のツールを使用できます：

### `set_competition`
コンペティションコンテキストを設定
```python
{
  "method": "set_competition",
  "params": {"competition_name": "titanic"}
}
```

### `read_file`
ファイルを読み込み
```python
{
  "method": "read_file",
  "params": {"file_path": "docs/README.md"}
}
```

### `write_code`
コードを書き込み
```python
{
  "method": "write_code",
  "params": {
    "file_path": "experiments/exp002_new.py",
    "content": "# experiment code..."
  }
}
```

### `run_experiment`
実験を実行
```python
{
  "method": "run_experiment",
  "params": {"exp_script": "experiments/exp002_new.py"}
}
```

### `analyze_results`
結果を分析
```python
{
  "method": "analyze_results",
  "params": {"exp_name": "exp002_new"}
}
```

### `get_experiment_history`
実験履歴を取得
```python
{
  "method": "get_experiment_history",
  "params": {"n": 5}
}
```

### `get_best_experiment`
ベスト実験を取得
```python
{
  "method": "get_best_experiment",
  "params": {"metric_name": "cv_score"}
}
```

### `submit_to_kaggle`
Kaggleに提出
```python
{
  "method": "submit_to_kaggle",
  "params": {
    "submission_path": "submissions/exp002_new_submission.csv",
    "message": "Improved feature engineering"
  }
}
```

## ワークフロー例

### 完全な実験サイクル

```
# 1. 初期化
コンペティションを"titanic"に設定してください。

# 2. 現状確認
以下を確認してください：
- docs/README.md の内容
- 最近の実験履歴（5件）
- 現在のベストスコア

# 3. 仮説生成
データと実験履歴を分析し、改善の仮説を3つ提案してください。

# 4. 実装
最も有望な仮説に基づいて、exp002_[name].py を実装してください。

# 5. 実行
実験を実行してください。

# 6. 分析
結果を分析し、改善の有無を判定してください。

# 7. 提出（条件付き）
改善が見られた場合のみ、Kaggleに提出してください。

# 8. 次のイテレーション
結果を踏まえて、次の仮説を提案してください。
```

## プロンプトのガイドライン

Antigravityに効果的に指示するためのガイドライン：

### 明確な指示
❌ 「実験を改善して」
✅ 「Age特徴量の欠損値処理を改善し、中央値の代わりにタイトルベースの推定を使用する実験を作成してください」

### 段階的な実行
❌ 「すべて自動でやって」
✅ 「まず仮説を3つ提案してください。その後、私が選んだ仮説を実装します」

### コンテキストの提供
❌ 「次の実験を実行して」
✅ 「前回の実験（exp001）ではCVスコア0.8372でした。Age特徴量の改善を試したいので、新しい実験を実装してください」

## ベストプラクティス

1. **段階的な進行**: 一度に1つの実験のみ実行
2. **明確なコミュニケーション**: 各ステップで期待する結果を明示
3. **履歴の確認**: 定期的に実験履歴を確認
4. **ドキュメント化**: 学びをドキュメントに記録
5. **検証**: 実験結果を必ず分析してから次へ進む

## トラブルシューティング

### ツールが見つからない
MCPサーバーが正しく設定されているか確認してください。

### 実験が失敗する
エラーログを確認し、段階的にデバッグしてください。

### 状態がおかしい
実験履歴と状態ファイルを確認してください：
- `competitions/titanic/output/experiment_history.json`
- `core/agent/state/titanic_state.json`
