# kauto - ドキュメント一覧

このディレクトリには、プロジェクト全体に関するドキュメントが格納されています。

## プロジェクトドキュメント

### [project_charter.md](project_charter.md)
プロジェクトの憲章。ビジョン、ゴール、ロードマップ、アーキテクチャ設計、技術スタック、運用ルールを定義しています。

### [agent_architecture.md](agent_architecture.md)
エージェントシステムのアーキテクチャ詳細。コンポーネント構成、データフロー、状態管理について説明しています。

### [agent_usage.md](agent_usage.md)
エージェントツール（AgentTools、ExperimentRunner、ExperimentTracker）の使用方法ガイド。

### [antigravity_agent_guide.md](antigravity_agent_guide.md) ⭐ 推奨
Antigravity（このAIアシスタント）を使用した自律実験ループの実行ガイド。MCPサーバー経由でツールを使用する方法を説明しています。

### [orchestrator_usage.md](orchestrator_usage.md)
外部LLM APIを使用したオーケストレーターの使用ガイド（参考用）。

## コンペ固有のドキュメント

各コンペティションのドキュメントは、`competitions/[competition]/docs/` に配置されています。

例: `competitions/titanic/docs/`
- `README.md`: コンペの概要とデータ構造
- `OVERVIEW.md`: Kaggleのoverviewページの内容
- `DATA.md`: Kaggleのdataページの内容
- `experiment_workflow.md`: 実験の進め方

## 推奨される読む順序

1. **初めての方**: `project_charter.md` → `antigravity_agent_guide.md`
2. **開発者**: `agent_architecture.md` → `agent_usage.md`
3. **実験を始める**: `antigravity_agent_guide.md` → コンペ固有の`experiment_workflow.md`
