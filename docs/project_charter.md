# Project Charter: kauto (Kaggle Autonomous Agent)

## 1. プロジェクト概要
**kauto** は、Kaggleコンペティションにおいて、人間が戦略的なディレクションを行い、AIエージェントが自律的に実験・改善・サブミットを行うための協働システムです。

## 2. ビジョン & ゴール
**"Human Director, Agent Experimenter"**
- **人間**: コンペの選定、大まかな方針策定、エージェントの成果物のレビュー、戦略的判断。
- **エージェント**: コーディング、特徴量エンジニアリング、モデル学習、ハイパーパラメータ探索、サブミット、実験レポート作成。

**中長期ゴール**:
エージェントが自律的に試行錯誤を繰り返し、KaggleのLeaderboardで競争力のあるスコアを達成できるシステムを構築する。

## 3. スコープ & ロードマップ

### Phase 1: Environment & Baseline (完了)
- **目的**: 手動でも動作する基本的なパイプラインの確立。
- **成果**:
    - `uv` による高速なPython環境構築。
    - `core` (汎用) と `competitions` (コンペ依存) のディレクトリ分離。
    - Titanicコンペにおけるベースライン作成とKaggle API経由のサブミット成功。
    - コンペ情報の自動取得・ドキュメント化ツール (`CompetitionManager`)。

### Phase 2: Experiment Management & Modularity (完了)
- **目的**: 実験の再現性と効率性を高めるための基盤整備。
- **成果**:
    - 実験管理構造の確立 (`src/` と `experiments/` の分離)。
    - 特徴量とモデルの基底クラス実装 (`BaseFeature`, `LGBMModel`)。
    - 実験ごとの成果物を個別管理する `output/` ディレクトリ構造。
    - 実験ワークフローのドキュメント化。

### Phase 3: Agent Integration (完了)
- **目的**: LLMエージェントによる自律的な実験ループの実現。
- **成果**:
    - エージェントコア実装 (`core/agent/`: base, tools, mcp_server)
    - 実験管理システム (`core/experiment/`: runner, tracker)
    - Hydra設定管理システム
    - エージェントプロンプト (`prompts/`)
    - Antigravity統合（MCPサーバー経由）
    - 包括的なドキュメント
    - エンドツーエンドテスト
- **備考**:
    - 外部LLM API呼び出しではなく、Antigravity（このAIアシスタント）を直接使用する方式を採用
    - MCPサーバーを介してツールを提供

## 4. アーキテクチャ設計
汎用性を高めるため、コアロジックとコンペ固有の実装を明確に分離します。

```text
kauto/
├── core/               # 汎用フレームワーク (コンペに依存しない)
│   ├── experiment/     # 実験実行・管理ロジック (Phase 3で実装予定)
│   ├── agent/          # エージェント制御・インターフェース (Phase 3で実装予定)
│   └── utils/          # 共通ユーティリティ (CompetitionManager, Submitter等)
├── competitions/       # コンペごとの実装
│   └── [competition]/  # (例: titanic)
│       ├── data/       # データセット (raw/processed)
│       ├── docs/       # コンペドキュメント (README, OVERVIEW, DATA)
│       ├── src/        # 共通ライブラリ (features, models, utils)
│       ├── experiments/# 実験スクリプト (exp001.py, exp002.py, ...)
│       ├── output/     # 実験結果 (gitignore対象)
│       ├── notebooks/  # 分析用ノートブック
│       └── submissions/# 提出用ファイル
├── docs/               # プロジェクトドキュメント (本ファイル等)
├── scripts/            # 開発・デバッグ用スクリプト
└── prompts/            # エージェント用システムプロンプト
```

## 5. 技術スタック
- **言語**: Python 3.11+
- **パッケージ管理**: uv
- **機械学習**: LightGBM, scikit-learn, pandas, numpy, joblib
- **実験管理**: 実験スクリプトベース (Phase 3でHydra/MLflow/WandB導入予定)
- **API**: Kaggle API
- **Web Scraping**: requests, BeautifulSoup4

## 6. 運用ルール
- **スモールスタート**: 最初から完璧を目指さず、動くものを作ってから拡張する。
- **ドキュメント指向**: 設計や仕様は `docs/` 配下にまとめ、エージェントが参照できるようにする。
- **実験の独立性**: 各実験は独立したスクリプトとして管理し、成果物は `output/[exp_name]/` に保存する。
