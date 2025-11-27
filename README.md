# kauto - Kaggle Autonomous Agent

**kauto** は、Kaggleコンペティションにおける実験・改善・提出を自律的に行うAIエージェントシステムのためのフレームワークです。
人間が「Director」として戦略を決定し、AIエージェントが「Experimenter」として実装と検証を行う協働モデルを目指しています。

## プロジェクトステータス

- **Phase 1 (Environment & Baseline)**: ✅ 完了
- **Phase 2 (Experiment Management)**: ✅ 完了
- **Phase 3 (Agent Integration)**: 🚧 次フェーズ

詳細は [Project Charter](docs/project_charter.md) を参照してください。

## 主な機能

- 🔄 コンペティション環境の自動セットアップ
- 📊 コンペ情報の自動取得・ドキュメント化
- 🧪 実験管理構造 (src/experiments分離)
- 🤖 特徴量・モデルの基底クラス
- 📤 Kaggle API経由の自動提出

## Prerequisites

- **Python 3.11+**
- **uv**: 高速なPythonパッケージマネージャ
  - [インストール方法](https://github.com/astral-sh/uv)
- **Kaggle API Token**: `kaggle.json` を `~/.kaggle/` に配置してください。

## Setup

1. **リポジトリのクローン**
   ```bash
   git clone <repository-url>
   cd kauto
   ```

2. **依存関係のインストール**
   ```bash
   uv sync
   ```

## Usage

### 1. 新しいコンペティションのセットアップ
指定したコンペティションのディレクトリ作成、データダウンロード、ドキュメント生成（Overview/Dataページの取得含む）を一括で行います。

```bash
# 例: Titanicコンペの場合
uv run python core/utils/competition_manager.py titanic
```

これにより `competitions/titanic/` 配下に以下の構造が生成されます：
- `data/raw/`: 生データ
- `data/processed/`: 加工済みデータ（空フォルダ）
- `src/`: 共通ライブラリ（空フォルダ）
- `experiments/`: 実験スクリプト（空フォルダ）
- `docs/`: コンペ情報のドキュメント (`README.md`, `OVERVIEW.md`, `DATA.md`)

### 2. ベースライン実験の実行 (Titanicの例)
Titanicコンペを例とした、実験スクリプトの実行手順です。

**実験の実行**
`experiments/` 配下のスクリプトを実行します。これにより、特徴量生成、学習、推論、提出ファイルの作成が一括で行われます。
```bash
uv run python competitions/titanic/experiments/exp001_baseline.py
```

実行後、以下の成果物が生成されます：
- `output/exp001_baseline/`: 学習済みモデル、ログ、加工済みデータ
- `submissions/exp001_baseline_submission.csv`: 提出用ファイル

**Kaggleへの提出**
作成された提出ファイルをKaggleにアップロードします。
```bash
uv run python core/utils/submitter.py titanic \
  competitions/titanic/submissions/exp001_baseline_submission.csv \
  "Baseline submission via kauto"
```

### 3. コンペ情報の再取得
OverviewやDataページの情報を再取得してドキュメントを更新したい場合は、以下のユーティリティを使用します。
```bash
uv run python core/utils/kaggle_page_fetcher.py titanic
```

## Directory Structure

```text
kauto/
├── core/               # 汎用フレームワーク (コンペ非依存)
│   ├── utils/          # ユーティリティ (CompetitionManager, Submitter等)
│   └── ...
├── competitions/       # コンペごとの実装
│   └── [competition]/  # (例: titanic)
│       ├── data/       # データセット (raw/processed)
│       ├── docs/       # コンペドキュメント (README, OVERVIEW, DATA)
│       ├── src/        # 共通ライブラリ (特徴量, モデル, ユーティリティ)
│       ├── experiments/# 実験スクリプト (exp001.py, ...)
│       ├── output/     # 実験結果 (モデル, ログ等 - gitignore)
│       ├── notebooks/  # 分析用ノートブック
│       └── submissions/# 提出用ファイル
├── docs/               # プロジェクトドキュメント
├── scripts/            # 開発・デバッグ用スクリプト
└── prompts/            # エージェント用システムプロンプト
```

## ライセンス

MIT License
