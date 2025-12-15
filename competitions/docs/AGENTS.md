# AI Agent Instructions (AGENTS.md)

このドキュメントは、このプロジェクトの開発を支援するAIエージェント（Gemini等）に向けた指示書です。
開発フロー、環境差異の吸収方針、コーディング規約などを定義します。

## 🛠 開発ワークフロー

このプロジェクトでは「ローカル開発 → Kaggle実行」のハイブリッドフローを採用しています。

### 1. ローカル環境 (Mac)
*   **用途**: ノートブックの作成、ロジックの実装、デバッグ、Small Dataでの動作検証
*   **計算資源**: CPU (またはMPS)
*   **データ**: フルデータまたはサブセット
*   **役割**: エラーなくコードが動くこと、ロジックが正しいことの確認

### 2. Kaggle環境 (Global)
*   **用途**: Full DataでのGPU学習、長時間学習、スコア計測
*   **計算資源**: NVIDIA GPU (P100/T4 x2)
*   **データ**: Read-only Input (`/kaggle/input`) + Writable Output (`/kaggle/working`)
*   **役割**: 本番学習とモデル生成

## 💻 コーディング規約・考慮事項

AIエージェントはコードを生成・修正する際、以下の点を常に考慮してください。

### パス管理の抽象化
ローカルとKaggle環境の両方でコードを変更せずに動作させるため、パス設定を分離してください。

*   **`setup_directories(base_dir, data_dir)`の使用を徹底する**
    *   `base_dir`: 出力先（ローカルならカレント、Kaggleなら `/kaggle/working`）
    *   `data_dir`: データ入力元（ローカルなら `.`、Kaggleなら `/kaggle/input/...`）

### デバイス管理 (CPU/GPU/MPS)
環境に応じてデバイスを自動選択するロジックを入れてください。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
```

### デバッグモードの実装
ローカルでの動作確認を高速化するため、データセットの一部だけを使用する `debug` フラグや引数を推奨します。

```python
# 例
if debug:
    train_df = train_df.iloc[:100]
```

## 🤖 エージェントへの特記事項

*   **Exp003以降**: GPU前提の実験はipynb形式で作成し、Kaggle Notebookとして実行することを想定してください。
*   **同期**: ローカルで作成した `.ipynb` や `.py` はGit管理されます。Kaggleで実行した結果（ログやモデル）は、重要なもののみローカルにダウンロードして `output/` に整理するか、Kaggle Dataset経由で管理します。
