# Exp006 Phase 2 検証計画

Exp006 Phase 2で実装された機能（構造変更、ArcFace、Embedding Head、EMA）の効果を検証し、最終的なモデル構成（ArcFace推論を採用するか否か）を決定するための手順です。

## 🔬 実験シナリオ一覧

以下のスクリプトをKaggle GPU環境で順番に実行してください。すべて `StratifiedGroupKFold` の1-Fold Hold-out (20%) 検証を行います。

| 実験ID | スクリプト名 | 設定内容 | 目的 |
| :--- | :--- | :--- | :--- |
| **(A) Baseline** | `experiments/exp006_phase2_structural.py` | `use_arcface=False`<br>`use_embedding_head=False` | リファクタリング後の正常性確認。<br>基準となるスコアを算出。 |
| **(B) ArcFace** | `experiments/exp006_phase2_arcface.py` | `use_arcface=True`<br>`use_embedding_head=False` | ArcFace Headの効果検証。<br>BGクラスの分離性能が上がるか？ |
| **(C) Embedding** | `experiments/exp006_phase2_embedding.py` | `use_arcface=False`<br>`use_embedding_head=True` | Embedding Head (BN->FC->BN) の効果検証。<br>Lossは変えずにHead構造だけで改善するか？ |
| **(D) EMA** | `experiments/exp006_phase2_ema.py` | `use_arcface=False`<br>`use_embedding_head=False`<br>`EMA=True` | EMAの効果検証。<br>Validationスコアの安定性と最大値の向上を確認。 |

## 📊 比較・評価ポイント

各実験のログ出力（`Val F1`など）を比較し、以下の観点で評価します。

### 1. **Baseline vs ArcFace** (A vs B)
*   **BG Stats (Recall/Precision)** に注目。
    *   ArcFaceはクラス間の角度を広げるため、Unknown（背景）とPlayerの境界が明確になり、**誤検知（Background False Positive）が減ることが期待**されます。
    *   もし `Player F1` が維持されつつ、`BG Precision` が向上していれば、ArcFace採用の強い根拠となります。

### 2. **Baseline vs Embedding** (A vs C)
*   ArcFaceを使わなくても、Embedding Head（BN-FC-BN）を追加するだけで精度が向上する場合があります。
*   ArcFaceよりも学習が安定しやすいメリットがあるため、Bと同等の精度ならCを採用する選択肢もあります。

### 3. **EMAの安定性** (A vs D)
*   `[EMA] F1` が `[Normal] F1` より安定して高いか、あるいはピーク性能が高いかを確認します。
*   基本的には無条件で採用して良いテクニックですが、計算コスト（メモリ）とのトレードオフを確認します。

## ✅ 採用判断（Decision Tree）

1.  **(B) ArcFace** の `Player F1` が (A) より著しく高い、または `BG Precision` が大幅に改善した？
    *   **YES** -> **ArcFace推論を採用**。Phase 3ではArcFaceベースの推論パイプライン構築へ。
    *   **NO** -> (C) を確認。

2.  **(C) Embedding** が (A) より良い？
    *   **YES** -> **Embedding Head + CrossEntropy** を採用。ArcFaceは不採用。
    *   **NO** -> **Standard Head (Baseline)** を維持。

3.  **(D) EMA** が有効？
    *   **YES** -> 採用モデルにEMAを統合。

## 📝 次のアクション
Kaggleで上記4つの実験を回し、結果（ログ）を共有してください。その結果に基づいて、Exp007以降の構成を確定します。
