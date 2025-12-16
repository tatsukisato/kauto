# Exp006 Phase 2 検証計画

Exp006 Phase 2で実装された機能（構造変更、ArcFace、Embedding Head、EMA）の効果を検証し、最終的なモデル構成（ArcFace推論を採用するか否か）を決定するための手順です。

## 🔬 実験シナリオ一覧

以下のスクリプトをKaggle GPU環境で順番に実行してください。すべて `StratifiedGroupKFold` の1-Fold Hold-out (20%) 検証を行います。

| 実験ID | スクリプト名 | 設定内容 | 目的 |
| :--- | :--- | :--- | :--- |
| **(A) Baseline** | `experiments/exp006_phase2_structural.py` | `use_arcface=False`<br>`use_embedding_head=False` | リファクタリング後の正常性確認。基準スコア算出。 |
| **(B) ArcFace** | `experiments/exp006_phase2_arcface.py` | `use_arcface=True`<br>`use_embedding_head=False` | ArcFace Headの効果検証（BGとPlayerの分離性能）。 |
| **(C) Embedding** | `experiments/exp006_phase2_embedding.py` | `use_arcface=False`<br>`use_embedding_head=True` | Embedding Head (BN->FC->BN) の効果検証（Head構造単体の寄与）。 |
| **(D) ArcFace + Embedding** | `experiments/exp006_phase2_arcface_embedding.py`* | `use_arcface=True`<br>`use_embedding_head=True` | ArcFace と Embedding Head を組み合わせた効果検証。両者の相乗効果を確認。 |
| **(E) D + EMA** | `experiments/exp006_phase2_arcface_embedding_ema.py`* | `use_arcface=True`<br>`use_embedding_head=True`<br>`EMA=True` | (D) に EMA を組み合わせた場合の安定性とピーク性能を検証。 |

\* D/E 用のラッパースクリプト（既存実験の設定を組み合わせる小スクリプト）を作成してください。既存 model 実装がフラグで組合せを許容する場合は、既存ファイルを利用してフラグだけ切り替える形でも可。

## 📊 比較・評価ポイント

各実験のログ出力（`Val F1`など）を比較し、以下の観点で評価します。

### 1. **ArcFace の効果** (A vs B)
*   **BG Stats (Recall/Precision)** に注目。ArcFace はクラス間の角度を広げるため、Unknown（背景）と Player の境界が明確になり、**背景誤検知（BG False Positive）が減ることが期待**されます。  
*   Player F1 が維持または向上しつつ BG Precision が改善すれば ArcFace に利点あり。

### 2. **Embedding Head の効果** (A vs C)
*   Head 構造変更のみで精度が上がるかを確認。学習の安定性や汎化差を重視。

### 3. **組合せの相乗効果** (B/C vs D)
*   ArcFace と Embedding を同時に使うと、どちらか単体より優れるかを確認。特に BG と Player の識別境界がさらに改善されるか、Player F1 が向上するかを評価。

### 4. **EMA の恩恵** (D vs E)
*   EMA を適用すると Validation スコアの安定化やピーク性能が改善されるかを確認。実運用での利点（モデル保存時の安定した最良モデル）を評価。

## ✅ 採用判断（Decision Tree）

1. まず A を基準として B/C を評価。次に D（B+C）を評価し、最後に E（D+EMA）を評価する順序で進めます。

2. 具体的判断フロー：
- Step 1: B (ArcFace) が A より Player F1 を維持/改善かつ BG Precision を改善しているか？  
  - YES → 進む（ArcFace は有力候補）。次に C と比較。  
  - NO → ArcFace 単体は不採用候補。次に C を重視。

- Step 2: C (Embedding) が A より優れているか？  
  - YES → Embedding Head を採用候補。次に D（組合せ）を確認。  
  - NO → Embedding 単体は不採用候補。

- Step 3: D (ArcFace + Embedding) が B/C のどちらよりも良いか（Player F1 と BG Precision 両面）？  
  - YES → D を採用候補（両者の組合せが有効）。次に E を確認して安定化を図る。  
  - NO → 単体で良好だった方（B または C）を採用候補。

- Step 4: E (D + EMA) が D より安定して高いピーク性能を出すか？  
  - YES → 最終採用は E（組合せ + EMA）。  
  - NO → 最終採用は D（または B/C の採用候補）、EMA はコストとのトレードオフを考慮して任意採用。

## 📝 実行順（推奨）
1. A → 2. B → 3. C → 4. D → 5. E  
（各ステップでログを共有し、Decision Tree に従って次を実行）

## 🧾 レポートに含める情報
- 各実験ログ（各エポックの Val Loss / Overall F1 / Player F1 / BG Recall / BG Precision）  
- 最終的に保存したベストモデルパス（および EMA モデルの有無）  
- 公開 LB（可能なら）との比較

## 📝 次のアクション
Kaggleで上記実験を順に実行し、各実験ログ（CSV またはノートブック出力）を共有してください。結果に基づき Phase 3 の構成（ArcFace/Embedding/EMA の採用）を確定します。
