# 実験ログ

このファイルには全ての実験の記録を時系列で記載します。

---

## exp001_baseline

**実施日時**: 2025-12-13 17:51

### 概要
Baseline LightGBM model with bbox position features only

### 設定
- **モデル**: LightGBM (多クラス分類)
- **特徴量**: x, y, w, h, area, aspect_ratio, center_x, center_y, quarter_num, quarter_sub, session, frame, angle
- **バリデーション**: Q1を学習、Q2をバリデーション
- **パラメータ**:
  ```yaml
  num_leaves: 31
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  num_boost_round: 1000
  early_stopping_rounds: 50
  ```

### 結果
- **Validation Macro F1**: 0.1695
- **Best Iteration**: 11
- **学習サンプル数**: 6,410
- **バリデーションサンプル数**: 18,510

### クラスごとのF1スコア

| Player | F1 Score | 備考 |
|--------|----------|------|
| Player_0 | 0.0000 | |
| Player_1 | 0.1502 | |
| Player_2 | 0.2071 | |
| Player_3 | 0.1397 | |
| Player_4 | 0.3086 | |
| Player_5 | 0.0000 | |
| Player_6 | 0.2226 | |
| Player_7 | 0.2103 | |
| Player_8 | 0.2274 | |
| Player_9 | 0.2047 | |
| Player_10 | 0.1937 | |

### 観察・学び
1. 位置情報のみでは限界: Macro F1が0.17と低い
2. Player 4が識別しやすい: F1=0.31で最高スコア
3. クラス不均衡の影響: Q1/Q2で登場しない選手のF1が0.0
4. 早期停止: 11イテレーションで停止

### 次のステップ
- [ ] 画像特徴の追加
- [ ] topカメラデータの活用
- [ ] 時系列特徴（前後フレームとの差分）
- [ ] ディープラーニングモデルの検討

---

## exp004_stratified_group_validation

**実施日時**: 2025-12-15 05:29

### 概要
Validation戦略の改善。Class Distribution MismatchとTemporal Leakageを同時に解消するため、`quarter`をグループとしたStratified Group K-Fold検証を実施。

### 設定
- **モデル**: SimpleCNN (ResNet18 backbone, frozen)
- **データ処理**: 
    - 完全画像ではなくCrop画像(224x224)を使用
    - データはQ1, Q2を使用
- **バリデーション**: Stratified Group K-Fold (Group=quarter, Hold-out 20%)
    - Train: 19880, Val: 5040
- **パラメータ**:
  ```python
  batch_size: 256
  lr: 4e-3
  optimizer: Adam
  scheduler: ReduceLROnPlateau
  ```

### 結果
- **Validation Macro F1**: 0.6604
- **Public LB**: 0.5434
- **学習サンプル数**: 19,880
- **バリデーションサンプル数**: 5,040

### 観察・学び
1.  **CVスコアの高騰**: 0.66という高いスコアが出たが、Public LB (0.54) との乖離が大きい。
2.  **原因**: テストデータの「ゴミbbox（未学習の背景など）」に対して、無理やり0-10の選手クラスを割り当てているため、実環境（テスト）ではスコアが下がる。
3.  **改善**: Validation戦略自体は正しいが、タスク設定（11クラス分類）が実態に即していない。

---

## exp005_background_aug

**実施日時**: 2025-12-15 16:10

### 概要
テストデータの特性（ゴミbbox、不明な選手）に合わせ、**Backgroundクラス(Label -1)** を導入。学習データから生成した「背景」「位置ずれbbox」を負例として学習させることで、頑健性を向上させる。

### 設定
- **モデル**: SimpleCNN (ResNet18 backbone, frozen) - **Output 12 classes**
- **データ拡張 (強化)**:
    - RandomResizedCrop (scale 0.6-1.0)
    - GaussianBlur
    - ColorJitter (stronger)
- **Background生成**:
    - 学習画像の正解bbox以外の領域をクロップ
    - 正解bboxをずらしてIoU < 0.5 としたものをクロップ
- **バリデーション**: Stratified Group K-Fold (Group=quarter, Hold-out 20%)

### 結果
- **Validation Macro F1**: 0.5390 (↓ from 0.6604)
- **Public LB**: 0.5043 (↓ from 0.5434)
- **Gap**: 0.0347 (以前は 0.117)

### 観察・学び
1.  **スコアの下落**: タスクが「選手識別」から「選手識別 + ゴミ検知」に難化したため、見かけのスコアは低下。
2.  **信頼性の向上**: CVとLBの差が縮まり、ローカルでの評価がLBと連動しやすくなった（CV戦略として成功）。
3.  **今後の方向**: このValidation環境をベースに、モデル自体の精度向上（Backbone変更、解像度アップ、Head改良など）を目指すフェーズに入った。

---

## テンプレート（次回実験用）

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
