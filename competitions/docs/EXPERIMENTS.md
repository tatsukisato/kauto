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
