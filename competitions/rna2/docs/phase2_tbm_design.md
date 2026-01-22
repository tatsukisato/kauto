# Phase2 Template-Based Modeling — 仕様設計書（清書）

## 要約（TL;DR）
Phase1 の `LengthFilter` → `GlobalAlign` ベースラインを壊さず、局所フォールバック（LocalAlign）、ギャップ補間、Kabsch による剛体調整、最終欠損補完を段階的に導入して coverage と RMSD を改善する。まずは補間と Kabsch の小さなユーティリティを実装して単体テストで検証し、その後 pipeline に統合する。

## 目次
- 目的とスコープ
- 背景と参照実装
- 主要コンポーネント設計（LocalAlign / Gap interpolation / Kabsch 等）
- API と設定（必須パラメータ）
- データフロー（概略図）
- 指標・ログ（coverage, missing-rate, RMSD, TM-score）
- 実装ロードマップ（優先度付きタスク）
- テスト計画
- リスクと対策
- 付録（擬似アルゴリズム・数式）

---

## 目的とスコープ
- 目的: テンプレート転写の coverage を上げ、構造評価指標（RMSD / TM-score）を改善すること。既存の安定した Phase1 フローに安全に組み込む。
- スコープ（Phase2）:
  - Local alignment のフォールバック
  - ギャップ補間（短い欠損の線形補間を優先）
  - Kabsch による剛体調整（条件付き適用）
  - 最終欠損補完ポリシー（補間→top-K マージ→近傍平均）

## 背景（参照実装）
- Template 保持: [competitions/rna2/src/baseline/template_model.py](competitions/rna2/src/baseline/template_model.py)
- Scoring/search: [competitions/rna2/src/baseline/search.py](competitions/rna2/src/baseline/search.py)
- 推論/座標転写: [competitions/rna2/src/baseline/predict.py](competitions/rna2/src/baseline/predict.py)
- 既存計画書: [competitions/rna2/docs/template_baseline_plan.md](competitions/rna2/docs/template_baseline_plan.md)

## 主要コンポーネント設計

1) LocalAlign（局所フォールバック）
- 役割: global alignment が弱い場合に、クエリとテンプレートの高一致ブロックを検出して部分転写する。
- 実装候補: Smith–Waterman（Biopython の local alignment）を推奨。代替として窓スライド法（短い window に対する local/global の繰返し）も利用可能。
- 運用ルール（推奨）:
  - まず global scoring を行う（既存）。`global_identity < global_threshold` の場合に local を実行。
  - local で見つかったブロックは `local_min_identity` と `min_local_length` を満たすもののみ採用。
  - global で既にマップ済みの領域は優先、local は未カバー領域の補完に限定。

（利点）短い挿入/欠失に強く、一部領域だけ保たれているケースを拾える。

2) ギャップ補間（Gap interpolation）
- 役割: 転写後に残る短い連続欠損（NaN）を埋めて coverage を向上させる。
- 手法:
  - 初期実装は線形補間（各軸 x/y/z を独立に線形補間）
  - オプションでスプラインや近傍平均を実装可能
- 適用条件:
  - 連続欠損長 <= `max_interp_length` の場合のみ補間
  - 境界（端部）や近傍データ不足時はスキップ

3) Kabsch（剛体調整）
- 役割: マップされた対応点群に対し最適な回転・並進を求め、テンプレ由来座標をクエリ座標系へ整列させる。
- 適用条件（推奨）:
  - `n_pairs >= kabsch_min_pairs`（例: 3）
  - coverage >= `kabsch_min_coverage`（例: 0.3）
- 注意点: Kabsch は剛体変換のみで局所変形を扱えない。誤った対応点があると逆効果になるため、外れ値除去や厳格な閾値が必要。

4) 最終欠損補完フロー
- フロー（順序）:
  1. 転写直後に線形補間（`max_interp_length`以内）
  2. 補間で埋まらない長ギャップは top-K テンプレからのマージを検討（`top_k_templates`）
  3. 最後に近傍残基（±k）の平均/中央値で保険的に埋める

## API・設定（必須項目）
- TemplateRepository:
  - add: `save(path)`, `load(path)`
  - extend: `find_best(sequence, scorer, top_k=1)` → returns top-K list
- predict pipeline (worker config):
  - `length_ratio_min`, `length_ratio_max`
  - `global_threshold`, `use_local_fallback`, `local_min_identity`, `min_local_length`
  - `use_kabsch`, `kabsch_min_pairs`, `kabsch_min_coverage`
  - `fill_gaps`, `max_interp_length`, `interp_method`, `top_k_templates`

## データフロー（概略）

1. repo.fit(train_sequences, train_labels) → templates
2. for each query:
   - apply length filter → candidates
   - score candidates with global aligner → best (or top-K)
   - transfer coordinates by alignment mapping
   - compute raw coverage
   - if global score low and `use_local_fallback`: run local align to map remaining regions
   - apply Kabsch if conditions met
   - apply gap interpolation (<= `max_interp_length`)
   - if still missing and `top_k_templates`>1: try merge from other templates
   - compute final coverage, log metrics

（図: 単純フロー）

query_seq → LengthFilter → candidates → GlobalAlign → best → CoordinateTransfer → [LocalAlign fallback] → [Kabsch] → GapInterp → SubmissionBuilder

## 指標・ログ
- coverage: 非欠損残基数 / 全残基数（raw と final を記録）
- missing-rate: 1 - coverage（gap 補間前後で比較）
- RMSD: 対応点でのユークリッド距離二乗平均の平方根
  - 数式: $$\mathrm{RMSD}=\sqrt{\frac{1}{N}\sum_{i=1}^N \lVert p_i - q_i\rVert^2}$$
- TM-score: 長さ正規化された構造類似度スコア（0〜1）。Kabsch は前処理として有益だが、TM-score は局所外れ値に強いため過信禁物。

ログ出力: 各クエリで以下を CSV に出力する
- `query_id, raw_coverage, final_coverage, raw_missing_rate, post_interp_missing_rate, global_score, used_local_blocks, kabsch_applied, n_mapped_pairs, rmsd` 

## テスト計画
- ユニットテスト:
  - `test_interp.py`: 線形補間の端部・最大長を検証
  - `test_kabsch.py`: 既知の回転・平行移動で復元できるか
  - `test_local_align.py`: simple synthetic cases で局所ブロックを検出・マッピング
- 統合評価:
  - Phase1 と Phase2 出力を `validation_rmsd_summary.json` ベースで比較（RMSD, coverage）

## 実装ロードマップ（優先度）
- 高: `predict.py` の worker config 伝搬修正、ギャップ補間ユーティリティ
- 中: LocalAlign 実装と統合、Kabsch 実装
- 低: top-K マージ、TemplateRepository 永続化

## リスクと対策
- 計算コスト: local align と top-K はコスト増 → k-mer プリフィルタで候補を絞る
- メモリ: テンプレ座標の大量コピー → 共有メモリ・インデックス参照を検討
- Kabsch の誤適用: 対応点閾値を厳格にし外れ値除去する

## 付録: 擬似アルゴリズム
- LocalAlign（概念）:
```python
if global_identity < global_threshold:
    local_blocks = smith_waterman(query, tpl)
    for block in local_blocks:
        if block.identity >= local_min_identity and block.length >= min_local_length:
            map_block_coordinates(block)
```

- 線形補間（概念）:
```python
for each gap of length L in coords:
    if L <= max_interp_length:
        interpolate linearly between neighboring known points for x,y,z
```

---
このドキュメントに問題がなければ、次は `predict.py` のワーカ初期化の修正（`config` 伝搬）を実装します。変更は小さな PR 単位で進め、ユニットテストを追加します。

## 実装状況（2026-01-22）

**完了:**
- **LocalAlign（局所フォールバック）:** `competitions/rna2/src/baseline/search.py` に実装済み。
- **ギャップ補間（線形＋近傍填充）:** `competitions/rna2/src/baseline/template_utils.py` に実装済み。ユニットテストあり（`competitions/rna2/tests/test_interp.py`）。
- **Kabsch（剛体合わせ）:** `competitions/rna2/src/baseline/math_utils.py` に実装済み。ユニットテストあり（`competitions/rna2/tests/test_kabsch.py`）。
- **パイプライン統合:** `predict.py` に LocalAlign→Kabsch→補間の統合を実装。後処理は `competitions/rna2/src/baseline/submission.py` に分離し、ワーカー実装は `competitions/rna2/src/baseline/generator.py` に移動。
- **ユニットテスト追加:** `competitions/rna2/tests/` に基本テストを追加済み（3 passed, 1 skipped ローカル実行時）。

**保留 / 未実装:**
- **top-K マージ（複数テンプレートの領域マージ）:** 設計には記載済みだが実装は未着手（優先度: 低）。
- **TemplateRepository 永続化（save/load）:** フェーズ2では保留。必要時に `numpy.savez_compressed` 方式で実装予定。
- **統合評価ワークフロー（validation RMSD/TM-score 比較）:** 単体テストはあるが、設計にある完全な統合評価は未実施。

**備考:**
- 主要な変更は小さなコミット/PR に分割済み。コード導入後の数値評価（validation set 上の RMSD/coverage）は次フェーズの作業項目です。

## CI・依存関係の注意

- 現状、簡易的な CI ワークフロー（`.github/workflows/ci.yml`）がドラフトで追加されていますが、本格運用は保留しています。
- フルテスト（local_align など）は Biopython に依存するため、CI に Biopython を追加するか、該当テストを環境依存でスキップするポリシーを決める必要があります。
- 推奨: `pyproject.toml` または `requirements-dev.txt` にテスト/実行に必要な依存（`biopython`, `pytest`, `numpy`, `pandas` 等）を明記し、CI でインストールするようにしてください。



