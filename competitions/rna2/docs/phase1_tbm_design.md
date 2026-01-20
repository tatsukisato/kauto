# RNA Template-Based Modeling Baseline — Phase1 設計書

## 1. 目的とスコープ
目的: Kaggle RNA コンペ向けに「壊れず動く」「低コスト」「最小構成」のテンプレートベースベースラインを構築する。  
スコープ（Phase1）:
- Length filter
- Global alignment によるテンプレ選択
- Single best template の座標転写
- ギャップ補完や幾何学最適化は除外（Phase2 以降）

## 2. 全体アーキテクチャ（概略）
train_sequences.csv + train_labels.csv → TemplateRepository.fit() → in-memory templates  
推論: query_seq → LengthFilter → GlobalAlignScorer → find_best → CoordinateTransfer → SubmissionBuilder

（簡易図）
train seq/labels → TemplateRepository.fit() → templates
query seq → LengthFilter → candidates → GlobalAlign → best → transfer → submission

## 3. コンポーネント（責務）

- TemplateRepository
  - 保持: templates = {target_id: {seq, coords}}
  - メソッド: fit(seq_df, labels_df), find_best(seq, scorer), get(), save()/load()
  - 設計: scorer/filter/transfer_strategy を差し替え可能にする

- LengthFilter
  - 責務: 明らかに長さ不一致のテンプレを除外
  - 条件: 0.7 <= tpl_len / query_len <= 1.3 (デフォルト)

- GlobalAlignmentScorer
  - 責務: 全配列グローバルアライメントで類似度算出（Biopython pairwise2 推奨）
  - 出力: 正規化された identity スコア（matches / max(len(a), len(b)））

- CoordinateTransfer
  - 責務: アライメントに従ってテンプレ座標をコピー
  - Phase1 制約: ギャップ補完なし、マッチ箇所のみコピー（未マッチは除外 or NaN）

- SubmissionBuilder
  - 責務: Kaggle 提出フォーマットの DataFrame を生成（ID,resname,resid,x_1,y_1,z_1,...）

## 4. データフロー（詳細）
学習:
- train_sequences.csv, train_labels.csv → TemplateRepository.fit() → templates

推論:
- for each query:
  - apply LengthFilter → candidate list
  - for each candidate: compute GlobalAlignmentScorer score
  - select best candidate (top_k=1)
  - transfer coordinates via CoordinateTransfer
  - append to SubmissionBuilder

## 5. コンフィグ（推奨初期値）
- length_ratio_min: 0.7
- length_ratio_max: 1.3
- alignment_method: global (pairwise2.globalxx)
- scorer: seq_identity
- top_k_templates: 1
- allow_partial: false
- use_kabsch: false
- fill_gaps: false

## 6. 実装上の注意点
- TemplateRepository.fit() は train_labels を target_id でグルーピングして seq と coords を関連付けて保持すること。
- find_best() は最初は線形走査で問題ない（将来的に k-mer prefilter を追加）。
- CoordinateTransfer はアライメント結果に基づきテンプレの残基座標をマッピングしてコピーする。ギャップは無視。

（転写簡易疑似コード）
```python
aln = pairwise2.align.globalxx(query_seq, tpl_seq)[0]
out = []
t_i = 0
for q, t in zip(aln.seqA, aln.seqB):
    if q != "-" and t != "-":
        out.append(tpl_coords[t_i])
        t_i += 1
    elif t != "-":
        t_i += 1
return np.array(out)
```

## 7. 完了条件（Done）
- ノートブック（run_baseline.ipynb）から end-to-end 実行できる
- Kaggle 提出形式の CSV を生成できる
- サンプルデータで予測が出力される（OOM/タイムアウト無し）

## 8. Phase2 への拡張ポイント（予告）
- LocalAlign fallback
- ギャップ補完（線形補間）
- Top-K templates + region merge
- Kabsch による幾何学最適化

---

次のステップ候補:
- Notebook セル構成への落とし込み
- TemplateRepository の雛形実装（Python）
- Submission builder の実装例

## 実装確認と差分コメント

以下は現行実装（`competitions/rna2/src/baseline` 下の実装）を確認した結果です。設計書との相違点・未実装点・今後の実装候補を記載します。

- **実装済み（主な項目）**:
  - `TemplateRepository.fit` / `get` / `find_best` が実装されており、テンプレートは in-memory に保持される（`template_model.py`）。
  - 長さフィルタと k-mer プレフィルタ（`prefilter_k`, `prefilter_top_n`）が導入されており、線形スキャン→短縮→スコア計算の流れになっている。
  - 配列類似度スコアは Biopython の `PairwiseAligner` を用いる `seq_identity` が実装されている（`search.py`）。
  - 座標転写は `predict.py` のワーカ `_worker_process_query` 内で実装されており、`PairwiseAligner` の aligned blocks を使ってテンプレの座標をマップしている。マップされない箇所は NaN として出力される。
  - SubmissionBuilder 相当の処理は `generate_submission_parallel` に実装されており、ID,resname,resid,x_1,y_1,z_1... の DataFrame を生成する。

- **ドキュメントと異なる点 / 注意点**:
  - 設計書では Biopython の `pairwise2.globalxx` を想定しているが、実装は `PairwiseAligner`（新しい API）を使用している。どちらでも良いが依存関係と説明を合わせる必要がある。
  - 設計に挙げた `TemplateRepository.save()/load()` は実装されていない（現状はメモリ保持のみ）。永続化が必要なら追加実装を行うべき。
  - ワーカ内での長さ比閾値（length ratio）が `_worker_process_query` にて `0.7/1.3` とハードコーディングされている。`TemplateRepository.length_ratio_min/max` をワーカに渡す設計に変更することを推奨する（現在 find_best では repo 側の値を利用するが、並列ワーカ側は利用していない）。
  - `TemplateRepository.fit` は座標列の自動検出（`x_1,y_1,z_1` または `x_{i}` パターン）や `-1e+18` を NaN に置換する前処理を行っている。データ前処理の仕様（欠損値の扱い）を設計書に明記しておくと良い。
  - `pyproject.toml` に `biopython` が明示されていないように見える（現状 `bio` という依存があるが `Bio.Align.PairwiseAligner` を提供するのは `biopython`）。ランタイムで `PairwiseAligner` が無いとエラーになるため、依存関係に `biopython` を追加することを推奨する。
  - `generate_submission_parallel` は Unix 系で `fork` を利用するが、Windows では挙動が異なる点に注意（ドキュメントでマルチプロセスの挙動差を注記すると親切）。

- **未実装 / 改善提案（今後の実装ポイント）**:
  - `TemplateRepository.save()` / `load()` の追加（pickle / CSV / ディレクトリ構造など、運用要件に合わせて選択）。
  - ワーカ初期化時に `length_ratio_min` / `length_ratio_max` を渡す（`initargs` に追加し、ハードコーディングを排除）。
  - 依存関係チェック時にわかりやすいエラーメッセージを出す（Biopython 未導入時の案内）。
  - `allow_partial` / `fill_gaps` / `use_kabsch` のフラグを将来的に有効化できるよう、`generate_submission` と `TemplateRepository` の API で受け渡し可能にする。
  - 転写アルゴリズムを示す擬似コードを `PairwiseAligner` の `aligned` ブロック版に更新する（より実装に近い説明にする）。

- **短い擬似コード（PairwiseAligner でのマッピング例）**:
```python
aligner = PairwiseAligner()
alns = aligner.align(query_seq, tpl_seq)
aln = next(iter(alns))
q_blocks, t_blocks = aln.aligned
for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
    for offset in range(qe-qs):
        q_idx = qs + offset
        t_idx = ts + offset
        # t_idx に対応するテンプレ座標をコピー
```

上記の差分と TODO をドキュメントに反映しました。実装側で修正を希望する箇所があれば指示ください。
