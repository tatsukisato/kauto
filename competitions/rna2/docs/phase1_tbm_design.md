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
