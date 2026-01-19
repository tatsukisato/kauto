# Template-based Baseline 計画書（整理版）

## 概要
学習データ中の既存構造（参照座標）を「テンプレート」として保持し、テスト/validation 配列に対して最も類似するテンプレートを選んで座標を転写するシンプルで壊れにくいベースライン。実行は必ずノートブック（competitions/rna2/experiments/run_baseline.ipynb）から行う。

## 目的
- 最小限の処理で「テンプレート抽出 → 評価 → submission 生成」フローを再現する。
- ノートブック中心で追跡・可視化を容易にする。

## 前提（データ構造）
- データ場所: `competitions/rna2/data/stanford-rna-3d-folding-2`
- 主要ファイル:
  - `train_sequences.csv`: `target_id,sequence,...`
  - `validation_sequences.csv` / `test_sequences.csv`
  - `train_labels.csv` / `validation_labels.csv`: `ID,resname,resid,x_1,y_1,z_1,...,chain,copy`（ID は `"{target_id}_{resid}"`）

## 候補手法（短名＋要点）
- GlobalAlign-TBM: 全配列グローバルアライメント→最類似テンプレート丸ごと転写
- LocalAlign-TBM: 部分一致領域のみ転写（後段で導入）
- kmer-Jaccard: k-mer で高速プリフィルタ
- LengthFiltered-Identity: 長さ比で絞る単純一致率
- Window-TBM: スライディング窓で局所テンプレ選択
- Hybrid-TopK-TBM: 上位Kテンプレを保持して best-of-N を生成

## 手法比較（実装初期に重視する軸）
- 安定性・実装容易さ重視：Length filter + GlobalAlign（初期推奨）
- 精度向上は LocalAlign や TopK、Kabsch を段階的に導入する

## 推奨初期構成（最重要）
- フロー: 長さ比フィルタ → Global alignment scoring → Best template 選択 → 座標コピー
- 理由: 壊れにくく再現性が高い。短時間で実装可能で拡張性あり。

## TemplateRepository（要点）
- 目的: テンプレートの抽出・検索・永続化を担う
- 推奨構成:
  - self.templates: { target_id: { "seq": str, "coords": DataFrame/ndarray } }
  - メソッド:
    - fit(seq_df, labels_df): テンプレ抽出
    - find_best(sequence, scorer): 最良テンプレ検索 → (best_tid, tpl, score)
    - get(target_id) / save(path) / load(path)

### fit()（概念）
- train_sequences と train_labels を結びつけ、各テンプレートに seq と coords を格納する。

### find_best()（シンプル実装例）
- Biopython の pairwise2.globalxx を scorer（デフォルト: identity）に使い全テンプレを走査して最良を選ぶ（初期は全件線形スキャンで可）。

## 転写ロジック（シンプル疑似コード）
- アライメントに従って 1:1 でマッチする残基のテンプレート座標をコピー。ギャップはトリミング（初期設定）。
- 必要なら欠損部を線形補間するフェーズ2実装へ。

疑似コード（説明用）
```python
def transfer_coords(query_seq, tpl_seq, tpl_coords):
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

## scorer（推奨）
- 初期: seq_identity（matches / max(len(a), len(b)））をデフォルト
- 拡張: local alignment score、k-mer 類似度（高速化用）

## 初期パラメータ案（デフォルト）
- alignment: globalxx
- length_handling: trim
- min_len_ratio: 0.7
- max_len_ratio: 1.3
- n_templates（best-of-N）: 1
- use_kabsch: False（Phase2で導入）

## ノートブック実行フロー（簡潔）
1. competitions/rna2/src を sys.path に追加
2. baseline.data, baseline.template_model, baseline.search, baseline.predict, baseline.evaluate を import
3. データ読み込み → repo.fit() → generate_submission() → 保存（submission_from_notebook.csv）
4. validation がある場合は evaluate() を呼ぶ

## 改善ロードマップ（段階）
- Phase1: Length filter + GlobalAlign + single template（安定版）
- Phase2: LocalAlign フォールバック、ギャップ補間、Kabsch
- Phase3: TopK、領域マージ、confidence weighting、motif DB

## 検証チェックリスト（簡潔）
- repo.templates が空でないこと（train が存在する場合）
- submission のカラム順が `ID,resname,resid,x_1,y_1,z_1,...` になっていること
- evaluate() が大きな異常値を返さないこと

## 参考
- Biopython pairwise alignments: https://biopython.org/wiki/Pairwise_alignments
- RNA 3D リソース: https://rna.bgsu.edu/rna3dhub

---

必要ならこの整理版をベースにノートブック用セル（実行例）や baseline パッケージの具体実装テンプレートを追加します。
