# Stanford RNA 3D Folding Part 2 — データ概要

## 概要
このコンペでは、各 RNA 配列について5つの3D構造を予測します。提出は各ターゲットにつき5セットの座標が必要です（x,y,z は評価前に -999.999〜9999.999 にクリップされます）。

---

## ファイル一覧と説明

### sequences ファイル
- train_sequences.csv / validation_sequences.csv / test_sequences.csv
  - target_id: 任意の識別子（train は通常 pdb_id_chain_id）
  - sequence: 予測対象の連結配列（複数チェーンの指定に従う）
  - temporal_cutoff: 公開日（yyyy-mm-dd）
  - description: 出自の説明（PDBならエントリタイトル等）
  - stoichiometry: 使うチェーンの指定（例 `{A:1;B:2}`）
  - all_sequences: FASTA 形式で構成される全チェーン配列（parse_fasta_py.py 参照）
  - ligand_ids / ligand_SMILES: 溶質情報（不要なら無視可）

### labels ファイル（train / validation）
- train_labels.csv / validation_labels.csv
  - ID: target_id と残基番号を“_”で結合（1-based）
  - resname: 塩基（A,C,G,U）
  - resid: 残基番号（整数）
  - x_1,y_1,z_1,...: 各実験構造の C1' 座標（train は各シーケンスに対し少なくとも1つの参照構造を持つ）
  - chain: 残基のチェーン ID（提出では chain/copy は省略可）
  - copy: 同一配列のコピー番号（複数コピーがある場合）

### sample_submission.csv
- train_labels と同形式だが、各ターゲットに対して5セットの座標を含める必要あり（x_1,y_1,z_1 ... x_5,y_5,z_5）。
- chain と copy は提出時必須ではない。

---

## MSA/
- 各ターゲット用 MSA（FAST A）: {target_id}.MSA.fasta
- マルチチェーンの場合、各ホモログはチェーンごとに別行で、他チェーンはギャップ（-）で埋められる。
- ヘッダに `chain={chain}`、必要なら `copies={copy}` のタグが付与（| 区切り）。

---

## PDB_RNA/
- PDB の RNA 含有エントリの 3D 情報
  - {PDB_id}.cif ファイル群
  - pdb_seqres_NA.fasta: PDB 内の核酸チェーン配列
  - pdb_release_dates_NA.csv: RNA を含むエントリのリリース日

---

## extra/
- parse_fasta_py.py: all_sequences フィールドを辞書化する helper（parse_fasta()）
- rna_metadata.csv: 2025-12-17 までの RNA / RNA-DNA ハイブリッド構造のメタデータ
- README.md: rna_metadata.csv の説明

---

## 追加ノート（フィルタ基準など）
- validation_sequences.csv は 2025-05-29 以降にリリースされたターゲット（最終提出日以降）で、さらに 2025-12-17 までを含む。最小 40% RNA 含有、重複除去（MMseqs2 90%）を実施。
- train_sequences.csv は冗長性がある（PDB の一部を含むが validation と重複しないようにフィルタ済み）。
- extra/rna_metadata.csv を用いた選定基準（要約）:
  - 正準 A/C/G/U またはマッピング可能な修飾残基
  - 未定義（N）や T を含まない
  - 修飾/非正準が 25% 以下
  - 配列中少なくとも 50% がモデル化されていること
  - 全 RNA チェーンの「調整済み構造性」合計が 20%以上
  - 全チェーン合計で最小 10 nt、C1' が解決されていること（P トレースのみは除外）
- パイプライン: https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline

---