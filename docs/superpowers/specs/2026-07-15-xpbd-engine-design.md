# XPBD Engine Design (v1)

Date: 2026-07-15
Status: Approved

## 目的

純粋なXPBD (Extended Position Based Dynamics) エンジンをPythonで実装し、
サンプルシミュレーションを3Dビューワで確認できるようにする。

主目的は**研究・実験の土台**である。新しい制約や手法を試すための、
拡張しやすく見通しの良い基盤を優先する。パフォーマンスは
「数千〜数万頂点がリアルタイムで動く」程度を目標とし、極限最適化はしない。

## スコープ (v1)

含むもの:

- XPBDソルバ本体(サブステップ方式)
- 距離制約・体積制約・地面衝突制約
- 四面体メッシュのソフトボディ
- 立方体格子ベースの四面体メッシュ自前生成
- polyscopeによるソフトボディ落下デモ
- Jacobi / Gauss-Seidel 両ソルバモード(収束比較実験用)

含まないもの(将来拡張):

- 自己衝突・物体間衝突
- 布・ロープ・剛体・流体
- 摩擦モデル(v1は法線方向の押し出しのみ)
- インタラクティブ操作(マウスドラッグ等)
- GPU / JIT バックエンド

## アーキテクチャ

制約クラス中心の小さなコア構成。エンジン本体は可視化に依存しない。

```
xpbd/
├── pyproject.toml           # uv管理。依存: numpy / dev: pytest, pytest-cov
├── src/xpbd/
│   ├── __init__.py
│   ├── body.py              # SoftBody: positions, velocities, inv_mass, tets, edges
│   ├── solver.py            # Solver: XPBDサブステップループ、モード切替
│   ├── mesh.py              # 四面体メッシュ生成(格子→tet 5分割)、表面三角形抽出
│   └── constraints/
│       ├── __init__.py
│       ├── base.py          # Constraint インターフェース (Protocol)
│       ├── distance.py      # DistanceConstraint(エッジ集合、バッチ)
│       ├── volume.py        # VolumeConstraint(四面体集合、バッチ)
│       └── ground.py        # GroundConstraint(y >= 0 の不等式制約)
├── examples/
│   └── softbody_drop.py     # polyscopeデモ(polyscope依存はここだけ)
└── tests/
```

## ソルバ設計

Macklin et al. 2016 (XPBD) に Müller et al. 2019 (Small Steps) の
サブステップ方式を組み合わせた現在の標準形を採用する。

```
step(dt):
  h = dt / n_substeps
  for each substep:
    x_prev = x
    v += g * h
    x += v * h                     # 予測位置
    for c in constraints:
      c.reset_lambda()
      c.project(...)               # λ蓄積 + Δx適用、α̃ = α / h²
    v = (x - x_prev) / h           # 速度再導出
```

- 反復回数はsubstepあたり1回を基本とし、収束はサブステップ数で稼ぐ
- compliance α は制約オブジェクトごとに保持。α=0 で剛、α>0 で物理単位の柔らかさ
- λ は制約ごとに substep 開始時にリセットし、substep 内で蓄積する

### ソルバモード

研究用に2モードを持ち、同一シーンで収束・挙動を比較できる。

- `jacobi`(デフォルト): 制約集合ごとにNumPyバッチで勾配・Δλを一括計算し、
  `np.add.at` で頂点補正を同時適用する。頂点を共有する制約間はJacobi風の
  同時更新になるため、収束はサブステップ数で補う
- `gauss_seidel`: 制約を1本ずつ逐次射影する参照実装。遅いが論文に忠実

## 制約インターフェース

各制約オブジェクトは「同種制約の集合」を保持する(1エッジ1オブジェクトではない)。

```python
class Constraint(Protocol):
    def reset_lambda(self) -> None:
        """Reset accumulated lambda at the start of a substep."""

    def project(self, x: ndarray, inv_mass: ndarray, h: float) -> None:
        """Batched projection (jacobi mode). Applies corrections in-place."""

    def project_one(self, i: int, x: ndarray, inv_mass: ndarray, h: float) -> None:
        """Project the i-th constraint (gauss_seidel mode)."""

    def __len__(self) -> int:
        """Number of individual constraints in this batch."""
```

### 制約の種類

- **DistanceConstraint**: エッジ集合 `(E, 2)`、静止長 `(E,)`。
  C = |x_i - x_j| - L0
- **VolumeConstraint**: 四面体集合 `(T, 4)`、静止体積 `(T,)`。
  C = 6 * (V - V0) 相当の標準的な体積制約
- **GroundConstraint**: 不等式制約 C = y >= 0。違反頂点のみ射影(α=0)。
  摩擦はv1では扱わない

## データ設計 (SoftBody)

- `positions: (N, 3) float64`
- `velocities: (N, 3) float64`
- `inv_mass: (N,) float64`(0 で固定頂点)
- `tets: (T, 4) int`、`edges: (E, 2) int`(tetsから重複除去で導出)
- 質量は四面体体積を密度で重み付けし頂点に分配する

## メッシュ生成 (mesh.py)

- 立方体格子 `(nx, ny, nz)` を1セルあたり5四面体に分割して生成
  (外部tetgen等への依存なし)
- 隣接セルで分割向きを反転し、面の対角線を整合させる
- 表面三角形抽出: 全四面体の面のうち1回しか現れない面を表面とする
  (polyscopeの表示・将来の衝突用)

## サンプル (examples/softbody_drop.py)

- 柔らかい立方体(またはビーム)を地面に落下させる
- polyscope の `register_volume_mesh` とフレームコールバックで
  `solver.step(dt)` → 頂点位置更新を毎フレーム実行
- CLI引数: compliance、サブステップ数、ソルバモード、格子解像度

## テスト計画 (TDD, pytest, カバレッジ80%以上)

1. **距離制約**: 2粒子1エッジが静止長へ収束 / α=0 で厳密、α大で伸びが残る
2. **体積制約**: 単一四面体の体積が保存される
3. **地面衝突**: 落下後の貫通量が閾値以下
4. **ソルバ**: 制約なし自由落下が解析解と一致 /
   jacobi と gauss_seidel が同一シーンで近い定常状態に収束
5. **メッシュ生成**: 全四面体の体積が正 / 表面抽出がEuler特性と整合

## 依存関係

- ランタイム: numpy(コア)、polyscope(examplesのみ)
- 開発: pytest, pytest-cov
- Python 3.11+ / uv でパッケージ管理

## 将来拡張の想定(設計上の受け皿)

- 新しい制約 = `constraints/` にクラスを1つ追加するだけ
- Neo-Hookean等の材料モデルも制約クラスとして追加可能
- 衝突検出は制約生成器(毎substepで制約集合を再構築)として追加する想定
