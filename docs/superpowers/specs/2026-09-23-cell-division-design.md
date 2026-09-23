# 細胞分裂シミュレーション設計仕様書 (Cell Division Simulation Design Specification)

- **日付**: 2026-09-23
- **対象**: `src/xpbd/` および `examples/cell_division.py`
- **目的**: XPBD (Extended Position Based Dynamics) 上で、紡錘体の伸長、星状微小管による細胞体の球状化、収縮環による分裂溝の形成、メッシュ切断・キャッピングによる2つの独立した `SoftBody` へのトポロジー分離、および膜頂点の接線緩和による均等配置を自律的かつ高精度に再現する。

---

## 1. 背景と生物学的力学メカニズム

細胞分裂（Cytokinesis）では、単なる幾何学的二分割ではなく、以下の力学的要素が協調して動作する：

1. **紡錘体と中心体（極）の伸長 (Spindle Elongation)**:
   - 2つの中心体（極）$P_0, P_1$ の距離が伸び、細胞の長軸方向へ突っ張る力を生み出す。
2. **星状体微小管によるテザリングと球状化 (Astral Microtubule Tethering & Rounding)**:
   - 極から放射状に伸びる星状微小管が細胞皮層（細胞膜）と係留し、各極の周囲の膜を引き寄せる。
   - これにより、伸長中の細胞の両半球がそれぞれの極を中心とした「丸い球状ローブ（娘細胞の核）」として整えられる。
3. **収縮環による分裂溝の絞り込み (Contractile Ring / Cleavage Furrow)**:
   - 紡錘体の中央（赤道面）においてアクトミオシン環が求心的に収縮し、内部圧力（体積保存）に抗してくびれを深める。
4. **トポロジー分離 (Abscission)**:
   - くびれが十分に閉じた段階で膜が切断・再封鎖（キャッピング）され、2つの独立した細胞膜・細胞質に分かれる。
5. **膜頂点の均等距離再配置 (Tangential Surface Relaxation)**:
   - 2次元流体である脂質二重膜の張力緩和を再現し、くびれ部で過密・極部で過疎になった膜頂点を表面接線方向にラプラシアン平滑化し、均等な格子配置へと復元する。

---

## 2. システム構成とモジュール設計

### 2.1 新規・更新モジュール構成

```
src/xpbd/
├── mesh.py                        # 既存メッシュユーティリティ
├── topology.py (新規)             # メッシュ切断・キャッピング・平滑化
│   ├── slice_and_cap_mesh()       # 赤道面での閉曲面切断および境界ループのキャッピング
│   ├── extract_boundary_loop()    # 開口部境界エッジループの検出と巡回ソート
│   └── tangential_relaxation()    # 表面接線方向ラプラシアン平滑化（均等距離化）
└── constraints/
    ├── spindle.py (新規)          # 紡錘体（可変長カプセル押し広げ ＋ 星状微小管テザー）
    └── contractile_ring.py (新規) # 赤道面収縮環拘束（求心XPBD拘束）

examples/
└── cell_division.py               # ステートマシン実行、2つのSoftBody連携、Polyscope UI
```

---

## 3. 詳細仕様

### 3.1 紡錘体と星状微小管 (`SpindleConstraint`)

- **位置づけ**: 両極 $P_0, P_1$ と、その周囲の膜頂点に対する複合拘束。
- **極間距離の伸長**:
  - 目標距離 $L(t)$ を初期距離 $L_0$ から最大長 $L_{max}$（例: 初期の約2.5倍）まで、速度 $v_{push}$ で毎ステップ漸増。
  - 極間エッジに対して剛性距離拘束（`compliance=0.0`）を適用。
- **極の衝突押し出し**:
  - 極粒子は有効半径 $R_{pole}$ を持ち、内部流体粒子および膜頂点と球体衝突（めり込み排除）。
- **星状微小管テザリング拘束**:
  - 左極 $P_0$ は初期に $x < x_{mid}$ にあった表面頂点群と、右極 $P_1$ は $x > x_{mid}$ にあった表面頂点群と放射状の距離拘束を構成。
  - 自然長 $R_{tether} \approx R_{cell} \times 0.6 \sim 0.75$、適度なコンプライアンス（`compliance=1e-3` 程度）を持ち、極を中心として膜をふっくらと丸く引き寄せる。

### 3.2 収縮環拘束 (`ContractileRingConstraint`)

- **対象頂点**: 赤道面近傍（$|x_i - x_{mid}| < w_{furrow}$）に位置する表面膜頂点。
- **定式化**:
  - 中心軸上の点 $C(x_i) = (x_i, y_0, z_0)$ に対する径方向距離 $r_i = \sqrt{(y_i - y_0)^2 + (z_i - z_0)^2}$。
  - 拘束条件: $C_i = r_i - r_{target}(t) = 0$。
  - 目標半径 $r_{target}(t)$ は、伸長完了後に $R_{cell} \to r_{thresh}$（例: $0.05 \times R_{cell}$）へと毎ステップ線形または指数的に縮小。
  - XPBDによる求心拘束射影により、膜頂点を中心軸へと安定して絞り込む。

### 3.3 メッシュ切断と開口部キャッピング (`topology.py`)

- **`slice_and_cap_mesh(positions, faces, cut_x=0.0)`**:
  1. **頂点分割**:
     - 左側頂点集合 $V_L = \{i \mid positions[i, 0] < cut_x\}$
     - 右側頂点集合 $V_R = \{i \mid positions[i, 0] \ge cut_x\}$
  2. **面分割**:
     - 3頂点すべてが $V_L$ に属する面 $\to Faces_L$
     - 3頂点すべてが $V_R$ に属する面 $\to Faces_R$
     - 赤道面を跨ぐ面（境界三角形）を除去。
  3. **境界ループ検出**:
     - $Faces_L$ において1回しか現れないエッジ群（境界エッジ）を抽出し、有向閉ループとして接続順に整列。
     - $Faces_R$ についても同様に境界エッジループを抽出。
  4. **中心頂点追加とキャッピング**:
     - 左側中心頂点 $P_{cap, L} = (cut_x, y_{mean}, z_{mean})$、右側中心頂点 $P_{cap, R} = (cut_x, y_{mean}, z_{mean})$ を追加。
     - 境界ループの各辺と中心頂点を結ぶ扇状の三角形（Triangle Fan）を生成。
     - 生成された三角形の巻き順（Winding Order）を既存の面に揃え、外向き法線を維持。
  5. **閉曲面（Watertight）検証**:
     - 各メッシュのオイラー標数 $\chi = V - E + F = 2$ を満たす完全な2次元多様体閉曲面であることを保証。

### 3.4 膜頂点の表面接線緩和 (`tangential_relaxation`)

- **目的**: 分裂後のくびれ部での頂点過密および極付近での過疎を解消し、正二十面体状の均等配置へ緩和する。
- **アルゴリズム**:
  - 各頂点 $i$ の1-ring隣接頂点平均 $\bar{x}_i = \frac{1}{|N(i)|} \sum_{j \in N(i)} x_j$。
  - 頂点法線 $n_i$（隣接面の面積加重法線平均）。
  - 接線変位量:
    $$\Delta x_i^{\text{tan}} = (\bar{x}_i - x_i) - \big((\bar{x}_i - x_i) \cdot n_i\big) n_i$$
  - 位置更新: $x_i \leftarrow x_i + \gamma \Delta x_i^{\text{tan}}$（$\gamma \approx 0.15$）。
  - 中心（極）からの距離射影（または `GlobalVolumeConstraint`）と併用することで、球状の外形および体積を一切崩さずに頂点間隔のみを均等化。
  - 緩和完了後、エッジ距離拘束の自然長 `rest_lengths` を現在の長さに更新。

### 3.5 2つの独立した `SoftBody` への移行

- 分裂完了時、単一の `SoftBody` から以下の2つの独立インスタンスを生成：
  - `body_left`: 左側メッシュ頂点＋左側内部粒子＋左極 $P_0$
  - `body_right`: 右側メッシュ頂点＋右側内部粒子＋右極 $P_1$
- それぞれに対して独立した `DistanceConstraint`, `GlobalVolumeConstraint`, `ParticleCollisionConstraint` を構築。
- 娘細胞同士の重なりを防ぐため、2つの細胞間での粒子間衝突拘束を設定。

---

## 4. ステートマシンと遷移フロー

シミュレーション状態 `SimulationState`（Enum）:

1. **`ELONGATING`**:
   - 紡錘体目標距離 $L(t)$ を伸長。
   - 星状微小管が膜を極周囲へ丸め込む。
   - 終了条件: 極間距離が $L_{max}$（例: $1.2 \times R_{cell}$）に到達。
2. **`CONSTRICTING`**:
   - 紡錘体長さを固定維持。
   - 収縮環が赤道面頂点を中心軸へ絞り込む。
   - 終了条件: 赤道面最小半径 $r_{min} \le r_{thresh}$（例: $0.08 \times R_{cell}$）。
3. **`DIVIDING`**:
   - `slice_and_cap_mesh` を実行。
   - 内部粒子を分配し、`body_left`, `body_right` をインスタンス化。
   - 即座に `RELAXING` へ遷移。
4. **`RELAXING`**:
   - 極への距離拘束を保ちつつ、各娘細胞の膜頂点に対して `tangential_relaxation` を各ステップで実行。
   - 終了条件: 一定ステップ数（例: 60ステップ）経過、またはエッジ長分散が閾値以下。
5. **`INDEPENDENT`**:
   - 2つの独立した完全な娘細胞としてシミュレーションを継続。

---

## 5. テスト・検証戦略

1. **トポロジー・幾何単体テスト (`tests/test_topology.py`)**:
   - 単位球（icosphere）の切断・キャッピングテスト。
   - 両メッシュがWatertightであること（Euler標数 $\chi = 2$、境界エッジなし）。
   - 切断後の体積の和が元の球体積とほぼ一致（$\pm 5\%$ 以内）すること。
   - 内部粒子の左右振り分けに重複や欠落がないこと。
2. **接線緩和テスト (`tests/test_tangential_relaxation.py`)**:
   - 意図的に頂点を歪ませたメッシュに対して接線緩和を適用。
   - エッジ長の標準偏差が単調減少し、かつ球の体積・半径変化が $1\%$ 未満であること。
3. **ヘッドレス統合テスト (`tests/test_cell_division_pipeline.py`)**:
   - Polyscope UIなしでシミュレーションを実行。
   - `ELONGATING` $\to$ `CONSTRICTING` $\to$ `DIVIDING` $\to$ `RELAXING` $\to$ `INDEPENDENT` の全ステップが例外なく完了し、最終的に2つの `SoftBody` が正常に更新されることを自動検証。
