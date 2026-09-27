# 自律的XPBD水風船モデルによる細胞分裂 実装計画 (Autonomous Cell Division Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 幾何学的変形（Kinematics関数の直接上書き）を完全に撤廃し、膜の張力（Distance制約）、水風船内圧（GlobalVolume制約）、真円収縮環、紡錘体伸長、流体粒子間反発の最小限の物理制約のみで自律的にくびれ・分裂・球形化するXPBDシミュレーションを構築する。

**Architecture:** 赤道面 $x = 0$ と交差するエッジを分割して完全な真円頂点ループ（収縮環）を挿入した母細胞メッシュを作成し、XPBDソルバーで内圧と張力を同時に解く。収縮環が絞られると内圧により膜が自律的に左右に丸く膨らみ、内部粒子はお互いを押し合って先端へ自然に退避する。くびれきった瞬間に体積 $V_0/2$ の均一な正多面体メッシュ（Icosphere）へと交代して娘細胞が安定した球体となる。

**Tech Stack:** Python 3.13, NumPy, pytest, Polyscope, XPBD

**Spec:** `docs/superpowers/specs/2026-09-27-autonomous-cell-division-physics-design.md`

## Global Constraints
- 純粋Python/NumPyで動作し、C++コンパイル等の追加バイナリ依存を導入しない。
- 全テスト（`uv run pytest`）がパスすること。
- ヘッドレスモード（`--headless`）で全300ステップが安定して完走すること。
- 人工的な座標クランプやボイド生成関数（Kinematics）を用いず、物理法則（XPBD制約）で解くこと。

## Review Focus
1. **赤道真円ループの完全性**: 挿入された収縮環の全頂点が $x = 0$ 平面上にあり、隙間なく一巡する単一閉ループであること。
2. **体積保存性**: 収縮環が絞られる全過程で、`GlobalVolumeConstraint` により総体積が初期体積の 1% 以内の誤差で保存されること。
3. **粒子の膜内包性**: 内部粒子が膜の変形に伴って外側へ漏れ出さない（突き抜けない）こと。
4. **粒子の均一分散**: 粒子間反発により、中央や特定の局所に不自然に密集せず、左右の膨らんだ空間を満遍なく満たすこと。
5. **娘細胞の球形化**: トポロジー交代後の2つの娘細胞が、独立した内圧と張力によって歪みなく球体へと落ち着くこと。

---

### Task 1: 赤道面真円リング（収縮環）のメッシュ挿入ユーティリティ

**Files:**
- Create: `tests/test_equatorial_ring.py`
- Modify: `src/xpbd/topology.py`

**Interfaces:**
- Consumes: `positions: np.ndarray`, `faces: np.ndarray`, `plane_x: float = 0.0`
- Produces: `insert_equatorial_ring(positions, faces, plane_x=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - 返り値: `(new_positions, new_faces, ring_vertex_indices)`
  - `ring_vertex_indices`: 赤道面 $x = 0$ 上で時計回り（または反時計回り）に連続して繋がった頂点インデックスの1次元配列

- [ ] **Step 1: Write the failing test**

```python
# tests/test_equatorial_ring.py
import numpy as np
import pytest
from xpbd.mesh import create_icosphere
from xpbd.topology import insert_equatorial_ring


def test_insert_equatorial_ring_properties():
    verts, faces = create_icosphere(subdivisions=2, radius=1.0)
    new_verts, new_faces, ring_indices = insert_equatorial_ring(verts, faces, plane_x=0.0)

    # 1. Ring vertices must lie strictly on plane x = 0
    assert len(ring_indices) >= 6
    np.testing.assert_allclose(new_verts[ring_indices, 0], 0.0, atol=1e-12)

    # 2. Ring vertices must be approximately circular
    radii = np.linalg.norm(new_verts[ring_indices, 1:], axis=1)
    np.testing.assert_allclose(radii, 1.0, rtol=0.05)

    # 3. New mesh must be watertight and manifold: Euler characteristic V - E + F = 2
    from xpbd.mesh import extract_surface_edges
    edges = extract_surface_edges(new_faces)
    euler = len(new_verts) - len(edges) + len(new_faces)
    assert euler == 2

    # 4. Volume must be preserved
    p1 = new_verts[new_faces[:, 0]]
    p2 = new_verts[new_faces[:, 1]]
    p3 = new_verts[new_faces[:, 2]]
    vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0
    assert vol > 0.0
    expected_vol = 4.0 / 3.0 * np.pi * (1.0 ** 3)
    np.testing.assert_allclose(vol, expected_vol, rtol=0.08)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_equatorial_ring.py -v`  
Expected: FAIL with `ImportError: cannot import name 'insert_equatorial_ring'`

- [ ] **Step 3: Implement `insert_equatorial_ring` in `src/xpbd/topology.py`**

```python
def insert_equatorial_ring(
    positions: np.ndarray, faces: np.ndarray, plane_x: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivides triangles crossing x = plane_x to insert an exact planar ring of vertices.

    Returns (new_positions, new_faces, ring_vertex_indices).
    """
    # 1. Identify crossing edges and calculate intersection points at x = plane_x
    # 2. Split crossing triangles (1 edge crossed -> 2 tris; 2 edges crossed -> 3 tris)
    # 3. Order the intersection vertices circularly around x = plane_x by polar angle
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_equatorial_ring.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_equatorial_ring.py src/xpbd/topology.py
git commit -m "feat(topology): add insert_equatorial_ring for planar contractile ring"
```

---

### Task 2: 内部流体粒子の反発および膜内壁拘束の実装

**Files:**
- Create: `tests/test_particle_fluid.py`
- Create: `src/xpbd/constraints/particle_repulsion.py`

**Interfaces:**
- Consumes: `particles: np.ndarray`, `radius: float`, `membrane_positions: np.ndarray`, `membrane_faces: np.ndarray`
- Produces: `ParticleFluidConstraint`
  - `project_repulsion(x: np.ndarray, radius: float)`: 粒子同士が $2r$ 未満の場合に反発
  - `project_membrane_confinement(x: np.ndarray, center_l: np.ndarray, r_l: float, center_r: np.ndarray, r_r: float, neck_r: float)`: 粒子を細胞膜内部に拘束

- [ ] **Step 1: Write the failing test**

```python
# tests/test_particle_fluid.py
import numpy as np
import pytest
from xpbd.constraints.particle_repulsion import ParticleFluidConstraint


def test_particle_repulsion_spreads_cluster():
    # 10 particles initialized very close together
    np.random.seed(42)
    pts = np.random.uniform(-0.01, 0.01, size=(10, 3))
    r = 0.05
    c = ParticleFluidConstraint(particle_radius=r)

    # Apply repulsion
    for _ in range(20):
        c.project_repulsion(pts)

    # Verify no two particles are within 2*r
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert np.min(dist) >= 2 * r * 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_particle_fluid.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ParticleFluidConstraint`**

Implement pairwise repulsion and membrane boundary confinement in `src/xpbd/constraints/particle_repulsion.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_particle_fluid.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_particle_fluid.py src/xpbd/constraints/particle_repulsion.py
git commit -m "feat(constraints): implement ParticleFluidConstraint for cytoplasm fluid"
```

---

### Task 3: 自律的細胞分裂シミュレーションへの刷新 (`examples/cell_division.py`)

**Files:**
- Modify: `examples/cell_division.py`

**Interfaces:**
- 使用コンポーネント:
  - `insert_equatorial_ring` (Task 1)
  - `DistanceConstraint` (膜の張力)
  - `GlobalVolumeConstraint` (水風船の内圧)
  - `ContractileRingConstraint` (真円収縮環)
  - `ParticleFluidConstraint` (流体粒子)
  - `create_icosphere` (娘細胞リセット用)

- [ ] **Step 1: 幾何学的変形（Kinematics）コードを削除し物理主導へ再構築**
  - `_apply_division_kinematics` や人工的なクランプ・排除関数を廃止。
  - 母細胞に `insert_equatorial_ring` で生成した真円収縮環をセット。
  - ソルバー各ステップで `GlobalVolumeConstraint` と `DistanceConstraint`、収縮環の締め付け、および粒子流体反発を評価。
  - 収縮環が閾値（$r_{neck} < 0.08 R_{cell}$）に達した瞬間、2つの新しい均一 Icosphere（体積 $V_0/2$）へとトポロジーを交代。

- [ ] **Step 2: ヘッドレスモードで実行確認**

Run: `uv run python examples/cell_division.py --headless --headless-steps 300`  
Expected: 正常終了し、`state=INDEPENDENT` に到達すること。

- [ ] **Step 3: Commit**

```bash
git add examples/cell_division.py
git commit -m "feat(cell_division): refactor to autonomous XPBD physics simulation"
```

---

### Task 4: 統合パイプラインテストの更新と全テスト検証

**Files:**
- Modify: `tests/test_cell_division_pipeline.py`

- [ ] **Step 1: 統合テストで自律的水風船シミュレーションの完走を検証**
  - 総体積保存（初期体積に対する変動 $< 1\%$）
  - 粒子の二分配分（左・右への配分合計が初期粒子数と一致）
  - 娘細胞の球形度の検証

- [ ] **Step 2: 全テストの実行**

Run: `uv run pytest`  
Expected: 全テスト PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cell_division_pipeline.py
git commit -m "test: update integration tests for autonomous cell division pipeline"
```
