# XPBD Engine v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 純粋なXPBDエンジン(NumPy)と、polyscopeで見られるソフトボディ落下デモを作る。

**Architecture:** 制約クラス中心の小さなコア。`Solver` がサブステップループ(Small Steps方式)を回し、制約オブジェクト(距離・体積・地面)がNumPyバッチ(jacobi)または逐次(gauss_seidel)で射影する。エンジン本体はnumpyのみに依存し、可視化はexamples側に隔離する。

**Tech Stack:** Python 3.11+, numpy, polyscope (examplesのみ), pytest + pytest-cov, uv

**Spec:** `docs/superpowers/specs/2026-07-15-xpbd-engine-design.md`

## Global Constraints

- Python `>=3.11`、パッケージ管理は `uv`(実行は `uv run ...`)
- コア(`src/xpbd/`)のランタイム依存は `numpy` のみ。polyscopeは `viz` extraとしてexamplesだけが使う
- コード内コメント・docstringは英語。絵文字禁止
- 1ファイル800行以内(目安200-400行)
- コミットはconventional commits(`feat:`, `test:`, `docs:`, `chore:`)
- 配列dtype: 位置・速度は `float64`、インデックスは `int64`
- テストは各タスクで先に書く(TDD)。最終カバレッジ80%以上

---

### Task 1: プロジェクト雛形 + メッシュ生成コア (`tet_grid`, `tet_volumes`)

**Files:**
- Create: `pyproject.toml`
- Create: `src/xpbd/__init__.py`
- Create: `src/xpbd/mesh.py`
- Test: `tests/test_mesh.py`

**Interfaces:**
- Consumes: なし(最初のタスク)
- Produces:
  - `xpbd.mesh.tet_grid(nx: int, ny: int, nz: int, spacing: float = 1.0, origin: tuple = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]` — 戻り値 `(positions (N,3) float64, tets (T,4) int64)`。全四面体の符号付き体積は正
  - `xpbd.mesh.tet_volumes(positions: np.ndarray, tets: np.ndarray) -> np.ndarray` — 符号付き体積 `(T,)`

- [ ] **Step 1: pyproject.toml とパッケージ雛形を作る**

`pyproject.toml`:

```toml
[project]
name = "xpbd"
version = "0.1.0"
description = "A pure XPBD physics engine for research and experimentation"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
viz = ["polyscope>=2.2"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/xpbd"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`src/xpbd/__init__.py`(この時点では空でよい):

```python
```

Run: `uv sync`
Expected: 依存解決が成功し `.venv` が作られる

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_mesh.py`:

```python
import numpy as np
import pytest

from xpbd.mesh import tet_grid, tet_volumes


def test_tet_volumes_unit_tet():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]])
    vols = tet_volumes(positions, tets)
    assert vols.shape == (1,)
    assert vols[0] == pytest.approx(1.0 / 6.0)


def test_tet_grid_single_cell():
    positions, tets = tet_grid(1, 1, 1)
    assert positions.shape == (8, 3)
    assert tets.shape == (5, 4)
    vols = tet_volumes(positions, tets)
    assert np.all(vols > 0)
    assert vols.sum() == pytest.approx(1.0)


def test_tet_grid_total_volume_and_orientation():
    positions, tets = tet_grid(2, 3, 2, spacing=0.5)
    vols = tet_volumes(positions, tets)
    assert np.all(vols > 0)
    assert vols.sum() == pytest.approx(2 * 3 * 2 * 0.5**3)


def test_tet_grid_origin_and_spacing():
    positions, _ = tet_grid(1, 1, 1, spacing=2.0, origin=(1.0, 2.0, 3.0))
    assert positions.min(axis=0) == pytest.approx([1.0, 2.0, 3.0])
    assert positions.max(axis=0) == pytest.approx([3.0, 4.0, 5.0])
```

- [ ] **Step 3: テストが失敗することを確認**

Run: `uv run pytest tests/test_mesh.py -v`
Expected: FAIL (`ModuleNotFoundError` または `ImportError`)

- [ ] **Step 4: `src/xpbd/mesh.py` を実装**

```python
"""Tetrahedral mesh generation and topology utilities."""

import numpy as np

# Alternating 5-tet decompositions of a cube cell. Corners are (di, dj, dk)
# offsets. Adjacent cells use opposite parity so face diagonals match.
_TETS_EVEN = (
    ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 0, 0), (0, 1, 1), (1, 1, 0)),
    ((0, 0, 1), (0, 0, 0), (1, 0, 1), (0, 1, 1)),
    ((1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)),
)
_TETS_ODD = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 1, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 0, 1), (1, 0, 0), (0, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (0, 1, 0), (1, 1, 1), (0, 0, 1)),
)


def tet_volumes(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volumes of tetrahedra, shape (T,)."""
    p = positions[tets]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0]) / 6.0


def tet_grid(
    nx: int,
    ny: int,
    nz: int,
    spacing: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize an (nx, ny, nz)-cell box grid with 5 tets per cell.

    Returns (positions (N, 3) float64, tets (T, 4) int64). All tets have
    positive signed volume.
    """
    gx, gy, gz = nx + 1, ny + 1, nz + 1
    ii, jj, kk = np.meshgrid(
        np.arange(gx), np.arange(gy), np.arange(gz), indexing="ij"
    )
    positions = (
        np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
        * spacing
        + np.asarray(origin, dtype=np.float64)
    )

    def vidx(i: int, j: int, k: int) -> int:
        return (i * gy + j) * gz + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pattern = _TETS_EVEN if (i + j + k) % 2 == 0 else _TETS_ODD
                for corners in pattern:
                    tets.append(
                        [vidx(i + di, j + dj, k + dk) for di, dj, dk in corners]
                    )
    tets = np.asarray(tets, dtype=np.int64)

    # Enforce positive orientation by swapping two vertices where needed.
    flip = tet_volumes(positions, tets) < 0
    flipped = tets[flip]
    flipped[:, [0, 1]] = flipped[:, [1, 0]]
    tets[flip] = flipped
    return positions, tets
```

- [ ] **Step 5: テストが通ることを確認**

Run: `uv run pytest tests/test_mesh.py -v`
Expected: 4 PASS

- [ ] **Step 6: .gitignore を作りコミット**

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.coverage
```

```bash
git add pyproject.toml uv.lock .gitignore src/xpbd/__init__.py src/xpbd/mesh.py tests/test_mesh.py
git commit -m "feat: project scaffolding and tet grid mesh generation"
```

---

### Task 2: メッシュトポロジー (`edges_from_tets`, `surface_from_tets`)

**Files:**
- Modify: `src/xpbd/mesh.py`(関数を追記)
- Test: `tests/test_mesh.py`(テストを追記)

**Interfaces:**
- Consumes: `tet_grid` (Task 1)
- Produces:
  - `xpbd.mesh.edges_from_tets(tets: np.ndarray) -> np.ndarray` — 重複のないエッジ `(E,2) int64`、各行は昇順
  - `xpbd.mesh.surface_from_tets(tets: np.ndarray) -> np.ndarray` — 境界三角形 `(F,3) int64`(1回しか現れない面)

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_mesh.py` に追記:

```python
from xpbd.mesh import edges_from_tets, surface_from_tets


def test_edges_from_single_tet():
    tets = np.array([[0, 1, 2, 3]])
    edges = edges_from_tets(tets)
    assert edges.shape == (6, 2)
    assert np.all(edges[:, 0] < edges[:, 1])


def test_edges_are_unique_on_grid():
    _, tets = tet_grid(2, 2, 2)
    edges = edges_from_tets(tets)
    keys = {tuple(e) for e in edges.tolist()}
    assert len(keys) == len(edges)


def test_surface_of_single_tet():
    tets = np.array([[0, 1, 2, 3]])
    faces = surface_from_tets(tets)
    assert faces.shape == (4, 3)


def test_surface_of_single_cell():
    # 5-tet cube: each of the 6 cube faces splits into 2 boundary triangles.
    _, tets = tet_grid(1, 1, 1)
    faces = surface_from_tets(tets)
    assert faces.shape == (12, 3)


def test_surface_internal_faces_cancel():
    # 2x1x1 grid: 10 exterior unit squares -> 20 boundary triangles.
    # This only holds if adjacent cells share matching face diagonals.
    _, tets = tet_grid(2, 1, 1)
    faces = surface_from_tets(tets)
    assert faces.shape == (20, 3)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_mesh.py -v`
Expected: 新規5件が FAIL (`ImportError`)、既存4件は PASS

- [ ] **Step 3: `src/xpbd/mesh.py` に実装を追記**

```python
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def edges_from_tets(tets: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a tet mesh, each row sorted ascending."""
    pairs = tets[:, _TET_EDGES].reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def surface_from_tets(tets: np.ndarray) -> np.ndarray:
    """Boundary triangles of a tet mesh (faces appearing exactly once)."""
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(
        keys, axis=0, return_index=True, return_counts=True
    )
    return faces[first[counts == 1]]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_mesh.py -v`
Expected: 9 PASS

- [ ] **Step 5: コミット**

```bash
git add src/xpbd/mesh.py tests/test_mesh.py
git commit -m "feat: edge and boundary surface extraction from tet meshes"
```

---

### Task 3: SoftBody (`body.py`)

**Files:**
- Create: `src/xpbd/body.py`
- Modify: `src/xpbd/__init__.py`
- Test: `tests/test_body.py`

**Interfaces:**
- Consumes: `tet_volumes`, `edges_from_tets` (Task 1, 2)
- Produces:
  - `xpbd.body.SoftBody` — dataclass。フィールド: `positions (N,3) float64`, `velocities (N,3) float64`, `inv_mass (N,) float64`, `tets (T,4) int64`, `edges (E,2) int64`
  - `SoftBody.from_tet_mesh(positions, tets, density: float = 1000.0) -> SoftBody` — 質量を四面体体積の密度重みで頂点に等分配(各頂点に `density * V / 4`)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_body.py`:

```python
import numpy as np
import pytest

from xpbd.body import SoftBody
from xpbd.mesh import tet_grid


def test_from_tet_mesh_shapes_and_dtypes():
    positions, tets = tet_grid(1, 1, 1)
    body = SoftBody.from_tet_mesh(positions, tets)
    n = len(positions)
    assert body.positions.shape == (n, 3)
    assert body.velocities.shape == (n, 3)
    assert np.all(body.velocities == 0)
    assert body.inv_mass.shape == (n,)
    assert body.positions.dtype == np.float64
    assert body.edges.ndim == 2 and body.edges.shape[1] == 2


def test_total_mass_matches_density_times_volume():
    positions, tets = tet_grid(2, 1, 1, spacing=0.5)
    body = SoftBody.from_tet_mesh(positions, tets, density=800.0)
    total_mass = (1.0 / body.inv_mass).sum()
    expected = 800.0 * (2 * 1 * 1 * 0.5**3)
    assert total_mass == pytest.approx(expected)


def test_positions_are_copied():
    positions, tets = tet_grid(1, 1, 1)
    body = SoftBody.from_tet_mesh(positions, tets)
    positions[0, 0] = 99.0
    assert body.positions[0, 0] != 99.0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_body.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'xpbd.body'`)

- [ ] **Step 3: `src/xpbd/body.py` を実装**

```python
"""Soft body state container."""

from dataclasses import dataclass

import numpy as np

from .mesh import edges_from_tets, tet_volumes


@dataclass
class SoftBody:
    """Particle state and topology of a tetrahedral soft body.

    inv_mass entries of 0.0 mark pinned (immovable) vertices.
    """

    positions: np.ndarray  # (N, 3) float64
    velocities: np.ndarray  # (N, 3) float64
    inv_mass: np.ndarray  # (N,) float64
    tets: np.ndarray  # (T, 4) int64
    edges: np.ndarray  # (E, 2) int64

    @classmethod
    def from_tet_mesh(
        cls,
        positions: np.ndarray,
        tets: np.ndarray,
        density: float = 1000.0,
    ) -> "SoftBody":
        """Build a body from a tet mesh, lumping tet mass onto vertices."""
        positions = np.asarray(positions, dtype=np.float64).copy()
        tets = np.asarray(tets, dtype=np.int64)
        vols = tet_volumes(positions, tets)
        mass = np.zeros(len(positions))
        np.add.at(mass, tets.ravel(), np.repeat(density * vols / 4.0, 4))
        inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-300), 0.0)
        return cls(
            positions=positions,
            velocities=np.zeros_like(positions),
            inv_mass=inv_mass,
            tets=tets,
            edges=edges_from_tets(tets),
        )
```

`src/xpbd/__init__.py` を更新:

```python
"""A pure XPBD physics engine for research and experimentation."""

from .body import SoftBody

__all__ = ["SoftBody"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_body.py -v`
Expected: 3 PASS

- [ ] **Step 5: コミット**

```bash
git add src/xpbd/body.py src/xpbd/__init__.py tests/test_body.py
git commit -m "feat: SoftBody state container with lumped vertex masses"
```

---

### Task 4: Constraintプロトコル + 距離制約

**Files:**
- Create: `src/xpbd/constraints/__init__.py`
- Create: `src/xpbd/constraints/base.py`
- Create: `src/xpbd/constraints/distance.py`
- Test: `tests/test_distance.py`

**Interfaces:**
- Consumes: なし(numpyのみ)
- Produces:
  - `xpbd.constraints.base.Constraint` — Protocol: `reset_lambda() -> None`, `project(x, inv_mass, h) -> None`, `project_one(i, x, inv_mass, h) -> None`, `__len__() -> int`
  - `xpbd.constraints.DistanceConstraint(edges, rest_lengths, compliance=0.0, relaxation=1.0)`
  - `DistanceConstraint.from_positions(positions, edges, compliance=0.0, relaxation=1.0)` — 現在位置から静止長を計算

XPBDの式(全制約共通): `α̃ = compliance / h²`, `Δλ = (-C - α̃ λ) / (Σ wᵢ|∇ᵢC|² + α̃)`, `Δxᵢ = wᵢ Δλ ∇ᵢC`。
`relaxation` はjacobiバッチ射影のΔxにのみ掛ける減速係数(頂点を共有する制約の同時更新による過補正を抑える。gauss_seidelでは使わない)。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_distance.py`:

```python
import numpy as np
import pytest

from xpbd.constraints import DistanceConstraint

H = 0.01


def two_particles(stretch=2.0):
    x = np.array([[0.0, 0.0, 0.0], [stretch, 0.0, 0.0]])
    inv_mass = np.array([1.0, 1.0])
    con = DistanceConstraint(
        edges=np.array([[0, 1]]), rest_lengths=np.array([1.0])
    )
    return x, inv_mass, con


def test_rigid_projection_restores_rest_length():
    x, inv_mass, con = two_particles(stretch=2.0)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    dist = np.linalg.norm(x[1] - x[0])
    assert dist == pytest.approx(1.0, abs=1e-9)


def test_projection_is_symmetric_for_equal_masses():
    x, inv_mass, con = two_particles(stretch=2.0)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    # Both endpoints move toward each other by the same amount.
    assert x[0, 0] == pytest.approx(0.5)
    assert x[1, 0] == pytest.approx(1.5)


def test_pinned_vertex_does_not_move():
    x, _, con = two_particles(stretch=2.0)
    inv_mass = np.array([0.0, 1.0])
    con.reset_lambda()
    con.project(x, inv_mass, H)
    assert np.all(x[0] == 0.0)
    assert x[1, 0] == pytest.approx(1.0)


def test_compliance_leaves_xpbd_residual():
    # One projection with compliance leaves C * alpha_tilde/(w + alpha_tilde).
    compliance = 1e-4
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_mass = np.array([1.0, 1.0])
    con = DistanceConstraint(
        edges=np.array([[0, 1]]),
        rest_lengths=np.array([1.0]),
        compliance=compliance,
    )
    con.reset_lambda()
    con.project(x, inv_mass, H)
    alpha_tilde = compliance / H**2
    expected_c = 1.0 * alpha_tilde / (2.0 + alpha_tilde)
    dist = np.linalg.norm(x[1] - x[0])
    assert dist - 1.0 == pytest.approx(expected_c, rel=1e-9)


def test_project_one_matches_batch_for_single_edge():
    x_batch, inv_mass, con_a = two_particles(stretch=1.7)
    x_seq, _, con_b = two_particles(stretch=1.7)
    con_a.reset_lambda()
    con_a.project(x_batch, inv_mass, H)
    con_b.reset_lambda()
    con_b.project_one(0, x_seq, inv_mass, H)
    np.testing.assert_allclose(x_batch, x_seq, atol=1e-12)


def test_from_positions_uses_current_lengths():
    x = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [4.0, 3.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    con = DistanceConstraint.from_positions(x, edges)
    np.testing.assert_allclose(con.rest_lengths, [3.0, 4.0])


def test_len_returns_edge_count():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    con = DistanceConstraint.from_positions(x, np.array([[0, 1], [1, 2]]))
    assert len(con) == 2
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_distance.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 実装**

`src/xpbd/constraints/base.py`:

```python
"""Constraint interface shared by all XPBD constraint batches."""

from typing import Protocol

import numpy as np


class Constraint(Protocol):
    """A homogeneous batch of XPBD constraints.

    One object holds many constraints of the same kind (e.g. all edges).
    The solver calls reset_lambda() once per substep, then either the
    batched project() (jacobi mode) or project_one() per constraint
    (gauss_seidel mode).
    """

    def reset_lambda(self) -> None:
        """Reset accumulated Lagrange multipliers at substep start."""
        ...

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        """Project all constraints at once, applying corrections in-place."""
        ...

    def project_one(
        self, i: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        """Project the i-th constraint, applying corrections in-place."""
        ...

    def __len__(self) -> int:
        """Number of individual constraints in this batch."""
        ...
```

`src/xpbd/constraints/distance.py`:

```python
"""Distance (edge) constraint: C = |xi - xj| - rest_length."""

import numpy as np

_EPS = 1e-12


class DistanceConstraint:
    """Batch of distance constraints over edges of shape (E, 2)."""

    def __init__(
        self,
        edges: np.ndarray,
        rest_lengths: np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> None:
        self.edges = np.asarray(edges, dtype=np.int64)
        self.rest_lengths = np.asarray(rest_lengths, dtype=np.float64)
        self.compliance = float(compliance)
        self.relaxation = float(relaxation)
        self.lam = np.zeros(len(self.edges))

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        edges: np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> "DistanceConstraint":
        edges = np.asarray(edges, dtype=np.int64)
        d = positions[edges[:, 1]] - positions[edges[:, 0]]
        rest = np.linalg.norm(d, axis=1)
        return cls(edges, rest, compliance=compliance, relaxation=relaxation)

    def __len__(self) -> int:
        return len(self.edges)

    def reset_lambda(self) -> None:
        self.lam[:] = 0.0

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1)
        n = d / np.maximum(dist, _EPS)[:, None]
        c = dist - self.rest_lengths
        alpha = self.compliance / h**2
        w = inv_mass[i] + inv_mass[j]
        dlam = -(c + alpha * self.lam) / np.maximum(w + alpha, _EPS)
        self.lam += dlam
        corr = (self.relaxation * dlam)[:, None] * n
        np.add.at(x, j, inv_mass[j][:, None] * corr)
        np.add.at(x, i, -inv_mass[i][:, None] * corr)

    def project_one(
        self, k: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        i, j = self.edges[k]
        d = x[j] - x[i]
        dist = float(np.linalg.norm(d))
        if dist < _EPS:
            return
        n = d / dist
        c = dist - self.rest_lengths[k]
        alpha = self.compliance / h**2
        w = inv_mass[i] + inv_mass[j]
        denom = w + alpha
        if denom < _EPS:
            return
        dlam = -(c + alpha * self.lam[k]) / denom
        self.lam[k] += dlam
        x[j] += inv_mass[j] * dlam * n
        x[i] -= inv_mass[i] * dlam * n
```

`src/xpbd/constraints/__init__.py`:

```python
"""XPBD constraint batches."""

from .base import Constraint
from .distance import DistanceConstraint

__all__ = ["Constraint", "DistanceConstraint"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_distance.py -v`
Expected: 8 PASS

- [ ] **Step 5: コミット**

```bash
git add src/xpbd/constraints/ tests/test_distance.py
git commit -m "feat: constraint protocol and XPBD distance constraint"
```

---

### Task 5: 体積制約

**Files:**
- Create: `src/xpbd/constraints/volume.py`
- Modify: `src/xpbd/constraints/__init__.py`
- Test: `tests/test_volume.py`

**Interfaces:**
- Consumes: `xpbd.mesh.tet_volumes`(テストとfrom_positionsで使用)
- Produces:
  - `xpbd.constraints.VolumeConstraint(tets, rest_volumes, compliance=0.0, relaxation=1.0)`
  - `VolumeConstraint.from_positions(positions, tets, compliance=0.0, relaxation=1.0)`

制約関数は `C = 6 (V - V0)`。勾配(標準形):
`g1 = (p2-p0)×(p3-p0)`, `g2 = (p3-p0)×(p1-p0)`, `g3 = (p1-p0)×(p2-p0)`, `g0 = -(g1+g2+g3) = (p3-p1)×(p2-p1)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_volume.py`:

```python
import numpy as np
import pytest

from xpbd.constraints import VolumeConstraint
from xpbd.mesh import tet_volumes

H = 0.01


def unit_tet():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


def test_squashed_tet_recovers_volume():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    x[:, 2] *= 0.3  # squash along z
    inv_mass = np.ones(4)
    for _ in range(30):
        con.reset_lambda()
        con.project(x, inv_mass, H)
    vol = tet_volumes(x, tets)[0]
    assert vol == pytest.approx(1.0 / 6.0, rel=1e-5)


def test_rest_state_is_fixed_point():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    con.reset_lambda()
    con.project(x, np.ones(4), H)
    np.testing.assert_allclose(x, rest, atol=1e-12)


def test_pinned_vertices_do_not_move():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    x[3, 2] = 0.5
    inv_mass = np.array([0.0, 0.0, 0.0, 1.0])
    con.reset_lambda()
    con.project(x, inv_mass, H)
    np.testing.assert_allclose(x[:3], rest[:3], atol=1e-12)


def test_project_one_matches_batch_for_single_tet():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    x_a = rest.copy()
    x_a[:, 2] *= 0.5
    x_b = x_a.copy()
    inv_mass = np.ones(4)
    con_a = VolumeConstraint.from_positions(rest, tets)
    con_b = VolumeConstraint.from_positions(rest, tets)
    con_a.reset_lambda()
    con_a.project(x_a, inv_mass, H)
    con_b.reset_lambda()
    con_b.project_one(0, x_b, inv_mass, H)
    np.testing.assert_allclose(x_a, x_b, atol=1e-12)


def test_len_returns_tet_count():
    rest = unit_tet()
    con = VolumeConstraint.from_positions(rest, np.array([[0, 1, 2, 3]]))
    assert len(con) == 1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_volume.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: `src/xpbd/constraints/volume.py` を実装**

```python
"""Tetrahedron volume constraint: C = 6 * (V - V0)."""

import numpy as np

from ..mesh import tet_volumes

_EPS = 1e-12


class VolumeConstraint:
    """Batch of volume constraints over tets of shape (T, 4)."""

    def __init__(
        self,
        tets: np.ndarray,
        rest_volumes: np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> None:
        self.tets = np.asarray(tets, dtype=np.int64)
        self.rest_volumes = np.asarray(rest_volumes, dtype=np.float64)
        self.compliance = float(compliance)
        self.relaxation = float(relaxation)
        self.lam = np.zeros(len(self.tets))

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        tets: np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> "VolumeConstraint":
        tets = np.asarray(tets, dtype=np.int64)
        rest = tet_volumes(positions, tets)
        return cls(tets, rest, compliance=compliance, relaxation=relaxation)

    def __len__(self) -> int:
        return len(self.tets)

    def reset_lambda(self) -> None:
        self.lam[:] = 0.0

    @staticmethod
    def _gradients(p0, p1, p2, p3):
        g1 = np.cross(p2 - p0, p3 - p0)
        g2 = np.cross(p3 - p0, p1 - p0)
        g3 = np.cross(p1 - p0, p2 - p0)
        g0 = -(g1 + g2 + g3)
        return g0, g1, g2, g3

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        t = self.tets
        p0, p1, p2, p3 = x[t[:, 0]], x[t[:, 1]], x[t[:, 2]], x[t[:, 3]]
        g0, g1, g2, g3 = self._gradients(p0, p1, p2, p3)
        vol = tet_volumes(x, t)
        c = 6.0 * (vol - self.rest_volumes)
        alpha = self.compliance / h**2
        w = (
            inv_mass[t[:, 0]] * np.einsum("ij,ij->i", g0, g0)
            + inv_mass[t[:, 1]] * np.einsum("ij,ij->i", g1, g1)
            + inv_mass[t[:, 2]] * np.einsum("ij,ij->i", g2, g2)
            + inv_mass[t[:, 3]] * np.einsum("ij,ij->i", g3, g3)
        )
        dlam = -(c + alpha * self.lam) / np.maximum(w + alpha, _EPS)
        self.lam += dlam
        s = self.relaxation * dlam
        for col, g in zip(range(4), (g0, g1, g2, g3)):
            idx = t[:, col]
            np.add.at(x, idx, (inv_mass[idx] * s)[:, None] * g)

    def project_one(
        self, k: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        t = self.tets[k]
        p0, p1, p2, p3 = x[t[0]], x[t[1]], x[t[2]], x[t[3]]
        g1 = np.cross(p2 - p0, p3 - p0)
        g2 = np.cross(p3 - p0, p1 - p0)
        g3 = np.cross(p1 - p0, p2 - p0)
        g0 = -(g1 + g2 + g3)
        vol = np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        c = 6.0 * (vol - self.rest_volumes[k])
        alpha = self.compliance / h**2
        grads = (g0, g1, g2, g3)
        w = sum(inv_mass[t[m]] * np.dot(grads[m], grads[m]) for m in range(4))
        denom = w + alpha
        if denom < _EPS:
            return
        dlam = -(c + alpha * self.lam[k]) / denom
        self.lam[k] += dlam
        for m in range(4):
            x[t[m]] += inv_mass[t[m]] * dlam * grads[m]
```

`src/xpbd/constraints/__init__.py` を更新:

```python
"""XPBD constraint batches."""

from .base import Constraint
from .distance import DistanceConstraint
from .volume import VolumeConstraint

__all__ = ["Constraint", "DistanceConstraint", "VolumeConstraint"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_volume.py -v`
Expected: 5 PASS

- [ ] **Step 5: コミット**

```bash
git add src/xpbd/constraints/volume.py src/xpbd/constraints/__init__.py tests/test_volume.py
git commit -m "feat: XPBD tetrahedron volume constraint"
```

---

### Task 6: 地面衝突制約

**Files:**
- Create: `src/xpbd/constraints/ground.py`
- Modify: `src/xpbd/constraints/__init__.py`
- Test: `tests/test_ground.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `xpbd.constraints.GroundConstraint(n_vertices: int, height: float = 0.0)` — 不等式制約 `y >= height` をα=0で射影(違反頂点をクランプ)。`inv_mass == 0` の頂点は動かさない

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_ground.py`:

```python
import numpy as np

from xpbd.constraints import GroundConstraint

H = 0.01


def test_penetrating_vertices_are_clamped():
    x = np.array([[0.0, -0.5, 0.0], [1.0, 2.0, 0.0], [2.0, -0.1, 3.0]])
    inv_mass = np.ones(3)
    con = GroundConstraint(3)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    np.testing.assert_allclose(x[:, 1], [0.0, 2.0, 0.0])
    # x and z are untouched.
    np.testing.assert_allclose(x[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(x[:, 2], [0.0, 0.0, 3.0])


def test_custom_height():
    x = np.array([[0.0, 0.2, 0.0]])
    con = GroundConstraint(1, height=0.5)
    con.reset_lambda()
    con.project(x, np.ones(1), H)
    assert x[0, 1] == 0.5


def test_pinned_vertex_is_not_clamped():
    x = np.array([[0.0, -1.0, 0.0]])
    con = GroundConstraint(1)
    con.reset_lambda()
    con.project(x, np.zeros(1), H)
    assert x[0, 1] == -1.0


def test_project_one_matches_batch():
    x_a = np.array([[0.0, -0.5, 0.0], [1.0, 2.0, 0.0]])
    x_b = x_a.copy()
    inv_mass = np.ones(2)
    con = GroundConstraint(2)
    con.reset_lambda()
    con.project(x_a, inv_mass, H)
    con.reset_lambda()
    for i in range(len(con)):
        con.project_one(i, x_b, inv_mass, H)
    np.testing.assert_allclose(x_a, x_b)


def test_len_is_vertex_count():
    assert len(GroundConstraint(7)) == 7
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_ground.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: `src/xpbd/constraints/ground.py` を実装**

```python
"""Ground plane collision: inequality constraint y >= height."""

import numpy as np


class GroundConstraint:
    """Rigid (compliance 0) unilateral constraint against a ground plane.

    Projection simply clamps violating movable vertices to the plane.
    """

    def __init__(self, n_vertices: int, height: float = 0.0) -> None:
        self.n_vertices = int(n_vertices)
        self.height = float(height)

    def __len__(self) -> int:
        return self.n_vertices

    def reset_lambda(self) -> None:
        # Rigid unilateral projection carries no lambda state.
        pass

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        violating = (inv_mass > 0.0) & (x[:, 1] < self.height)
        x[violating, 1] = self.height

    def project_one(
        self, i: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        if inv_mass[i] > 0.0 and x[i, 1] < self.height:
            x[i, 1] = self.height
```

`src/xpbd/constraints/__init__.py` を更新:

```python
"""XPBD constraint batches."""

from .base import Constraint
from .distance import DistanceConstraint
from .ground import GroundConstraint
from .volume import VolumeConstraint

__all__ = [
    "Constraint",
    "DistanceConstraint",
    "GroundConstraint",
    "VolumeConstraint",
]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_ground.py -v`
Expected: 5 PASS

- [ ] **Step 5: コミット**

```bash
git add src/xpbd/constraints/ground.py src/xpbd/constraints/__init__.py tests/test_ground.py
git commit -m "feat: ground plane collision constraint"
```

---

### Task 7: XPBDソルバ (`solver.py`)

**Files:**
- Create: `src/xpbd/solver.py`
- Modify: `src/xpbd/__init__.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Consumes: `SoftBody` (Task 3)、`Constraint` protocol と各制約 (Task 4-6)
- Produces:
  - `xpbd.solver.Solver(body: SoftBody, constraints: list, gravity=(0.0, -9.81, 0.0), n_substeps: int = 10, mode: str = "jacobi")` — `mode` は `"jacobi" | "gauss_seidel"`、不正値は `ValueError`
  - `Solver.step(dt: float) -> None` — 1フレーム進める

アルゴリズム(spec準拠、Small Steps方式):

```
h = dt / n_substeps
for each substep:
    x_prev = x.copy()
    v[movable] += gravity * h
    x += v * h
    for c in constraints:
        c.reset_lambda()
        jacobi:       c.project(x, inv_mass, h)
        gauss_seidel: for i in range(len(c)): c.project_one(i, x, inv_mass, h)
    v = (x - x_prev) / h
```

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_solver.py`:

```python
import numpy as np
import pytest

from xpbd.body import SoftBody
from xpbd.constraints import (
    DistanceConstraint,
    GroundConstraint,
    VolumeConstraint,
)
from xpbd.mesh import tet_grid
from xpbd.solver import Solver

DT = 1.0 / 60.0


def single_particle_body():
    # A degenerate body with one free particle and no constraints.
    return SoftBody(
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        inv_mass=np.ones(1),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
    )


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Solver(single_particle_body(), [], mode="nonsense")


def test_free_fall_matches_analytic_solution():
    body = single_particle_body()
    solver = Solver(body, [], n_substeps=10)
    for _ in range(60):
        solver.step(DT)
    # y = -0.5 g t^2 with t = 1s; symplectic Euler error is O(g t h / 2).
    assert body.positions[0, 1] == pytest.approx(-0.5 * 9.81, abs=0.02)


def test_pinned_particle_does_not_fall():
    body = single_particle_body()
    body.inv_mass[0] = 0.0
    solver = Solver(body, [])
    for _ in range(30):
        solver.step(DT)
    np.testing.assert_allclose(body.positions, 0.0)
    np.testing.assert_allclose(body.velocities, 0.0)


def hanging_pair(mode):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    body = SoftBody(
        positions=positions.copy(),
        velocities=np.zeros((2, 3)),
        inv_mass=np.array([0.0, 1.0]),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.array([[0, 1]]),
    )
    con = DistanceConstraint.from_positions(body.positions, body.edges)
    return body, Solver(body, [con], n_substeps=10, mode=mode)


@pytest.mark.parametrize("mode", ["jacobi", "gauss_seidel"])
def test_hanging_particle_keeps_rest_length(mode):
    body, solver = hanging_pair(mode)
    for _ in range(120):
        solver.step(DT)
    dist = np.linalg.norm(body.positions[1] - body.positions[0])
    assert dist == pytest.approx(1.0, abs=1e-6)


def test_compliant_edge_stretches_under_gravity():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    body = SoftBody(
        positions=positions.copy(),
        velocities=np.zeros((2, 3)),
        inv_mass=np.array([0.0, 1.0]),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.array([[0, 1]]),
    )
    con = DistanceConstraint.from_positions(
        body.positions, body.edges, compliance=1e-2
    )
    solver = Solver(body, [con], n_substeps=10)
    for _ in range(240):
        solver.step(DT)
    dist = np.linalg.norm(body.positions[1] - body.positions[0])
    assert dist > 1.001


def dropped_cube_scene(mode):
    positions, tets = tet_grid(2, 2, 2, spacing=0.25)
    positions[:, 1] += 1.0
    body = SoftBody.from_tet_mesh(positions, tets)
    constraints = [
        DistanceConstraint.from_positions(
            body.positions, body.edges, relaxation=0.35
        ),
        VolumeConstraint.from_positions(
            body.positions, body.tets, relaxation=0.35
        ),
        GroundConstraint(len(body.positions)),
    ]
    return body, Solver(body, constraints, n_substeps=20, mode=mode)


def test_dropped_cube_rests_on_ground_without_penetration():
    body, solver = dropped_cube_scene("jacobi")
    for _ in range(120):
        solver.step(DT)
    assert body.positions[:, 1].min() >= -1e-9


def test_jacobi_and_gauss_seidel_reach_similar_states():
    body_j, solver_j = dropped_cube_scene("jacobi")
    body_g, solver_g = dropped_cube_scene("gauss_seidel")
    for _ in range(120):
        solver_j.step(DT)
        solver_g.step(DT)
    com_j = body_j.positions.mean(axis=0)
    com_g = body_g.positions.mean(axis=0)
    np.testing.assert_allclose(com_j, com_g, atol=0.05)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'xpbd.solver'`)

- [ ] **Step 3: `src/xpbd/solver.py` を実装**

```python
"""XPBD solver with substepping (Macklin 2016, Mueller 2019 Small Steps)."""

import numpy as np

from .body import SoftBody

_MODES = ("jacobi", "gauss_seidel")


class Solver:
    """Advances a soft body by projecting constraint batches each substep.

    jacobi mode projects each batch vectorized (simultaneous updates via
    np.add.at); gauss_seidel mode projects constraints one by one and is
    the slow, paper-faithful reference.
    """

    def __init__(
        self,
        body: SoftBody,
        constraints: list,
        gravity: tuple = (0.0, -9.81, 0.0),
        n_substeps: int = 10,
        mode: str = "jacobi",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.body = body
        self.constraints = list(constraints)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.n_substeps = int(n_substeps)
        self.mode = mode

    def step(self, dt: float) -> None:
        """Advance the simulation by one frame of duration dt."""
        body = self.body
        h = dt / self.n_substeps
        movable = body.inv_mass > 0.0
        for _ in range(self.n_substeps):
            x_prev = body.positions.copy()
            body.velocities[movable] += self.gravity * h
            body.positions += body.velocities * h
            for con in self.constraints:
                con.reset_lambda()
                if self.mode == "jacobi":
                    con.project(body.positions, body.inv_mass, h)
                else:
                    for i in range(len(con)):
                        con.project_one(i, body.positions, body.inv_mass, h)
            body.velocities = (body.positions - x_prev) / h
```

`src/xpbd/__init__.py` を更新:

```python
"""A pure XPBD physics engine for research and experimentation."""

from .body import SoftBody
from .solver import Solver

__all__ = ["SoftBody", "Solver"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run pytest tests/test_solver.py -v`
Expected: 8 PASS(gauss_seidelの落下テストは数秒かかる場合がある)

- [ ] **Step 5: 全テストとカバレッジを確認**

Run: `uv run pytest --cov=xpbd --cov-report=term-missing`
Expected: 全テストPASS、カバレッジ80%以上

- [ ] **Step 6: コミット**

```bash
git add src/xpbd/solver.py src/xpbd/__init__.py tests/test_solver.py
git commit -m "feat: XPBD substep solver with jacobi and gauss_seidel modes"
```

---

### Task 8: サンプルデモ + README

**Files:**
- Create: `examples/softbody_drop.py`
- Create: `README.md`

**Interfaces:**
- Consumes: 全モジュール (Task 1-7)
- Produces: `uv run --extra viz python examples/softbody_drop.py` で起動するデモ

- [ ] **Step 1: viz依存を入れる**

Run: `uv sync --extra viz`
Expected: polyscopeがインストールされる

- [ ] **Step 2: `examples/softbody_drop.py` を実装**

```python
"""Soft body drop demo rendered with polyscope.

Run: uv run --extra viz python examples/softbody_drop.py
Adjust compliance, substeps and solver mode via CLI flags.
"""

import argparse

import numpy as np
import polyscope as ps

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    DistanceConstraint,
    GroundConstraint,
    VolumeConstraint,
)
from xpbd.mesh import tet_grid

DT = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--spacing", type=float, default=0.15)
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--edge-compliance", type=float, default=1e-4)
    parser.add_argument("--volume-compliance", type=float, default=0.0)
    parser.add_argument("--relaxation", type=float, default=0.35)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["jacobi", "gauss_seidel"], default="jacobi"
    )
    return parser.parse_args()


def tilt(positions: np.ndarray, angle_x: float, angle_z: float) -> np.ndarray:
    """Rotate positions about their centroid for a more interesting fall."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    center = positions.mean(axis=0)
    return (positions - center) @ (rot_z @ rot_x).T + center


def build_solver(args: argparse.Namespace) -> Solver:
    positions, tets = tet_grid(args.nx, args.ny, args.nz, spacing=args.spacing)
    positions = tilt(positions, angle_x=0.4, angle_z=0.3)
    positions[:, 1] += args.drop_height - positions[:, 1].min()
    body = SoftBody.from_tet_mesh(positions, tets)
    constraints = [
        DistanceConstraint.from_positions(
            body.positions,
            body.edges,
            compliance=args.edge_compliance,
            relaxation=args.relaxation,
        ),
        VolumeConstraint.from_positions(
            body.positions,
            body.tets,
            compliance=args.volume_compliance,
            relaxation=args.relaxation,
        ),
        GroundConstraint(len(body.positions)),
    ]
    return Solver(
        body, constraints, n_substeps=args.substeps, mode=args.mode
    )


def register_floor() -> None:
    size = 4.0
    verts = np.array(
        [[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    floor = ps.register_surface_mesh("floor", verts, faces)
    floor.set_color((0.75, 0.75, 0.78))


def main() -> None:
    args = parse_args()
    solver = build_solver(args)
    body = solver.body

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_floor()
    mesh = ps.register_volume_mesh("softbody", body.positions, tets=body.tets)

    def frame() -> None:
        solver.step(DT)
        mesh.update_vertex_positions(body.positions)

    ps.set_user_callback(frame)
    ps.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: デモをヘッドレスに近い形で動作確認**

GUIを開かずロジックだけ検証する(polyscope未初期化でsolver部分を回す):

Run: `uv run --extra viz python -c "
import examples.softbody_drop as demo
import sys
sys.argv = ['demo']
args = demo.parse_args()
solver = demo.build_solver(args)
for _ in range(30):
    solver.step(demo.DT)
print('min y:', solver.body.positions[:, 1].min())
print('OK')
"`
Expected: `OK` が出力され、min y が -1e-9 以上

- [ ] **Step 4: GUIデモの手動確認(ユーザー確認項目)**

Run: `uv run --extra viz python examples/softbody_drop.py`
Expected: polyscopeウィンドウが開き、柔らかい立方体が落下して床でつぶれて弾む。ウィンドウを閉じて終了

- [ ] **Step 5: `README.md` を書く**

````markdown
# xpbd

A pure XPBD (Extended Position Based Dynamics) engine in Python, built as a
small, hackable foundation for physics research and experimentation.

- Substep-based XPBD solver (Macklin et al. 2016 + "Small Steps" 2019)
- Constraint batches: distance, tetrahedron volume, ground collision
- Two solver modes: vectorized `jacobi` (fast) and sequential
  `gauss_seidel` (paper-faithful reference)
- Core depends on numpy only; visualization lives in `examples/`

## Setup

```bash
uv sync --extra viz
```

## Run the demo

```bash
uv run --extra viz python examples/softbody_drop.py
```

Useful flags: `--mode gauss_seidel`, `--substeps 30`,
`--edge-compliance 1e-3`, `--nx 8 --ny 8 --nz 8`.

## Run tests

```bash
uv run pytest --cov=xpbd
```

## Layout

```
src/xpbd/
  body.py          # SoftBody state container
  solver.py        # XPBD substep loop
  mesh.py          # tet grid generation, edges, boundary surface
  constraints/     # distance, volume, ground (one class per kind)
examples/
  softbody_drop.py # polyscope demo
```

## Adding a new constraint

Implement the `Constraint` protocol (`reset_lambda`, `project`,
`project_one`, `__len__`) in `src/xpbd/constraints/` and pass an instance
to `Solver`. See `distance.py` for the canonical XPBD update.
````

- [ ] **Step 6: 全テスト最終確認とコミット**

Run: `uv run pytest --cov=xpbd --cov-report=term-missing`
Expected: 全PASS、カバレッジ80%以上

```bash
git add examples/softbody_drop.py README.md uv.lock
git commit -m "feat: polyscope soft body drop demo and README"
```

---

## Self-Review Notes

- Spec coverage: ソルバ/3制約/2モード/メッシュ生成/SoftBody/デモ/テスト計画/カバレッジ — 全てタスク1-8に対応済み
- 型整合: `project(x, inv_mass, h)` / `project_one(i, x, inv_mass, h)` / `reset_lambda()` / `__len__()` を全制約で統一。`from_positions` の引数順も統一
- relaxation はspecに明記がないが、Jacobi同時更新の過補正対策として設計承認時に合意済みの範囲(バッチ射影の実装詳細)
