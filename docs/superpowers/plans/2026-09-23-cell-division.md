# Cell Division Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a physiologically grounded cell division simulation in XPBD featuring spindle elongation, astral microtubule cortex rounding, contractile ring cleavage furrowing, mesh slicing/capping into two independent `SoftBody` instances, and post-division tangential surface relaxation.

**Architecture:** Pure NumPy core extensions in `src/xpbd/` (`topology.py`, `constraints/spindle.py`, `constraints/contractile_ring.py`) accompanied by complete unit test coverage, and an interactive state-machine driven Polyscope demonstration in `examples/cell_division.py`.

**Tech Stack:** Python >= 3.11, NumPy >= 1.26, Polyscope >= 2.2, pytest, pytest-cov.

**Spec:** `docs/superpowers/specs/2026-09-23-cell-division-design.md`

## Global Constraints

- Pure NumPy in `src/xpbd/`; no Polyscope or GUI imports in core library code.
- Closed surface meshes must be valid 2-manifold triangle meshes (Euler characteristic $\chi = V - E + F = 2$, positive volume, outward normals).
- All new constraints must implement the `xpbd.constraints.base.Constraint` protocol (`reset_lambda`, `project`, `project_one`, `__len__`).
- Time-integration and substepping must stay consistent with XPBD compliance formulation ($\alpha = \text{compliance} / h^2$).

## Review Focus

1. Boundary loop ordering during slicing: boundary edges must form a continuous oriented closed loop to avoid non-manifold self-intersecting caps.
2. Watertight cap normals: cap triangles must have counter-clockwise winding when viewed from outside each daughter cell so volume calculations are positive and correct.
3. Internal particle allocation: boundary particles near $x = cut\_x$ must not be duplicated or lost.
4. Tangential relaxation volume preservation: relaxation must be strictly tangential to surface normals so cells do not collapse or inflate during smoothing.
5. Post-division constraint validity: new `DistanceConstraint` and `GlobalVolumeConstraint` rest lengths and volumes must reflect the daughter cell geometry accurately.

---

### Task 1: Mesh Slicing and Capping Topology Utility

**Files:**
- Create: `src/xpbd/topology.py`
- Test: `tests/test_topology.py`

**Interfaces:**
- Consumes: `xpbd.mesh.edges_from_faces`
- Produces:
  - `slice_and_cap_mesh(positions: np.ndarray, faces: np.ndarray, cut_x: float = 0.0) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]`
  - `partition_particles(positions: np.ndarray, cut_x: float = 0.0) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Write the failing test for mesh slicing, capping, and particle partitioning**

```python
# tests/test_topology.py
import numpy as np
import pytest
from xpbd.mesh import icosphere_mesh, tet_volumes
from xpbd.topology import slice_and_cap_mesh, partition_particles

def test_slice_and_cap_icosphere():
    pos, faces = icosphere_mesh(subdivisions=2, radius=1.0, origin=(0.0, 0.0, 0.0))
    (pos_l, faces_l), (pos_r, faces_r) = slice_and_cap_mesh(pos, faces, cut_x=0.0)

    # 1. Check vertex coordinates
    assert np.all(pos_l[:, 0] <= 1e-12)
    assert np.all(pos_r[:, 0] >= -1e-12)

    # 2. Check manifoldness / Euler characteristic: chi = V - E + F = 2 for closed spheres
    from xpbd.mesh import edges_from_faces
    e_l = edges_from_faces(faces_l)
    chi_l = len(pos_l) - len(e_l) + len(faces_l)
    assert chi_l == 2, f"Left mesh Euler characteristic is {chi_l}, expected 2"

    e_r = edges_from_faces(faces_r)
    chi_r = len(pos_r) - len(e_r) + len(faces_r)
    assert chi_r == 2, f"Right mesh Euler characteristic is {chi_r}, expected 2"

    # 3. Check positive volume (outward normals)
    def compute_volume(p, f):
        p1 = p[f[:, 0]]
        p2 = p[f[:, 1]]
        p3 = p[f[:, 2]]
        return np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0

    vol_l = compute_volume(pos_l, faces_l)
    vol_r = compute_volume(pos_r, faces_r)
    assert vol_l > 0.0
    assert vol_r > 0.0
    total_vol = compute_volume(pos, faces)
    assert np.isclose(vol_l + vol_r, total_vol, rtol=0.1)

def test_partition_particles():
    pts = np.array([
        [-0.5, 0.0, 0.0],
        [-0.1, 0.2, 0.1],
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.0],
        [0.5, 0.3, -0.1],
    ])
    left_idx, right_idx = partition_particles(pts, cut_x=0.0)
    assert set(left_idx.tolist() + right_idx.tolist()) == set(range(len(pts)))
    assert len(set(left_idx).intersection(set(right_idx))) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_topology.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'xpbd.topology')

- [ ] **Step 3: Write minimal implementation in `src/xpbd/topology.py`**

Implement `partition_particles` and `slice_and_cap_mesh`:
- Identify vertices with $x < cut\_x$ and $x \ge cut\_x$.
- Filter faces belonging entirely to left and right.
- For faces straddling the cut plane, find the open boundary edges.
- Sort the boundary edges into a contiguous oriented loop.
- Insert a center vertex at the centroid of the boundary loop.
- Create fan triangles connecting the center vertex to boundary edges with consistent outward normals.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_topology.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/topology.py tests/test_topology.py
git commit -m "feat: add mesh slicing and capping topology utility"
```

---

### Task 2: Tangential Surface Relaxation

**Files:**
- Modify: `src/xpbd/topology.py`
- Test: `tests/test_tangential_relaxation.py`

**Interfaces:**
- Consumes: `src/xpbd/topology.py`
- Produces:
  - `tangential_surface_relaxation(positions: np.ndarray, faces: np.ndarray, center: np.ndarray, factor: float = 0.15, n_iters: int = 1) -> np.ndarray`

- [ ] **Step 1: Write the failing test for tangential relaxation**

```python
# tests/test_tangential_relaxation.py
import numpy as np
import pytest
from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.topology import tangential_surface_relaxation

def test_tangential_surface_relaxation_reduces_variance():
    pos, faces = icosphere_mesh(subdivisions=2, radius=1.0, origin=(0.0, 0.0, 0.0))
    # Distort positions: push vertices on one side closer together
    distorted = pos.copy()
    mask = distorted[:, 0] > 0
    distorted[mask, 1:] *= 0.5
    
    edges = edges_from_faces(faces)
    lens_before = np.linalg.norm(distorted[edges[:, 1]] - distorted[edges[:, 0]], axis=1)
    std_before = np.std(lens_before)

    center = np.mean(distorted, axis=0)
    relaxed = tangential_surface_relaxation(distorted, faces, center=center, factor=0.2, n_iters=10)

    lens_after = np.linalg.norm(relaxed[edges[:, 1]] - relaxed[edges[:, 0]], axis=1)
    std_after = np.std(lens_after)

    # Edge length variance must decrease
    assert std_after < std_before
    # Centroid and overall scale must be preserved
    assert np.allclose(np.mean(relaxed, axis=0), center, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tangential_relaxation.py -v`
Expected: FAIL (ImportError: cannot import name 'tangential_surface_relaxation')

- [ ] **Step 3: Implement `tangential_surface_relaxation` in `src/xpbd/topology.py`**

- Compute vertex normals from incident face normals.
- For each vertex, compute average neighbor position $\bar{x}_i$.
- Compute tangential shift: $\Delta x_i^{\text{tan}} = (\bar{x}_i - x_i) - ((\bar{x}_i - x_i) \cdot n_i) n_i$.
- Update positions: $x_i \leftarrow x_i + \text{factor} \times \Delta x_i^{\text{tan}}$.
- Project radial distance back to preserve radius profile from center: $x_i \leftarrow \text{center} + \frac{x_i - \text{center}}{\|x_i - \text{center}\|} \cdot r_{orig}$.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tangential_relaxation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/topology.py tests/test_tangential_relaxation.py
git commit -m "feat: add tangential surface relaxation utility"
```

---

### Task 3: Spindle and Astral Microtubule Constraint

**Files:**
- Create: `src/xpbd/constraints/spindle.py`
- Modify: `src/xpbd/constraints/__init__.py`
- Test: `tests/test_spindle.py`

**Interfaces:**
- Consumes: `xpbd.constraints.base.Constraint`
- Produces:
  - `SpindleConstraint(pole_0: int, pole_1: int, left_tethers: np.ndarray, right_tethers: np.ndarray, ...)`

- [ ] **Step 1: Write the failing test for SpindleConstraint**

```python
# tests/test_spindle.py
import numpy as np
import pytest
from xpbd.constraints.spindle import SpindleConstraint

def test_spindle_elongation_and_astral_tethers():
    # 2 poles and 4 membrane vertices
    # 0: pole_l, 1: pole_r, 2,3: left membrane, 4,5: right membrane
    x = np.array([
        [-0.2, 0.0, 0.0],
        [ 0.2, 0.0, 0.0],
        [-0.5, 0.2, 0.0],
        [-0.5, -0.2, 0.0],
        [ 0.5, 0.2, 0.0],
        [ 0.5, -0.2, 0.0],
    ], dtype=np.float64)
    inv_mass = np.ones(len(x))
    
    con = SpindleConstraint(
        pole_0=0,
        pole_1=1,
        left_membrane_idx=np.array([2, 3]),
        right_membrane_idx=np.array([4, 5]),
        pole_rest_length=0.4,
        tether_rest_length=0.25,
        tether_compliance=1e-3,
    )
    
    assert len(con) > 0
    # Increase target length of spindle
    con.set_target_length(0.8)
    assert con.pole_rest_length == 0.8

    # Project should push poles apart and pull tethers
    h = 0.01
    x_cur = x.copy()
    con.reset_lambda()
    con.project(x_cur, inv_mass, h)
    
    # Poles must move further apart
    assert (x_cur[1, 0] - x_cur[0, 0]) > (x[1, 0] - x[0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spindle.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `SpindleConstraint` in `src/xpbd/constraints/spindle.py`**

- Conforms to `Constraint` protocol.
- Handles pole-to-pole distance constraint.
- Handles radial astral microtubule distance constraints between each pole and its respective hemisphere vertices.
- Export in `src/xpbd/constraints/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spindle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/constraints/spindle.py src/xpbd/constraints/__init__.py tests/test_spindle.py
git commit -m "feat: add spindle and astral microtubule constraint"
```

---

### Task 4: Contractile Ring Constraint

**Files:**
- Create: `src/xpbd/constraints/contractile_ring.py`
- Modify: `src/xpbd/constraints/__init__.py`
- Test: `tests/test_contractile_ring.py`

**Interfaces:**
- Consumes: `xpbd.constraints.base.Constraint`
- Produces:
  - `ContractileRingConstraint(ring_indices: np.ndarray, center_axis: tuple[float, float] = (0.0, 0.0), target_radius: float = 0.5, compliance: float = 0.0)`

- [ ] **Step 1: Write the failing test for ContractileRingConstraint**

```python
# tests/test_contractile_ring.py
import numpy as np
import pytest
from xpbd.constraints.contractile_ring import ContractileRingConstraint

def test_contractile_ring_constriction():
    # 4 ring vertices arranged in circle of radius 1.0 around (y=0, z=0) at x=0
    x = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    inv_mass = np.ones(4)

    ring = ContractileRingConstraint(
        ring_indices=np.arange(4),
        center_y=0.0,
        center_z=0.0,
        target_radius=0.5,
        compliance=0.0,
    )
    
    h = 0.01
    x_cur = x.copy()
    ring.reset_lambda()
    ring.project(x_cur, inv_mass, h)

    radii = np.sqrt(x_cur[:, 1]**2 + x_cur[:, 2]**2)
    assert np.allclose(radii, 0.5, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contractile_ring.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `ContractileRingConstraint` in `src/xpbd/constraints/contractile_ring.py`**

- Conforms to `Constraint` protocol.
- Projects ring vertices radially toward the specified central longitudinal axis $(x_i, y_0, z_0)$ to achieve `target_radius`.
- Export in `src/xpbd/constraints/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contractile_ring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/xpbd/constraints/contractile_ring.py src/xpbd/constraints/__init__.py tests/test_contractile_ring.py
git commit -m "feat: add contractile ring constraint"
```

---

### Task 5: Cell Division Headless Pipeline Integration Test

**Files:**
- Create: `tests/test_cell_division_pipeline.py`

**Interfaces:**
- Consumes: `SoftBody`, `Solver`, `SpindleConstraint`, `ContractileRingConstraint`, `slice_and_cap_mesh`, `tangential_surface_relaxation`

- [ ] **Step 1: Write integration test executing full cycle headlessly**

```python
# tests/test_cell_division_pipeline.py
import numpy as np
import pytest
from xpbd import SoftBody, Solver
from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.constraints import (
    DistanceConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
    SpindleConstraint,
    ContractileRingConstraint,
)
from xpbd.topology import slice_and_cap_mesh, partition_particles, tangential_surface_relaxation

def test_full_cell_division_lifecycle():
    # Setup cell
    r_cell = 0.5
    origin = (0.0, 1.5, 0.0)
    surf_pos, faces = icosphere_mesh(subdivisions=2, radius=r_cell, origin=origin)
    n_surf = len(surf_pos)

    # 2 poles
    poles_pos = np.array([
        [-0.1, origin[1], origin[2]],
        [ 0.1, origin[1], origin[2]],
    ], dtype=np.float64)
    total_pos = np.concatenate([surf_pos, poles_pos], axis=0)
    p0_idx, p1_idx = n_surf, n_surf + 1

    inv_mass = np.ones(len(total_pos))
    body = SoftBody(
        positions=total_pos,
        velocities=np.zeros_like(total_pos),
        inv_mass=inv_mass,
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=edges_from_faces(faces),
    )

    left_mask = surf_pos[:, 0] < origin[0]
    right_mask = surf_pos[:, 0] >= origin[0]
    left_surf_idx = np.where(left_mask)[0]
    right_surf_idx = np.where(right_mask)[0]

    spindle = SpindleConstraint(
        p0_idx, p1_idx, left_surf_idx, right_surf_idx,
        pole_rest_length=0.2, tether_rest_length=r_cell * 0.7
    )

    equator_mask = np.abs(surf_pos[:, 0] - origin[0]) < 0.1
    ring_idx = np.where(equator_mask)[0]
    ring = ContractileRingConstraint(ring_idx, center_y=origin[1], center_z=origin[2], target_radius=r_cell)

    constraints = [
        DistanceConstraint.from_positions(body.positions, body.edges),
        GlobalVolumeConstraint.from_positions(body.positions, [faces]),
        spindle,
        ring,
    ]
    solver = Solver(body, constraints, gravity=(0.0, 0.0, 0.0))

    # Phase 1: Elongate
    for _ in range(20):
        spindle.set_target_length(spindle.pole_rest_length + 0.02)
        solver.step(0.016)
    assert (body.positions[p1_idx, 0] - body.positions[p0_idx, 0]) > 0.3

    # Phase 2: Constrict
    for _ in range(30):
        ring.set_target_radius(max(0.05, ring.target_radius - 0.015))
        solver.step(0.016)
    
    # Phase 3: Divide
    cur_surf = body.positions[:n_surf]
    (pos_l, faces_l), (pos_r, faces_r) = slice_and_cap_mesh(cur_surf, faces, cut_x=origin[0])
    assert len(pos_l) > 0 and len(pos_r) > 0
    assert len(faces_l) > 0 and len(faces_r) > 0

    # Build daughter bodies
    body_l = SoftBody.from_surface_mesh(pos_l, faces_l)
    body_r = SoftBody.from_surface_mesh(pos_r, faces_r)

    # Phase 4: Relax
    center_l = np.mean(body_l.positions, axis=0)
    pos_l_relaxed = tangential_surface_relaxation(body_l.positions, faces_l, center=center_l, factor=0.15, n_iters=5)
    body_l.positions[:] = pos_l_relaxed

    # Phase 5: Independent simulation step
    solver_l = Solver(body_l, [DistanceConstraint.from_positions(body_l.positions, body_l.edges), GlobalVolumeConstraint.from_positions(body_l.positions, [faces_l])])
    solver_r = Solver(body_r, [DistanceConstraint.from_positions(body_r.positions, body_r.edges), GlobalVolumeConstraint.from_positions(body_r.positions, [faces_r])])
    solver_l.step(0.016)
    solver_r.step(0.016)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_cell_division_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cell_division_pipeline.py
git commit -m "test: add integration test for full cell division lifecycle"
```

---

### Task 6: Interactive Polyscope Demonstration Application

**Files:**
- Modify: `examples/cell_division.py`

**Interfaces:**
- Consumes: all above modules + Polyscope UI

- [ ] **Step 1: Refactor `examples/cell_division.py` to implement the complete 5-stage state machine**
- Stages: `ELONGATING` $\to$ `CONSTRICTING` $\to$ `DIVIDING` $\to$ `RELAXING` $\to$ `INDEPENDENT`.
- Handle dynamic mesh updates in Polyscope (transition from 1 "bag" surface mesh to 2 separate meshes "daughter_left" and "daughter_right").
- Display state machine status and parameters in ImGui.
- Add manual trigger buttons ("Force Divide", "Restart / Reset").

- [ ] **Step 2: Syntax and headless execution verification**

Run: `uv run python -c "import examples.cell_division"`
Run: `uv run pytest tests/test_example_script_syntax.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add examples/cell_division.py
git commit -m "feat: complete interactive cell division simulation with Polyscope GUI"
```
