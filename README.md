# xpbd

A pure XPBD (Extended Position Based Dynamics) engine in Python, built as a
small, hackable foundation for physics research and experimentation.

- Substep-based XPBD solver (Macklin et al. 2016 + "Small Steps" 2019)
- Soft bodies from tetrahedral meshes or closed triangle surfaces ("bags")
- Constraint batches: distance, tetrahedron volume, global (bag) volume,
  ground, box container, particle collision, plus cell-division specific
  constraints (spindle, contractile ring, cytoplasm fluid particles)
- Two solver modes: vectorized `jacobi` (fast, averaged simultaneous
  projection) and sequential `gauss_seidel` (paper-faithful reference)
- Core depends on numpy only; visualization lives in `examples/`

## Setup

```bash
uv sync --extra viz
```

## Run the demos

All demos use polyscope and accept `--mode jacobi|gauss_seidel`,
`--substeps N`, `--transparency` and `--point-radius`.

### Soft body drop

```bash
uv run --extra viz python examples/softbody_drop.py
```

A tetrahedral grid falls onto the ground. Useful flags:
`--edge-compliance 1e-3`, `--volume-compliance`, `--nx 8 --ny 8 --nz 8`.

### Colliding bags

```bash
uv run --extra viz python examples/bags_collision.py
```

Several inflated bags (closed surfaces with internal fluid particles) collide
inside a box. Useful flags: `--num-bags`, `--mesh-type
icosphere|tet_sphere|cube_sphere`, `--internal-particles`,
`--inflation-factor`, `--no-self-collision`.

Mouse: drag normally to move the camera; hold Ctrl (Cmd on macOS) and drag to
grab and pull vertices. The GUI panel has Reset, Play/Pause and display
toggles.

### Cell division

```bash
uv run --extra viz python examples/cell_division.py
```

Autonomous cytokinesis: the cell elongates along the spindle, the contractile
ring constricts the equatorial furrow, the membrane splits, and two daughter
cells relax into independent spheres
(`ELONGATING -> CONSTRICTING -> DIVIDING -> RELAXING -> INDEPENDENT`).
Useful flags: `--internal-particles`, `--bag-radius`, `--subdivisions`,
`--headless --headless-steps 300` (run without a window).

## Run tests

```bash
uv run pytest --cov=xpbd
```

## Layout

```
src/xpbd/
  body.py          # SoftBody state container (tet mesh / surface mesh)
  solver.py        # XPBD substep loop
  mesh.py          # tet grid, sphere / icosphere meshes, edges, boundary surface
  topology.py      # mesh slicing / capping, ring insertion, surface relaxation
  constraints/     # one class per constraint kind
    distance.py, volume.py, global_volume.py, ground.py, box.py,
    collision.py, spindle.py, contractile_ring.py, particle_repulsion.py
examples/
  softbody_drop.py   # tet soft body falling on the ground
  bags_collision.py  # colliding bags with mouse grab
  cell_division.py   # autonomous cytokinesis
docs/
  BACKLOG.md         # deferred improvements
  superpowers/       # design specs and implementation plans
```

## Adding a new constraint

Implement the `Constraint` protocol (`reset_lambda`, `project`,
`project_one`, `__len__`) in `src/xpbd/constraints/` and pass an instance
to `Solver`. See `distance.py` for the canonical XPBD update.

Note: `ParticleFluidConstraint` (`particle_repulsion.py`) does not implement
this protocol and is not passed to `Solver`. It operates on a separate
fluid-particle array and is called directly from the simulation loop via
`project_repulsion` and `project_confinement_mesh` (see
`examples/cell_division.py`).
