# xpbd

A pure XPBD (Extended Position Based Dynamics) engine in Python, built as a
small, hackable foundation for physics research and experimentation.

- Substep-based XPBD solver (Macklin et al. 2016 + "Small Steps" 2019)
- Constraint batches: distance, tetrahedron volume, ground collision
- Two solver modes: vectorized `jacobi` (fast, averaged simultaneous
  projection) and sequential `gauss_seidel` (paper-faithful reference)
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
