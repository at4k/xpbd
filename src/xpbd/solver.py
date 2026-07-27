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

    def reset(self) -> None:
        """Reset internal constraint states (e.g. Lagrange multipliers)."""
        for con in self.constraints:
            con.reset_lambda()

