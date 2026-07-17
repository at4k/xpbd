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
