"""3D Box Container Constraint for confining particles."""

import numpy as np


class BoxConstraint:
    """Confines all particles within a specified 3D box."""

    def __init__(
        self,
        n_vertices: int,
        bounds_min: tuple[float, float, float] = (-1.0, 0.0, -1.0),
        bounds_max: tuple[float, float, float] = (1.0, 2.0, 1.0),
    ) -> None:
        self.n_vertices = int(n_vertices)
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)

    def __len__(self) -> int:
        return self.n_vertices

    def reset_lambda(self) -> None:
        pass

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        movable = inv_mass > 0.0
        # Confine each axis
        x[movable] = np.clip(x[movable], self.bounds_min, self.bounds_max)

    def project_one(
        self, i: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        if inv_mass[i] > 0.0:
            x[i] = np.clip(x[i], self.bounds_min, self.bounds_max)
