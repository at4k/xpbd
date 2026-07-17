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
