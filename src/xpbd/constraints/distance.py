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
