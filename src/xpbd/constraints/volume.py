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
        # Averaged Jacobi: accumulate corrections and divide by the number
        # of constraints touching each vertex so simultaneous updates stay
        # stable regardless of vertex valence.
        delta = np.zeros_like(x)
        count = np.zeros(len(x))
        for col, g in zip(range(4), (g0, g1, g2, g3)):
            idx = t[:, col]
            np.add.at(delta, idx, (inv_mass[idx] * dlam)[:, None] * g)
            np.add.at(count, idx, 1.0)
        x += self.relaxation * delta / np.maximum(count, 1.0)[:, None]

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
