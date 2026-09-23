"""Spindle constraint: manages pole elongation and astral microtubule tethers."""

import numpy as np

_EPS = 1e-12


class SpindleConstraint:
    """Manages spindle elongation between two poles and astral microtubule tethers.

    The central pole-to-pole constraint pushes the two centrosomes apart along the
    elongation axis, while astral tethers pull the surrounding cell cortex (membrane)
    towards each pole to maintain rounded lobes in each daughter hemisphere.
    """

    def __init__(
        self,
        pole_0: int,
        pole_1: int,
        left_membrane_idx: np.ndarray | list[int] | None = None,
        right_membrane_idx: np.ndarray | list[int] | None = None,
        pole_rest_length: float = 0.4,
        pole_compliance: float = 0.0,
        tether_rest_length: float = 0.3,
        tether_compliance: float = 1e-3,
        relaxation: float = 1.0,
    ) -> None:
        self.pole_0 = int(pole_0)
        self.pole_1 = int(pole_1)
        self.pole_rest_length = float(pole_rest_length)
        self.pole_compliance = float(pole_compliance)

        if left_membrane_idx is not None:
            self.left_membrane = np.asarray(left_membrane_idx, dtype=np.int64)
        else:
            self.left_membrane = np.zeros(0, dtype=np.int64)

        if right_membrane_idx is not None:
            self.right_membrane = np.asarray(right_membrane_idx, dtype=np.int64)
        else:
            self.right_membrane = np.zeros(0, dtype=np.int64)

        self.tether_rest_length = float(tether_rest_length)
        self.tether_compliance = float(tether_compliance)
        self.relaxation = float(relaxation)

        self._build_edges()

    def _build_edges(self) -> None:
        edges = [[self.pole_0, self.pole_1]]
        for idx in self.left_membrane:
            edges.append([self.pole_0, int(idx)])
        for idx in self.right_membrane:
            edges.append([self.pole_1, int(idx)])
        self.edges = np.asarray(edges, dtype=np.int64)
        self.lam = np.zeros(len(self.edges), dtype=np.float64)

    def set_target_length(self, length: float) -> None:
        """Update the spindle pole-to-pole target rest length."""
        self.pole_rest_length = float(length)

    def set_tether_rest_length(self, length: float) -> None:
        """Update the astral tether target rest length."""
        self.tether_rest_length = float(length)

    def __len__(self) -> int:
        return len(self.edges)

    def reset_lambda(self) -> None:
        self.lam[:] = 0.0

    def _get_rest_and_compliance(self) -> tuple[np.ndarray, np.ndarray]:
        n_edges = len(self.edges)
        rests = np.full(n_edges, self.tether_rest_length, dtype=np.float64)
        compliances = np.full(n_edges, self.tether_compliance, dtype=np.float64)
        rests[0] = self.pole_rest_length
        compliances[0] = self.pole_compliance
        return rests, compliances

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        if len(self.edges) == 0:
            return

        rests, compliances = self._get_rest_and_compliance()
        i = self.edges[:, 0]
        j = self.edges[:, 1]

        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1)
        n = d / np.maximum(dist, _EPS)[:, None]
        c = dist - rests

        alpha = compliances / (h**2)
        w = inv_mass[i] + inv_mass[j]
        denom = w + alpha
        valid = denom > _EPS

        dlam = np.zeros(len(self.edges))
        dlam[valid] = -(c[valid] + alpha[valid] * self.lam[valid]) / denom[valid]
        self.lam += dlam

        corr = dlam[:, None] * n
        delta = np.zeros_like(x)
        count = np.zeros(len(x))

        np.add.at(delta, j, inv_mass[j][:, None] * corr)
        np.add.at(delta, i, -inv_mass[i][:, None] * corr)
        np.add.at(count, i, 1.0)
        np.add.at(count, j, 1.0)

        # For the poles, we don't want the stiff pole constraint diluted by many tethers
        active = count > 0
        x[active] += self.relaxation * delta[active] / np.sqrt(count[active])[:, None]

    def project_one(
        self, k: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        i, j = self.edges[k]
        rest = self.pole_rest_length if k == 0 else self.tether_rest_length
        comp = self.pole_compliance if k == 0 else self.tether_compliance

        d = x[j] - x[i]
        dist = float(np.linalg.norm(d))
        if dist < _EPS:
            return

        n = d / dist
        c = dist - rest
        alpha = comp / (h**2)
        w = inv_mass[i] + inv_mass[j]
        denom = w + alpha
        if denom < _EPS:
            return

        dlam = -(c + alpha * self.lam[k]) / denom
        self.lam[k] += dlam

        x[j] += inv_mass[j] * dlam * n
        x[i] -= inv_mass[i] * dlam * n
