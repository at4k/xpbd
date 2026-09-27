"""Contractile ring constraint: constricts equatorial vertices toward the central axis."""

import numpy as np

_EPS = 1e-12


class ContractileRingConstraint:
    """XPBD constraint that constricts equatorial membrane vertices towards a longitudinal axis.

    Models the actomyosin contractile ring that drives cytokinesis (cleavage furrowing).
    Constraint function: C(x_i) = sqrt((y_i - center_y)^2 + (z_i - center_z)^2) - target_radius.
    """

    def __init__(
        self,
        ring_indices: np.ndarray | list[int],
        center_y: float = 0.0,
        center_z: float = 0.0,
        center_x: float | None = None,
        target_radius: float = 0.5,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> None:
        self.ring_indices = np.asarray(ring_indices, dtype=np.int64)
        self.center_y = float(center_y)
        self.center_z = float(center_z)
        self.center_x = float(center_x) if center_x is not None else None
        self.target_radius = float(target_radius)
        self.compliance = float(compliance)
        self.relaxation = float(relaxation)
        self.lam = np.zeros(len(self.ring_indices), dtype=np.float64)

    def set_target_radius(self, radius: float) -> None:
        """Update the target radius of the contractile ring."""
        self.target_radius = max(0.0, float(radius))

    def __len__(self) -> int:
        return len(self.ring_indices)

    def reset_lambda(self) -> None:
        self.lam[:] = 0.0

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        if len(self.ring_indices) == 0:
            return

        idx = self.ring_indices
        w = inv_mass[idx]
        movable = w > _EPS
        if not np.any(movable):
            return

        idx_m = idx[movable]
        w_m = w[movable]
        lam_m = self.lam[movable]

        dy = x[idx_m, 1] - self.center_y
        dz = x[idx_m, 2] - self.center_z
        r = np.sqrt(dy**2 + dz**2)

        valid_r = r > _EPS
        if not np.any(valid_r):
            return

        idx_m = idx_m[valid_r]
        w_m = w_m[valid_r]
        lam_m = lam_m[valid_r]
        r = r[valid_r]
        dy = dy[valid_r]
        dz = dz[valid_r]

        gy = dy / r
        gz = dz / r
        c = r - self.target_radius

        # Unilateral constraint: only pull inward if radius exceeds target
        active = c > 0
        if not np.any(active):
            return

        idx_m = idx_m[active]
        w_m = w_m[active]
        lam_m = lam_m[active]
        c = c[active]
        gy = gy[active]
        gz = gz[active]

        alpha = self.compliance / (h**2)
        denom = w_m + alpha
        dlam = -(c + alpha * lam_m) / np.maximum(denom, _EPS)
        self.lam[np.where(movable)[0][valid_r][active]] += dlam

        corr = dlam * self.relaxation
        x[idx_m, 1] += w_m * corr * gy
        x[idx_m, 2] += w_m * corr * gz

        if self.center_x is not None:
            dx = x[idx, 0] - self.center_x
            x[idx, 0] -= dx * self.relaxation

    def project_one(
        self, k: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        idx = self.ring_indices[k]
        w = inv_mass[idx]
        if w < _EPS:
            return

        dy = x[idx, 1] - self.center_y
        dz = x[idx, 2] - self.center_z
        r = float(np.sqrt(dy**2 + dz**2))
        if r < _EPS:
            return

        gy = dy / r
        gz = dz / r
        c = r - self.target_radius
        if c <= 0:
            if self.center_x is not None:
                dx = x[idx, 0] - self.center_x
                x[idx, 0] -= dx * self.relaxation
            return

        alpha = self.compliance / (h**2)
        denom = w + alpha
        if denom < _EPS:
            return

        dlam = -(c + alpha * self.lam[k]) / denom
        self.lam[k] += dlam

        corr = dlam * self.relaxation
        x[idx, 1] += w * corr * gy
        x[idx, 2] += w * corr * gz

        if self.center_x is not None:
            dx = x[idx, 0] - self.center_x
            x[idx, 0] -= dx * self.relaxation
