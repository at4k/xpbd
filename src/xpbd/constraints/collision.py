"""Particle-to-particle unilateral collision constraint."""

import numpy as np

_EPS = 1e-12


class ParticleCollisionConstraint:
    """Handles unilateral particle-to-particle collisions.

    If group_ids is provided, collisions are optionally restricted to particles
    belonging to different groups (preventing intra-bag self-collision issues).
    """

    def __init__(
        self,
        n_vertices: int,
        radius: float | np.ndarray = 0.05,
        group_ids: np.ndarray | None = None,
    ) -> None:
        self.n_vertices = int(n_vertices)
        if isinstance(radius, (int, float)):
            self.radii = np.full(n_vertices, float(radius), dtype=np.float64)
        else:
            self.radii = np.asarray(radius, dtype=np.float64)

        if group_ids is not None:
            self.group_ids = np.asarray(group_ids, dtype=np.int64)
        else:
            self.group_ids = None

    def __len__(self) -> int:
        return self.n_vertices

    def reset_lambda(self) -> None:
        pass

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        # Vectorized pairwise distance calculation
        # x shape: (N, 3)
        n = len(x)
        if n <= 1:
            return

        # We can do a simple upper-triangular or full pair search.
        # Since N is typically small (e.g. < 1000) for the demo,
        # vectorized O(N^2) in NumPy is very fast.
        diff = x[:, None, :] - x[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (N, N)

        sum_radii = self.radii[:, None] + self.radii[None, :]  # (N, N)

        # Collision mask: dist < sum_radii, and i < j (to handle each pair once)
        mask = (dist < sum_radii) & (dist > _EPS)
        tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask &= tri_mask

        if self.group_ids is not None:
            # Only collide different groups
            group_mask = self.group_ids[:, None] != self.group_ids[None, :]
            mask &= group_mask

        ii, jj = np.where(mask)
        if len(ii) == 0:
            return

        # Resolve collisions
        d_val = dist[ii, jj]
        overlap = sum_radii[ii, jj] - d_val
        dir_vec = diff[ii, jj] / d_val[:, None]  # from j to i

        w_i = inv_mass[ii]
        w_j = inv_mass[jj]
        w_sum = w_i + w_j

        valid = w_sum > _EPS
        if not np.any(valid):
            return

        ii = ii[valid]
        jj = jj[valid]
        overlap = overlap[valid]
        dir_vec = dir_vec[valid]
        w_i = w_i[valid]
        w_j = w_j[valid]
        w_sum = w_sum[valid]

        # Positional correction
        corr = (overlap / w_sum)[:, None] * dir_vec  # correction vector for i (moves away from j)

        # We can accumulate updates using np.add.at
        delta = np.zeros_like(x)
        count = np.zeros(len(x))

        np.add.at(delta, ii, w_i[:, None] * corr)
        np.add.at(delta, jj, -w_j[:, None] * corr)
        np.add.at(count, ii, 1.0)
        np.add.at(count, jj, 1.0)

        # Average corrections per particle to maintain stability
        active = count > 0
        x[active] += delta[active] / count[active][:, None]

    def project_one(
        self, i: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        # Sequential project_one for Gauss-Seidel mode
        # Projects collision for particle i against all other particles j
        if inv_mass[i] < _EPS:
            return

        diff = x[i] - x
        dist = np.linalg.norm(diff, axis=-1)
        sum_radii = self.radii[i] + self.radii

        mask = (dist < sum_radii) & (dist > _EPS)
        mask[i] = False

        if self.group_ids is not None:
            group_mask = self.group_ids[i] != self.group_ids
            mask &= group_mask

        jj = np.where(mask)[0]
        for j in jj:
            w_i = inv_mass[i]
            w_j = inv_mass[j]
            w_sum = w_i + w_j
            if w_sum < _EPS:
                continue

            d_val = dist[j]
            overlap = sum_radii[j] - d_val
            dir_vec = diff[j] / d_val

            corr = (overlap / w_sum) * dir_vec
            x[i] += w_i * corr
            x[j] -= w_j * corr
