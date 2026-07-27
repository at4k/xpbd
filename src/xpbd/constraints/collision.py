"""Particle-to-particle unilateral collision constraint with self-collision neighbor filtering."""

import numpy as np

_EPS = 1e-12


class ParticleCollisionConstraint:
    """Handles unilateral particle-to-particle collisions.

    Allows self-collision for flexible structures while excluding directly
    connected mesh neighbors (1-ring neighbors) to prevent constraint jittering.
    """

    def __init__(
        self,
        n_vertices: int,
        radius: float | np.ndarray = 0.05,
        edges: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
        self_collision: bool = True,
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

        self.self_collision = self_collision

        # Setup adjacency matrix for neighbor filtering
        self.adj = np.zeros((self.n_vertices, self.n_vertices), dtype=bool)
        if edges is not None:
            edges = np.asarray(edges, dtype=np.int64)
            # Clip edges to make sure they are within valid vertex index range
            valid = (edges[:, 0] < self.n_vertices) & (edges[:, 1] < self.n_vertices)
            valid_edges = edges[valid]
            self.adj[valid_edges[:, 0], valid_edges[:, 1]] = True
            self.adj[valid_edges[:, 1], valid_edges[:, 0]] = True

    def __len__(self) -> int:
        return self.n_vertices

    def reset_lambda(self) -> None:
        pass

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        n = len(x)
        if n <= 1:
            return

        diff = x[:, None, :] - x[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (N, N)

        sum_radii = self.radii[:, None] + self.radii[None, :]  # (N, N)

        # Collision mask: dist < sum_radii, and i < j (to handle each pair once)
        mask = (dist < sum_radii) & (dist > _EPS)
        tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask &= tri_mask

        # Exclude direct edge-connected neighbors
        mask &= ~self.adj

        if not self.self_collision and self.group_ids is not None:
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
        corr = (overlap / w_sum)[:, None] * dir_vec

        delta = np.zeros_like(x)
        count = np.zeros(len(x))

        np.add.at(delta, ii, w_i[:, None] * corr)
        np.add.at(delta, jj, -w_j[:, None] * corr)
        np.add.at(count, ii, 1.0)
        np.add.at(count, jj, 1.0)

        # Average corrections per particle to maintain stability
        active = count > 0
        # In highly packed systems, dividing fully by count can over-damp collision response.
        # Dividing by np.sqrt(count) provides a much stronger and more robust collision
        # barrier while maintaining perfect stability under XPBD.
        x[active] += delta[active] / np.sqrt(count[active])[:, None]

    def project_one(
        self, i: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        if inv_mass[i] < _EPS:
            return

        # Pre-filter candidate neighbors using current positions
        diff = x[i] - x
        dist = np.linalg.norm(diff, axis=-1)
        sum_radii = self.radii[i] + self.radii

        mask = (dist < sum_radii) & (dist > _EPS)
        mask[i] = False

        # Ensure we only process each pair once by enforcing j > i
        mask &= (np.arange(self.n_vertices) > i)

        # Exclude direct edge-connected neighbors of vertex i
        mask &= ~self.adj[i]

        if not self.self_collision and self.group_ids is not None:
            group_mask = self.group_ids[i] != self.group_ids
            mask &= group_mask

        jj = np.where(mask)[0]
        for j in jj:
            w_i = inv_mass[i]
            w_j = inv_mass[j]
            w_sum = w_i + w_j
            if w_sum < _EPS:
                continue

            # Dynamically recompute current difference and distance to be mathematically correct in Gauss-Seidel
            cur_diff = x[i] - x[j]
            cur_dist = float(np.linalg.norm(cur_diff))
            cur_sum_radii = self.radii[i] + self.radii[j]

            if cur_dist >= cur_sum_radii or cur_dist < _EPS:
                continue

            overlap = cur_sum_radii - cur_dist
            dir_vec = cur_diff / cur_dist

            corr = (overlap / w_sum) * dir_vec
            x[i] += w_i * corr
            x[j] -= w_j * corr
