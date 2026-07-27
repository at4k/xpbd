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

        # 1D Sweep and Prune Broadphase filtering to avoid N^2 distance computations
        order = np.argsort(x[:, 0])
        x_s = x[order]
        r_s = self.radii[order]
        idx_s = np.arange(n)[order]

        # Find candidate pairs (i < j) whose X-interval overlaps
        pairs_i = []
        pairs_j = []

        for i in range(n - 1):
            x_i = x_s[i, 0]
            r_i = r_s[i]
            # Search right bound
            limit = x_i + r_i + np.max(r_s[i + 1:])
            j_end = np.searchsorted(x_s[:, 0], limit, side="right")
            if j_end > i + 1:
                js = np.arange(i + 1, j_end)
                # Quick Y and Z bounding box check
                dx = x_s[js, 0] - x_i
                dy = np.abs(x_s[js, 1] - x_s[i, 1])
                dz = np.abs(x_s[js, 2] - x_s[i, 2])
                sum_r = r_i + r_s[js]

                valid = (dx < sum_r) & (dy < sum_r) & (dz < sum_r)
                if np.any(valid):
                    valid_js = js[valid]
                    pairs_i.append(np.full(len(valid_js), i, dtype=np.int64))
                    pairs_j.append(valid_js)

        if not pairs_i:
            return

        ii_s = np.concatenate(pairs_i)
        jj_s = np.concatenate(pairs_j)

        # Map sorted indices back to original indices
        ii = idx_s[ii_s]
        jj = idx_s[jj_s]

        # Enforce canonical ordering (ii < jj)
        swap = ii > jj
        ii[swap], jj[swap] = jj[swap], ii[swap]

        # Neighbor mesh filtering (exclude 1-ring neighbors)
        mask = ~self.adj[ii, jj]

        if not self.self_collision and self.group_ids is not None:
            mask &= (self.group_ids[ii] != self.group_ids[jj])

        ii = ii[mask]
        jj = jj[mask]

        if len(ii) == 0:
            return

        # Precise 3D distance check & resolution
        diff = x[ii] - x[jj]  # vector from j to i
        dist_sq = np.sum(diff**2, axis=-1)
        sum_radii = self.radii[ii] + self.radii[jj]

        colliding = (dist_sq < sum_radii**2) & (dist_sq > _EPS)
        if not np.any(colliding):
            return

        ii = ii[colliding]
        jj = jj[colliding]
        diff = diff[colliding]
        d_val = np.sqrt(dist_sq[colliding])
        sum_r = sum_radii[colliding]

        overlap = sum_r - d_val
        dir_vec = diff / d_val[:, None]

        w_i = inv_mass[ii]
        w_j = inv_mass[jj]
        w_sum = w_i + w_j

        valid_w = w_sum > _EPS
        if not np.any(valid_w):
            return

        ii = ii[valid_w]
        jj = jj[valid_w]
        overlap = overlap[valid_w]
        dir_vec = dir_vec[valid_w]
        w_i = w_i[valid_w]
        w_j = w_j[valid_w]
        w_sum = w_sum[valid_w]

        # Positional correction
        corr = (overlap / w_sum)[:, None] * dir_vec

        delta = np.zeros_like(x)
        count = np.zeros(len(x))

        np.add.at(delta, ii, w_i[:, None] * corr)
        np.add.at(delta, jj, -w_j[:, None] * corr)
        np.add.at(count, ii, 1.0)
        np.add.at(count, jj, 1.0)

        active = count > 0
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
