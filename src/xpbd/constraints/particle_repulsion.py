"""Particle fluid repulsion and membrane boundary confinement constraint."""

import numpy as np


class ParticleFluidConstraint:
    """Pairwise repulsion and membrane confinement for cytoplasm fluid particles."""

    def __init__(
        self,
        particle_radius: float = 0.05,
        repulsion_stiffness: float = 0.5,
    ) -> None:
        self.particle_radius = float(particle_radius)
        self.repulsion_stiffness = float(repulsion_stiffness)

    def project_repulsion(
        self, particles: np.ndarray, relaxation: float = 0.5
    ) -> None:
        """Projects pairwise repulsive distance constraint so ||p_i - p_j|| >= 2*r."""
        n = len(particles)
        if n < 2:
            return

        diff = particles[:, None, :] - particles[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (N, N)
        np.fill_diagonal(dist, np.inf)

        min_dist = 2.0 * self.particle_radius
        overlap = dist < min_dist
        if not np.any(overlap):
            return

        force = np.zeros_like(particles)
        for i in range(n):
            cl = overlap[i]
            if np.any(cl):
                d = dist[i, cl, None]
                # Direction from neighbor to i
                dir_vec = diff[i, cl] / np.maximum(d, 1e-9)
                penetration = min_dist - d
                f = dir_vec * penetration
                force[i] = np.sum(f, axis=0) * 0.5

        particles += force * relaxation

    def project_confinement_mesh(
        self,
        particles: np.ndarray,
        membrane_positions: np.ndarray,
        margin: float = 0.05,
    ) -> None:
        """Projects particles to stay within the inner boundary of the membrane mesh."""
        if len(particles) == 0 or len(membrane_positions) == 0:
            return

        diff_mesh = particles[:, None, :] - membrane_positions[None, :, :]  # (N, M, 3)
        dist_sq = np.sum(diff_mesh**2, axis=-1)
        closest_idx = np.argmin(dist_sq, axis=1)

        closest_v = membrane_positions[closest_idx]
        norm_v = np.linalg.norm(closest_v, axis=1, keepdims=True)
        out_dir = closest_v / np.maximum(norm_v, 1e-12)

        rel = particles - closest_v
        dot_out = np.sum(rel * out_dir, axis=1)

        over = dot_out > -margin
        if np.any(over):
            correction = (dot_out[over] + margin)[:, None] * out_dir[over]
            particles[over] -= correction
