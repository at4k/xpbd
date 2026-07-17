"""Soft body state container."""

from dataclasses import dataclass

import numpy as np

from .mesh import edges_from_tets, tet_volumes


@dataclass
class SoftBody:
    """Particle state and topology of a tetrahedral soft body.

    inv_mass entries of 0.0 mark pinned (immovable) vertices.
    """

    positions: np.ndarray  # (N, 3) float64
    velocities: np.ndarray  # (N, 3) float64
    inv_mass: np.ndarray  # (N,) float64
    tets: np.ndarray  # (T, 4) int64
    edges: np.ndarray  # (E, 2) int64

    @classmethod
    def from_tet_mesh(
        cls,
        positions: np.ndarray,
        tets: np.ndarray,
        density: float = 1000.0,
    ) -> "SoftBody":
        """Build a body from a tet mesh, lumping tet mass onto vertices."""
        positions = np.asarray(positions, dtype=np.float64).copy()
        tets = np.asarray(tets, dtype=np.int64)
        vols = tet_volumes(positions, tets)
        mass = np.zeros(len(positions))
        np.add.at(mass, tets.ravel(), np.repeat(density * vols / 4.0, 4))
        inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-300), 0.0)
        return cls(
            positions=positions,
            velocities=np.zeros_like(positions),
            inv_mass=inv_mass,
            tets=tets,
            edges=edges_from_tets(tets),
        )
