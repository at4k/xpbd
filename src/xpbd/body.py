"""Soft body state container."""

from dataclasses import dataclass

import numpy as np

from .mesh import edges_from_faces, edges_from_tets, tet_volumes


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

    @classmethod
    def from_surface_mesh(
        cls,
        positions: np.ndarray,
        faces: np.ndarray,
        density: float = 1000.0,
        thickness: float = 0.01,
    ) -> "SoftBody":
        """Build a body from a closed triangle surface mesh, lumping mass onto vertices."""
        positions = np.asarray(positions, dtype=np.float64).copy()
        faces = np.asarray(faces, dtype=np.int64)

        # Calculate triangle areas
        p1 = positions[faces[:, 0]]
        p2 = positions[faces[:, 1]]
        p3 = positions[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)

        mass = np.zeros(len(positions))
        # Lump mass: each triangle area * thickness * density / 3
        tri_mass = density * areas * thickness / 3.0
        np.add.at(mass, faces.ravel(), np.repeat(tri_mass, 3))

        inv_mass = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1e-300), 0.0)

        return cls(
            positions=positions,
            velocities=np.zeros_like(positions),
            inv_mass=inv_mass,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=edges_from_faces(faces),
        )
