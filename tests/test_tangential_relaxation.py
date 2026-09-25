import numpy as np
import pytest

from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.topology import tangential_surface_relaxation


def test_tangential_surface_relaxation_reduces_variance():
    pos, faces = icosphere_mesh(subdivisions=2, radius=1.0, origin=(0.0, 0.0, 0.0))
    # Distort positions: push vertices on one side closer together
    distorted = pos.copy()
    mask = distorted[:, 0] > 0
    distorted[mask, 1:] *= 0.5
    
    edges = edges_from_faces(faces)
    lens_before = np.linalg.norm(distorted[edges[:, 1]] - distorted[edges[:, 0]], axis=1)
    std_before = np.std(lens_before)

    center = np.mean(distorted, axis=0)
    relaxed = tangential_surface_relaxation(distorted, faces, center=center, factor=0.2, n_iters=10)

    lens_after = np.linalg.norm(relaxed[edges[:, 1]] - relaxed[edges[:, 0]], axis=1)
    std_after = np.std(lens_after)

    def compute_volume(p, f):
        p1 = p[f[:, 0]]
        p2 = p[f[:, 1]]
        p3 = p[f[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0)

    vol_before = compute_volume(distorted, faces)
    vol_after = compute_volume(relaxed, faces)
    vol_err = abs(vol_after - vol_before) / vol_before

    # Edge length variance must decrease
    assert std_after < std_before, f"std_after ({std_after}) should be < std_before ({std_before})"
    # Centroid and overall scale must be preserved
    assert np.allclose(np.mean(relaxed, axis=0), center, atol=0.05)
    # Volume must be strictly preserved within 1%
    assert vol_err < 0.01, f"Volume error is {vol_err * 100:.2f}%, expected < 1%"

