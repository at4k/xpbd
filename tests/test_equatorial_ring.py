import numpy as np
import pytest
from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.topology import insert_equatorial_ring


def test_insert_equatorial_ring_properties():
    verts, faces = icosphere_mesh(subdivisions=2, radius=1.0)
    new_verts, new_faces, ring_indices = insert_equatorial_ring(verts, faces, plane_x=0.0)

    # 1. Ring vertices must lie strictly on plane x = 0
    assert len(ring_indices) >= 6
    np.testing.assert_allclose(new_verts[ring_indices, 0], 0.0, atol=1e-12)

    # 2. Ring vertices must be approximately circular
    radii = np.linalg.norm(new_verts[ring_indices, 1:], axis=1)
    np.testing.assert_allclose(radii, 1.0, rtol=0.05)

    # 3. New mesh must be watertight and manifold: Euler characteristic V - E + F = 2
    edges = edges_from_faces(new_faces)
    euler = len(new_verts) - len(edges) + len(new_faces)
    assert euler == 2

    # 4. Volume must be preserved
    p1 = new_verts[new_faces[:, 0]]
    p2 = new_verts[new_faces[:, 1]]
    p3 = new_verts[new_faces[:, 2]]
    vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0
    assert vol > 0.0
    expected_vol = 4.0 / 3.0 * np.pi * (1.0 ** 3)
    np.testing.assert_allclose(vol, expected_vol, rtol=0.08)
