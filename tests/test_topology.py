import numpy as np
import pytest

from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.topology import slice_and_cap_mesh, partition_particles


def test_slice_and_cap_icosphere():
    pos, faces = icosphere_mesh(subdivisions=2, radius=1.0, origin=(0.0, 0.0, 0.0))
    (pos_l, faces_l), (pos_r, faces_r) = slice_and_cap_mesh(pos, faces, cut_x=0.0)

    # 1. Check vertex coordinates
    assert np.all(pos_l[:, 0] <= 1e-12)
    assert np.all(pos_r[:, 0] >= -1e-12)

    # 2. Check manifoldness / Euler characteristic: chi = V - E + F = 2 for closed spheres
    e_l = edges_from_faces(faces_l)
    chi_l = len(pos_l) - len(e_l) + len(faces_l)
    assert chi_l == 2, f"Left mesh Euler characteristic is {chi_l}, expected 2"

    e_r = edges_from_faces(faces_r)
    chi_r = len(pos_r) - len(e_r) + len(faces_r)
    assert chi_r == 2, f"Right mesh Euler characteristic is {chi_r}, expected 2"

    # 3. Check positive volume (outward normals)
    def compute_volume(p, f):
        p1 = p[f[:, 0]]
        p2 = p[f[:, 1]]
        p3 = p[f[:, 2]]
        return np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0

    vol_l = compute_volume(pos_l, faces_l)
    vol_r = compute_volume(pos_r, faces_r)
    assert vol_l > 0.0, f"Left volume {vol_l} must be positive"
    assert vol_r > 0.0, f"Right volume {vol_r} must be positive"
    total_vol = compute_volume(pos, faces)
    assert np.isclose(vol_l + vol_r, total_vol, rtol=0.1)


def test_partition_particles():
    pts = np.array([
        [-0.5, 0.0, 0.0],
        [-0.1, 0.2, 0.1],
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.0],
        [0.5, 0.3, -0.1],
    ])
    left_idx, right_idx = partition_particles(pts, cut_x=0.0)
    assert set(left_idx.tolist() + right_idx.tolist()) == set(range(len(pts)))
    assert len(set(left_idx).intersection(set(right_idx))) == 0
    assert np.all(pts[left_idx, 0] <= 0.0)
    assert np.all(pts[right_idx, 0] >= 0.0)
