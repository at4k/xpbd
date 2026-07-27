import numpy as np
import pytest

from xpbd import SoftBody
from xpbd.constraints import (
    BoxConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
)
from xpbd.mesh import edges_from_faces, sphere_mesh


def test_sphere_mesh_generation():
    positions, faces = sphere_mesh(subdivisions=2, radius=1.0)
    assert len(positions) > 0
    assert len(faces) > 0
    # Sphere radius should be approx 1.0
    norms = np.linalg.norm(positions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)

    # Test edges_from_faces
    edges = edges_from_faces(faces)
    assert len(edges) > 0


def test_global_volume_constraint_recovery():
    positions, faces = sphere_mesh(subdivisions=2, radius=1.0)
    # Scale sphere to squash it
    x_squashed = positions.copy()
    x_squashed[:, 2] *= 0.5  # Squash along Z axis

    con = GlobalVolumeConstraint.from_positions(positions, faces)
    rest_vol = con.rest_volumes[0]

    # Run projection iterations
    inv_mass = np.ones(len(positions))
    h = 0.01
    for _ in range(50):
        con.reset_lambda()
        con.project(x_squashed, inv_mass, h)

    # Calculate final volume
    p1 = x_squashed[faces[:, 0]]
    p2 = x_squashed[faces[:, 1]]
    p3 = x_squashed[faces[:, 2]]
    final_vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0

    assert final_vol == pytest.approx(rest_vol, rel=1e-3)


def test_particle_collision():
    # Two particles with radius 0.1 at (0, 0, 0) and (0.1, 0, 0)
    # Overlap is 0.1 because total radii is 0.2
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    inv_mass = np.ones(2)
    con = ParticleCollisionConstraint(n_vertices=2, radius=0.1)

    con.project(x, inv_mass, h=0.01)

    # They should be pushed apart so distance is 0.2
    dist = np.linalg.norm(x[1] - x[0])
    assert dist == pytest.approx(0.2, abs=1e-5)


def test_particle_collision_group_ids():
    # Overlapping particles, but with the same group_id (no collision expected when self_collision is False)
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    inv_mass = np.ones(2)
    con = ParticleCollisionConstraint(
        n_vertices=2, radius=0.1, group_ids=np.array([1, 1]), self_collision=False
    )

    con.project(x, inv_mass, h=0.01)

    # They should NOT move because they belong to the same group
    dist = np.linalg.norm(x[1] - x[0])
    assert dist == pytest.approx(0.1, abs=1e-5)


def test_box_constraint():
    x = np.array([[2.0, -1.0, 3.5], [0.0, 0.5, 0.0]], dtype=np.float64)
    inv_mass = np.ones(2)
    con = BoxConstraint(
        n_vertices=2, bounds_min=(-1.0, 0.0, -1.0), bounds_max=(1.0, 1.0, 1.0)
    )

    con.project(x, inv_mass, h=0.01)

    # Check bounds containment
    assert np.all(x >= np.array([-1.0, 0.0, -1.0]))
    assert np.all(x <= np.array([1.0, 1.0, 1.0]))


def test_soft_body_from_surface_mesh():
    positions, faces = sphere_mesh(subdivisions=2, radius=1.0)
    body = SoftBody.from_surface_mesh(positions, faces)

    assert len(body.positions) == len(positions)
    assert len(body.edges) > 0
    assert len(body.tets) == 0
    assert np.all(body.inv_mass > 0.0)
