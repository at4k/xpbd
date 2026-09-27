import numpy as np
import pytest
from xpbd.constraints.particle_repulsion import ParticleFluidConstraint
from xpbd.mesh import icosphere_mesh


def test_particle_repulsion_spreads_cluster():
    # 10 particles initialized very close together
    np.random.seed(42)
    pts = np.random.uniform(-0.01, 0.01, size=(10, 3))
    r = 0.05
    c = ParticleFluidConstraint(particle_radius=r)

    # Apply repulsion iterations
    for _ in range(25):
        c.project_repulsion(pts)

    # Verify no two particles are severely overlapping (at least 85% of 2*r)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert np.min(dist) >= 2 * r * 0.85


def test_particle_confinement_sphere_mesh():
    # Sphere of radius 1.0
    verts, faces = icosphere_mesh(subdivisions=2, radius=1.0)
    c = ParticleFluidConstraint(particle_radius=0.05)

    # 10 particles, some inside, some outside (e.g. at radius 1.2)
    pts = np.array([
        [1.2, 0.0, 0.0],
        [0.0, 1.3, 0.0],
        [0.0, 0.0, 1.1],
        [-1.15, 0.0, 0.0],
        [0.2, 0.3, 0.1],
    ])

    c.project_confinement_mesh(pts, verts, margin=0.05)

    # All particles must now be inside radius 1.0 - margin = 0.95
    radii = np.linalg.norm(pts, axis=1)
    assert np.all(radii <= 1.0)


def test_particle_confinement_with_face_normals():
    # Sphere displaced to origin (0, 1.5, 0)
    origin = (0.0, 1.5, 0.0)
    verts, faces = icosphere_mesh(subdivisions=2, radius=0.5, origin=origin)
    c = ParticleFluidConstraint(particle_radius=0.04)

    # Particle outside the top pole at y = 2.2 (radius 0.7 > 0.5)
    pts = np.array([[0.0, 2.2, 0.0]])
    c.project_confinement_mesh(pts, verts, faces=faces, margin=0.04)

    # Must be pushed inside along true normal (downward in Y)
    dist = np.linalg.norm(pts[0] - origin)
    assert dist <= 0.50


def test_particle_confinement_with_array_centers():
    # Two spheres/lobes with centers at (-1, 0, 0) and (+1, 0, 0)
    verts_l, _ = icosphere_mesh(subdivisions=1, radius=0.5, origin=(-1.0, 0.0, 0.0))
    verts_r, _ = icosphere_mesh(subdivisions=1, radius=0.5, origin=(1.0, 0.0, 0.0))
    all_verts = np.concatenate([verts_l, verts_r], axis=0)

    pts = np.array([
        [-1.7, 0.0, 0.0],  # outside left lobe (dist from -1 is 0.7 > 0.5)
        [1.7, 0.0, 0.0],   # outside right lobe (dist from +1 is 0.7 > 0.5)
    ])
    centers = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    c = ParticleFluidConstraint(particle_radius=0.04)
    c.project_confinement_mesh(pts, all_verts, margin=0.04, center=centers)

    assert np.linalg.norm(pts[0] - centers[0]) <= 0.50
    assert np.linalg.norm(pts[1] - centers[1]) <= 0.50
