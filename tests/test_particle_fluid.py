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
