import numpy as np
from xpbd.constraints import ParticleCollisionConstraint


def test_particle_collision_broadphase_detection():
    # Place 3 particles where (0, 1) overlap and (0, 2) do not
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    inv_mass = np.ones(3, dtype=np.float64)
    radius = 0.05
    con = ParticleCollisionConstraint(n_vertices=3, radius=radius)

    x_before = pos.copy()
    con.project(pos, inv_mass, h=0.01)

    # Particle 0 and 1 should be pushed apart along X axis
    assert pos[0, 0] < x_before[0, 0]
    assert pos[1, 0] > x_before[1, 0]
    # Particle 2 should remain untouched
    np.testing.assert_allclose(pos[2], x_before[2])


def test_particle_collision_no_nan():
    np.random.seed(42)
    pos = np.random.uniform(-0.5, 0.5, size=(100, 3))
    inv_mass = np.ones(100, dtype=np.float64)
    con = ParticleCollisionConstraint(n_vertices=100, radius=0.08)

    con.project(pos, inv_mass, h=0.01)

    # Verify no NaN values in positions
    assert not np.any(np.isnan(pos))
    assert not np.any(np.isinf(pos))
