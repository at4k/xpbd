import numpy as np
from xpbd import SoftBody, Solver
from xpbd.mesh import icosphere_mesh


def test_apply_push_up_impulse():
    pos, faces = icosphere_mesh(subdivisions=2, radius=0.5, origin=(0.0, 1.0, 0.0))
    body = SoftBody.from_surface_mesh(pos, faces)

    # Apply upward impulse to bottom vertices (y < 1.0)
    bottom_mask = body.positions[:, 1] < 1.0
    body.velocities[bottom_mask, 1] += 5.0

    # Verify upward velocity was applied
    assert np.all(body.velocities[bottom_mask, 1] == 5.0)


def test_apply_squash_impulse():
    pos, faces = icosphere_mesh(subdivisions=2, radius=0.5, origin=(0.0, 1.0, 0.0))
    body = SoftBody.from_surface_mesh(pos, faces)

    # Apply downward impulse to top vertices (y > 1.0)
    top_mask = body.positions[:, 1] > 1.0
    body.velocities[top_mask, 1] -= 5.0

    # Verify downward velocity was applied
    assert np.all(body.velocities[top_mask, 1] == -5.0)
