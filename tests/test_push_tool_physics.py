import numpy as np


def test_push_sphere_constraint_repulsion():
    # Test pushing vertices out of a sphere radius
    positions = np.array([
        [0.0, 0.0, 0.0],   # Inside push sphere
        [0.05, 0.0, 0.0],  # Inside push sphere
        [0.5, 0.0, 0.0],   # Outside push sphere
    ], dtype=np.float64)

    push_center = np.array([0.0, 0.0, 0.0])
    push_radius = 0.2

    # Apply repulsion pushing outwards from center
    vecs = positions - push_center
    dists = np.linalg.norm(vecs, axis=1)
    inside = dists < push_radius

    for idx in np.where(inside)[0]:
        d = dists[idx]
        if d < 1e-6:
            # Shift along X if directly at center
            dir_vec = np.array([1.0, 0.0, 0.0])
        else:
            dir_vec = vecs[idx] / d
        positions[idx] = push_center + dir_vec * push_radius

    # Vertices 0 and 1 should now be at distance >= push_radius from center
    new_dists = np.linalg.norm(positions - push_center, axis=1)
    assert new_dists[0] >= push_radius - 1e-6
    assert new_dists[1] >= push_radius - 1e-6
    # Vertex 2 should remain unaffected
    assert new_dists[2] == 0.5
