import numpy as np


def test_mouse_pusher_direct_repulsion():
    # Test pushing body positions directly with a 3D mouse pusher sphere
    positions = np.array([
        [0.0, 0.5, 0.0],
        [0.1, 0.5, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)

    velocities = np.zeros_like(positions)
    pusher_pos = np.array([0.0, 0.5, 0.0])
    push_radius = 0.3
    dt = 1.0 / 60.0

    # Simulate mouse direct push frame
    vecs = positions - pusher_pos
    dists = np.linalg.norm(vecs, axis=1)
    inside = dists < push_radius

    for idx in np.where(inside)[0]:
        d = dists[idx]
        dir_vec = np.array([0.0, 1.0, 0.0]) if d < 1e-6 else vecs[idx] / d
        target_p = pusher_pos + dir_vec * push_radius
        velocities[idx] = (target_p - positions[idx]) / dt
        positions[idx] = target_p

    # Position 0 should be pushed upwards
    assert positions[0, 1] == 0.5 + push_radius
    assert velocities[0, 1] > 0.0
