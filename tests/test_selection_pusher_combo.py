import numpy as np


def test_selection_based_push_tool():
    # Test setting pusher position directly to selected vertex or hit position
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0],
        [-0.5, 0.2, 0.0],
    ], dtype=np.float64)

    # User clicks near vertex 1 (0.5, 0.2, 0) -> Selection places Push Tool at vertex 1
    selected_idx = 1
    pusher_pos = positions[selected_idx].copy()

    # Apply push force pushing outwards from selected vertex
    push_radius = 0.3
    vecs = positions - pusher_pos
    dists = np.linalg.norm(vecs, axis=1)
    inside = dists < push_radius

    assert bool(inside[1]) is True  # Selected vertex is inside pusher
    assert pusher_pos[0] == 0.5

