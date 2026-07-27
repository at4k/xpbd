import numpy as np


def test_ray_nearest_vertex_finding():
    # Test Ray-to-point distance and index selection logic
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.2, 0.2, 0.0],
    ], dtype=np.float64)

    # Ray origin and direction pointing directly near (0.2, 0.2, 0)
    ray_o = np.array([0.2, 0.2, -1.0])
    ray_d = np.array([0.0, 0.0, 1.0])

    diff = positions - ray_o
    t = np.maximum(0.0, np.dot(diff, ray_d))
    closest_pts = ray_o + t[:, None] * ray_d
    dists = np.linalg.norm(positions - closest_pts, axis=1)

    min_idx = np.argmin(dists)
    assert min_idx == 2
    assert dists[min_idx] < 1e-5
