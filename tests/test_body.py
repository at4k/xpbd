import numpy as np
import pytest

from xpbd.body import SoftBody
from xpbd.mesh import tet_grid


def test_from_tet_mesh_shapes_and_dtypes():
    positions, tets = tet_grid(1, 1, 1)
    body = SoftBody.from_tet_mesh(positions, tets)
    n = len(positions)
    assert body.positions.shape == (n, 3)
    assert body.velocities.shape == (n, 3)
    assert np.all(body.velocities == 0)
    assert body.inv_mass.shape == (n,)
    assert body.positions.dtype == np.float64
    assert body.edges.ndim == 2 and body.edges.shape[1] == 2


def test_total_mass_matches_density_times_volume():
    positions, tets = tet_grid(2, 1, 1, spacing=0.5)
    body = SoftBody.from_tet_mesh(positions, tets, density=800.0)
    total_mass = (1.0 / body.inv_mass).sum()
    expected = 800.0 * (2 * 1 * 1 * 0.5**3)
    assert total_mass == pytest.approx(expected)


def test_positions_are_copied():
    positions, tets = tet_grid(1, 1, 1)
    body = SoftBody.from_tet_mesh(positions, tets)
    positions[0, 0] = 99.0
    assert body.positions[0, 0] != 99.0
