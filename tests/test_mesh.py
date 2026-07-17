import numpy as np
import pytest

from xpbd.mesh import tet_grid, tet_volumes


def test_tet_volumes_unit_tet():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tets = np.array([[0, 1, 2, 3]])
    vols = tet_volumes(positions, tets)
    assert vols.shape == (1,)
    assert vols[0] == pytest.approx(1.0 / 6.0)


def test_tet_grid_single_cell():
    positions, tets = tet_grid(1, 1, 1)
    assert positions.shape == (8, 3)
    assert tets.shape == (5, 4)
    vols = tet_volumes(positions, tets)
    assert np.all(vols > 0)
    assert vols.sum() == pytest.approx(1.0)


def test_tet_grid_total_volume_and_orientation():
    positions, tets = tet_grid(2, 3, 2, spacing=0.5)
    vols = tet_volumes(positions, tets)
    assert np.all(vols > 0)
    assert vols.sum() == pytest.approx(2 * 3 * 2 * 0.5**3)


def test_tet_grid_origin_and_spacing():
    positions, _ = tet_grid(1, 1, 1, spacing=2.0, origin=(1.0, 2.0, 3.0))
    assert positions.min(axis=0) == pytest.approx([1.0, 2.0, 3.0])
    assert positions.max(axis=0) == pytest.approx([3.0, 4.0, 5.0])
