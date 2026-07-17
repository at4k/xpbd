import numpy as np
import pytest

from xpbd.constraints import VolumeConstraint
from xpbd.mesh import tet_volumes

H = 0.01


def unit_tet():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


def test_squashed_tet_recovers_volume():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    x[:, 2] *= 0.3  # squash along z
    inv_mass = np.ones(4)
    for _ in range(30):
        con.reset_lambda()
        con.project(x, inv_mass, H)
    vol = tet_volumes(x, tets)[0]
    assert vol == pytest.approx(1.0 / 6.0, rel=1e-5)


def test_rest_state_is_fixed_point():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    con.reset_lambda()
    con.project(x, np.ones(4), H)
    np.testing.assert_allclose(x, rest, atol=1e-12)


def test_pinned_vertices_do_not_move():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    con = VolumeConstraint.from_positions(rest, tets)
    x = rest.copy()
    x[3, 2] = 0.5
    inv_mass = np.array([0.0, 0.0, 0.0, 1.0])
    con.reset_lambda()
    con.project(x, inv_mass, H)
    np.testing.assert_allclose(x[:3], rest[:3], atol=1e-12)


def test_project_one_matches_batch_for_single_tet():
    rest = unit_tet()
    tets = np.array([[0, 1, 2, 3]])
    x_a = rest.copy()
    x_a[:, 2] *= 0.5
    x_b = x_a.copy()
    inv_mass = np.ones(4)
    con_a = VolumeConstraint.from_positions(rest, tets)
    con_b = VolumeConstraint.from_positions(rest, tets)
    con_a.reset_lambda()
    con_a.project(x_a, inv_mass, H)
    con_b.reset_lambda()
    con_b.project_one(0, x_b, inv_mass, H)
    np.testing.assert_allclose(x_a, x_b, atol=1e-12)


def test_len_returns_tet_count():
    rest = unit_tet()
    con = VolumeConstraint.from_positions(rest, np.array([[0, 1, 2, 3]]))
    assert len(con) == 1
