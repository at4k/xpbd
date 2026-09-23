import numpy as np
import pytest

from xpbd.constraints.spindle import SpindleConstraint


def test_spindle_elongation_and_astral_tethers():
    # 2 poles and 4 membrane vertices
    # 0: pole_l, 1: pole_r, 2,3: left membrane, 4,5: right membrane
    x = np.array([
        [-0.2, 0.0, 0.0],
        [ 0.2, 0.0, 0.0],
        [-0.6, 0.3, 0.0],
        [-0.6, -0.3, 0.0],
        [ 0.6, 0.3, 0.0],
        [ 0.6, -0.3, 0.0],
    ], dtype=np.float64)
    inv_mass = np.ones(len(x))

    con = SpindleConstraint(
        pole_0=0,
        pole_1=1,
        left_membrane_idx=np.array([2, 3]),
        right_membrane_idx=np.array([4, 5]),
        pole_rest_length=0.4,
        tether_rest_length=0.3,
        tether_compliance=1e-3,
    )

    assert len(con) == 1 + 2 + 2  # 1 pole edge + 2 left tethers + 2 right tethers

    # Increase target length of spindle
    con.set_target_length(0.8)
    assert con.pole_rest_length == 0.8

    # Project should push poles apart and pull tethers
    h = 0.01
    x_cur = x.copy()
    con.reset_lambda()
    con.project(x_cur, inv_mass, h)

    # Poles must move further apart
    assert (x_cur[1, 0] - x_cur[0, 0]) > (x[1, 0] - x[0, 0])

    # Left membrane vertices (2, 3) must be pulled toward pole 0
    d_before = np.linalg.norm(x[2] - x[0])
    d_after = np.linalg.norm(x_cur[2] - x_cur[0])
    assert d_after < d_before


def test_spindle_project_one():
    x = np.array([
        [-0.2, 0.0, 0.0],
        [ 0.2, 0.0, 0.0],
        [-0.6, 0.0, 0.0],
        [ 0.6, 0.0, 0.0],
    ], dtype=np.float64)
    inv_mass = np.ones(len(x))

    con = SpindleConstraint(
        pole_0=0,
        pole_1=1,
        left_membrane_idx=np.array([2]),
        right_membrane_idx=np.array([3]),
        pole_rest_length=0.8,
        tether_rest_length=0.2,
    )

    h = 0.01
    x_cur = x.copy()
    con.reset_lambda()
    for i in range(len(con)):
        con.project_one(i, x_cur, inv_mass, h)

    assert (x_cur[1, 0] - x_cur[0, 0]) > (x[1, 0] - x[0, 0])
