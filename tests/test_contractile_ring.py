import numpy as np
import pytest

from xpbd.constraints.contractile_ring import ContractileRingConstraint


def test_contractile_ring_constriction():
    # 4 ring vertices arranged in circle of radius 1.0 around (y=0, z=0) at x=0
    x = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    inv_mass = np.ones(4)

    ring = ContractileRingConstraint(
        ring_indices=np.arange(4),
        center_y=0.0,
        center_z=0.0,
        target_radius=0.5,
        compliance=0.0,
    )

    assert len(ring) == 4
    assert ring.target_radius == 0.5

    h = 0.01
    x_cur = x.copy()
    ring.reset_lambda()
    ring.project(x_cur, inv_mass, h)

    radii = np.sqrt(x_cur[:, 1] ** 2 + x_cur[:, 2] ** 2)
    assert np.allclose(radii, 0.5, atol=1e-3)


def test_contractile_ring_project_one():
    x = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    inv_mass = np.ones(2)

    ring = ContractileRingConstraint(
        ring_indices=np.arange(2),
        center_y=0.0,
        center_z=0.0,
        target_radius=0.3,
        compliance=0.0,
    )

    h = 0.01
    x_cur = x.copy()
    ring.reset_lambda()
    for i in range(len(ring)):
        ring.project_one(i, x_cur, inv_mass, h)

    radii = np.sqrt(x_cur[:, 1] ** 2 + x_cur[:, 2] ** 2)
    assert np.allclose(radii, 0.3, atol=1e-3)
