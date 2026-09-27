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


def test_contractile_ring_center_x_anchoring():
    # 2 movable points, 1 fixed point (inv_mass=0)
    x = np.array([
        [0.2, 0.4, 0.0],  # radius = 0.4 < target_radius 0.5 (inactive radially)
        [0.3, 0.8, 0.0],  # radius = 0.8 > target_radius 0.5 (active radially)
        [0.5, 0.8, 0.0],  # fixed point (inv_mass=0.0)
    ], dtype=np.float64)
    inv_mass = np.array([1.0, 1.0, 0.0])

    ring = ContractileRingConstraint(
        ring_indices=np.arange(3),
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        target_radius=0.5,
        compliance=0.0,
    )

    x_cur = x.copy()
    ring.project(x_cur, inv_mass, h=0.01)

    # Point 0 is movable, anchored to x=0
    assert np.isclose(x_cur[0, 0], 0.0, atol=1e-4)
    # Point 1 is movable, anchored to x=0
    assert np.isclose(x_cur[1, 0], 0.0, atol=1e-4)
    # Point 2 has inv_mass=0, so its x must NOT move!
    assert np.isclose(x_cur[2, 0], 0.5, atol=1e-4)


def test_contractile_ring_bilateral_vs_unilateral():
    x_in = np.array([[0.0, 0.2, 0.0]], dtype=np.float64)  # radius = 0.2 < 0.5
    inv_m = np.ones(1)

    # Bilateral pulls or pushes to target radius
    ring_bi = ContractileRingConstraint(
        ring_indices=[0], target_radius=0.5, compliance=0.0, unilateral=False
    )
    x1 = x_in.copy()
    ring_bi.project(x1, inv_m, h=0.01)
    assert np.isclose(x1[0, 1], 0.5, atol=1e-3)

    # Unilateral only pulls inward if radius > target
    ring_uni = ContractileRingConstraint(
        ring_indices=[0], target_radius=0.5, compliance=0.0, unilateral=True
    )
    x2 = x_in.copy()
    ring_uni.project(x2, inv_m, h=0.01)
    assert np.isclose(x2[0, 1], 0.2, atol=1e-3)
