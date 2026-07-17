import numpy as np
import pytest

from xpbd.body import SoftBody
from xpbd.constraints import (
    DistanceConstraint,
    GroundConstraint,
    VolumeConstraint,
)
from xpbd.mesh import tet_grid
from xpbd.solver import Solver

DT = 1.0 / 60.0


def single_particle_body():
    # A degenerate body with one free particle and no constraints.
    return SoftBody(
        positions=np.zeros((1, 3)),
        velocities=np.zeros((1, 3)),
        inv_mass=np.ones(1),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
    )


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Solver(single_particle_body(), [], mode="nonsense")


def test_free_fall_matches_analytic_solution():
    body = single_particle_body()
    solver = Solver(body, [], n_substeps=10)
    for _ in range(60):
        solver.step(DT)
    # y = -0.5 g t^2 with t = 1s; symplectic Euler error is O(g t h / 2).
    assert body.positions[0, 1] == pytest.approx(-0.5 * 9.81, abs=0.02)


def test_pinned_particle_does_not_fall():
    body = single_particle_body()
    body.inv_mass[0] = 0.0
    solver = Solver(body, [])
    for _ in range(30):
        solver.step(DT)
    np.testing.assert_allclose(body.positions, 0.0)
    np.testing.assert_allclose(body.velocities, 0.0)


def hanging_pair(mode):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    body = SoftBody(
        positions=positions.copy(),
        velocities=np.zeros((2, 3)),
        inv_mass=np.array([0.0, 1.0]),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.array([[0, 1]]),
    )
    con = DistanceConstraint.from_positions(body.positions, body.edges)
    return body, Solver(body, [con], n_substeps=10, mode=mode)


@pytest.mark.parametrize("mode", ["jacobi", "gauss_seidel"])
def test_hanging_particle_keeps_rest_length(mode):
    body, solver = hanging_pair(mode)
    for _ in range(120):
        solver.step(DT)
    dist = np.linalg.norm(body.positions[1] - body.positions[0])
    assert dist == pytest.approx(1.0, abs=1e-6)


def test_compliant_edge_stretches_under_gravity():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    body = SoftBody(
        positions=positions.copy(),
        velocities=np.zeros((2, 3)),
        inv_mass=np.array([0.0, 1.0]),
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=np.array([[0, 1]]),
    )
    con = DistanceConstraint.from_positions(
        body.positions, body.edges, compliance=1e-2
    )
    solver = Solver(body, [con], n_substeps=10)
    for _ in range(240):
        solver.step(DT)
    dist = np.linalg.norm(body.positions[1] - body.positions[0])
    assert dist > 1.001


def dropped_cube_scene(mode):
    positions, tets = tet_grid(2, 2, 2, spacing=0.25)
    positions[:, 1] += 1.0
    body = SoftBody.from_tet_mesh(positions, tets)
    constraints = [
        DistanceConstraint.from_positions(body.positions, body.edges),
        VolumeConstraint.from_positions(body.positions, body.tets),
        GroundConstraint(len(body.positions)),
    ]
    return body, Solver(body, constraints, n_substeps=20, mode=mode)


def test_dropped_cube_rests_on_ground_without_penetration():
    body, solver = dropped_cube_scene("jacobi")
    for _ in range(120):
        solver.step(DT)
    assert body.positions[:, 1].min() >= -1e-9


def test_jacobi_and_gauss_seidel_reach_similar_states():
    body_j, solver_j = dropped_cube_scene("jacobi")
    body_g, solver_g = dropped_cube_scene("gauss_seidel")
    for _ in range(120):
        solver_j.step(DT)
        solver_g.step(DT)
    com_j = body_j.positions.mean(axis=0)
    com_g = body_g.positions.mean(axis=0)
    np.testing.assert_allclose(com_j, com_g, atol=0.05)
