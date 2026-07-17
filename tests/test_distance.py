import numpy as np
import pytest

from xpbd.constraints import DistanceConstraint

H = 0.01


def two_particles(stretch=2.0):
    x = np.array([[0.0, 0.0, 0.0], [stretch, 0.0, 0.0]])
    inv_mass = np.array([1.0, 1.0])
    con = DistanceConstraint(
        edges=np.array([[0, 1]]), rest_lengths=np.array([1.0])
    )
    return x, inv_mass, con


def test_rigid_projection_restores_rest_length():
    x, inv_mass, con = two_particles(stretch=2.0)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    dist = np.linalg.norm(x[1] - x[0])
    assert dist == pytest.approx(1.0, abs=1e-9)


def test_projection_is_symmetric_for_equal_masses():
    x, inv_mass, con = two_particles(stretch=2.0)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    # Both endpoints move toward each other by the same amount.
    assert x[0, 0] == pytest.approx(0.5)
    assert x[1, 0] == pytest.approx(1.5)


def test_pinned_vertex_does_not_move():
    x, _, con = two_particles(stretch=2.0)
    inv_mass = np.array([0.0, 1.0])
    con.reset_lambda()
    con.project(x, inv_mass, H)
    assert np.all(x[0] == 0.0)
    assert x[1, 0] == pytest.approx(1.0)


def test_compliance_leaves_xpbd_residual():
    # One projection with compliance leaves C * alpha_tilde/(w + alpha_tilde).
    compliance = 1e-4
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_mass = np.array([1.0, 1.0])
    con = DistanceConstraint(
        edges=np.array([[0, 1]]),
        rest_lengths=np.array([1.0]),
        compliance=compliance,
    )
    con.reset_lambda()
    con.project(x, inv_mass, H)
    alpha_tilde = compliance / H**2
    expected_c = 1.0 * alpha_tilde / (2.0 + alpha_tilde)
    dist = np.linalg.norm(x[1] - x[0])
    assert dist - 1.0 == pytest.approx(expected_c, rel=1e-9)


def test_project_one_matches_batch_for_single_edge():
    x_batch, inv_mass, con_a = two_particles(stretch=1.7)
    x_seq, _, con_b = two_particles(stretch=1.7)
    con_a.reset_lambda()
    con_a.project(x_batch, inv_mass, H)
    con_b.reset_lambda()
    con_b.project_one(0, x_seq, inv_mass, H)
    np.testing.assert_allclose(x_batch, x_seq, atol=1e-12)


def test_from_positions_uses_current_lengths():
    x = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [4.0, 3.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    con = DistanceConstraint.from_positions(x, edges)
    np.testing.assert_allclose(con.rest_lengths, [3.0, 4.0])


def test_len_returns_edge_count():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    con = DistanceConstraint.from_positions(x, np.array([[0, 1], [1, 2]]))
    assert len(con) == 2
