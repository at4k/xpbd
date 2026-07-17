import numpy as np

from xpbd.constraints import GroundConstraint

H = 0.01


def test_penetrating_vertices_are_clamped():
    x = np.array([[0.0, -0.5, 0.0], [1.0, 2.0, 0.0], [2.0, -0.1, 3.0]])
    inv_mass = np.ones(3)
    con = GroundConstraint(3)
    con.reset_lambda()
    con.project(x, inv_mass, H)
    np.testing.assert_allclose(x[:, 1], [0.0, 2.0, 0.0])
    # x and z are untouched.
    np.testing.assert_allclose(x[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(x[:, 2], [0.0, 0.0, 3.0])


def test_custom_height():
    x = np.array([[0.0, 0.2, 0.0]])
    con = GroundConstraint(1, height=0.5)
    con.reset_lambda()
    con.project(x, np.ones(1), H)
    assert x[0, 1] == 0.5


def test_pinned_vertex_is_not_clamped():
    x = np.array([[0.0, -1.0, 0.0]])
    con = GroundConstraint(1)
    con.reset_lambda()
    con.project(x, np.zeros(1), H)
    assert x[0, 1] == -1.0


def test_project_one_matches_batch():
    x_a = np.array([[0.0, -0.5, 0.0], [1.0, 2.0, 0.0]])
    x_b = x_a.copy()
    inv_mass = np.ones(2)
    con = GroundConstraint(2)
    con.reset_lambda()
    con.project(x_a, inv_mass, H)
    con.reset_lambda()
    for i in range(len(con)):
        con.project_one(i, x_b, inv_mass, H)
    np.testing.assert_allclose(x_a, x_b)


def test_len_is_vertex_count():
    assert len(GroundConstraint(7)) == 7
