import numpy as np
from xpbd import Solver, SoftBody
from xpbd.constraints import DistanceConstraint
from xpbd.mesh import sphere_mesh


def test_solver_reset():
    pos, faces = sphere_mesh(subdivisions=2)
    body = SoftBody.from_surface_mesh(pos, faces)
    con = DistanceConstraint.from_positions(body.positions, body.edges)
    solver = Solver(body, [con])

    # Run a step so lagrange multipliers get non-zero values
    solver.step(1.0 / 60.0)

    # Call reset
    solver.reset()

    # Check lagrange multipliers are zeroed
    assert np.allclose(con.lam, 0.0)
