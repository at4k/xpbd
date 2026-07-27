import numpy as np
from xpbd.mesh import sphere_tet_mesh, tet_volumes


def test_sphere_tet_mesh():
    subdivisions = 2
    radius = 1.0
    origin = (0.0, 0.0, 0.0)
    pos, boundary_faces, tets = sphere_tet_mesh(subdivisions=subdivisions, radius=radius, origin=origin)

    assert len(pos) > 0
    assert boundary_faces.shape[1] == 3
    assert tets.shape[1] == 4

    # All tets must have strictly positive volume
    vols = tet_volumes(pos, tets)
    assert np.all(vols > 0.0)
