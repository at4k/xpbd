import numpy as np
from xpbd.mesh import icosphere_mesh, edges_from_faces


def test_icosphere_mesh_geometry():
    subdivisions = 2
    radius = 0.5
    origin = (1.0, 2.0, 3.0)
    verts, faces = icosphere_mesh(subdivisions=subdivisions, radius=radius, origin=origin)

    # Verify positions shape and faces shape
    assert len(verts) > 0
    assert faces.shape[1] == 3

    # Verify radius from origin
    dists = np.linalg.norm(verts - np.array(origin), axis=1)
    np.testing.assert_allclose(dists, radius, rtol=1e-5)


def test_icosphere_mesh_uniform_valence():
    verts, faces = icosphere_mesh(subdivisions=2, radius=1.0)
    edges = edges_from_faces(faces)

    # Count valence per vertex
    valence = np.zeros(len(verts), dtype=int)
    for e in edges:
        valence[e[0]] += 1
        valence[e[1]] += 1

    # Exactly 12 vertices have valence 5 (original icosahedron vertices), all others have valence 6
    assert np.sum(valence == 5) == 12
    assert np.sum(valence == 6) == len(verts) - 12
    assert np.all((valence == 5) | (valence == 6))
