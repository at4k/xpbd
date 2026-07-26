"""Tetrahedral mesh generation and topology utilities."""

import numpy as np

# Alternating 5-tet decompositions of a cube cell. Corners are (di, dj, dk)
# offsets. Adjacent cells use opposite parity so face diagonals match.
_TETS_EVEN = (
    ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (1, 0, 1)),
    ((0, 1, 0), (0, 0, 0), (0, 1, 1), (1, 1, 0)),
    ((0, 0, 1), (0, 0, 0), (1, 0, 1), (0, 1, 1)),
    ((1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)),
)
_TETS_ODD = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 1, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 0, 1), (1, 0, 0), (0, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (0, 1, 0), (1, 1, 1), (0, 0, 1)),
)


_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def edges_from_tets(tets: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a tet mesh, each row sorted ascending."""
    pairs = tets[:, _TET_EDGES].reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def surface_from_tets(tets: np.ndarray) -> np.ndarray:
    """Boundary triangles of a tet mesh (faces appearing exactly once)."""
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(
        keys, axis=0, return_index=True, return_counts=True
    )
    return faces[first[counts == 1]]


def tet_volumes(positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volumes of tetrahedra, shape (T,)."""
    p = positions[tets]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0]) / 6.0


_TRI_EDGES = np.array([[0, 1], [1, 2], [2, 0]])


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle mesh, each row sorted ascending."""
    pairs = faces[:, _TRI_EDGES].reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def extract_boundary_mesh(
    positions: np.ndarray, tets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts boundary surface mesh (positions, faces) and removes internal vertices."""
    faces = surface_from_tets(tets)
    unique_v, inverse = np.unique(faces, return_inverse=True)
    new_positions = positions[unique_v]
    new_faces = inverse.reshape(faces.shape)
    return new_positions, new_faces


def sphere_mesh(
    subdivisions: int = 4,
    radius: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a spherified box surface mesh."""
    pos, tets = tet_grid(
        subdivisions,
        subdivisions,
        subdivisions,
        spacing=2.0 / subdivisions,
        origin=(-1.0, -1.0, -1.0),
    )
    pos_b, faces = extract_boundary_mesh(pos, tets)
    norms = np.linalg.norm(pos_b, axis=1)
    norms = np.maximum(norms, 1e-12)
    pos_sphere = (pos_b / norms[:, None]) * radius + np.asarray(
        origin, dtype=np.float64
    )
    return pos_sphere, faces


def tet_grid(
    nx: int,
    ny: int,
    nz: int,
    spacing: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize an (nx, ny, nz)-cell box grid with 5 tets per cell.

    Returns (positions (N, 3) float64, tets (T, 4) int64). All tets have
    positive signed volume.
    """
    gx, gy, gz = nx + 1, ny + 1, nz + 1
    ii, jj, kk = np.meshgrid(
        np.arange(gx), np.arange(gy), np.arange(gz), indexing="ij"
    )
    positions = (
        np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.float64)
        * spacing
        + np.asarray(origin, dtype=np.float64)
    )

    def vidx(i: int, j: int, k: int) -> int:
        return (i * gy + j) * gz + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pattern = _TETS_EVEN if (i + j + k) % 2 == 0 else _TETS_ODD
                for corners in pattern:
                    tets.append(
                        [vidx(i + di, j + dj, k + dk) for di, dj, dk in corners]
                    )
    tets = np.asarray(tets, dtype=np.int64)

    # Enforce positive orientation by swapping two vertices where needed.
    flip = tet_volumes(positions, tets) < 0
    flipped = tets[flip]
    flipped[:, [0, 1]] = flipped[:, [1, 0]]
    tets[flip] = flipped
    return positions, tets
