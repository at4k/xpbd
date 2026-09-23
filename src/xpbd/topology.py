"""Topology and geometry utilities for mesh slicing, capping, and relaxation."""

import numpy as np

from .mesh import edges_from_faces


def partition_particles(
    positions: np.ndarray, cut_x: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Partitions particle indices into left (x <= cut_x) and right (x > cut_x)."""
    positions = np.asarray(positions, dtype=np.float64)
    left_indices = np.where(positions[:, 0] <= cut_x)[0]
    right_indices = np.where(positions[:, 0] > cut_x)[0]
    return left_indices, right_indices


def slice_and_cap_mesh(
    positions: np.ndarray,
    faces: np.ndarray,
    cut_x: float = 0.0,
    eps: float = 1e-9,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Slices a closed 2-manifold surface mesh at plane x = cut_x and caps both sides.

    Returns:
        (left_positions, left_faces), (right_positions, right_faces)
        Both meshes are closed, watertight 2-manifolds (chi = 2).
    """
    positions = np.asarray(positions, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    # Slight perturbation if vertices lie exactly on cut plane to avoid degenerate cuts
    on_plane = np.abs(positions[:, 0] - cut_x) < eps
    if np.any(on_plane):
        positions[on_plane, 0] += eps * 1.5

    # Lists for left and right meshes
    left_verts = list(positions)
    right_verts = list(positions)

    left_faces = []
    right_faces = []

    # Cache for edge intersection vertices: (min_idx, max_idx) -> (left_v_idx, right_v_idx, pos)
    cut_cache: dict[tuple[int, int], tuple[int, int, np.ndarray]] = {}

    def get_cut_vertex(u: int, v: int) -> tuple[int, int]:
        key = (min(u, v), max(u, v))
        if key in cut_cache:
            return cut_cache[key][0], cut_cache[key][1]
        pu = positions[u]
        pv = positions[v]
        denom = pv[0] - pu[0]
        t = (cut_x - pu[0]) / denom if abs(denom) > 1e-15 else 0.5
        t = np.clip(t, 0.0, 1.0)
        p_cut = pu + t * (pv - pu)
        p_cut[0] = cut_x  # Exact coordinate on cut plane

        l_idx = len(left_verts)
        left_verts.append(p_cut)
        r_idx = len(right_verts)
        right_verts.append(p_cut)

        cut_cache[key] = (l_idx, r_idx, p_cut)
        return l_idx, r_idx

    # Directed segments on the cut plane for capping
    left_cap_segments = []  # (v0_idx, v1_idx)
    right_cap_segments = []

    for f in faces:
        v0, v1, v2 = f[0], f[1], f[2]
        p0, p1, p2 = positions[v0], positions[v1], positions[v2]
        s0, s1, s2 = p0[0] < cut_x, p1[0] < cut_x, p2[0] < cut_x

        num_left = int(s0) + int(s1) + int(s2)

        if num_left == 3:
            left_faces.append([v0, v1, v2])
        elif num_left == 0:
            right_faces.append([v0, v1, v2])
        elif num_left == 1:
            # Single vertex on left, 2 on right
            if s0:
                va, vb, vc = v0, v1, v2
            elif s1:
                va, vb, vc = v1, v2, v0
            else:
                va, vb, vc = v2, v0, v1

            l_ab, r_ab = get_cut_vertex(va, vb)
            l_ac, r_ac = get_cut_vertex(va, vc)

            # Left gets triangle (va, l_ab, l_ac)
            left_faces.append([va, l_ab, l_ac])
            left_cap_segments.append((l_ab, l_ac))

            # Right gets quad (r_ab, vb, vc, r_ac) -> split into 2 triangles
            right_faces.append([r_ab, vb, vc])
            right_faces.append([r_ab, vc, r_ac])
            right_cap_segments.append((r_ac, r_ab))

        elif num_left == 2:
            # Single vertex on right, 2 on left
            if not s0:
                va, vb, vc = v0, v1, v2
            elif not s1:
                va, vb, vc = v1, v2, v0
            else:
                va, vb, vc = v2, v0, v1

            l_ab, r_ab = get_cut_vertex(va, vb)
            l_ac, r_ac = get_cut_vertex(va, vc)

            # Right gets triangle (va, r_ab, r_ac)
            right_faces.append([va, r_ab, r_ac])
            right_cap_segments.append((r_ab, r_ac))

            # Left gets quad (l_ab, vb, vc, l_ac) -> 2 triangles
            left_faces.append([l_ab, vb, vc])
            left_faces.append([l_ab, vc, l_ac])
            left_cap_segments.append((l_ac, l_ab))

    # Add center vertices for capping
    all_cut_pos = [item[2] for item in cut_cache.values()]
    if all_cut_pos:
        cap_center = np.mean(all_cut_pos, axis=0)
        cap_center[0] = cut_x
    else:
        cap_center = np.array([cut_x, 0.0, 0.0], dtype=np.float64)

    l_center_idx = len(left_verts)
    left_verts.append(cap_center)
    r_center_idx = len(right_verts)
    right_verts.append(cap_center)

    # Cap left mesh (normal should point +X)
    for seg_start, seg_end in left_cap_segments:
        tri = [l_center_idx, seg_start, seg_end]
        p_c = left_verts[l_center_idx]
        p_s = left_verts[seg_start]
        p_e = left_verts[seg_end]
        normal = np.cross(p_s - p_c, p_e - p_c)
        if normal[0] < 0:
            tri = [l_center_idx, seg_end, seg_start]
        left_faces.append(tri)

    # Cap right mesh (normal should point -X)
    for seg_start, seg_end in right_cap_segments:
        tri = [r_center_idx, seg_start, seg_end]
        p_c = right_verts[r_center_idx]
        p_s = right_verts[seg_start]
        p_e = right_verts[seg_end]
        normal = np.cross(p_s - p_c, p_e - p_c)
        if normal[0] > 0:
            tri = [r_center_idx, seg_end, seg_start]
        right_faces.append(tri)

    # Clean unreferenced vertices for both meshes
    def clean_mesh(verts_list, faces_list):
        arr_v = np.array(verts_list, dtype=np.float64)
        arr_f = np.array(faces_list, dtype=np.int64)
        used_idx, inverse = np.unique(arr_f, return_inverse=True)
        new_v = arr_v[used_idx]
        new_f = inverse.reshape(arr_f.shape)
        return new_v, new_f

    mesh_l = clean_mesh(left_verts, left_faces)
    mesh_r = clean_mesh(right_verts, right_faces)

    return mesh_l, mesh_r
