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
        p_c = left_verts[l_center_idx]
        p_s = left_verts[seg_start]
        p_e = left_verts[seg_end]
        normal = np.cross(p_s - p_c, p_e - p_c)
        if abs(normal[0]) < 1e-12:
            continue
        tri = [l_center_idx, seg_start, seg_end]
        if normal[0] < 0:
            tri = [l_center_idx, seg_end, seg_start]
        left_faces.append(tri)

    # Cap right mesh (normal should point -X)
    for seg_start, seg_end in right_cap_segments:
        p_c = right_verts[r_center_idx]
        p_s = right_verts[seg_start]
        p_e = right_verts[seg_end]
        normal = np.cross(p_s - p_c, p_e - p_c)
        if abs(normal[0]) < 1e-12:
            continue
        tri = [r_center_idx, seg_start, seg_end]
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


def tangential_surface_relaxation(
    positions: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray | None = None,
    factor: float = 0.15,
    n_iters: int = 1,
) -> np.ndarray:
    """Equilibrates surface vertices along surface tangents to reduce vertex bunching.

    Preserves the radial profile and volume by constraining movement strictly to
    tangents and restoring radius from center.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    n_verts = len(pos)

    if center is None:
        center = np.mean(pos, axis=0)
    else:
        center = np.asarray(center, dtype=np.float64)

    r_orig = np.linalg.norm(pos - center, axis=1, keepdims=True)
    r_orig = np.maximum(r_orig, 1e-12)

    # Edge pairs for 1-ring neighbor averaging
    edges_0 = faces[:, 0]
    edges_1 = faces[:, 1]
    edges_2 = faces[:, 2]

    # Symmetric directed pairs: (0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)
    src = np.concatenate([edges_0, edges_1, edges_1, edges_2, edges_2, edges_0])
    dst = np.concatenate([edges_1, edges_0, edges_2, edges_1, edges_0, edges_2])

    # Remove duplicate directed edges if any
    unique_pairs = np.unique(np.stack([src, dst], axis=-1), axis=0)
    src_u, dst_u = unique_pairs[:, 0], unique_pairs[:, 1]

    counts = np.zeros((n_verts, 1), dtype=np.float64)
    np.add.at(counts, src_u, 1.0)
    counts = np.maximum(counts, 1.0)

    def _calc_volume(p: np.ndarray, f: np.ndarray) -> float:
        p1 = p[f[:, 0]]
        p2 = p[f[:, 1]]
        p3 = p[f[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0)

    target_vol = _calc_volume(pos, faces)

    for _ in range(n_iters):
        # 1. Compute vertex normals
        p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
        f_normals = np.cross(p1 - p0, p2 - p0)

        v_normals = np.zeros_like(pos)
        np.add.at(v_normals, faces[:, 0], f_normals)
        np.add.at(v_normals, faces[:, 1], f_normals)
        np.add.at(v_normals, faces[:, 2], f_normals)

        v_norm = np.linalg.norm(v_normals, axis=1, keepdims=True)
        normals = np.divide(
            v_normals, v_norm, out=np.zeros_like(v_normals), where=v_norm > 1e-12
        )

        # 2. Compute 1-ring neighbor averages
        sums = np.zeros_like(pos)
        np.add.at(sums, src_u, pos[dst_u])
        avg_neighbors = sums / counts

        # 3. Laplacian vector
        laplacian = avg_neighbors - pos

        # 4. Tangential projection (remove component along surface normal)
        normal_comp = (
            np.sum(laplacian * normals, axis=1, keepdims=True) * normals
        )
        tangent_disp = laplacian - normal_comp

        # 5. Tangential update
        pos += factor * tangent_disp

        # 6. Restore radial distance from center
        vec = pos - center
        cur_r = np.linalg.norm(vec, axis=1, keepdims=True)
        pos = center + (vec / np.maximum(cur_r, 1e-12)) * r_orig

        # 7. Preserve volume strictly by scaling positions
        if target_vol > 1e-12:
            cur_vol = _calc_volume(pos, faces)
            if cur_vol > 1e-12:
                vol_scale = (target_vol / cur_vol) ** (1.0 / 3.0)
                pos = center + (pos - center) * vol_scale

    return pos

