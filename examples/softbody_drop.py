"""Soft body drop demo rendered with polyscope.

Run: uv run --extra viz python examples/softbody_drop.py
Adjust compliance, substeps and solver mode via CLI flags.
"""

import argparse

import numpy as np
import polyscope as ps

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    DistanceConstraint,
    GroundConstraint,
    VolumeConstraint,
)
from xpbd.mesh import tet_grid

DT = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--spacing", type=float, default=0.15)
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--edge-compliance", type=float, default=1e-4)
    parser.add_argument("--volume-compliance", type=float, default=0.0)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["jacobi", "gauss_seidel"], default="jacobi"
    )
    parser.add_argument(
        "--transparency", type=float, default=0.4,
        help="Mesh surface transparency (0.0=invisible, 1.0=opaque, default=0.4)"
    )
    parser.add_argument(
        "--point-radius", type=float, default=0.015,
        help="Radius of internal node particles"
    )
    return parser.parse_args()


def tilt(positions: np.ndarray, angle_x: float, angle_z: float) -> np.ndarray:
    """Rotate positions about their centroid for a more interesting fall."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    center = positions.mean(axis=0)
    return (positions - center) @ (rot_z @ rot_x).T + center


def build_solver(args: argparse.Namespace) -> Solver:
    positions, tets = tet_grid(args.nx, args.ny, args.nz, spacing=args.spacing)
    positions = tilt(positions, angle_x=0.4, angle_z=0.3)
    positions[:, 1] += args.drop_height - positions[:, 1].min()
    body = SoftBody.from_tet_mesh(positions, tets)
    constraints = [
        DistanceConstraint.from_positions(
            body.positions,
            body.edges,
            compliance=args.edge_compliance,
            relaxation=args.relaxation,
        ),
        VolumeConstraint.from_positions(
            body.positions,
            body.tets,
            compliance=args.volume_compliance,
            relaxation=args.relaxation,
        ),
        GroundConstraint(len(body.positions)),
    ]
    return Solver(
        body, constraints, n_substeps=args.substeps, mode=args.mode
    )


def register_floor() -> None:
    size = 4.0
    verts = np.array(
        [[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    floor = ps.register_surface_mesh("floor", verts, faces)
    floor.set_color((0.75, 0.75, 0.78))


def main() -> None:
    args = parse_args()
    solver = build_solver(args)
    body = solver.body

    init_positions = body.positions.copy()
    is_running = True
    transparency = args.transparency
    point_radius = args.point_radius
    show_mesh = True
    show_nodes = True

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_floor()

    mesh = ps.register_volume_mesh("softbody", body.positions, tets=body.tets)
    mesh.set_transparency(transparency)

    nodes = ps.register_point_cloud("nodes", body.positions)
    nodes.set_radius(point_radius)
    nodes.set_color((0.2, 0.5, 0.9))

    def reset_simulation() -> None:
        body.positions[:] = init_positions.copy()
        body.velocities[:] = np.zeros_like(init_positions)
        solver.reset()
        mesh.update_vertex_positions(body.positions)
        nodes.update_point_positions(body.positions)

    def frame() -> None:
        nonlocal is_running, transparency, point_radius, show_mesh, show_nodes
        import polyscope.imgui as imgui

        if imgui.Button("Reset"):
            reset_simulation()
        imgui.SameLine()
        if imgui.Button("Pause" if is_running else "Play"):
            is_running = not is_running

        imgui.SeparatorText("Visualization")

        changed_trans, new_trans = imgui.SliderFloat("Surface Transparency", transparency, 0.0, 1.0)
        if changed_trans:
            transparency = new_trans
            mesh.set_transparency(transparency)

        changed_rad, new_rad = imgui.SliderFloat("Point Radius", point_radius, 0.001, 0.05)
        if changed_rad:
            point_radius = new_rad
            nodes.set_radius(point_radius)

        changed_m, new_m = imgui.Checkbox("Show Mesh", show_mesh)
        if changed_m:
            show_mesh = new_m
            mesh.set_enabled(show_mesh)

        changed_n, new_n = imgui.Checkbox("Show Nodes", show_nodes)
        if changed_n:
            show_nodes = new_n
            nodes.set_enabled(show_nodes)

        if is_running:
            solver.step(DT)
            mesh.update_vertex_positions(body.positions)
            nodes.update_point_positions(body.positions)

    ps.set_user_callback(frame)
    ps.show()


if __name__ == "__main__":
    main()
