"""Demo simulating soft cells/bags colliding and deforming in a confined 3D container.

Run: uv run --extra viz python examples/bags_collision.py
Adjust compliance, substeps, and number of bags via CLI flags.
"""

import argparse

import numpy as np
import polyscope as ps

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    BoxConstraint,
    DistanceConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
)
from xpbd.mesh import sphere_mesh

DT = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bags", type=int, default=4)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--bag-radius", type=float, default=0.45)
    parser.add_argument("--particle-radius", type=float, default=0.0)
    parser.add_argument("--inflation-factor", type=float, default=1.15)
    parser.add_argument("--edge-compliance", type=float, default=1e-3)
    parser.add_argument("--volume-compliance", type=float, default=0.0)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=35)
    parser.add_argument("--no-self-collision", action="store_true", help="Disable self-collision for each bag")
    parser.add_argument(
        "--mode", choices=["jacobi", "gauss_seidel"], default="gauss_seidel"
    )
    return parser.parse_args()


def combine_bags(
    bags_data: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[SoftBody, list[np.ndarray], np.ndarray, list[tuple[int, int]]]:
    """Combines multiple (positions, faces) into a single SoftBody and returns tracking lists."""
    all_positions = []
    all_edges = []
    all_inv_mass = []

    faces_list = []
    group_ids_list = []
    vertex_ranges = []

    vertex_offset = 0
    for bag_idx, (pos, faces) in enumerate(bags_data):
        n_verts = len(pos)

        temp_body = SoftBody.from_surface_mesh(pos, faces)

        all_positions.append(temp_body.positions)
        all_edges.append(temp_body.edges + vertex_offset)
        all_inv_mass.append(temp_body.inv_mass)

        shifted_faces = faces + vertex_offset
        faces_list.append(shifted_faces)

        group_ids_list.append(np.full(n_verts, bag_idx, dtype=np.int64))
        vertex_ranges.append((vertex_offset, vertex_offset + n_verts))

        vertex_offset += n_verts

    combined_positions = np.concatenate(all_positions, axis=0)
    combined_edges = np.concatenate(all_edges, axis=0)
    combined_inv_mass = np.concatenate(all_inv_mass, axis=0)
    group_ids = np.concatenate(group_ids_list, axis=0)

    combined_body = SoftBody(
        positions=combined_positions,
        velocities=np.zeros_like(combined_positions),
        inv_mass=combined_inv_mass,
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=combined_edges,
    )

    return combined_body, faces_list, group_ids, vertex_ranges


def build_solver(
    args: argparse.Namespace,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> tuple[Solver, list[np.ndarray], list[tuple[int, int]]]:
    # Generate initial positions and faces for multiple bags
    bags_data = []

    # Simple layout strategy: stack them vertically with slight horizontal offsets
    np.random.seed(42)
    for i in range(args.num_bags):
        # Slightly randomized initial centroids inside the upper part of the box
        cx = np.random.uniform(-0.3, 0.3)
        cy = 0.5 + i * (args.bag_radius * 1.8)
        cz = np.random.uniform(-0.3, 0.3)

        pos, faces = sphere_mesh(
            subdivisions=args.subdivisions,
            radius=args.bag_radius,
            origin=(cx, cy, cz),
        )
        bags_data.append((pos, faces))

    body, faces_list, group_ids, vertex_ranges = combine_bags(bags_data)

    # Automatically calculate particle collision radius if not specified
    if args.particle_radius <= 0.0:
        # Calculate average edge length of first bag to make the surface watertight
        first_pos, first_faces = bags_data[0]
        from xpbd.mesh import edges_from_faces
        first_edges = edges_from_faces(first_faces)
        edge_diffs = first_pos[first_edges[:, 1]] - first_pos[first_edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_diffs, axis=1)
        avg_edge_length = np.mean(edge_lengths)
        # Having radius >= avg_edge_length * 0.6 ensures that spheres of adjacent vertices touch/overlap,
        # preventing any vertex from slipping through the triangular faces while avoiding self-collision buckling!
        p_radius = avg_edge_length * 0.60
        print(f"Auto-calculated watertight particle collision radius: {p_radius:.4f} (avg edge length: {avg_edge_length:.4f})")
    else:
        p_radius = args.particle_radius

    constraints = [
        # 1. Distance constraints to keep the surface structure
        DistanceConstraint.from_positions(
            body.positions,
            body.edges,
            compliance=args.edge_compliance,
            relaxation=args.relaxation,
        ),
        # 2. Global volume constraints to keep bags inflated
        GlobalVolumeConstraint.from_positions(
            body.positions,
            faces_list,
            compliance=args.volume_compliance,
            relaxation=args.relaxation,
            inflation_factor=args.inflation_factor,
        ),
        # 3. Particle-to-particle collision handling (with self-collision and neighbor filtering enabled)
        ParticleCollisionConstraint(
            len(body.positions),
            radius=p_radius,
            edges=body.edges,
            group_ids=group_ids,
            self_collision=not args.no_self_collision,
        ),
        # 4. Box container boundaries
        BoxConstraint(len(body.positions), bounds_min, bounds_max),
    ]

    solver = Solver(
        body, constraints, n_substeps=args.substeps, mode=args.mode
    )
    return solver, faces_list, vertex_ranges


def register_box_wireframe(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> None:
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    box_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # columns
        ]
    )
    ps.register_curve_network(
        "container", corners, box_edges, radius=0.01, color=(0.1, 0.1, 0.1)
    )


def main() -> None:
    args = parse_args()

    # Define container limits
    bounds_min = (-1.0, 0.0, -1.0)
    bounds_max = (1.0, 3.0, 1.0)

    solver, faces_list, vertex_ranges = build_solver(
        args, bounds_min, bounds_max
    )
    body = solver.body

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_box_wireframe(bounds_min, bounds_max)

    # Register each bag as a separate Polyscope surface mesh for clear, colored visualization
    ps_meshes = []
    colors = [
        (0.85, 0.35, 0.25),  # reddish coral
        (0.25, 0.65, 0.85),  # sky blue
        (0.35, 0.75, 0.45),  # soft green
        (0.85, 0.75, 0.25),  # soft yellow
        (0.65, 0.35, 0.85),  # lavender purple
        (0.85, 0.55, 0.25),  # soft orange
    ]

    for bag_idx, (faces, (start, end)) in enumerate(
        zip(faces_list, vertex_ranges)
    ):
        bag_name = f"bag_{bag_idx}"
        # Shift faces to reference local vertices for registration
        local_faces = faces - start
        local_pos = body.positions[start:end]

        mesh = ps.register_surface_mesh(bag_name, local_pos, local_faces)
        color = colors[bag_idx % len(colors)]
        mesh.set_color(color)
        mesh.set_edge_width(1.0)
        ps_meshes.append((mesh, start, end))

    def frame() -> None:
        solver.step(DT)
        # Update vertex positions for each registered bag
        for mesh, start, end in ps_meshes:
            mesh.update_vertex_positions(body.positions[start:end])

    ps.set_user_callback(frame)
    ps.show()


if __name__ == "__main__":
    main()
