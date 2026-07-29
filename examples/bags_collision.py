import argparse

import numpy as np
import polyscope as ps

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    BoxConstraint,
    DistanceConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
    VolumeConstraint,
)
from xpbd.mesh import edges_from_faces, edges_from_tets, icosphere_mesh, sphere_mesh, sphere_tet_mesh


DT = 1.0 / 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bags", type=int, default=4)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--bag-radius", type=float, default=0.45)
    parser.add_argument("--particle-radius", type=float, default=0.0)
    parser.add_argument("--inflation-factor", type=float, default=1.0)
    parser.add_argument("--edge-compliance", type=float, default=0.0)  # Rigid edges for smooth surface
    parser.add_argument("--volume-compliance", type=float, default=0.0)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--no-self-collision", action="store_true", help="Disable self-collision for each bag")
    parser.add_argument(
        "--mesh-type", choices=["icosphere", "tet_sphere", "cube_sphere"], default="icosphere",
        help="icosphere (uniform surface), tet_sphere (3D tetrahedral mesh), cube_sphere (legacy)"
    )
    parser.add_argument(
        "--internal-particles", type=int, default=15,
        help="Number of internal fluid particles per bag (0 to disable)"
    )
    parser.add_argument(
        "--mode", choices=["jacobi", "gauss_seidel"], default="jacobi",
        help="jacobi mode is vectorized and dramatically faster"
    )
    return parser.parse_args()



def generate_internal_particles(
    origin: tuple[float, float, float],
    radius: float,
    num_particles: int,
) -> np.ndarray:
    """Generates internal fluid particles inside a safe inner sphere to avoid initial overlaps."""
    if num_particles <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Fill within 45% radius of sphere to maintain clear safety gap from shell
    r_max = radius * 0.45

    side = int(np.ceil(num_particles ** (1.0 / 3.0)))
    if side < 2:
        side = 2
    grid_1d = np.linspace(-r_max, r_max, side)
    gx, gy, gz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    dists = np.linalg.norm(pts, axis=1)
    inside = pts[dists <= r_max]

    if len(inside) > num_particles:
        indices = np.linspace(0, len(inside) - 1, num_particles, dtype=int)
        inside = inside[indices]
    elif len(inside) < num_particles:
        np.random.seed(123)
        while len(inside) < num_particles:
            vec = np.random.uniform(-r_max, r_max, size=(num_particles * 2, 3))
            norms = np.linalg.norm(vec, axis=1)
            valid = vec[norms <= r_max]
            inside = np.concatenate([inside, valid], axis=0)[:num_particles]

    return inside + np.asarray(origin, dtype=np.float64)


def build_solver(
    args: argparse.Namespace,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> tuple[Solver, list[np.ndarray], list[tuple[int, int]], list[tuple[int, int]]]:
    all_positions = []
    all_edges = []
    all_tets = []
    all_inv_mass = []
    all_radii = []
    faces_list = []
    group_ids_list = []

    surf_vertex_ranges = []
    particle_vertex_ranges = []

    np.random.seed(42)
    vertex_offset = 0

    for bag_idx in range(args.num_bags):
        cx = np.random.uniform(-0.3, 0.3)
        cy = 0.5 + bag_idx * (args.bag_radius * 1.8)
        cz = np.random.uniform(-0.3, 0.3)
        origin = (cx, cy, cz)

        if args.mesh_type == "tet_sphere":
            surf_pos, faces, tets = sphere_tet_mesh(
                subdivisions=args.subdivisions,
                radius=args.bag_radius,
                origin=origin,
            )
            n_surf = len(surf_pos)
            surf_edges = edges_from_tets(tets)
            shifted_tets = tets + vertex_offset
            all_tets.append(shifted_tets)
            internal_pos = np.zeros((0, 3), dtype=np.float64)
            n_internal = 0
        else:
            if args.mesh_type == "icosphere":
                surf_pos, faces = icosphere_mesh(
                    subdivisions=args.subdivisions,
                    radius=args.bag_radius,
                    origin=origin,
                )
            else:
                surf_pos, faces = sphere_mesh(
                    subdivisions=args.subdivisions,
                    radius=args.bag_radius,
                    origin=origin,
                )
            n_surf = len(surf_pos)
            surf_edges = edges_from_faces(faces)
            internal_pos = generate_internal_particles(
                origin, args.bag_radius, args.internal_particles
            )
            n_internal = len(internal_pos)

        edge_diffs = surf_pos[surf_edges[:, 1]] - surf_pos[surf_edges[:, 0]]
        avg_edge_len = np.mean(np.linalg.norm(edge_diffs, axis=1))

        r_surf = avg_edge_len * 0.70 if args.particle_radius <= 0 else args.particle_radius
        r_int = avg_edge_len * 0.50 if args.particle_radius <= 0 else args.particle_radius

        total_bag_pos = np.concatenate([surf_pos, internal_pos], axis=0) if n_internal > 0 else surf_pos

        all_positions.append(total_bag_pos)
        all_edges.append(surf_edges + vertex_offset)

        # Give internal fluid particles slightly lower mass impact so shell stays smooth
        inv_mass = np.ones(len(total_bag_pos), dtype=np.float64)
        if n_internal > 0:
            inv_mass[n_surf:] = 2.0  # lighter fluid particles
        all_inv_mass.append(inv_mass)

        bag_radii = np.empty(len(total_bag_pos), dtype=np.float64)
        bag_radii[:n_surf] = r_surf
        if n_internal > 0:
            bag_radii[n_surf:] = r_int
        all_radii.append(bag_radii)

        shifted_faces = faces + vertex_offset
        faces_list.append(shifted_faces)

        group_ids_list.append(np.full(len(total_bag_pos), bag_idx, dtype=np.int64))

        surf_start = vertex_offset
        surf_end = vertex_offset + n_surf
        internal_start = surf_end
        internal_end = surf_end + n_internal

        surf_vertex_ranges.append((surf_start, surf_end))
        particle_vertex_ranges.append((internal_start, internal_end))

        vertex_offset += len(total_bag_pos)

    combined_positions = np.concatenate(all_positions, axis=0)
    combined_edges = np.concatenate(all_edges, axis=0)
    combined_inv_mass = np.concatenate(all_inv_mass, axis=0)
    combined_radii = np.concatenate(all_radii, axis=0)
    group_ids = np.concatenate(group_ids_list, axis=0)
    combined_tets = np.concatenate(all_tets, axis=0) if len(all_tets) > 0 else np.zeros((0, 4), dtype=np.int64)

    combined_body = SoftBody(
        positions=combined_positions,
        velocities=np.zeros_like(combined_positions),
        inv_mass=combined_inv_mass,
        tets=combined_tets,
        edges=combined_edges,
    )

    constraints = [
        # 1. Distance constraints for surface mesh (rigid compliance 0.0 for smooth surface)
        DistanceConstraint.from_positions(
            combined_body.positions,
            combined_body.edges,
            compliance=args.edge_compliance,
            relaxation=args.relaxation,
        ),
        # 2. Global volume constraint for surface meshes
        GlobalVolumeConstraint.from_positions(
            combined_body.positions,
            faces_list,
            compliance=args.volume_compliance,
            relaxation=args.relaxation,
            inflation_factor=args.inflation_factor,
        ),
    ]

    if len(combined_tets) > 0:
        constraints.append(
            VolumeConstraint.from_positions(
                combined_body.positions,
                combined_tets,
                compliance=args.volume_compliance,
                relaxation=args.relaxation,
            )
        )

    constraints.extend([
        # 3. Particle collisions with watertight radii
        ParticleCollisionConstraint(
            len(combined_body.positions),
            radius=combined_radii,
            edges=combined_body.edges,
            group_ids=group_ids,
            self_collision=not args.no_self_collision,
        ),
        # 4. Box container boundary
        BoxConstraint(len(combined_body.positions), bounds_min, bounds_max),
    ])

    solver = Solver(
        combined_body, constraints, n_substeps=args.substeps, mode=args.mode
    )
    return solver, faces_list, surf_vertex_ranges, particle_vertex_ranges





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
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    ps.register_curve_network(
        "container", corners, box_edges, radius=0.01, color=(0.1, 0.1, 0.1)
    )


def main() -> None:
    args = parse_args()

    bounds_min = (-1.0, 0.0, -1.0)
    bounds_max = (1.0, 3.0, 1.0)

    solver, faces_list, surf_vertex_ranges, particle_vertex_ranges = build_solver(
        args, bounds_min, bounds_max
    )
    body = solver.body

    # Backup initial positions & velocities for instant reset
    is_running = True
    push_radius = 0.35
    grabbed_idx = None
    grabbed_depth = 2.0
    interaction_active = False  # toggled only on Ctrl press/release edges, not every frame

    def reset_simulation() -> None:
        nonlocal is_running, grabbed_idx
        grabbed_idx = None
        body.positions[:] = init_positions.copy()
        body.velocities[:] = init_velocities.copy()
        solver.reset()
        update_visuals()


    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_box_wireframe(bounds_min, bounds_max)

    ps_meshes = []
    ps_clouds = []
    colors = [
        (0.85, 0.35, 0.25),
        (0.25, 0.65, 0.85),
        (0.35, 0.75, 0.45),
        (0.85, 0.75, 0.25),
        (0.65, 0.35, 0.85),
        (0.85, 0.55, 0.25),
    ]

    for bag_idx, (faces, (s_start, s_end), (p_start, p_end)) in enumerate(
        zip(faces_list, surf_vertex_ranges, particle_vertex_ranges)
    ):
        bag_name = f"bag_{bag_idx}"
        local_faces = faces - s_start
        local_pos = body.positions[s_start:s_end]

        mesh = ps.register_surface_mesh(bag_name, local_pos, local_faces)
        color = colors[bag_idx % len(colors)]
        mesh.set_color(color)
        mesh.set_edge_width(1.0)
        mesh.set_smooth_shade(True)
        ps_meshes.append((mesh, s_start, s_end))

        if p_end > p_start:
            cloud_name = f"particles_{bag_idx}"
            cloud = ps.register_point_cloud(cloud_name, body.positions[p_start:p_end])
            cloud.set_color(color)
            cloud.set_radius(0.010)
            ps_clouds.append((cloud, p_start, p_end))

    def clamp_internal_particles() -> None:
        """Enforces that internal fluid particles strictly stay inside their corresponding bag's outer shell."""
        pos = body.positions
        for (s_start, s_end), (p_start, p_end) in zip(surf_vertex_ranges, particle_vertex_ranges):
            if p_end <= p_start:
                continue
            surf_pos = pos[s_start:s_end]
            centroid = np.mean(surf_pos, axis=0)
            r_avg = np.mean(np.linalg.norm(surf_pos - centroid, axis=1))
            r_max_allowed = r_avg * 0.85

            part_pos = pos[p_start:p_end]
            vecs = part_pos - centroid
            dists = np.linalg.norm(vecs, axis=1)
            too_far = dists > r_max_allowed
            if np.any(too_far):
                scale = (r_max_allowed / np.maximum(dists[too_far], 1e-12))[:, None]
                pos[p_start:p_end][too_far] = centroid + vecs[too_far] * scale

    def handle_direct_mouse_drag(io, interaction_active: bool) -> None:
        nonlocal grabbed_idx, grabbed_depth

        is_down = interaction_active and (bool(io.MouseDown[0]) or bool(io.MouseDown[1]))

        if not interaction_active and grabbed_idx is None:
            return

        try:
            m_pos = io.MousePos
            px = float(m_pos.x) if hasattr(m_pos, "x") else float(m_pos[0])
            py = float(m_pos.y) if hasattr(m_pos, "y") else float(m_pos[1])

            # screen_coords_to_world_ray returns only the ray DIRECTION (3-vector);
            # the origin is the camera eye position, constant for every pixel this frame.
            ray_d = ps.screen_coords_to_world_ray((px, py))
            ray_o = ps.get_view_camera_parameters().get_position()
            if ray_o is None or ray_d is None or np.any(np.isnan(ray_o)) or np.any(np.isnan(ray_d)):
                return

            norm_d = np.linalg.norm(ray_d)
            if norm_d < 1e-6:
                return
            ray_d = ray_d / norm_d

            pos = body.positions

            if is_down:
                if grabbed_idx is None:
                    # Find nearest surface vertex to ray
                    diff = pos - ray_o
                    t_proj = np.maximum(0.0, np.dot(diff, ray_d))
                    closest_pts = ray_o + t_proj[:, None] * ray_d
                    dists = np.linalg.norm(pos - closest_pts, axis=1)
                    min_idx = np.argmin(dists)

                    if dists[min_idx] < 1.5:
                        grabbed_idx = int(min_idx)
                        grabbed_depth = float(t_proj[min_idx])
                else:
                    # Directly move grabbed vertex with mouse ray
                    target_pos = ray_o + ray_d * grabbed_depth
                    vel = (target_pos - pos[grabbed_idx]) / DT
                    body.velocities[grabbed_idx] = vel
                    pos[grabbed_idx] = target_pos

                    # Also repulse surrounding vertices within push_radius
                    vecs = pos - target_pos
                    dists_to_target = np.linalg.norm(vecs, axis=1)
                    inside = dists_to_target < push_radius
                    for idx in np.where(inside)[0]:
                        if idx != grabbed_idx:
                            d = dists_to_target[idx]
                            dir_vec = np.array([0.0, 1.0, 0.0]) if d < 1e-6 else vecs[idx] / d
                            p_target = target_pos + dir_vec * push_radius
                            body.velocities[idx] = (p_target - pos[idx]) / DT
                            pos[idx] = p_target

                    # Visual feedback marker
                    marker = ps.register_point_cloud("mouse_grab_marker", np.array([target_pos]))
                    marker.set_color((1.0, 0.3, 0.1))
                    marker.set_radius(0.08)
            else:
                if grabbed_idx is not None:
                    grabbed_idx = None
                    ps.remove_point_cloud("mouse_grab_marker", error_if_absent=False)
        except Exception as e:
            print(f"[mouse drag] error: {e!r}")

    def step_simulation() -> None:
        solver.step(DT)
        clamp_internal_particles()
        update_visuals()

    def update_visuals() -> None:
        for mesh, s_start, s_end in ps_meshes:
            mesh.update_vertex_positions(body.positions[s_start:s_end])
        for cloud, p_start, p_end in ps_clouds:
            cloud.update_point_positions(body.positions[p_start:p_end])

    def frame() -> None:
        nonlocal is_running, interaction_active
        import polyscope.imgui as imgui

        io = imgui.GetIO()

        # Ctrl (or Cmd on macOS, where the physical Ctrl key doesn't reach this app) held
        # = physics grab mode, with camera navigation suspended.
        # Toggled only on the press/release edge so a drag never straddles both modes.
        modifier_down = bool(
            imgui.IsKeyDown(imgui.ImGuiKey_LeftCtrl)
            or imgui.IsKeyDown(imgui.ImGuiKey_RightCtrl)
            or imgui.IsKeyDown(imgui.ImGuiKey_LeftSuper)
            or imgui.IsKeyDown(imgui.ImGuiKey_RightSuper)
        )
        if modifier_down != interaction_active:
            interaction_active = modifier_down
            ps.set_do_default_mouse_interaction(not interaction_active)

        handle_direct_mouse_drag(io, interaction_active)

        if imgui.Button("Reset / Restart"):
            reset_simulation()
        imgui.SameLine()
        if imgui.Button("Pause" if is_running else "Play"):
            is_running = not is_running

        if is_running:
            step_simulation()

    ps.set_user_callback(frame)
    ps.show()
















if __name__ == "__main__":
    main()
