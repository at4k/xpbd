"""Cell division (cytokinesis) simulation demonstrating spindle elongation,

cleavage furrow constriction, watertight topological splitting into two
independent SoftBodies, and post-division tangential surface relaxation.
"""

from enum import Enum
import argparse

import numpy as np
import polyscope as ps

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    BoxConstraint,
    ContractileRingConstraint,
    DistanceConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
    SpindleConstraint,
)
from xpbd.mesh import edges_from_faces, icosphere_mesh
from xpbd.topology import (
    partition_particles,
    slice_and_cap_mesh,
    tangential_surface_relaxation,
)

DT = 1.0 / 60.0


class CellDivisionState(Enum):
    ELONGATING = "ELONGATING"
    CONSTRICTING = "CONSTRICTING"
    DIVIDING = "DIVIDING"
    RELAXING = "RELAXING"
    INDEPENDENT = "INDEPENDENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--bag-radius", type=float, default=0.5)
    parser.add_argument("--internal-particles", type=int, default=80)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--mode", choices=["jacobi", "gauss_seidel"], default="jacobi")
    parser.add_argument("--transparency", type=float, default=0.45)
    parser.add_argument("--point-radius", type=float, default=0.015)
    parser.add_argument("--headless", action="store_true", help="Run simulation headlessly")
    parser.add_argument("--headless-steps", type=int, default=300, help="Number of steps in headless mode")
    return parser.parse_args()


def generate_internal_particles(origin, radius, num_particles):
    if num_particles <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    r_max = radius * 0.78
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


class CellDivisionSim:
    """Manages the full lifecycle of cell division across XPBD constraints and bodies."""

    def __init__(self, args, bounds_min, bounds_max):
        self.args = args
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.origin = np.array([0.0, 1.5, 0.0], dtype=np.float64)
        self.r_cell = args.bag_radius

        self.push_speed = 0.25
        self.constrict_speed = 0.20
        self.relax_factor = 0.15
        self.max_elongation = self.r_cell * 1.4
        self.constrict_thresh = self.r_cell * 0.12

        self.reset()

    def reset(self):
        self.state = CellDivisionState.ELONGATING
        self.relax_step_count = 0
        self.is_divided = False

        # 1. Mother cell surface mesh
        surf_pos, self.faces = icosphere_mesh(
            subdivisions=self.args.subdivisions, radius=self.r_cell, origin=self.origin
        )
        self.n_surf = len(surf_pos)
        self.edges = edges_from_faces(self.faces)

        # 2. Internal fluid particles
        internal_pos = generate_internal_particles(
            self.origin, self.r_cell, self.args.internal_particles
        )
        self.n_internal = len(internal_pos)

        # 3. Two spindle poles
        pole_dist_init = self.r_cell * 0.25
        poles_pos = np.array(
            [
                [-pole_dist_init, self.origin[1], self.origin[2]],
                [pole_dist_init, self.origin[1], self.origin[2]],
            ],
            dtype=np.float64,
        )
        self.pole_idx_0 = self.n_surf + self.n_internal
        self.pole_idx_1 = self.pole_idx_0 + 1

        if self.n_internal > 0:
            total_pos = np.concatenate([surf_pos, internal_pos, poles_pos], axis=0)
        else:
            total_pos = np.concatenate([surf_pos, poles_pos], axis=0)

        # Masses
        inv_mass = np.ones(len(total_pos), dtype=np.float64)
        if self.n_internal > 0:
            inv_mass[self.n_surf : self.pole_idx_0] = 2.0
        inv_mass[self.pole_idx_0 :] = 0.5  # Heavy poles

        # Radii for collision
        radii = np.full(len(total_pos), 0.04, dtype=np.float64)
        radii[: self.n_surf] = 0.03
        radii[self.pole_idx_0 :] = 0.08

        self.mother_body = SoftBody(
            positions=total_pos,
            velocities=np.zeros_like(total_pos),
            inv_mass=inv_mass,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=self.edges,
        )

        # Spindle & astral tethers
        left_mask = surf_pos[:, 0] < self.origin[0]
        right_mask = surf_pos[:, 0] >= self.origin[0]
        left_idx = np.where(left_mask)[0]
        right_idx = np.where(right_mask)[0]

        self.spindle_con = SpindleConstraint(
            self.pole_idx_0,
            self.pole_idx_1,
            left_membrane_idx=left_idx,
            right_membrane_idx=right_idx,
            pole_rest_length=pole_dist_init * 2.0,
            tether_rest_length=self.r_cell * 0.70,
            tether_compliance=1e-3,
        )

        # Equatorial contractile ring
        equator_mask = np.abs(surf_pos[:, 0] - self.origin[0]) < 0.15
        self.ring_idx = np.where(equator_mask)[0]
        self.ring_con = ContractileRingConstraint(
            self.ring_idx,
            center_y=self.origin[1],
            center_z=self.origin[2],
            target_radius=self.r_cell,
            compliance=0.0,
        )

        self.dist_con = DistanceConstraint.from_positions(
            self.mother_body.positions, self.mother_body.edges, compliance=0.0
        )
        self.vol_con = GlobalVolumeConstraint.from_positions(
            self.mother_body.positions, [self.faces], compliance=0.0
        )
        self.initial_volume = float(self.vol_con.rest_volumes[0])

        self.coll_con = ParticleCollisionConstraint(
            len(self.mother_body.positions),
            radius=radii,
            edges=self.mother_body.edges,
            self_collision=True,
        )
        self.box_con = BoxConstraint(
            len(self.mother_body.positions), self.bounds_min, self.bounds_max
        )

        constraints = [
            self.dist_con,
            self.vol_con,
            self.spindle_con,
            self.ring_con,
            self.coll_con,
            self.box_con,
        ]

        self.mother_solver = Solver(
            self.mother_body,
            constraints,
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        # Daughter bodies (initially None)
        self.body_left: SoftBody | None = None
        self.body_right: SoftBody | None = None
        self.faces_left: np.ndarray | None = None
        self.faces_right: np.ndarray | None = None
        self.solver_left: Solver | None = None
        self.solver_right: Solver | None = None
        self.dist_con_l: DistanceConstraint | None = None
        self.dist_con_r: DistanceConstraint | None = None

    def execute_division(self):
        """Topological separation into two independent SoftBodies."""
        cur_surf = self.mother_body.positions[: self.n_surf]

        # Slicing plane: midpoint between poles
        p0_x = self.mother_body.positions[self.pole_idx_0, 0]
        p1_x = self.mother_body.positions[self.pole_idx_1, 0]
        cut_x = float((p0_x + p1_x) / 2.0)

        # Slicing and capping
        (pos_l, self.faces_left), (pos_r, self.faces_right) = slice_and_cap_mesh(
            cur_surf, self.faces, cut_x=cut_x
        )
        n_surf_l = len(pos_l)
        n_surf_r = len(pos_r)

        # Partition internal particles
        if self.n_internal > 0:
            int_pos = self.mother_body.positions[self.n_surf : self.pole_idx_0]
            l_int_idx, r_int_idx = partition_particles(int_pos, cut_x=cut_x)
            int_l = int_pos[l_int_idx]
            int_r = int_pos[r_int_idx]
        else:
            int_l = np.zeros((0, 3), dtype=np.float64)
            int_r = np.zeros((0, 3), dtype=np.float64)

        # Assign poles to respective daughters
        pole_l = self.mother_body.positions[self.pole_idx_0 : self.pole_idx_0 + 1]
        pole_r = self.mother_body.positions[self.pole_idx_1 : self.pole_idx_1 + 1]

        total_l = np.concatenate([pos_l, int_l, pole_l], axis=0)
        total_r = np.concatenate([pos_r, int_r, pole_r], axis=0)

        edges_l = edges_from_faces(self.faces_left)
        edges_r = edges_from_faces(self.faces_right)

        inv_m_l = np.ones(len(total_l), dtype=np.float64)
        if len(int_l) > 0:
            inv_m_l[n_surf_l : n_surf_l + len(int_l)] = 2.0
        inv_m_l[-1] = 0.5

        inv_m_r = np.ones(len(total_r), dtype=np.float64)
        if len(int_r) > 0:
            inv_m_r[n_surf_r : n_surf_r + len(int_r)] = 2.0
        inv_m_r[-1] = 0.5

        self.body_left = SoftBody(
            positions=total_l,
            velocities=np.zeros_like(total_l),
            inv_mass=inv_m_l,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=edges_l,
        )
        self.body_right = SoftBody(
            positions=total_r,
            velocities=np.zeros_like(total_r),
            inv_mass=inv_m_r,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=edges_r,
        )

        daughter_target_vol = self.initial_volume * 0.5
        self.dist_con_l = DistanceConstraint.from_positions(
            self.body_left.positions, edges_l, compliance=0.0
        )
        self.dist_con_r = DistanceConstraint.from_positions(
            self.body_right.positions, edges_r, compliance=0.0
        )

        vol_con_l = GlobalVolumeConstraint(
            [self.faces_left], [daughter_target_vol], compliance=0.0
        )
        vol_con_r = GlobalVolumeConstraint(
            [self.faces_right], [daughter_target_vol], compliance=0.0
        )

        box_l = BoxConstraint(len(total_l), self.bounds_min, self.bounds_max)
        box_r = BoxConstraint(len(total_r), self.bounds_min, self.bounds_max)

        self.solver_left = Solver(
            self.body_left,
            [self.dist_con_l, vol_con_l, box_l],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )
        self.solver_right = Solver(
            self.body_right,
            [self.dist_con_r, vol_con_r, box_r],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        self.n_surf_l = n_surf_l
        self.n_surf_r = n_surf_r
        self.n_int_l = len(int_l)
        self.n_int_r = len(int_r)
        self.is_divided = True
        self.state = CellDivisionState.RELAXING

    def step(self):
        """Advances the simulation by one frame DT."""
        if not self.is_divided:
            # Phase 1: Elongating
            if self.state == CellDivisionState.ELONGATING:
                new_len = self.spindle_con.pole_rest_length + self.push_speed * DT
                self.spindle_con.set_target_length(new_len)
                cur_dist = (
                    self.mother_body.positions[self.pole_idx_1, 0]
                    - self.mother_body.positions[self.pole_idx_0, 0]
                )
                if cur_dist >= self.max_elongation:
                    self.state = CellDivisionState.CONSTRICTING

            # Phase 2: Constricting
            elif self.state == CellDivisionState.CONSTRICTING:
                new_r = max(
                    0.03, self.ring_con.target_radius - self.constrict_speed * DT
                )
                self.ring_con.set_target_radius(new_r)

                dy = self.mother_body.positions[self.ring_idx, 1] - self.origin[1]
                dz = self.mother_body.positions[self.ring_idx, 2] - self.origin[2]
                neck_r = float(np.min(np.sqrt(dy**2 + dz**2)))
                if neck_r <= self.constrict_thresh or new_r <= 0.04:
                    self.state = CellDivisionState.DIVIDING

            # Phase 3: Dividing
            elif self.state == CellDivisionState.DIVIDING:
                self.execute_division()
                return

            self.mother_solver.step(DT)

        else:
            # Post-division
            if self.state == CellDivisionState.RELAXING:
                self.relax_step_count += 1
                # Tangential surface relaxation on daughter cell surfaces
                c_l = np.mean(self.body_left.positions[: self.n_surf_l], axis=0)
                rel_l = tangential_surface_relaxation(
                    self.body_left.positions[: self.n_surf_l],
                    self.faces_left,
                    center=c_l,
                    factor=self.relax_factor,
                    n_iters=2,
                )
                self.body_left.positions[: self.n_surf_l] = rel_l

                c_r = np.mean(self.body_right.positions[: self.n_surf_r], axis=0)
                rel_r = tangential_surface_relaxation(
                    self.body_right.positions[: self.n_surf_r],
                    self.faces_right,
                    center=c_r,
                    factor=self.relax_factor,
                    n_iters=2,
                )
                self.body_right.positions[: self.n_surf_r] = rel_r

                if self.relax_step_count >= 30:
                    # Update edge rest lengths to relaxed configuration
                    d_l = (
                        self.body_left.positions[self.body_left.edges[:, 1]]
                        - self.body_left.positions[self.body_left.edges[:, 0]]
                    )
                    self.dist_con_l.rest_lengths = np.linalg.norm(d_l, axis=1)

                    d_r = (
                        self.body_right.positions[self.body_right.edges[:, 1]]
                        - self.body_right.positions[self.body_right.edges[:, 0]]
                    )
                    self.dist_con_r.rest_lengths = np.linalg.norm(d_r, axis=1)

                    self.state = CellDivisionState.INDEPENDENT

            self.solver_left.step(DT)
            self.solver_right.step(DT)

            # Mutual collision separation between daughter cells
            c_l = np.mean(self.body_left.positions[: self.n_surf_l], axis=0)
            c_r = np.mean(self.body_right.positions[: self.n_surf_r], axis=0)
            diff = c_r - c_l
            dist = np.linalg.norm(diff)
            min_dist = self.r_cell * 1.05
            if dist < min_dist and dist > 1e-6:
                overlap = min_dist - dist
                n = diff / dist
                self.body_left.positions -= 0.5 * overlap * n
                self.body_right.positions += 0.5 * overlap * n


def run_headless(args):
    """Headless simulation runner for automated testing and benchmarking."""
    bounds_min = (-1.5, 0.0, -1.5)
    bounds_max = (1.5, 3.0, 1.5)
    sim = CellDivisionSim(args, bounds_min, bounds_max)

    for i in range(args.headless_steps):
        sim.step()

    assert sim.is_divided, "Simulation failed to reach division within step limit"
    assert sim.body_left is not None and sim.body_right is not None
    print(f"Headless simulation succeeded: {args.headless_steps} steps, state={sim.state.value}")


def register_box_wireframe(bounds_min, bounds_max):
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
        "container", corners, box_edges, radius=0.008, color=(0.15, 0.15, 0.15)
    )


def main():
    args = parse_args()

    if args.headless:
        run_headless(args)
        return

    bounds_min = (-1.5, 0.0, -1.5)
    bounds_max = (1.5, 3.0, 1.5)
    sim = CellDivisionSim(args, bounds_min, bounds_max)

    is_running = True
    transparency = args.transparency
    point_radius = args.point_radius
    show_surface = True
    show_points = True
    show_spindle = True

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_box_wireframe(bounds_min, bounds_max)

    # Initial mother cell meshes
    mesh_mother = ps.register_surface_mesh(
        "cell_membrane",
        sim.mother_body.positions[: sim.n_surf],
        sim.faces,
    )
    mesh_mother.set_color((0.25, 0.65, 0.85))
    mesh_mother.set_edge_width(1.0)
    mesh_mother.set_smooth_shade(True)
    mesh_mother.set_transparency(transparency)

    cloud_mother = None
    if sim.n_internal > 0:
        cloud_mother = ps.register_point_cloud(
            "internal_fluid",
            sim.mother_body.positions[sim.n_surf : sim.pole_idx_0],
        )
        cloud_mother.set_color((0.25, 0.65, 0.85))
        cloud_mother.set_radius(point_radius)

    poles_curve = ps.register_curve_network(
        "spindle_poles",
        sim.mother_body.positions[[sim.pole_idx_0, sim.pole_idx_1]],
        np.array([[0, 1]]),
        radius=0.02,
        color=(0.9, 0.25, 0.25),
    )

    mesh_l = None
    mesh_r = None
    cloud_l = None
    cloud_r = None

    def update_visuals():
        nonlocal mesh_l, mesh_r, cloud_l, cloud_r
        if not sim.is_divided:
            mesh_mother.update_vertex_positions(
                sim.mother_body.positions[: sim.n_surf]
            )
            if cloud_mother is not None:
                cloud_mother.update_point_positions(
                    sim.mother_body.positions[sim.n_surf : sim.pole_idx_0]
                )
            poles_curve.update_node_positions(
                sim.mother_body.positions[[sim.pole_idx_0, sim.pole_idx_1]]
            )
        else:
            # Hide mother visuals on first divided frame
            if mesh_mother.is_enabled():
                mesh_mother.set_enabled(False)
                if cloud_mother:
                    cloud_mother.set_enabled(False)
                poles_curve.set_enabled(False)

                # Register daughter meshes
                mesh_l = ps.register_surface_mesh(
                    "daughter_left",
                    sim.body_left.positions[: sim.n_surf_l],
                    sim.faces_left,
                )
                mesh_l.set_color((0.2, 0.75, 0.85))
                mesh_l.set_edge_width(1.0)
                mesh_l.set_smooth_shade(True)
                mesh_l.set_transparency(transparency)

                mesh_r = ps.register_surface_mesh(
                    "daughter_right",
                    sim.body_right.positions[: sim.n_surf_r],
                    sim.faces_right,
                )
                mesh_r.set_color((0.45, 0.85, 0.45))
                mesh_r.set_edge_width(1.0)
                mesh_r.set_smooth_shade(True)
                mesh_r.set_transparency(transparency)

                if sim.n_int_l > 0:
                    cloud_l = ps.register_point_cloud(
                        "fluid_left",
                        sim.body_left.positions[
                            sim.n_surf_l : sim.n_surf_l + sim.n_int_l
                        ],
                    )
                    cloud_l.set_color((0.2, 0.75, 0.85))
                    cloud_l.set_radius(point_radius)

                if sim.n_int_r > 0:
                    cloud_r = ps.register_point_cloud(
                        "fluid_right",
                        sim.body_right.positions[
                            sim.n_surf_r : sim.n_surf_r + sim.n_int_r
                        ],
                    )
                    cloud_r.set_color((0.45, 0.85, 0.45))
                    cloud_r.set_radius(point_radius)

            # Update daughter positions
            if mesh_l is not None:
                mesh_l.update_vertex_positions(
                    sim.body_left.positions[: sim.n_surf_l]
                )
            if mesh_r is not None:
                mesh_r.update_vertex_positions(
                    sim.body_right.positions[: sim.n_surf_r]
                )
            if cloud_l is not None:
                cloud_l.update_point_positions(
                    sim.body_left.positions[
                        sim.n_surf_l : sim.n_surf_l + sim.n_int_l
                    ]
                )
            if cloud_r is not None:
                cloud_r.update_point_positions(
                    sim.body_right.positions[
                        sim.n_surf_r : sim.n_surf_r + sim.n_int_r
                    ]
                )

    def frame():
        nonlocal is_running, transparency, point_radius, show_surface, show_points, show_spindle
        import polyscope.imgui as imgui

        imgui.Text(f"Phase: {sim.state.value}")
        if not sim.is_divided:
            p0 = sim.mother_body.positions[sim.pole_idx_0, 0]
            p1 = sim.mother_body.positions[sim.pole_idx_1, 0]
            imgui.Text(f"Spindle Distance: {p1 - p0:.3f}")
        else:
            imgui.Text(f"Relax Steps: {sim.relax_step_count}")

        if imgui.Button("Reset"):
            sim.reset()
            if mesh_mother:
                mesh_mother.set_enabled(True)
            if cloud_mother:
                cloud_mother.set_enabled(True)
            if poles_curve:
                poles_curve.set_enabled(True)
            if mesh_l:
                ps.remove_surface_mesh("daughter_left")
            if mesh_r:
                ps.remove_surface_mesh("daughter_right")
            if cloud_l:
                ps.remove_point_cloud("fluid_left")
            if cloud_r:
                ps.remove_point_cloud("fluid_right")
            mesh_l = mesh_r = cloud_l = cloud_r = None

        imgui.SameLine()
        if imgui.Button("Pause" if is_running else "Play"):
            is_running = not is_running

        imgui.SameLine()
        if imgui.Button("Force Divide") and not sim.is_divided:
            sim.execute_division()

        imgui.SeparatorText("Division Controls")
        _, sim.push_speed = imgui.SliderFloat("Push Speed", sim.push_speed, 0.0, 1.0)
        _, sim.constrict_speed = imgui.SliderFloat(
            "Constriction Speed", sim.constrict_speed, 0.0, 1.0
        )
        _, sim.relax_factor = imgui.SliderFloat(
            "Relaxation Factor", sim.relax_factor, 0.0, 0.5
        )

        imgui.SeparatorText("Visualization")
        changed_trans, new_trans = imgui.SliderFloat(
            "Surface Transparency", transparency, 0.0, 1.0
        )
        if changed_trans:
            transparency = new_trans
            if mesh_mother:
                mesh_mother.set_transparency(transparency)
            if mesh_l:
                mesh_l.set_transparency(transparency)
            if mesh_r:
                mesh_r.set_transparency(transparency)

        if is_running:
            sim.step()

        update_visuals()

    ps.set_user_callback(frame)
    ps.show()


if __name__ == "__main__":
    main()
