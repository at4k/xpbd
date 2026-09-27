"""Autonomous cell division (cytokinesis) simulation in XPBD.

Physical principles:
1. Membrane surface tension via DistanceConstraint.
2. Incompressible cytoplasm fluid pressure via GlobalVolumeConstraint.
3. Mitotic spindle elongation via SpindleConstraint.
4. Contractile ring purse-string constriction via ContractileRingConstraint on an exact planar ring.
5. Internal cytoplasm fluid particles via ParticleFluidConstraint (pairwise repulsion + mesh confinement).
6. Post-division transition to pristine daughter Icospheres with preserved volume and surface tension.
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
    ParticleFluidConstraint,
    SpindleConstraint,
)
from xpbd.mesh import edges_from_faces, icosphere_mesh
from xpbd.topology import insert_equatorial_ring

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
    parser.add_argument("--internal-particles", type=int, default=160)
    parser.add_argument("--substeps", type=int, default=15)
    parser.add_argument("--mode", choices=["jacobi", "gauss_seidel"], default="gauss_seidel")
    parser.add_argument("--transparency", type=float, default=0.45)
    parser.add_argument("--point-radius", type=float, default=0.012)
    parser.add_argument(
        "--headless", action="store_true", help="Run simulation headlessly"
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=300,
        help="Number of steps in headless mode",
    )
    return parser.parse_args()


def generate_internal_particles(origin, radius, num_particles):
    """Generates internal cytoplasm fluid particles uniformly inside the mother cell."""
    if num_particles <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    r_max = radius * 0.75
    np.random.seed(42)

    # Uniform random distribution inside sphere
    u = np.random.normal(size=(num_particles, 3))
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    radii = (np.random.uniform(0.1, 1.0, size=(num_particles, 1)) ** (1.0 / 3.0)) * r_max
    pts = np.array(origin, dtype=np.float64) + u * radii

    # Initial repulsion relaxation to prevent clumping
    fluid = ParticleFluidConstraint(particle_radius=0.035)
    for _ in range(25):
        fluid.project_repulsion(pts)
        d = np.linalg.norm(pts - origin, axis=1)
        over = d > r_max
        if np.any(over):
            pts[over] = np.array(origin) + (pts[over] - origin) / d[over, None] * r_max

    return pts


class CellDivisionSim:
    """Manages the full lifecycle of cell division across XPBD constraints and bodies."""

    def __init__(self, args, bounds_min, bounds_max):
        self.args = args
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.origin = np.array([0.0, 1.5, 0.0], dtype=np.float64)
        self.r_cell = args.bag_radius
        self.r_daughter = self.r_cell / (2.0 ** (1.0 / 3.0))

        # Speed parameters
        self.push_speed = 0.85
        self.constrict_speed = 0.75
        self.spindle_radius = 0.030

        self.reset()

    def reset(self):
        """Rewinds the simulation to the initial time t = 0."""
        self.state = CellDivisionState.ELONGATING
        self.is_divided = False
        self.time = 0.0
        self.elong_prog = 0.0
        self.furrow_prog = 0.0

        # 1. Base mesh with exact planar equatorial ring
        base_pos, base_faces = icosphere_mesh(
            subdivisions=self.args.subdivisions, radius=self.r_cell, origin=self.origin
        )
        self.surf_pos, self.faces, self.ring_idx = insert_equatorial_ring(
            base_pos, base_faces, plane_x=float(self.origin[0])
        )
        self.n_surf = len(self.surf_pos)
        self.edges = edges_from_faces(self.faces)

        # 2. Spindle poles
        self.d_init = self.r_daughter * 0.20
        self.d_max = self.r_daughter * 0.55
        self.spindle_dist = self.d_init

        p0 = np.array([self.origin[0] - self.d_init, self.origin[1], self.origin[2]], dtype=np.float64)
        p1 = np.array([self.origin[0] + self.d_init, self.origin[1], self.origin[2]], dtype=np.float64)
        self.p0_idx = self.n_surf
        self.p1_idx = self.n_surf + 1

        # 3. Internal cytoplasm particles
        self.int_pos = generate_internal_particles(
            self.origin, self.r_cell, self.args.internal_particles
        )
        self.n_internal = len(self.int_pos)

        total_pos = np.concatenate([self.surf_pos, [p0, p1], self.int_pos], axis=0)
        inv_mass = np.ones(len(total_pos), dtype=np.float64)
        # Spindle poles have mass 1.0, internal particles moved via fluid constraint
        inv_mass[self.n_surf + 2 :] = 0.0

        self.mother_body = SoftBody(
            positions=total_pos,
            velocities=np.zeros_like(total_pos),
            inv_mass=inv_mass,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=self.edges,
        )

        # 4. XPBD Constraints
        self.dist_con = DistanceConstraint.from_positions(
            self.mother_body.positions, self.edges, compliance=1e-3
        )
        self.vol_con = GlobalVolumeConstraint.from_positions(
            self.mother_body.positions, [self.faces], compliance=0.0
        )
        self.initial_volume = float(self.vol_con.rest_volumes[0])

        left_surf = np.where(self.surf_pos[:, 0] < self.origin[0] - 0.15)[0]
        right_surf = np.where(self.surf_pos[:, 0] > self.origin[0] + 0.15)[0]

        self.spindle_con = SpindleConstraint(
            self.p0_idx,
            self.p1_idx,
            left_surf,
            right_surf,
            pole_rest_length=2.0 * self.d_init,
            pole_compliance=0.0,
            tether_rest_length=self.r_cell * 0.70,
            tether_compliance=1e-3,
        )

        self.ring_con = ContractileRingConstraint(
            self.ring_idx,
            center_x=float(self.origin[0]),
            center_y=float(self.origin[1]),
            center_z=float(self.origin[2]),
            target_radius=float(self.r_cell),
            compliance=0.0,
        )

        self.particle_con = ParticleFluidConstraint(particle_radius=0.035)

        self.mother_solver = Solver(
            self.mother_body,
            [self.dist_con, self.vol_con, self.spindle_con, self.ring_con],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        # Daughter bodies
        self.body_left: SoftBody | None = None
        self.body_right: SoftBody | None = None
        self.faces_left: np.ndarray | None = None
        self.faces_right: np.ndarray | None = None
        self.solver_left: Solver | None = None
        self.solver_right: Solver | None = None
        self.dist_con_l: DistanceConstraint | None = None
        self.dist_con_r: DistanceConstraint | None = None
        self.vol_con_l: GlobalVolumeConstraint | None = None
        self.vol_con_r: GlobalVolumeConstraint | None = None
        self.n_surf_l = 0
        self.n_surf_r = 0
        self.n_int_l = 0
        self.n_int_r = 0

    @property
    def pole_0(self):
        if not self.is_divided:
            return self.mother_body.positions[self.p0_idx]
        return np.array([self.origin[0] - self.d_max, self.origin[1], self.origin[2]])

    @property
    def pole_1(self):
        if not self.is_divided:
            return self.mother_body.positions[self.p1_idx]
        return np.array([self.origin[0] + self.d_max, self.origin[1], self.origin[2]])

    def execute_division(self):
        """Topological transition to two pristine Icosphere daughter cells."""
        v_daughter = self.initial_volume / 2.0
        r_daughter = (v_daughter / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)

        # Center positions: placing centers so daughters touch at x = origin[0] with physiological contact
        contact_offset = r_daughter * 0.85
        center_l = np.array([self.origin[0] - contact_offset, self.origin[1], self.origin[2]])
        center_r = np.array([self.origin[0] + contact_offset, self.origin[1], self.origin[2]])

        pos_l, self.faces_left = icosphere_mesh(
            subdivisions=self.args.subdivisions, radius=r_daughter, origin=center_l
        )
        pos_r, self.faces_right = icosphere_mesh(
            subdivisions=self.args.subdivisions, radius=r_daughter, origin=center_r
        )

        # Pristine spherical positions for rest geometry and target volume V0 / 2
        pristine_pos_l = pos_l.copy()
        pristine_pos_r = pos_r.copy()

        # Pre-clamp contact septum so initial simulation geometry starts flush without velocity jumps
        pos_l[:, 0] = np.minimum(pos_l[:, 0], self.origin[0])
        pos_r[:, 0] = np.maximum(pos_r[:, 0], self.origin[0])

        self.n_surf_l = len(pos_l)
        self.n_surf_r = len(pos_r)

        # Partition internal particles
        pts = self.mother_body.positions[self.n_surf + 2 :]
        is_left = pts[:, 0] < self.origin[0]
        pts_l = pts[is_left].copy()
        pts_r = pts[~is_left].copy()
        self.n_int_l = len(pts_l)
        self.n_int_r = len(pts_r)

        total_l = np.concatenate([pos_l, pts_l], axis=0) if self.n_int_l > 0 else pos_l
        total_r = np.concatenate([pos_r, pts_r], axis=0) if self.n_int_r > 0 else pos_r

        inv_m_l = np.ones(len(total_l), dtype=np.float64)
        inv_m_r = np.ones(len(total_r), dtype=np.float64)
        if self.n_int_l > 0:
            inv_m_l[self.n_surf_l :] = 0.0
        if self.n_int_r > 0:
            inv_m_r[self.n_surf_r :] = 0.0

        # Construct daughter SoftBodies
        edges_l = edges_from_faces(self.faces_left)
        edges_r = edges_from_faces(self.faces_right)

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

        self.dist_con_l = DistanceConstraint.from_positions(
            pristine_pos_l, edges_l, compliance=1e-3
        )
        self.dist_con_r = DistanceConstraint.from_positions(
            pristine_pos_r, edges_r, compliance=1e-3
        )

        self.vol_con_l = GlobalVolumeConstraint.from_positions(
            pristine_pos_l, [self.faces_left], compliance=0.0
        )
        self.vol_con_r = GlobalVolumeConstraint.from_positions(
            pristine_pos_r, [self.faces_right], compliance=0.0
        )

        self.solver_left = Solver(
            self.body_left,
            [self.dist_con_l, self.vol_con_l],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )
        self.solver_right = Solver(
            self.body_right,
            [self.dist_con_r, self.vol_con_r],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        self.is_divided = True
        self.state = CellDivisionState.INDEPENDENT

    def step(self):
        """Advances the simulation by one frame DT under XPBD physics."""
        self.time += DT

        if not self.is_divided:
            # Phase 1: Spindle Elongation
            if self.state == CellDivisionState.ELONGATING:
                self.elong_prog += self.push_speed * DT
                if self.elong_prog >= 1.0:
                    self.elong_prog = 1.0
                    self.state = CellDivisionState.CONSTRICTING

                self.spindle_dist = self.d_init + (self.d_max - self.d_init) * self.elong_prog
                self.spindle_con.set_target_length(2.0 * self.spindle_dist)
                self.mother_solver.step(DT)

                # Update internal particles
                pts = self.mother_body.positions[self.n_surf + 2 :]
                self.particle_con.project_repulsion(pts)
                self.particle_con.project_confinement_mesh(
                    pts, self.mother_body.positions[: self.n_surf], faces=self.faces, margin=0.04
                )

            # Phase 2: Cleavage Furrow Constriction
            elif self.state == CellDivisionState.CONSTRICTING:
                self.furrow_prog += self.constrict_speed * DT
                target_r = max(0.035, self.r_cell * (1.0 - min(1.0, self.furrow_prog) * 0.93))
                self.ring_con.set_target_radius(target_r)
                self.mother_solver.step(DT)

                pts = self.mother_body.positions[self.n_surf + 2 :]
                self.particle_con.project_repulsion(pts)
                self.particle_con.project_confinement_mesh(
                    pts, self.mother_body.positions[: self.n_surf], faces=self.faces, margin=0.04
                )

                # Check if ring constricted to division threshold
                ring_radii = np.sqrt(
                    (self.mother_body.positions[self.ring_idx, 1] - self.origin[1]) ** 2
                    + (self.mother_body.positions[self.ring_idx, 2] - self.origin[2]) ** 2
                )
                if np.mean(ring_radii) <= 0.05 * self.r_cell or self.furrow_prog >= 1.0:
                    self.state = CellDivisionState.DIVIDING

            # Phase 3: Division Trigger
            elif self.state == CellDivisionState.DIVIDING:
                self.execute_division()

        else:
            # Phase 4: Independent daughter cell dynamics
            self.solver_left.step(DT)
            self.solver_right.step(DT)

            # Septum non-penetration contact
            self.body_left.positions[:, 0] = np.minimum(
                self.body_left.positions[:, 0], self.origin[0]
            )
            self.body_right.positions[:, 0] = np.maximum(
                self.body_right.positions[:, 0], self.origin[0]
            )

            # Internal particles in daughters
            if self.n_int_l > 0:
                pts_l = self.body_left.positions[self.n_surf_l :]
                self.particle_con.project_repulsion(pts_l)
                self.particle_con.project_confinement_mesh(
                    pts_l, self.body_left.positions[: self.n_surf_l], faces=self.faces_left, margin=0.04
                )
            if self.n_int_r > 0:
                pts_r = self.body_right.positions[self.n_surf_r :]
                self.particle_con.project_repulsion(pts_r)
                self.particle_con.project_confinement_mesh(
                    pts_r, self.body_right.positions[: self.n_surf_r], faces=self.faces_right, margin=0.04
                )

    @property
    def int_l(self):
        if self.body_left is not None and self.n_int_l > 0:
            return self.body_left.positions[self.n_surf_l :]
        return np.zeros((0, 3), dtype=np.float64)

    @property
    def int_r(self):
        if self.body_right is not None and self.n_int_r > 0:
            return self.body_right.positions[self.n_surf_r :]
        return np.zeros((0, 3), dtype=np.float64)


def run_headless(args):
    """Headless simulation runner for automated testing and benchmarking."""
    bounds_min = (-1.5, 0.0, -1.5)
    bounds_max = (1.5, 3.0, 1.5)
    sim = CellDivisionSim(args, bounds_min, bounds_max)

    for _ in range(args.headless_steps):
        sim.step()

    assert sim.is_divided, "Simulation failed to reach division within step limit"
    assert sim.body_left is not None and sim.body_right is not None
    assert len(sim.int_l) + len(sim.int_r) == args.internal_particles
    print(
        f"Headless simulation succeeded: {args.headless_steps} steps, state={sim.state.value}"
    )


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

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("y_up")
    register_box_wireframe(bounds_min, bounds_max)

    # Initial mother cell visuals
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
            sim.mother_body.positions[sim.n_surf + 2 :],
        )
        cloud_mother.set_color((0.25, 0.70, 0.90))
        cloud_mother.set_radius(point_radius)

    # Central mitotic spindle rod connecting centrosomes
    spindle_curve = ps.register_curve_network(
        "spindle_apparatus",
        np.array([sim.pole_0, sim.pole_1]),
        np.array([[0, 1]]),
        radius=sim.spindle_radius,
        color=(0.95, 0.25, 0.25),
    )

    # Centrosomes (spindle poles)
    centrosomes = ps.register_point_cloud(
        "centrosomes",
        np.array([sim.pole_0, sim.pole_1]),
        radius=sim.spindle_radius * 1.8,
        color=(1.0, 0.85, 0.20),
    )

    mesh_l = None
    mesh_r = None
    cloud_l = None
    cloud_r = None

    def update_visuals():
        nonlocal mesh_l, mesh_r, cloud_l, cloud_r
        if not sim.is_divided:
            mesh_mother.set_enabled(True)
            if cloud_mother:
                cloud_mother.set_enabled(True)
            spindle_curve.set_enabled(True)
            centrosomes.set_enabled(True)

            if mesh_l is not None:
                mesh_l.set_enabled(False)
            if mesh_r is not None:
                mesh_r.set_enabled(False)
            if cloud_l is not None:
                cloud_l.set_enabled(False)
            if cloud_r is not None:
                cloud_r.set_enabled(False)

            mesh_mother.update_vertex_positions(
                sim.mother_body.positions[: sim.n_surf]
            )
            if cloud_mother is not None:
                cloud_mother.update_point_positions(
                    sim.mother_body.positions[sim.n_surf + 2 :]
                )
            spindle_curve.update_node_positions(
                np.array([sim.pole_0, sim.pole_1])
            )
            centrosomes.update_point_positions(
                np.array([sim.pole_0, sim.pole_1])
            )
        else:
            if mesh_mother.is_enabled():
                mesh_mother.set_enabled(False)
                if cloud_mother:
                    cloud_mother.set_enabled(False)
                spindle_curve.set_enabled(False)
                centrosomes.set_enabled(False)

                mesh_l = ps.register_surface_mesh(
                    "daughter_left",
                    sim.body_left.positions,
                    sim.faces_left,
                )
                mesh_l.set_color((0.20, 0.75, 0.85))
                mesh_l.set_edge_width(1.0)
                mesh_l.set_smooth_shade(True)
                mesh_l.set_transparency(transparency)

                mesh_r = ps.register_surface_mesh(
                    "daughter_right",
                    sim.body_right.positions,
                    sim.faces_right,
                )
                mesh_r.set_color((0.45, 0.85, 0.45))
                mesh_r.set_edge_width(1.0)
                mesh_r.set_smooth_shade(True)
                mesh_r.set_transparency(transparency)

                if len(sim.int_l) > 0:
                    cloud_l = ps.register_point_cloud(
                        "fluid_left",
                        sim.int_l,
                    )
                    cloud_l.set_color((0.20, 0.75, 0.85))
                    cloud_l.set_radius(point_radius)

                if len(sim.int_r) > 0:
                    cloud_r = ps.register_point_cloud(
                        "fluid_right",
                        sim.int_r,
                    )
                    cloud_r.set_color((0.45, 0.85, 0.45))
                    cloud_r.set_radius(point_radius)

            if mesh_l is not None:
                mesh_l.update_vertex_positions(sim.body_left.positions)
            if mesh_r is not None:
                mesh_r.update_vertex_positions(sim.body_right.positions)
            if cloud_l is not None:
                cloud_l.update_point_positions(sim.int_l)
            if cloud_r is not None:
                cloud_r.update_point_positions(sim.int_r)

    def frame():
        nonlocal is_running, transparency, point_radius, mesh_l, mesh_r, cloud_l, cloud_r
        import polyscope.imgui as imgui

        if imgui.Button("最初に戻る (Rewind to Start)"):
            sim.reset()
            update_visuals()

        imgui.SameLine()
        if imgui.Button("Pause" if is_running else "Resume"):
            is_running = not is_running

        imgui.SameLine()
        if imgui.Button("Step"):
            sim.step()
            update_visuals()

        imgui.Text(f"State: {sim.state.value}")
        imgui.Text(f"Time: {sim.time:.2f}s")

        if not sim.is_divided:
            imgui.Text(f"Elongation: {sim.elong_prog * 100:.1f}%")
            imgui.Text(f"Furrow Constriction: {sim.furrow_prog * 100:.1f}%")
        else:
            imgui.Text(f"Daughter Left: {len(sim.int_l)} particles")
            imgui.Text(f"Daughter Right: {len(sim.int_r)} particles")

        changed, transparency = imgui.SliderFloat(
            "Transparency", transparency, 0.0, 1.0
        )
        if changed:
            mesh_mother.set_transparency(transparency)
            if mesh_l:
                mesh_l.set_transparency(transparency)
            if mesh_r:
                mesh_r.set_transparency(transparency)

        changed_rad, point_radius = imgui.SliderFloat(
            "Particle Radius", point_radius, 0.005, 0.03
        )
        if changed_rad:
            if cloud_mother:
                cloud_mother.set_radius(point_radius)
            if cloud_l:
                cloud_l.set_radius(point_radius)
            if cloud_r:
                cloud_r.set_radius(point_radius)

        if is_running:
            sim.step()
            update_visuals()

    ps.set_user_callback(frame)
    ps.show()


if __name__ == "__main__":
    main()
