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
    DistanceConstraint,
    GlobalVolumeConstraint,
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
    """Generates internal cytoplasm particles strictly bounded within a spherical radius."""
    if num_particles <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    r_max = radius * 0.70
    np.random.seed(42)
    inside = []
    while len(inside) < num_particles:
        cand = np.random.uniform(-r_max, r_max, size=(num_particles * 2, 3))
        valid = cand[np.linalg.norm(cand, axis=1) <= r_max]
        inside.extend(valid)
    pts = np.array(inside[:num_particles], dtype=np.float64)
    return pts + np.asarray(origin, dtype=np.float64)


class CellDivisionSim:
    """Manages the full lifecycle of cell division across XPBD constraints and bodies."""

    def __init__(self, args, bounds_min, bounds_max):
        self.args = args
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.origin = np.array([0.0, 1.5, 0.0], dtype=np.float64)
        self.r_cell = args.bag_radius
        self.r_daughter = self.r_cell / (2.0 ** (1.0 / 3.0))  # ~0.3969 for r_cell=0.5

        # Speed parameters
        self.push_speed = 0.55
        self.constrict_speed = 0.55
        self.relax_factor = 0.15
        self.spindle_radius = 0.030

        self.reset()

    def reset(self):
        """Rewinds the simulation to the initial time t = 0."""
        self.state = CellDivisionState.ELONGATING
        self.relax_step_count = 0
        self.is_divided = False
        self.time = 0.0

        # Progress tracking
        self.elong_prog = 0.0
        self.furrow_prog = 0.0

        # Initial mother sphere mesh
        self.base_surf_pos, self.faces = icosphere_mesh(
            subdivisions=self.args.subdivisions, radius=self.r_cell, origin=self.origin
        )
        self.n_surf = len(self.base_surf_pos)
        self.edges = edges_from_faces(self.faces)
        self.cur_surf_pos = self.base_surf_pos.copy()

        # Normalized coordinates relative to origin (unit sphere)
        self.u_surf = (self.base_surf_pos - self.origin) / self.r_cell

        # Initial internal cytoplasm particles
        self.base_internal_pos = generate_internal_particles(
            self.origin, self.r_cell, self.args.internal_particles
        )
        self.n_internal = len(self.base_internal_pos)
        self.cur_internal_pos = self.base_internal_pos.copy()
        if self.n_internal > 0:
            self.u_internal = (self.base_internal_pos - self.origin) / self.r_cell
        else:
            self.u_internal = np.zeros((0, 3), dtype=np.float64)

        # Spindle parameters (two centrosomes separating inside the cell)
        # Closer daughter distance so lobes remain in close contact across cleavage furrow
        self.d_max = self.r_daughter * 0.65  # ~0.26 (lobes in close contact)
        self.d_init = self.r_daughter * 0.15  # ~0.06
        self.spindle_dist = self.d_init

        self.pole_0 = np.array(
            [self.origin[0] - self.spindle_dist, self.origin[1], self.origin[2]],
            dtype=np.float64,
        )
        self.pole_1 = np.array(
            [self.origin[0] + self.spindle_dist, self.origin[1], self.origin[2]],
            dtype=np.float64,
        )

        # Mother SoftBody container
        total_pos = np.concatenate(
            [self.cur_surf_pos, self.cur_internal_pos], axis=0
        )
        inv_mass = np.ones(len(total_pos), dtype=np.float64)
        if self.n_internal > 0:
            inv_mass[self.n_surf :] = 2.0

        self.mother_body = SoftBody(
            positions=total_pos,
            velocities=np.zeros_like(total_pos),
            inv_mass=inv_mass,
            tets=np.zeros((0, 4), dtype=np.int64),
            edges=self.edges,
        )

        self.dist_con = DistanceConstraint.from_positions(
            self.mother_body.positions, self.edges, compliance=1e-3
        )
        self.vol_con = GlobalVolumeConstraint.from_positions(
            self.mother_body.positions, [self.faces], compliance=0.0
        )
        self.initial_volume = float(self.vol_con.rest_volumes[0])

        self.mother_solver = Solver(
            self.mother_body,
            [self.dist_con, self.vol_con],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        # Daughter bodies (allocated upon division)
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

    def _compute_deformed_geometry(self, elong_t: float, constrict_t: float):
        """Computes the physiologically realistic two-lobed morphology and internal positions.

        During elongation and constriction, the cell maintains rounded spherical daughter lobes
        centered at the separating spindle poles rather than a thin elongated sausage.
        """
        d = 0.02 + (self.d_max - 0.02) * elong_t
        r_l = self.r_cell + (self.r_daughter - self.r_cell) * elong_t

        # 1. Surface deformation
        sign_x = np.where(self.u_surf[:, 0] >= 0, 1.0, -1.0)
        abs_ux = np.abs(self.u_surf[:, 0])
        r_yz_u = np.sqrt(self.u_surf[:, 1] ** 2 + self.u_surf[:, 2] ** 2)
        norm_yz = np.maximum(r_yz_u, 1e-12)
        dir_y = self.u_surf[:, 1] / norm_yz
        dir_z = self.u_surf[:, 2] / norm_yz

        # Smooth axial expansion along spindle
        x = sign_x * (d * (abs_ux ** 0.8) + abs_ux * r_l)
        abs_x = np.abs(x)

        # Spherical lobe base radius
        r_base = np.zeros_like(x)
        outer_mask = abs_x >= d
        dx_outer = abs_x[outer_mask] - d
        r_base[outer_mask] = np.sqrt(np.maximum(0.0, r_l ** 2 - dx_outer ** 2))

        inner_mask = ~outer_mask
        u = abs_x[inner_mask] / max(d, 1e-6)
        s = u * u * (3.0 - 2.0 * u)
        r_eq = np.sqrt(np.maximum(0.05, r_l ** 2 - d ** 2))
        r_base[inner_mask] = r_eq + (r_l - r_eq) * s

        # Furrow constriction: localized sharp notch at equator
        neck_r = r_eq * (1.0 - constrict_t * 0.92)
        w_furrow = 0.12
        xi = np.clip(abs_x / w_furrow, 0.0, 1.0)
        s_furrow = xi * xi * (3.0 - 2.0 * xi)
        scale_neck = (neck_r / r_eq) + (1.0 - neck_r / r_eq) * s_furrow
        r_final = r_base * scale_neck

        surf = np.zeros_like(self.base_surf_pos)
        surf[:, 0] = self.origin[0] + x
        surf[:, 1] = self.origin[1] + dir_y * r_final
        surf[:, 2] = self.origin[2] + dir_z * r_final

        # 2. Internal cytoplasm particles
        if self.n_internal > 0:
            sign_xi = np.where(self.u_internal[:, 0] >= 0, 1.0, -1.0)
            abs_uxi = np.abs(self.u_internal[:, 0])
            r_yzi_u = np.sqrt(self.u_internal[:, 1] ** 2 + self.u_internal[:, 2] ** 2)
            norm_yzi = np.maximum(r_yzi_u, 1e-12)
            dir_yi = self.u_internal[:, 1] / norm_yzi
            dir_zi = self.u_internal[:, 2] / norm_yzi

            xi = sign_xi * (d * (abs_uxi ** 0.8) + abs_uxi * r_l)
            abs_xi = np.abs(xi)
            near_eq = abs_xi < 0.12
            xi[near_eq] += np.where(sign_xi[near_eq] >= 0, 1.0, -1.0) * constrict_t * 0.04

            r_scale = r_yzi_u * (r_l / 1.0)
            r_final_i = r_scale * (1.0 - constrict_t * 0.3 * np.exp(-abs_xi ** 2 / 0.01))

            pts = np.zeros_like(self.base_internal_pos)
            pts[:, 0] = self.origin[0] + xi
            pts[:, 1] = self.origin[1] + dir_yi * r_final_i
            pts[:, 2] = self.origin[2] + dir_zi * r_final_i
        else:
            pts = np.zeros((0, 3), dtype=np.float64)

        return surf, pts

    def _clamp_particles_inside_daughter(
        self,
        body: SoftBody,
        n_surf: int,
        center: np.ndarray,
        max_radius: float,
        is_left: bool,
    ) -> None:
        """Strictly ensures internal particles cannot escape the daughter cell sphere or cross the contact plane."""
        pts = body.positions[n_surf:]
        if len(pts) == 0:
            return
        diff = pts - center
        dists = np.linalg.norm(diff, axis=1)
        escaped = dists > max_radius
        if np.any(escaped):
            pts[escaped] = (
                center
                + diff[escaped]
                * (max_radius / np.maximum(dists[escaped], 1e-12))[:, None]
            )
        # Keep left particles strictly on left side of contact plane (x <= origin[0] - 0.015)
        # and right particles strictly on right side (x >= origin[0] + 0.015)
        if is_left:
            pts[pts[:, 0] > self.origin[0] - 0.015, 0] = self.origin[0] - 0.015
        else:
            pts[pts[:, 0] < self.origin[0] + 0.015, 0] = self.origin[0] + 0.015
        # Zero out particle velocities so they cannot accumulate momentum or drift
        body.velocities[n_surf:] = 0.0

    def execute_division(self):
        """Topological separation into two independent SoftBodies."""
        cut_x = float(self.origin[0])

        # Slicing from the full two-lobed geometry at constrict_t = 0 ensures the contact septum
        # covers the broad physiological contact disk r <= R_contact across the dividing boundary
        surf_lobed, _ = self._compute_deformed_geometry(1.0, 0.0)
        (pos_l, self.faces_left), (pos_r, self.faces_right) = slice_and_cap_mesh(
            surf_lobed, self.faces, cut_x=cut_x
        )
        self.n_surf_l = len(pos_l)
        self.n_surf_r = len(pos_r)

        # Partition internal particles strictly by cut plane
        if self.n_internal > 0:
            int_pos = self.mother_body.positions[self.n_surf :]
            l_int_idx, r_int_idx = partition_particles(int_pos, cut_x=cut_x)
            int_l = int_pos[l_int_idx].copy()
            int_r = int_pos[r_int_idx].copy()
        else:
            int_l = np.zeros((0, 3), dtype=np.float64)
            int_r = np.zeros((0, 3), dtype=np.float64)

        self.n_int_l = len(int_l)
        self.n_int_r = len(int_r)

        total_l = (
            np.concatenate([pos_l, int_l], axis=0) if self.n_int_l > 0 else pos_l
        )
        total_r = (
            np.concatenate([pos_r, int_r], axis=0) if self.n_int_r > 0 else pos_r
        )

        edges_l = edges_from_faces(self.faces_left)
        edges_r = edges_from_faces(self.faces_right)

        inv_m_l = np.ones(len(total_l), dtype=np.float64)
        if self.n_int_l > 0:
            inv_m_l[self.n_surf_l :] = 2.0

        inv_m_r = np.ones(len(total_r), dtype=np.float64)
        if self.n_int_r > 0:
            inv_m_r[self.n_surf_r :] = 2.0

        # Construct 2 distinct SoftBody instances initialized with zero velocities
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
            self.body_left.positions, edges_l, compliance=1e-3
        )
        self.dist_con_r = DistanceConstraint.from_positions(
            self.body_right.positions, edges_r, compliance=1e-3
        )

        # Rest volume initialized from the newly capped daughter meshes to eliminate velocity spikes
        self.vol_con_l = GlobalVolumeConstraint.from_positions(
            self.body_left.positions, [self.faces_left], compliance=0.0
        )
        self.vol_con_r = GlobalVolumeConstraint.from_positions(
            self.body_right.positions, [self.faces_right], compliance=0.0
        )

        box_l = BoxConstraint(len(total_l), self.bounds_min, self.bounds_max)
        box_r = BoxConstraint(len(total_r), self.bounds_min, self.bounds_max)

        self.solver_left = Solver(
            self.body_left,
            [self.dist_con_l, self.vol_con_l, box_l],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )
        self.solver_right = Solver(
            self.body_right,
            [self.dist_con_r, self.vol_con_r, box_r],
            n_substeps=self.args.substeps,
            mode=self.args.mode,
            gravity=(0.0, 0.0, 0.0),
        )

        self.is_divided = True
        self.state = CellDivisionState.RELAXING

    def step(self):
        """Advances the simulation by one frame DT."""
        self.time += DT

        if not self.is_divided:
            # Phase 1: Spindle Elongation
            if self.state == CellDivisionState.ELONGATING:
                self.elong_prog += self.push_speed * DT
                if self.elong_prog >= 1.0:
                    self.elong_prog = 1.0
                    self.state = CellDivisionState.CONSTRICTING

                self.spindle_dist = (
                    self.d_init + (self.d_max - self.d_init) * self.elong_prog
                )
                self.pole_0[0] = self.origin[0] - self.spindle_dist
                self.pole_1[0] = self.origin[0] + self.spindle_dist

                surf, pts = self._compute_deformed_geometry(
                    self.elong_prog, 0.0
                )
                self.mother_body.positions[: self.n_surf] = surf
                if self.n_internal > 0:
                    self.mother_body.positions[self.n_surf :] = pts

            # Phase 2: Cleavage Furrow Constriction
            elif self.state == CellDivisionState.CONSTRICTING:
                self.furrow_prog += self.constrict_speed * DT
                if self.furrow_prog >= 1.0:
                    self.furrow_prog = 1.0
                    self.state = CellDivisionState.DIVIDING

                self.spindle_dist = self.d_max
                self.pole_0[0] = self.origin[0] - self.spindle_dist
                self.pole_1[0] = self.origin[0] + self.spindle_dist

                surf, pts = self._compute_deformed_geometry(
                    1.0, self.furrow_prog
                )
                self.mother_body.positions[: self.n_surf] = surf
                if self.n_internal > 0:
                    self.mother_body.positions[self.n_surf :] = pts

            # Phase 3: Division Trigger
            elif self.state == CellDivisionState.DIVIDING:
                self.execute_division()
                return

        else:
            # Phase 4: Post-division Relaxation & Contact State (Double-Bubble Equilibrium)
            sphere_center_l = np.array(
                [self.origin[0] - self.d_max, self.origin[1], self.origin[2]]
            )
            sphere_center_r = np.array(
                [self.origin[0] + self.d_max, self.origin[1], self.origin[2]]
            )

            if self.state == CellDivisionState.RELAXING:
                self.relax_step_count += 1

                # 1. Active cortical rounding force: free outer membrane rounds toward a smooth sphere
                free_idx_l = np.where(
                    self.body_left.positions[: self.n_surf_l, 0] < self.origin[0] - 0.005
                )[0]
                if len(free_idx_l) > 0:
                    v_l = self.body_left.positions[free_idx_l] - sphere_center_l
                    r_l = np.linalg.norm(v_l, axis=1)
                    target_r_l = r_l + (self.r_daughter - r_l) * 0.12
                    self.body_left.positions[free_idx_l] = sphere_center_l + (
                        v_l / np.maximum(r_l[:, None], 1e-12)
                    ) * target_r_l[:, None]
                    self.body_left.positions[free_idx_l, 0] = np.minimum(
                        self.body_left.positions[free_idx_l, 0], self.origin[0]
                    )

                free_idx_r = np.where(
                    self.body_right.positions[: self.n_surf_r, 0] > self.origin[0] + 0.005
                )[0]
                if len(free_idx_r) > 0:
                    v_r = self.body_right.positions[free_idx_r] - sphere_center_r
                    r_r = np.linalg.norm(v_r, axis=1)
                    target_r_r = r_r + (self.r_daughter - r_r) * 0.12
                    self.body_right.positions[free_idx_r] = sphere_center_r + (
                        v_r / np.maximum(r_r[:, None], 1e-12)
                    ) * target_r_r[:, None]
                    self.body_right.positions[free_idx_r, 0] = np.maximum(
                        self.body_right.positions[free_idx_r, 0], self.origin[0]
                    )

                # 2. Keep contact septum flush at x = origin[0] (contact plane non-penetration)
                cap_l = np.where(
                    self.body_left.positions[: self.n_surf_l, 0] >= self.origin[0] - 0.005
                )[0]
                self.body_left.positions[cap_l, 0] = self.origin[0]

                cap_r = np.where(
                    self.body_right.positions[: self.n_surf_r, 0] <= self.origin[0] + 0.005
                )[0]
                self.body_right.positions[cap_r, 0] = self.origin[0]

                # 3. Tangential surface relaxation on both daughter meshes (point arrangement optimization)
                rel_l = tangential_surface_relaxation(
                    self.body_left.positions[: self.n_surf_l],
                    self.faces_left,
                    center=sphere_center_l,
                    factor=self.relax_factor,
                    n_iters=1,
                )
                self.body_left.positions[: self.n_surf_l] = rel_l

                rel_r = tangential_surface_relaxation(
                    self.body_right.positions[: self.n_surf_r],
                    self.faces_right,
                    center=sphere_center_r,
                    factor=self.relax_factor,
                    n_iters=1,
                )
                self.body_right.positions[: self.n_surf_r] = rel_r

                self.body_left.positions[cap_l, 0] = self.origin[0]
                self.body_right.positions[cap_r, 0] = self.origin[0]

                # 4. Adapt edge rest lengths and volume constraint to the rounded geometry
                d_l = (
                    self.body_left.positions[self.body_left.edges[:, 1]]
                    - self.body_left.positions[self.body_left.edges[:, 0]]
                )
                self.dist_con_l.rest_lengths = (
                    self.dist_con_l.rest_lengths * 0.85
                    + np.linalg.norm(d_l, axis=1) * 0.15
                )

                d_r = (
                    self.body_right.positions[self.body_right.edges[:, 1]]
                    - self.body_right.positions[self.body_right.edges[:, 0]]
                )
                self.dist_con_r.rest_lengths = (
                    self.dist_con_r.rest_lengths * 0.85
                    + np.linalg.norm(d_r, axis=1) * 0.15
                )

                p1_l = self.body_left.positions[self.faces_left[:, 0]]
                p2_l = self.body_left.positions[self.faces_left[:, 1]]
                p3_l = self.body_left.positions[self.faces_left[:, 2]]
                self.vol_con_l.rest_volumes[0] = (
                    np.sum(np.einsum("ij,ij->i", np.cross(p1_l, p2_l), p3_l)) / 6.0
                )

                p1_r = self.body_right.positions[self.faces_right[:, 0]]
                p2_r = self.body_right.positions[self.faces_right[:, 1]]
                p3_r = self.body_right.positions[self.faces_right[:, 2]]
                self.vol_con_r.rest_volumes[0] = (
                    np.sum(np.einsum("ij,ij->i", np.cross(p1_r, p2_r), p3_r)) / 6.0
                )

                # Damp membrane velocities during relaxation
                self.body_left.velocities[: self.n_surf_l] *= 0.5
                self.body_right.velocities[: self.n_surf_r] *= 0.5

                if self.relax_step_count >= 40:
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

            # Enforce internal particle confinement and zero out particle velocities
            if self.n_int_l > 0:
                self._clamp_particles_inside_daughter(
                    self.body_left,
                    self.n_surf_l,
                    sphere_center_l,
                    self.r_daughter * 0.80,
                    is_left=True,
                )
            if self.n_int_r > 0:
                self._clamp_particles_inside_daughter(
                    self.body_right,
                    self.n_surf_r,
                    sphere_center_r,
                    self.r_daughter * 0.80,
                    is_left=False,
                )

            self.solver_left.step(DT)
            self.solver_right.step(DT)

            # Re-enforce internal particle confinement after solver step
            if self.n_int_l > 0:
                self._clamp_particles_inside_daughter(
                    self.body_left,
                    self.n_surf_l,
                    sphere_center_l,
                    self.r_daughter * 0.80,
                    is_left=True,
                )
            if self.n_int_r > 0:
                self._clamp_particles_inside_daughter(
                    self.body_right,
                    self.n_surf_r,
                    sphere_center_r,
                    self.r_daughter * 0.80,
                    is_left=False,
                )


def run_headless(args):
    """Headless simulation runner for automated testing and benchmarking."""
    bounds_min = (-1.5, 0.0, -1.5)
    bounds_max = (1.5, 3.0, 1.5)
    sim = CellDivisionSim(args, bounds_min, bounds_max)

    for _ in range(args.headless_steps):
        sim.step()

    assert sim.is_divided, "Simulation failed to reach division within step limit"
    assert sim.body_left is not None and sim.body_right is not None
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
            sim.mother_body.positions[sim.n_surf :],
        )
        cloud_mother.set_color((0.30, 0.70, 0.90))
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
            mesh_mother.update_vertex_positions(
                sim.mother_body.positions[: sim.n_surf]
            )
            if cloud_mother is not None:
                cloud_mother.update_point_positions(
                    sim.mother_body.positions[sim.n_surf :]
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
                    sim.body_left.positions[: sim.n_surf_l],
                    sim.faces_left,
                )
                mesh_l.set_color((0.20, 0.75, 0.85))
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
                        sim.body_left.positions[sim.n_surf_l :],
                    )
                    cloud_l.set_color((0.20, 0.75, 0.85))
                    cloud_l.set_radius(point_radius)

                if sim.n_int_r > 0:
                    cloud_r = ps.register_point_cloud(
                        "fluid_right",
                        sim.body_right.positions[sim.n_surf_r :],
                    )
                    cloud_r.set_color((0.45, 0.85, 0.45))
                    cloud_r.set_radius(point_radius)

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
                    sim.body_left.positions[sim.n_surf_l :]
                )
            if cloud_r is not None:
                cloud_r.update_point_positions(
                    sim.body_right.positions[sim.n_surf_r :]
                )

    def frame():
        nonlocal is_running, transparency, point_radius, mesh_l, mesh_r, cloud_l, cloud_r
        import polyscope.imgui as imgui

        # Prominent rewind and control bar
        if imgui.Button("最初に戻る (Rewind to Start)"):
            sim.reset()
            if mesh_mother:
                mesh_mother.set_enabled(True)
            if cloud_mother:
                cloud_mother.set_enabled(True)
            if spindle_curve:
                spindle_curve.set_enabled(True)
            if centrosomes:
                centrosomes.set_enabled(True)
            if mesh_l:
                ps.remove_surface_mesh("daughter_left")
            if mesh_r:
                ps.remove_surface_mesh("daughter_right")
            if cloud_l:
                ps.remove_point_cloud("fluid_left")
            if cloud_r:
                ps.remove_point_cloud("fluid_right")
            mesh_l = mesh_r = cloud_l = cloud_r = None
            update_visuals()

        imgui.SameLine()
        if imgui.Button("Pause" if is_running else "Play"):
            is_running = not is_running

        imgui.SameLine()
        if imgui.Button("Step"):
            is_running = False
            sim.step()
            update_visuals()

        imgui.SameLine()
        if imgui.Button("Force Divide") and not sim.is_divided:
            sim.execute_division()
            update_visuals()

        imgui.SeparatorText("Simulation Status")
        imgui.Text(f"Phase: {sim.state.value}")
        if not sim.is_divided:
            if sim.state == CellDivisionState.ELONGATING:
                imgui.Text(f"Spindle Elongation: {sim.elong_prog * 100:.1f}%")
            elif sim.state == CellDivisionState.CONSTRICTING:
                imgui.Text(f"Furrow Constriction: {sim.furrow_prog * 100:.1f}%")
        else:
            imgui.Text(f"Relaxation Steps: {sim.relax_step_count} / 40")

        imgui.SeparatorText("Simulation Parameters")
        _, sim.push_speed = imgui.SliderFloat(
            "Elongation Speed", sim.push_speed, 0.05, 1.0
        )
        _, sim.constrict_speed = imgui.SliderFloat(
            "Constriction Speed", sim.constrict_speed, 0.05, 1.0
        )
        _, sim.relax_factor = imgui.SliderFloat(
            "Relaxation Factor", sim.relax_factor, 0.05, 0.5
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
