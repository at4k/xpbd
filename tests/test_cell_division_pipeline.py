import numpy as np
import pytest

from xpbd import SoftBody, Solver
from xpbd.constraints import (
    DistanceConstraint,
    GlobalVolumeConstraint,
    ParticleCollisionConstraint,
    SpindleConstraint,
    ContractileRingConstraint,
)
from xpbd.mesh import icosphere_mesh, edges_from_faces
from xpbd.topology import slice_and_cap_mesh, partition_particles, tangential_surface_relaxation


def test_full_cell_division_lifecycle():
    # Setup cell
    r_cell = 0.5
    origin = (0.0, 1.5, 0.0)
    surf_pos, faces = icosphere_mesh(subdivisions=2, radius=r_cell, origin=origin)
    n_surf = len(surf_pos)

    # 2 poles
    poles_pos = np.array([
        [-0.1, origin[1], origin[2]],
        [ 0.1, origin[1], origin[2]],
    ], dtype=np.float64)
    total_pos = np.concatenate([surf_pos, poles_pos], axis=0)
    p0_idx, p1_idx = n_surf, n_surf + 1

    inv_mass = np.ones(len(total_pos), dtype=np.float64)
    body = SoftBody(
        positions=total_pos,
        velocities=np.zeros_like(total_pos),
        inv_mass=inv_mass,
        tets=np.zeros((0, 4), dtype=np.int64),
        edges=edges_from_faces(faces),
    )

    left_mask = surf_pos[:, 0] < origin[0]
    right_mask = surf_pos[:, 0] >= origin[0]
    left_surf_idx = np.where(left_mask)[0]
    right_surf_idx = np.where(right_mask)[0]

    spindle = SpindleConstraint(
        p0_idx,
        p1_idx,
        left_surf_idx,
        right_surf_idx,
        pole_rest_length=0.2,
        tether_rest_length=r_cell * 0.7,
        tether_compliance=1e-3,
    )

    equator_mask = np.abs(surf_pos[:, 0] - origin[0]) < 0.15
    ring_idx = np.where(equator_mask)[0]
    ring = ContractileRingConstraint(
        ring_idx,
        center_y=origin[1],
        center_z=origin[2],
        target_radius=r_cell,
        compliance=0.0,
    )

    constraints = [
        DistanceConstraint.from_positions(body.positions, body.edges),
        GlobalVolumeConstraint.from_positions(body.positions, [faces]),
        spindle,
        ring,
    ]
    solver = Solver(body, constraints, gravity=(0.0, 0.0, 0.0))

    # Phase 1: Elongate
    for _ in range(25):
        spindle.set_target_length(spindle.pole_rest_length + 0.015)
        solver.step(0.016)
    
    pole_dist = body.positions[p1_idx, 0] - body.positions[p0_idx, 0]
    assert pole_dist > 0.35, f"Poles did not elongate sufficiently: {pole_dist}"

    # Phase 2: Constrict
    for _ in range(30):
        ring.set_target_radius(max(0.04, ring.target_radius - 0.015))
        solver.step(0.016)

    # Check that ring constricted
    ring_radii = np.sqrt(
        (body.positions[ring_idx, 1] - origin[1]) ** 2
        + (body.positions[ring_idx, 2] - origin[2]) ** 2
    )
    assert np.mean(ring_radii) < r_cell * 0.6

    # Phase 3: Divide (Abscission) at the midpoint between the two poles
    cur_surf = body.positions[:n_surf]
    cut_x = float((body.positions[p0_idx, 0] + body.positions[p1_idx, 0]) / 2.0)
    (pos_l, faces_l), (pos_r, faces_r) = slice_and_cap_mesh(cur_surf, faces, cut_x=cut_x)

    assert len(pos_l) > 0 and len(pos_r) > 0
    assert len(faces_l) > 0 and len(faces_r) > 0

    # Verify both daughter meshes are closed 2-manifolds
    e_l = edges_from_faces(faces_l)
    assert len(pos_l) - len(e_l) + len(faces_l) == 2
    e_r = edges_from_faces(faces_r)
    assert len(pos_r) - len(e_r) + len(faces_r) == 2

    # Phase 4: Build daughter SoftBodies and relax
    body_l = SoftBody.from_surface_mesh(pos_l, faces_l)
    body_r = SoftBody.from_surface_mesh(pos_r, faces_r)

    center_l = np.mean(body_l.positions, axis=0)
    pos_l_relaxed = tangential_surface_relaxation(
        body_l.positions, faces_l, center=center_l, factor=0.15, n_iters=5
    )
    body_l.positions[:] = pos_l_relaxed

    center_r = np.mean(body_r.positions, axis=0)
    pos_r_relaxed = tangential_surface_relaxation(
        body_r.positions, faces_r, center=center_r, factor=0.15, n_iters=5
    )
    body_r.positions[:] = pos_r_relaxed

    # Phase 5: Independent simulation step for both daughters
    solver_l = Solver(
        body_l,
        [
            DistanceConstraint.from_positions(body_l.positions, body_l.edges),
            GlobalVolumeConstraint.from_positions(body_l.positions, [faces_l]),
        ],
        gravity=(0.0, 0.0, 0.0),
    )
    solver_r = Solver(
        body_r,
        [
            DistanceConstraint.from_positions(body_r.positions, body_r.edges),
            GlobalVolumeConstraint.from_positions(body_r.positions, [faces_r]),
        ],
        gravity=(0.0, 0.0, 0.0),
    )

    for _ in range(10):
        solver_l.step(0.016)
        solver_r.step(0.016)

    assert not np.any(np.isnan(body_l.positions))
    assert not np.any(np.isnan(body_r.positions))


def test_cell_division_sim_example():
    import argparse
    import sys
    from pathlib import Path

    root_dir = str(Path(__file__).resolve().parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from examples.cell_division import CellDivisionSim, CellDivisionState

    args = argparse.Namespace(
        subdivisions=2,
        bag_radius=0.5,
        internal_particles=60,
        substeps=20,
        mode="jacobi",
        transparency=0.45,
        point_radius=0.015,
        headless=True,
        headless_steps=300,
    )
    sim = CellDivisionSim(args, (-1.5, 0.0, -1.5), (1.5, 3.0, 1.5))
    sim.push_speed = 0.55
    sim.constrict_speed = 0.55

    # Run through full cycle
    for i in range(300):
        sim.step()
        if sim.is_divided:
            # Verify velocities remain stable without explosive divergence
            vel_l = np.max(np.linalg.norm(sim.body_left.velocities, axis=1))
            vel_r = np.max(np.linalg.norm(sim.body_right.velocities, axis=1))
            assert vel_l < 5.0, f"Step {i}: Left daughter velocity exploded: {vel_l}"
            assert vel_r < 5.0, f"Step {i}: Right daughter velocity exploded: {vel_r}"

    assert sim.is_divided
    assert sim.state == CellDivisionState.INDEPENDENT

    # Verify daughter cells are rounded/spherical rather than skinny hotdogs
    pos_l = sim.body_left.positions[: sim.n_surf_l]
    span_x = np.max(pos_l[:, 0]) - np.min(pos_l[:, 0])
    span_y = np.max(pos_l[:, 1]) - np.min(pos_l[:, 1])
    span_z = np.max(pos_l[:, 2]) - np.min(pos_l[:, 2])
    aspect_xy = span_x / span_y
    aspect_xz = span_x / span_z
    assert 0.75 < aspect_xy < 1.35, f"Left daughter not spherical: aspect_xy={aspect_xy:.2f}"
    assert 0.75 < aspect_xz < 1.35, f"Left daughter not spherical: aspect_xz={aspect_xz:.2f}"

    # Verify internal particles are strictly confined inside daughter cells
    c_l = np.mean(pos_l, axis=0)
    int_l = sim.body_left.positions[sim.n_surf_l :]
    dist_l = np.linalg.norm(int_l - c_l, axis=1)
    assert np.all(dist_l < sim.r_daughter * 1.05), "Internal particles escaped left daughter"

    # Verify reset/rewind functionality
    sim.reset()
    assert sim.state == CellDivisionState.ELONGATING
    assert not sim.is_divided
    assert sim.time == 0.0

