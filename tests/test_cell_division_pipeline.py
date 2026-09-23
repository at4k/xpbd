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
