"""A pure XPBD physics engine for research and experimentation."""

from .body import SoftBody
from .mesh import icosphere_mesh, sphere_mesh, sphere_tet_mesh
from .solver import Solver

__all__ = ["SoftBody", "Solver", "sphere_mesh", "icosphere_mesh", "sphere_tet_mesh"]


