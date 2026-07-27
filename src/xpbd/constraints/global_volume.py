"""Global volume (pressure) constraint for closed triangle meshes: C = V - V0."""

import numpy as np

_EPS = 1e-12


class GlobalVolumeConstraint:
    """Batch of global volume constraints over closed surface meshes.

    Each constraint corresponds to a single bag (a closed triangle mesh).
    """

    def __init__(
        self,
        faces_list: list[np.ndarray] | np.ndarray,
        rest_volumes: list[float] | np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
    ) -> None:
        if isinstance(faces_list, np.ndarray):
            self.faces_list = [faces_list]
        else:
            self.faces_list = [np.asarray(f, dtype=np.int64) for f in faces_list]

        self.rest_volumes = np.atleast_1d(np.asarray(rest_volumes, dtype=np.float64))
        self.compliance = float(compliance)
        self.relaxation = float(relaxation)
        self.lam = np.zeros(len(self.faces_list))

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        faces_list: list[np.ndarray] | np.ndarray,
        compliance: float = 0.0,
        relaxation: float = 1.0,
        inflation_factor: float = 1.0,
    ) -> "GlobalVolumeConstraint":
        if isinstance(faces_list, np.ndarray):
            flist = [faces_list]
        else:
            flist = [np.asarray(f, dtype=np.int64) for f in faces_list]

        rest = []
        for faces in flist:
            p1 = positions[faces[:, 0]]
            p2 = positions[faces[:, 1]]
            p3 = positions[faces[:, 2]]
            vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0
            rest.append(vol * inflation_factor)

        return cls(flist, rest, compliance=compliance, relaxation=relaxation)

    def __len__(self) -> int:
        return len(self.faces_list)

    def reset_lambda(self) -> None:
        self.lam[:] = 0.0

    def project(self, x: np.ndarray, inv_mass: np.ndarray, h: float) -> None:
        delta = np.zeros_like(x)
        count = np.zeros(len(x))

        for k in range(len(self.faces_list)):
            faces = self.faces_list[k]
            p1 = x[faces[:, 0]]
            p2 = x[faces[:, 1]]
            p3 = x[faces[:, 2]]

            vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0
            c = vol - self.rest_volumes[k]

            alpha = self.compliance / h**2

            g1 = np.cross(p2, p3) / 6.0
            g2 = np.cross(p3, p1) / 6.0
            g3 = np.cross(p1, p2) / 6.0

            grad = np.zeros_like(x)
            np.add.at(grad, faces[:, 0], g1)
            np.add.at(grad, faces[:, 1], g2)
            np.add.at(grad, faces[:, 2], g3)

            grad_sq = np.sum(grad**2, axis=1)
            w = np.sum(inv_mass * grad_sq)

            denom = w + alpha
            if denom < _EPS:
                continue

            dlam = -(c + alpha * self.lam[k]) / denom
            self.lam[k] += dlam

            # Accumulate corrections
            idx = np.unique(faces)
            delta[idx] += inv_mass[idx, None] * dlam * grad[idx]
            count[idx] += 1.0

        x += self.relaxation * delta / np.maximum(count, 1.0)[:, None]

    def project_one(
        self, k: int, x: np.ndarray, inv_mass: np.ndarray, h: float
    ) -> None:
        faces = self.faces_list[k]
        p1 = x[faces[:, 0]]
        p2 = x[faces[:, 1]]
        p3 = x[faces[:, 2]]

        vol = np.sum(np.einsum("ij,ij->i", np.cross(p1, p2), p3)) / 6.0
        c = vol - self.rest_volumes[k]

        alpha = self.compliance / h**2

        g1 = np.cross(p2, p3) / 6.0
        g2 = np.cross(p3, p1) / 6.0
        g3 = np.cross(p1, p2) / 6.0

        grad = np.zeros_like(x)
        np.add.at(grad, faces[:, 0], g1)
        np.add.at(grad, faces[:, 1], g2)
        np.add.at(grad, faces[:, 2], g3)

        grad_sq = np.sum(grad**2, axis=1)
        w = np.sum(inv_mass * grad_sq)

        denom = w + alpha
        if denom < _EPS:
            return

        dlam = -(c + alpha * self.lam[k]) / denom
        self.lam[k] += dlam

        x += inv_mass[:, None] * dlam * grad
