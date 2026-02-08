"""Rigid body geometry using quaternion rotations."""

import math
import torch
from torch import Tensor


def _normalize(v: Tensor, dim: int = -1) -> Tensor:
    return v / (v.norm(dim=dim, keepdim=True) + 1e-8)


def _quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_to_rot(q: Tensor) -> Tensor:
    """Convert unit quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q = _normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _rot_to_quat(R: Tensor) -> Tensor:
    """Convert 3x3 rotation matrix to unit quaternion (w,x,y,z)."""
    xx, yy, zz = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = xx + yy + zz
    w = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
    x = torch.sqrt(torch.clamp(1 + xx - yy - zz, min=1e-8)) / 2
    y = torch.sqrt(torch.clamp(1 - xx + yy - zz, min=1e-8)) / 2
    z = torch.sqrt(torch.clamp(1 - xx - yy + zz, min=1e-8)) / 2
    x = x * torch.sign(R[..., 2, 1] - R[..., 1, 2] + 1e-12)
    y = y * torch.sign(R[..., 0, 2] - R[..., 2, 0] + 1e-12)
    z = z * torch.sign(R[..., 1, 0] - R[..., 0, 1] + 1e-12)
    return _normalize(torch.stack([w, x, y, z], dim=-1))


class Rigid:
    """Rigid body transformation: rotation (quaternion) + translation."""

    def __init__(self, quat: Tensor, trans: Tensor):
        self.quat = _normalize(quat)  # (..., 4) unit quaternion (w,x,y,z)
        self.trans = trans  # (..., 3)

    @staticmethod
    def identity(shape=(), device=None) -> "Rigid":
        quat = torch.zeros(*shape, 4, device=device)
        quat[..., 0] = 1.0
        return Rigid(quat, torch.zeros(*shape, 3, device=device))

    @staticmethod
    def from_3_points(n: Tensor, ca: Tensor, c: Tensor) -> "Rigid":
        """Build local frame from backbone atoms N, CA, C.  Origin at CA."""
        e1 = _normalize(c - ca)
        v2 = n - ca
        e2 = _normalize(v2 - (v2 * e1).sum(-1, keepdim=True) * e1)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-1)  # columns = basis vectors
        return Rigid(_rot_to_quat(R), ca)

    def apply(self, pts: Tensor) -> Tensor:
        """Transform points: R @ pts + t."""
        R = _quat_to_rot(self.quat)
        return torch.einsum("...ij,...j->...i", R, pts) + self.trans

    def inverse(self) -> "Rigid":
        """Return T^{-1}."""
        inv_q = self.quat * torch.tensor([1, -1, -1, -1], device=self.quat.device)
        R = _quat_to_rot(self.quat)
        inv_t = -torch.einsum("...ji,...j->...i", R, self.trans)
        return Rigid(inv_q, inv_t)

    def compose(self, other: "Rigid") -> "Rigid":
        """self ∘ other: first apply other, then self."""
        new_q = _quat_multiply(self.quat, other.quat)
        R = _quat_to_rot(self.quat)
        new_t = torch.einsum("...ij,...j->...i", R, other.trans) + self.trans
        return Rigid(new_q, new_t)


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 90° rotation around z-axis:  q = (cos45, 0, 0, sin45)
    a = math.pi / 4
    q = torch.tensor([math.cos(a), 0.0, 0.0, math.sin(a)])
    t = torch.tensor([1.0, 2.0, 3.0])
    T = Rigid(q, t)

    pt = torch.tensor([1.0, 0.0, 0.0])
    result = T.apply(pt)
    expected = torch.tensor([1.0, 3.0, 3.0])  # R@[1,0,0]=[0,1,0] + t
    assert torch.allclose(result, expected, atol=1e-6), f"{result} != {expected}"

    # inverse round-trip
    pt2 = T.inverse().apply(result)
    assert torch.allclose(pt2, pt, atol=1e-6), f"inverse failed: {pt2}"

    # compose: T ∘ T^{-1} ≈ identity
    I = T.compose(T.inverse())
    assert torch.allclose(I.apply(pt), pt, atol=1e-6), "compose failed"

    # from_3_points smoke test
    n = torch.tensor([0.0, 1.0, 0.0])
    ca = torch.tensor([0.0, 0.0, 0.0])
    c = torch.tensor([1.0, 0.0, 0.0])
    frame = Rigid.from_3_points(n, ca, c)
    assert frame.trans.allclose(ca), "from_3_points origin should be CA"

    print("All tests passed.")
