"""Variable decode contract — the uniform [0,1] norm frame + integer in-bounds.

Regression guard for the integer-param decode scale bug: ``StaticVariable``'s
integer decode must land in the same [0,1] norm frame as continuous variables
(so the shared norm→raw step does not double-scale), and ``to_real`` must invert
it to an in-bounds integer. See the ``PFAB - Coordinate Spaces`` KB note.
"""
from __future__ import annotations

import torch

from pred_fab.core import Parameter
from pred_fab.orchestration.calibration.space import StaticVariable


def _int_var(lo: int, hi: int) -> StaticVariable:
    return StaticVariable(
        data_object=Parameter.integer("p", lo, hi),
        lo=float(lo), hi=float(hi), is_integer=True,
    )


def test_integer_decode_lands_in_unit_norm_frame():
    """Integer decode is uniform [0,1] (not [0,range]) so norm→raw is one rule."""
    v = _int_var(0, 5)
    z = torch.linspace(0.0, 5.0, 21)  # the integer optimiser frame is [0, range]
    norm = v.decode(z)
    assert torch.all(norm >= -1e-9) and torch.all(norm <= 1.0 + 1e-9)


def test_integer_norm_to_raw_is_in_bounds():
    """The shared norm·span+lo rule yields in-bounds raw for integers.

    This is the exact step that double-scaled before the fix: decode returning
    round(z) (in [0,range]) made raw = round(z)·range+lo, far out of bounds.
    """
    v = _int_var(0, 5)
    for zi in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        norm = float(v.decode(torch.tensor(zi)).item())
        raw = norm * v.span + v.lo  # the _decode_frames conversion
        assert v.lo <= raw <= v.hi


def test_integer_to_real_inverts_decode_in_bounds():
    """to_real(decode(z)) = round(z)+lo, clamped to [lo, hi], as a Python int."""
    v = _int_var(2, 7)
    for zi in (0.0, 0.4, 1.6, 2.5, 4.9, 5.0):
        real = v.to_real(float(v.decode(torch.tensor(zi)).item()))
        assert isinstance(real, int)
        assert v.lo <= real <= v.hi
        assert real == int(round(zi)) + int(v.lo)


def test_integer_decode_is_differentiable():
    """STE keeps decode differentiable so the gradient optimiser gets a signal."""
    v = _int_var(0, 4)
    z = torch.tensor(2.3, requires_grad=True)
    v.decode(z).backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


def test_integer_decode_clamps_out_of_range_u():
    """LBFGS is unconstrained, so decode must clamp u outside [0, range] to a
    valid integer — the objective is never evaluated outside [lo, hi]."""
    v = _int_var(1, 4)  # span = 3
    for u in (-5.0, -0.4, 3.4, 9.0):
        norm = float(v.decode(torch.tensor(u)).item())
        assert -1e-9 <= norm <= 1.0 + 1e-9, f"norm {norm} out of [0,1] for u={u}"
        raw = norm * v.span + v.lo
        assert v.lo <= raw <= v.hi, f"raw {raw} out of [{v.lo},{v.hi}] for u={u}"
    # boundary maps exactly
    assert float(v.decode(torch.tensor(-2.0)).item()) == 0.0
    assert float(v.decode(torch.tensor(5.0)).item()) == 1.0
