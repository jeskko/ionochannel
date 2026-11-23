"""Layer-modes spread over delay, after Vogler & Hoffmeyer (NTIA Report
92-284, 1992, "A new approach to HF channel modeling and simulation,
Part III"). A mode is given by its delays tau_L < tau_c < tau_U, the
Doppler shift fs at tau_c and fsL at tau_L (the "slant"), the Doppler
half-width sigma_D, and the threshold s_v (both spreads are measured
where the power has fallen to s_v of the peak; the report gives no
value). It becomes a set of taps across [tau_L, tau_U]:

- power over delay T(tau) = A exp(alpha (ln g + 1 - g)), g = (tau -
  tau_l)/(tau_c - tau_l) (eq. 8b), with alpha and tau_l from the
  appendix so that T(tau_L) = T(tau_U) = s_v A;
- Doppler shift fs + b (tau - tau_c), b = (fs - fsL)/(tau_c - tau_L)
  (eq. 8c);
- a Gaussian Doppler spectrum exp(-pi f^2 / sigma_f^2), sigma_f =
  sigma_D (-ln s_v / pi)^-1/2 (eq. 7a), which falls to s_v at +-sigma_D:
  in F.1487 terms a spread (2 sigma) of sigma_f sqrt(2 / pi).

The taps then run through the ordinary Channel. Only Gaussian Doppler
shapes; the report's Lorentzian option (eq. 7b) is not implemented.
"""
from __future__ import annotations

import math

import numpy as np

from .channel import Component, Path


def _z_l(ratio: float) -> float:
    """z_L in (0, 1) with ln z_L - z_L = ln z_U - z_U and
    (1 - z_L)/(z_U - 1) = ratio (appendix), by bisection."""
    if not 0 < ratio < 1:
        raise ValueError("need tau_c - tau_L < tau_U - tau_c (the report's sigma_c < sigma_T / 2)")

    def z_u(zl):                     # the other root of ln z - z = ln zl - zl, above 1
        c = math.log(zl) - zl
        lo, hi = 1.0, 2.0
        while math.log(hi) - hi > c:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if math.log(mid) - mid > c:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo, hi = 1e-12, 1 - 1e-6
    for _ in range(200):
        mid = (lo + hi) / 2
        r = (1 - mid) / (z_u(mid) - 1)
        if r < ratio:              # r rises towards 1 as z_L rises towards 1
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def vh_shape(tau_l: float, tau_c: float, tau_u: float, threshold: float):
    """(alpha, tau_ell) of the delay shape (the report's appendix)."""
    if not (tau_l < tau_c < tau_u):
        raise ValueError("need tau_L < tau_c < tau_U")
    if not 0 < threshold < 1:
        raise ValueError("threshold s_v must be in (0, 1)")
    zl = _z_l((tau_c - tau_l) / (tau_u - tau_c))
    alpha = math.log(threshold) / (math.log(zl) + 1 - zl)
    tau_ell = tau_c - (tau_c - tau_l) / (1 - zl)
    return alpha, tau_ell


def vh_mode(delay_s: float, spread_s: float, peak_s: float, shift_hz: float, shift_low_hz: float,
            doppler_halfwidth_hz: float, threshold: float, amplitude: float = 1.0,
            taps: int = 33) -> list[Path]:
    """Taps for one layer-mode: tau_L = delay_s, tau_U = delay_s + spread_s
    (the report's sigma_T), tau_c = delay_s + peak_s (sigma_c); shifts
    fs at tau_c and fsL at tau_L; Doppler half-width sigma_D at the
    threshold s_v; peak power density `amplitude` (the report's A).
    `taps` taps across the spread (33 as in the Rust crate); powers are
    the integral of T over each step, so they add up to the mode's power."""
    tl, tc, tu = delay_s, delay_s + peak_s, delay_s + spread_s
    alpha, tell = vh_shape(tl, tc, tu, threshold)
    k = max(2, int(taps))
    taus = np.linspace(tl, tu, k)
    g = (taus - tell) / (tc - tell)
    T = amplitude * np.exp(alpha * (np.log(g) + 1 - g))
    w = np.full(k, spread_s / (k - 1))
    w[0] = w[-1] = w[0] / 2
    b = (shift_hz - shift_low_hz) / (tc - tl)
    sigma_f = doppler_halfwidth_hz / math.sqrt(-math.log(threshold) / math.pi)
    spread = sigma_f * math.sqrt(2 / math.pi)
    return [Path(float(t), (Component(float(p * wi), spread, float(shift_hz + b * (t - tc))),))
            for t, p, wi in zip(taus, T, w)]


def normalize(paths) -> list[Path]:
    """The same paths scaled to a total power of 1."""
    total = sum(p.power for p in paths)
    return [Path(p.delay_s, tuple(Component(c.power / total, c.spread_hz, c.shift_hz, c.specular)
                                  for c in p.components)) for p in paths]
