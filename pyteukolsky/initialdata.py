"""
Initial-data helpers for the Teukolsky solver.
"""

import numpy as np
from math import factorial


def swsh(spin, ell, m, mu):
    """Spin-weighted spherical harmonic _{spin}Y_{ell,m}(mu) at phi=0.

    Implemented for spin=-2, any ell >= 2, via the general Wigner d-matrix
    k-sum:
        _{s}Y_{lm}(theta) = sqrt((2l+1)/4pi) * d^l_{-s,m}(theta)

        d^l_{m',m}(theta) = sum_k (-1)^k
              sqrt((l+m')!(l-m')!(l+m)!(l-m)!)
            / [(l+m'-k)! k! (l-m-k)! (k-m'+m)!]
            * cos(theta/2)^(2l-2k+m'-m) * sin(theta/2)^(2k-m'+m)

    with m' = -spin, k ranging over max(0, m'-m) .. min(l+m', l-m) (the
    range that keeps every factorial argument non-negative), and
        c2 = cos(theta/2) = sqrt((1+mu)/2),  s2 = sin(theta/2) = sqrt((1-mu)/2)
    expressed in mu = cos(theta).

    This reproduces the module's previous hardcoded ell=2 expressions
    exactly (verified to <=1e-15, see tests/test_validation.py): e.g. for
    l=2, m'=2, m=2 only k=0 survives, giving c2^4, matching the old
    ``norm * c2**4``.

    Parameters
    ----------
    spin : int  (must be -2; only s=-2 gravitational SWSHs are implemented)
    ell  : int  (>= |spin|)
    m    : int  (-ell <= m <= ell)
    mu   : array_like  mu = cos(theta) in [-1, 1]

    Returns
    -------
    ndarray, real, same shape as mu.
    """
    mu = np.asarray(mu, dtype=float)
    if spin != -2:
        raise NotImplementedError("Only spin=-2 implemented")
    if ell < abs(spin):
        raise ValueError(f"ell={ell} must be >= |spin|={abs(spin)}")
    if abs(m) > ell:
        raise ValueError(f"|m|={abs(m)} > ell={ell}")

    mp = -spin   # m' = -s, the module's Wigner-d convention

    c2 = np.sqrt(np.maximum((1.0 + mu) / 2.0, 0.0))  # cos(theta/2)
    s2 = np.sqrt(np.maximum((1.0 - mu) / 2.0, 0.0))  # sin(theta/2)

    kmin = max(0, mp - m)
    kmax = min(ell + mp, ell - m)

    prefactor = np.sqrt(
        factorial(ell + mp) * factorial(ell - mp)
        * factorial(ell + m) * factorial(ell - m)
    )

    d = np.zeros_like(mu)
    for k in range(kmin, kmax + 1):
        denom = (factorial(ell + mp - k) * factorial(k)
                 * factorial(ell - m - k) * factorial(k - mp + m))
        p_c = 2 * ell - 2 * k + mp - m
        p_s = 2 * k - mp + m
        d = d + (-1.0)**k * (prefactor / denom) * c2**p_c * s2**p_s

    norm = np.sqrt((2 * ell + 1) / (4.0 * np.pi))
    return norm * d


def gaussian_pulse(grid, r0, sigma_r, ell=2, m=2, spin=-2,
                   sigma_mu=None, amplitude=1.0):
    """Gaussian pulse in r times SWSH angular profile.

    Returns
        psi = amplitude * exp(-((R - r0)/sigma_r)^2) * swsh(spin, ell, m, MU)

    For a time-symmetric start pass psi as both psi0 and psi1 to
    Evolution.set_initial_data (v = 0 exactly).

    Parameters
    ----------
    grid     : Grid
    r0       : float, pulse center radius
    sigma_r  : float, pulse width in r
    ell, m   : int (default 2, 2)
    spin     : int (default -2)
    sigma_mu : float or None
        If given, multiply by exp(-(MU/sigma_mu)^2) to suppress the field
        near the poles (useful when the potential has a near-pole singularity).
    amplitude : float (default 1.0)

    Returns
    -------
    complex128 array of shape grid.shape
    """
    R  = grid.R
    MU = grid.MU

    f_r   = np.exp(-((R - r0) / sigma_r)**2)
    f_ang = swsh(spin, ell, m, MU)

    psi = amplitude * f_r * f_ang
    if sigma_mu is not None:
        psi = psi * np.exp(-(MU / sigma_mu)**2)

    return psi.astype(complex)
