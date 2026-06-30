"""
Initial-data helpers for the Teukolsky solver.
"""

import numpy as np


def swsh(spin, ell, m, mu):
    """Spin-weighted spherical harmonic _{spin}Y_{ell,m}(mu) at phi=0.

    Implemented analytically for spin=-2, ell=2 using Wigner d-matrix elements:
        _{s}Y_{lm}(theta) = sqrt((2l+1)/4pi) * d^l_{-s,m}(theta)

    with c2 = cos(theta/2), s2 = sin(theta/2) expressed in mu = cos(theta):
        c2 = sqrt((1+mu)/2),  s2 = sqrt((1-mu)/2)

    Parameters
    ----------
    spin : int  (must be -2)
    ell  : int  (must be 2)
    m    : int  (-ell <= m <= ell)
    mu   : array_like  mu = cos(theta) in [-1, 1]

    Returns
    -------
    ndarray, real, same shape as mu.
    """
    mu = np.asarray(mu, dtype=float)
    if spin != -2 or ell != 2:
        raise NotImplementedError("Only spin=-2, ell=2 implemented")
    if abs(m) > ell:
        raise ValueError(f"|m|={abs(m)} > ell={ell}")

    c2 = np.sqrt(np.maximum((1.0 + mu) / 2.0, 0.0))  # cos(theta/2)
    s2 = np.sqrt(np.maximum((1.0 - mu) / 2.0, 0.0))  # sin(theta/2)
    norm = np.sqrt(5.0 / (4.0 * np.pi))

    # d^2_{2,m}(theta) elements computed from the Wigner formula (k-sum):
    #   m= 2:  c2^4
    #   m= 1: -2 c2^3 s2
    #   m= 0:  sqrt(6) c2^2 s2^2
    #   m=-1: -2 c2 s2^3
    #   m=-2:  s2^4
    if m == 2:
        return norm * c2**4
    elif m == 1:
        return -2.0 * norm * c2**3 * s2
    elif m == 0:
        return np.sqrt(6.0) * norm * c2**2 * s2**2
    elif m == -1:
        return -2.0 * norm * c2 * s2**3
    else:  # m == -2
        return norm * s2**4


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
