"""
Waveform diagnostics: SWSH projection, QNM frequency extraction, and
reconstruction of the Weyl scalar psi_4 from the evolved mode field psi_m.
"""

import numpy as np
from scipy.optimize import curve_fit


def psi4_kinnersley(psi_m, r, mu, M, a, m, phi=0.0):
    r"""Kinnersley-frame Weyl scalar psi_4 from the evolved mode field psi_m.

    The solver evolves psi_m, the azimuthal-m mode of psi = zeta^4 * psi_4 in
    *ingoing Kerr-Schild* (horizon-penetrating) coordinates, with
    zeta = r - i a cos(theta).  Two useful Weyl scalars follow (see
    scripts/check_equations.py, CHECK 4):

        psi_4^KS  = psi_m e^{i m phi} / zeta^4          (horizon-penetrating tetrad)
        psi_4^Kin = Delta^2 * psi_4^KS                  (Kinnersley / radiation frame)

    The horizon-penetrating tetrad is the Kinnersley one with the ingoing leg
    regularised, l -> Delta l (Kinnersley l^mu ~ 1/Delta diverges at r_+).
    Because psi_4 is quadratic in the ingoing null leg n -> n/Delta, the scalar
    picks up psi_4 -> psi_4 / Delta^2.  Undoing that boost (multiplying by
    Delta^2) recovers the standard s=-2 *peeling* behaviour psi_4^Kin ~ 1/r at
    large r, whereas the bare psi_m/zeta^4 falls as 1/r^5.  (At the horizon
    Delta -> 0, so psi_4^Kin -> 0 while the code field psi_m/zeta^4 stays finite
    — the regularity that motivates the horizon-penetrating formulation.)

    This returns the Kinnersley scalar; drop the ``Delta**2`` factor for the
    horizon-penetrating one.

    Parameters
    ----------
    psi_m : ndarray, complex
        Evolved mode field, e.g. detector data of shape (Nt, Nmu) or a full
        grid slice.  ``r`` and ``mu`` are broadcast against it.
    r : float or ndarray
        Extraction radius / radial coordinate (broadcastable to psi_m).
    mu : float or ndarray
        mu = cos(theta) (broadcastable to psi_m).
    M, a : float
        Black-hole mass and spin.
    m : int
        Azimuthal mode number.
    phi : float, optional
        Azimuthal angle at which to evaluate the e^{i m phi} factor (default 0).

    Returns
    -------
    ndarray, complex — psi_4^Kinnersley, same broadcast shape as the inputs.
    """
    zeta  = r - 1j * a * mu
    Delta = r**2 - 2.0 * M * r + a**2
    return psi_m * np.exp(1j * m * phi) * Delta**2 / zeta**4


def project_swsh(psi_mu, mu, swsh_profile):
    """Project psi_mu onto a SWSH profile using midpoint-rule quadrature.

    Parameters
    ----------
    psi_mu       : ndarray shape (..., Nmu), complex
        Field values on the uniform staggered interior mu grid.
    mu           : ndarray shape (Nmu,), real
        Interior mu grid points (uniform, staggered).
    swsh_profile : ndarray shape (Nmu,), real
        Pre-evaluated SWSH values at mu grid points.

    Returns
    -------
    ndarray shape (...), complex — the projected amplitude(s).
    """
    dmu = mu[1] - mu[0]
    return np.sum(psi_mu * np.conj(swsh_profile), axis=-1) * dmu


def fit_qnm_frequency(times, psi_t, t_start, t_end):
    """Extract QNM frequency from a waveform in the fitting window [t_start, t_end].

    Two code paths depending on whether the input is real or complex:

    Real signal (Schwarzschild / real initial data):
        psi(t) ~ A exp(omega_I t) cos(omega_R t + phi)
        Fitted by nonlinear least squares (scipy.optimize.curve_fit) with
        initial guesses from the zero-crossing rate and log-amplitude slope.
        Time is centered at t0 = (t_start + t_end)/2 for numerical stability.

    Complex signal (Kerr / complex initial data):
        psi(t) ~ A exp(-i omega_R t + omega_I t)
        Fitted by linear regression of log|psi| (-> omega_I) and the
        unwrapped complex phase (-> -omega_R).

    Parameters
    ----------
    times  : 1D array, shape (Nt,)
    psi_t  : 1D array, shape (Nt,), real or complex
    t_start, t_end : float, fitting window

    Returns
    -------
    (omega_R, omega_I) : floats
        omega_R > 0 (oscillation frequency), omega_I < 0 (decay rate).
    """
    mask = (times >= t_start) & (times <= t_end)
    t   = times[mask]
    psi = psi_t[mask]

    if len(t) < 4:
        raise ValueError("Fitting window contains fewer than 4 points.")

    is_real = (np.isrealobj(psi)
               or np.max(np.abs(np.imag(psi))) < 1e-10 * np.max(np.abs(psi)))

    if is_real:
        psi_r = np.real(psi)
        # Center time for numerical stability in curve_fit
        t0 = 0.5 * (t[0] + t[-1])
        tc = t - t0

        # Initial guess for omega_I from log-amplitude slope
        oI_0 = float(np.polyfit(tc, np.log(np.abs(psi_r) + 1e-30), 1)[0])

        # Initial guess for omega_R from zero-crossing half-period
        sc = np.where(np.diff(np.sign(psi_r)))[0]
        if len(sc) >= 2:
            oR_0 = float(np.pi / np.mean(np.diff(t[sc])))
        else:
            oR_0 = 2.0 * np.pi / (t[-1] - t[0])  # fallback: one full period

        # Initial amplitude near the envelope maximum
        i_max = int(np.argmax(np.abs(psi_r)))
        A_0 = float(np.abs(psi_r[i_max]) / np.exp(oI_0 * tc[i_max]))

        def _model(tc, A, phi, oR, oI):
            return A * np.exp(oI * tc) * np.cos(oR * tc + phi)

        popt, _ = curve_fit(_model, tc, psi_r,
                            p0=[A_0, 0.0, oR_0, oI_0], maxfev=20000)
        return float(abs(popt[2])), float(popt[3])

    else:
        # Complex signal: linear fits on log-amplitude and unwrapped phase
        z = psi.astype(complex)
        log_amp = np.log(np.abs(z))
        omega_I = float(np.polyfit(t, log_amp, 1)[0])
        phase   = np.unwrap(np.angle(z))
        omega_R = float(-np.polyfit(t, phase, 1)[0])
        return omega_R, omega_I
