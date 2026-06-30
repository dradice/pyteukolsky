"""
Waveform diagnostics: SWSH projection and QNM frequency extraction.
"""

import numpy as np
from scipy.optimize import curve_fit


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
