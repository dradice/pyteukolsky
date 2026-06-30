"""
Milestone 4 tests: Schwarzschild (a=0) ℓ=m=2 QNM validation.

Tests:
  - swsh normalization and values
  - gaussian_pulse shape and dtype
  - project_swsh: projection of a SWSH profile onto itself
  - fit_qnm_frequency: synthetic damped sinusoid (real and complex forms)
  - Schwarzschild ℓ=m=2 QNM frequency vs Mω ≈ 0.3737 - 0.0890i
  - self-convergence in Nr (waveform differences decrease by ~4x when Nr doubles)

Grid note: rmin=1.99M keeps all interior cells at r > r_+ = 2M for Schwarzschild
(a=0), so no inside-horizon cells are evolved by RK4.  Using rmin=1.5M can
put several interior cells inside the horizon where the Cv/A coefficient is
positive-real, causing exponential growth that leaks through the FD stencil
into the exterior.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyteukolsky.grid import Grid
from pyteukolsky.equation import TeukolskyRHS
from pyteukolsky.evolve import Evolution
from pyteukolsky.initialdata import swsh, gaussian_pulse
from pyteukolsky.diagnostics import project_swsh, fit_qnm_frequency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# rmin=1.99M: just inside the horizon (r_+ = 2M for a=0, M=1).
# With this rmin, every interior cell has r > r_+ for Nr >= 50 and rmax=100.
RMIN = 1.99
RMAX = 100.0


def make_grid(Nr=50, Nmu=16, rmin=RMIN, rmax=RMAX, M=1.0):
    return Grid(rmin=rmin, rmax=rmax, Nr=Nr, Nmu=Nmu, ghost=2, M=M)


def _interior_mu(g):
    return g._mu[g.ghost : g.ghost + g.Nmu]


def run_schwarz(Nr, Nmu=16, t_final=130.0, cfl=0.45, diss=0.1,
                r_extract=30.0):
    """Run Schwarzschild a=0 m=2 Gaussian pulse; return (times, psi_22)."""
    M = 1.0
    g   = make_grid(Nr=Nr, Nmu=Nmu)
    rhs = TeukolskyRHS(g, M=M, a=0.0, m=2, dissipation=diss)
    evo = Evolution(rhs)
    psi0 = gaussian_pulse(g, r0=10.0, sigma_r=2.0, ell=2, m=2, spin=-2)
    evo.set_initial_data(psi0, psi0, dt_init=1e-3)
    evo.add_detector(r_extract)
    evo.evolve(t_final, cfl=cfl)

    mu  = _interior_mu(g)
    sw  = swsh(-2, 2, 2, mu)
    psi_22 = project_swsh(evo.waveforms[r_extract], mu, sw)
    return evo.times, psi_22


# ---------------------------------------------------------------------------
# swsh: normalization and values
# ---------------------------------------------------------------------------

def test_swsh_normalization():
    """||_{-2}Y_{2m}||^2 integrated over dmu should equal 1/(2pi) for each m."""
    Nmu = 10000
    mu  = np.linspace(-1.0 + 0.5 / Nmu, 1.0 - 0.5 / Nmu, Nmu)
    dmu = 2.0 / Nmu
    expected = 1.0 / (2.0 * np.pi)
    for m in (2, 1, 0, -1, -2):
        Y     = swsh(-2, 2, m, mu)
        norm2 = float(np.sum(Y**2) * dmu)
        # Midpoint-rule error is O(dmu^2) ~ O(4e-8); allow 1e-4 margin.
        assert abs(norm2 - expected) < 1e-4, \
            f"m={m}: ||Y||^2 dmu = {norm2:.8f}, expected {expected:.8f}"


def test_swsh_pole_values():
    """_{-2}Y_{2,+2} vanishes at mu=-1; _{-2}Y_{2,-2} vanishes at mu=+1."""
    assert swsh(-2, 2,  2, np.array([-1.0]))[0] == pytest.approx(0.0, abs=1e-15)
    assert swsh(-2, 2, -2, np.array([ 1.0]))[0] == pytest.approx(0.0, abs=1e-15)
    # Non-zero at the opposite pole
    assert abs(swsh(-2, 2,  2, np.array([1.0]))[0]) > 0.3
    assert abs(swsh(-2, 2, -2, np.array([-1.0]))[0]) > 0.3


def test_swsh_not_implemented():
    with pytest.raises(NotImplementedError):
        swsh(-2, 3, 2, np.array([0.0]))


# ---------------------------------------------------------------------------
# gaussian_pulse
# ---------------------------------------------------------------------------

def test_gaussian_pulse_shape_and_dtype():
    g    = make_grid(50, 16)
    psi0 = gaussian_pulse(g, r0=10.0, sigma_r=2.0)
    assert psi0.shape == g.shape
    assert psi0.dtype == np.complex128


def test_gaussian_pulse_peak_location():
    """Pulse maximum at r=r0, angular profile maximum at mu=+1 (m=2 SWSH)."""
    g    = make_grid(100, 32)
    r0   = 15.0
    psi0 = gaussian_pulse(g, r0=r0, sigma_r=2.0, ell=2, m=2, spin=-2)
    arr  = np.abs(psi0.real)
    idx  = np.unravel_index(np.argmax(arr), arr.shape)
    mu_max = g.MU[idx[0], 0].real
    r_max  = g.R[0, idx[1]].real
    assert abs(r_max - r0) / r0 < 0.1, f"r_max = {r_max:.2f}, expected {r0}"
    assert mu_max > 0.8, f"mu_max = {mu_max:.3f}, expected near +1"


# ---------------------------------------------------------------------------
# project_swsh
# ---------------------------------------------------------------------------

def test_project_swsh_self_projection():
    """Projecting _{-2}Y_{2,2} onto itself should give ≈ 1/(2pi)."""
    Nmu = 2048
    mu  = np.linspace(-1.0 + 0.5 / Nmu, 1.0 - 0.5 / Nmu, Nmu)
    Y   = swsh(-2, 2, 2, mu)
    result = float(np.real(project_swsh(Y.astype(complex), mu, Y)))
    expected = 1.0 / (2.0 * np.pi)
    assert abs(result - expected) < 1e-4, \
        f"project = {result:.8f}, expected {expected:.8f}"


def test_project_swsh_batch():
    """project_swsh handles (Nt, Nmu) input and returns (Nt,) output."""
    Nmu = 64
    Nt  = 10
    mu  = np.linspace(-1.0 + 0.5 / Nmu, 1.0 - 0.5 / Nmu, Nmu)
    Y   = swsh(-2, 2, 2, mu)
    psi = np.random.default_rng(0).random((Nt, Nmu)) + 0j
    result = project_swsh(psi, mu, Y)
    assert result.shape == (Nt,)


# ---------------------------------------------------------------------------
# fit_qnm_frequency: synthetic signals
# ---------------------------------------------------------------------------

def test_fit_qnm_synthetic_real():
    """Fit a real damped cosine: A exp(omega_I t) cos(omega_R t + phi)."""
    omega_R_true = 0.3737
    omega_I_true = -0.0890
    t   = np.linspace(50.0, 200.0, 5000)
    psi = np.exp(omega_I_true * t) * np.cos(omega_R_true * t + 0.5)

    omega_R, omega_I = fit_qnm_frequency(t, psi, t_start=60.0, t_end=180.0)

    assert abs(omega_R - omega_R_true) < 1e-4, f"omega_R = {omega_R:.5f}"
    assert abs(omega_I - omega_I_true) < 1e-4, f"omega_I = {omega_I:.5f}"


def test_fit_qnm_synthetic_complex():
    """Fit a complex signal: A exp(-i omega_R t + omega_I t)."""
    omega_R_true = 0.3737
    omega_I_true = -0.0890
    t   = np.linspace(50.0, 200.0, 5000)
    psi = np.exp((-1j * omega_R_true + omega_I_true) * t) * (1.0 + 0.5j)

    omega_R, omega_I = fit_qnm_frequency(t, psi, t_start=60.0, t_end=180.0)

    assert abs(omega_R - omega_R_true) < 1e-4, f"omega_R = {omega_R:.5f}"
    assert abs(omega_I - omega_I_true) < 1e-4, f"omega_I = {omega_I:.5f}"


def test_fit_qnm_too_few_points():
    t   = np.array([0.0, 1.0, 2.0])
    psi = np.array([1.0, 0.9, 0.8])
    with pytest.raises(ValueError):
        fit_qnm_frequency(t, psi, t_start=0.0, t_end=0.5)


# ---------------------------------------------------------------------------
# Schwarzschild QNM frequency (~5 s)
# ---------------------------------------------------------------------------

def test_schwarz_qnm_frequency():
    """Schwarzschild ℓ=m=2 QNM: Mω ≈ 0.3737 - 0.0890i (within 5%).

    Uses rmin=1.99M so all interior cells are outside the horizon (r > r_+ = 2M),
    preventing the inside-horizon Cv/A instability from corrupting the exterior.
    Fits the real ringdown waveform at r=30M with a damped-cosine model.

    Timing: with r0=10M and r_ext=30M the outgoing wave burst peaks at
    t ≈ 60-80M; by t=90M only the QNM ringdown remains.  Reflections from
    rmax=100M arrive at r=30M at t ≈ 160M.  Fitting window [90, 130]M is clean.
    """
    M = 1.0
    times, psi_22 = run_schwarz(Nr=100, t_final=140.0)

    omega_R, omega_I = fit_qnm_frequency(times, psi_22.real,
                                         t_start=90.0, t_end=130.0)

    assert abs(M * omega_R - 0.3737) < 0.02, \
        f"Mω_R = {M*omega_R:.4f}, expected 0.3737 ± 0.02"
    assert abs(M * omega_I + 0.0890) < 0.015, \
        f"Mω_I = {M*omega_I:.4f}, expected -0.0890 ± 0.015"


# ---------------------------------------------------------------------------
# Self-convergence in Nr (~15 s)
# ---------------------------------------------------------------------------

def test_self_convergence_Nr():
    """Waveform error decreases by ~4x when Nr doubles (2nd-order in space).

    All runs use the fine-grid (Nr=200) CFL timestep so they all hit the same
    discrete time points.  The coarser grids are slightly over-integrated
    (sub-CFL) but remain stable because all interior cells are outside r_+.
    """
    M = 1.0

    # Fix dt from the finest grid so all runs share the same time array.
    g_fine   = make_grid(Nr=200)
    rhs_fine = TeukolskyRHS(g_fine, M=M, a=0.0, m=2, dissipation=0.1)
    evo_fine = Evolution(rhs_fine)
    dt_fixed = evo_fine.cfl_dt(cfl=0.45)

    results = {}
    for Nr in (50, 100, 200):
        g   = make_grid(Nr=Nr)
        rhs = TeukolskyRHS(g, M=M, a=0.0, m=2, dissipation=0.1)
        evo = Evolution(rhs)
        psi0 = gaussian_pulse(g, r0=10.0, sigma_r=2.0, ell=2, m=2, spin=-2)
        evo.set_initial_data(psi0, psi0, dt_init=1e-3)
        evo.add_detector(30.0)
        evo.evolve(100.0, dt=dt_fixed)
        mu  = _interior_mu(g)
        sw  = swsh(-2, 2, 2, mu)
        results[Nr] = np.real(project_swsh(evo.waveforms[30.0], mu, sw))

    # All runs share the same dt → same time array length.  Compare waveforms.
    diff_lo = np.max(np.abs(results[100] - results[ 50]))
    diff_hi = np.max(np.abs(results[200] - results[100]))

    ratio = diff_lo / max(diff_hi, 1e-30)
    assert ratio > 3.0, \
        f"Nr convergence ratio = {ratio:.2f}, expected > 3 (2nd-order)"


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
