"""
Milestone 3 tests: Evolution class (RK4 driver, CFL, detectors, save).

Tests cover:
  - construction and state initialisation
  - CFL time step (positive, scales with resolution)
  - initial-data seeding (array and callable forms)
  - detector registration and interpolation weights
  - single RK4 step changes the state
  - full evolve loop produces correct output shapes
  - zero initial data stays zero (linear PDE)
  - save() writes a readable .npz with the expected keys
  - Sommerfeld BC: existing milestone-2 linearity still holds at interior
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyteukolsky.grid import Grid
from pyteukolsky.equation import TeukolskyRHS
from pyteukolsky.evolve import Evolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evo(Nr=30, Nmu=24, M=1.0, a=0.0, m=2, dissipation=0.0):
    g   = Grid(rmin=1.5, rmax=40.0, Nmu=Nmu, Nr=Nr, ghost=2, M=M)
    rhs = TeukolskyRHS(g, M=M, a=a, m=m, dissipation=dissipation)
    return Evolution(rhs), g, rhs


def gaussian_psi(g, r0=10.0, sigma=1.5):
    """Smooth Gaussian pulse in r, uniform in mu (complex128)."""
    return np.exp(-((g.R - r0) / sigma)**2).astype(complex)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_state_shape():
    evo, g, _ = make_evo()
    assert evo.psi.shape == g.shape
    assert evo.v.shape   == g.shape


def test_init_state_zero():
    evo, _, _ = make_evo()
    assert np.all(evo.psi == 0)
    assert np.all(evo.v   == 0)
    assert evo.t == 0.0


def test_init_no_detectors():
    evo, _, _ = make_evo()
    assert evo._detectors == []
    assert evo.waveforms  == {}


# ---------------------------------------------------------------------------
# CFL time step
# ---------------------------------------------------------------------------

def test_cfl_positive():
    evo, _, _ = make_evo()
    dt = evo.cfl_dt(cfl=0.5)
    assert dt > 0.0


def test_cfl_scales_with_resolution():
    """Halving Nr should roughly double the CFL step (radial-dominated)."""
    evo1, _, _ = make_evo(Nr=60)
    evo2, _, _ = make_evo(Nr=30)
    dt1 = evo1.cfl_dt(cfl=0.5)
    dt2 = evo2.cfl_dt(cfl=0.5)
    # coarser grid → larger step
    assert dt2 > dt1

def test_cfl_cfl_factor():
    """cfl_dt(cfl=c) == c * cfl_dt(cfl=1)."""
    evo, _, _ = make_evo()
    dt1 = evo.cfl_dt(cfl=1.0)
    dt2 = evo.cfl_dt(cfl=0.4)
    assert np.isclose(dt2, 0.4 * dt1, rtol=1e-10)


# ---------------------------------------------------------------------------
# Initial data
# ---------------------------------------------------------------------------

def test_set_initial_data_array():
    evo, g, _ = make_evo()
    psi0 = gaussian_psi(g, r0=10.0)
    psi1 = gaussian_psi(g, r0=10.1)
    dt_init = 0.1
    evo.set_initial_data(psi0, psi1, dt_init)
    assert np.allclose(evo.psi, psi1)
    assert np.allclose(evo.v,   (psi1 - psi0) / dt_init)
    assert evo.t == 0.0


def test_set_initial_data_callable():
    evo, g, _ = make_evo()
    f = lambda R, MU: np.exp(-((R - 10.0) / 2.0)**2).astype(complex)
    evo.set_initial_data(f, f, dt_init=1e-3)
    expected = f(g.R, g.MU)
    assert np.allclose(evo.psi, expected)
    assert np.allclose(evo.v,   np.zeros_like(expected))


def test_set_initial_data_time_symmetric():
    """psi0 == psi1 → v = 0 everywhere."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    assert np.allclose(evo.v, 0.0)


def test_set_initial_data_resets_time():
    evo, g, _ = make_evo()
    evo.t = 99.0
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    assert evo.t == 0.0


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def test_add_detector_registers():
    evo, g, _ = make_evo()
    evo.add_detector(15.0)
    assert len(evo._detectors) == 1
    assert 15.0 in evo.waveforms


def test_add_detector_weights_sum_to_one():
    evo, g, _ = make_evo()
    evo.add_detector(15.0)
    _, _, _, w0, w1 = evo._detectors[0]
    assert np.isclose(w0 + w1, 1.0)


def test_add_detector_bracket():
    """The two bracketing radii must straddle r_extract."""
    evo, g, _ = make_evo()
    r_ext = 15.0
    evo.add_detector(r_ext)
    _, abs_i0, abs_i1, w0, w1 = evo._detectors[0]
    r0 = g.r[abs_i0]
    r1 = g.r[abs_i1]
    assert r0 <= r_ext <= r1


def test_add_multiple_detectors():
    evo, _, _ = make_evo()
    for r in (8.0, 15.0, 25.0):
        evo.add_detector(r)
    assert len(evo._detectors) == 3
    assert len(evo.waveforms)  == 3


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def test_step_changes_psi():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    psi_before = evo.psi.copy()
    dt = evo.cfl_dt(cfl=0.3)
    evo.step(dt)
    assert not np.allclose(evo.psi, psi_before)


def test_step_advances_time():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.step(dt)
    assert np.isclose(evo.t, dt)


def test_step_zero_stays_zero():
    """Zero initial data must remain zero (linearity of the PDE)."""
    evo, g, _ = make_evo()
    evo.set_initial_data(
        np.zeros(g.shape, dtype=complex),
        np.zeros(g.shape, dtype=complex),
        dt_init=1e-3,
    )
    dt = evo.cfl_dt(cfl=0.3)
    for _ in range(5):
        evo.step(dt)
    assert np.allclose(evo.psi, 0.0)
    assert np.allclose(evo.v,   0.0)


# ---------------------------------------------------------------------------
# Full evolve loop
# ---------------------------------------------------------------------------

def _short_evolve(Nr=30, Nmu=24, n_steps=4):
    """Run evolve() for exactly n_steps steps and return the Evolution."""
    evo, g, _ = make_evo(Nr=Nr, Nmu=Nmu)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=n_steps * dt, dt=dt, record_every=1)
    return evo, g


def test_evolve_times_length():
    n = 4
    evo, _ = _short_evolve(n_steps=n)
    assert len(evo.times) == n


def test_evolve_times_increasing():
    evo, _ = _short_evolve(n_steps=5)
    assert np.all(np.diff(evo.times) > 0)


def test_evolve_waveform_shape():
    n = 4
    evo, g = _short_evolve(n_steps=n)
    arr = evo.waveforms[15.0]
    assert arr.shape == (n, g.Nmu)


def test_evolve_waveform_dtype():
    evo, _ = _short_evolve()
    assert evo.waveforms[15.0].dtype == np.complex128


def test_evolve_record_every():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    n_steps = 6
    evo.evolve(t_final=n_steps * dt, dt=dt, record_every=2)
    assert len(evo.times) == n_steps // 2


def test_evolve_final_time():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 5 * dt
    evo.evolve(t_final=t_final, dt=dt)
    assert np.isclose(evo.t, t_final, rtol=1e-10)


def test_evolve_snapshots():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=4 * dt, dt=dt, snapshot_every=2)
    assert len(evo.snapshots) == 2
    t_snap, psi_snap = evo.snapshots[0]
    assert psi_snap.shape == g.shape


# ---------------------------------------------------------------------------
# Norm stability (loose sanity check — not a convergence test)
# ---------------------------------------------------------------------------

def test_evolve_norm_bounded():
    """L2 norm of psi on interior should not blow up in a short run."""
    evo, g, _ = make_evo(Nr=40, Nmu=32)
    psi = gaussian_psi(g, r0=10.0, sigma=1.0)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    norm0 = np.sqrt(np.sum(np.abs(evo.psi[g.interior])**2))
    dt = evo.cfl_dt(cfl=0.4)
    evo.evolve(t_final=20 * dt, dt=dt)
    norm1 = np.sqrt(np.sum(np.abs(evo.psi[g.interior])**2))
    assert norm1 < 10 * norm0  # very loose — just checks no explosion


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_waveforms_keys():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, dt=dt)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "waveforms")
        evo.save_waveforms(path)
        loaded = np.load(path + ".npz")
        assert 'times'   in loaded
        assert 'mu_grid' in loaded
        assert 'M' in loaded
        assert 'a' in loaded
        assert 'm' in loaded
        key = 'waveform_15.000000'
        assert key in loaded
        assert loaded[key].shape == (3, g.Nmu)


def test_save_waveforms_no_grid_arrays():
    """save_waveforms should not include the full radial grid (kept small)."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, dt=dt)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "waveforms2")
        evo.save_waveforms(path)
        loaded = np.load(path + ".npz")
        assert 'r_grid' not in loaded


def test_save_snapshots_keys():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=4 * dt, dt=dt, snapshot_every=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "snapshots")
        evo.save_snapshots(path)
        loaded = np.load(path + ".npz")
        assert 'times_snap' in loaded
        assert 'psi'        in loaded
        assert 'r_grid'     in loaded
        assert loaded['psi'].shape == (2,) + g.shape


def test_save_snapshots_empty():
    """save_snapshots with no snapshots produces an empty psi array."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, dt=dt)  # no snapshot_every

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "snapshots_empty")
        evo.save_snapshots(path)
        loaded = np.load(path + ".npz")
        assert loaded['psi'].shape[0] == 0


# ---------------------------------------------------------------------------
# Sommerfeld BC: milestone-2 linearity still holds
# ---------------------------------------------------------------------------

def test_rhs_linearity_with_sommerfeld():
    """With Sommerfeld BC, rhs(alpha*psi, alpha*v) == alpha*rhs(psi,v) interior."""
    from pyteukolsky.equation import TeukolskyRHS
    g   = Grid(rmin=1.5, rmax=40.0, Nmu=24, Nr=30, ghost=2, M=1.0)
    rhs = TeukolskyRHS(g, M=1.0, a=0.0, m=2)
    rng = np.random.default_rng(17)
    psi = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    v   = rng.standard_normal(g.shape) + 1j*rng.standard_normal(g.shape)
    alpha = 2.0 + 1j
    dp1, dv1 = rhs.rhs(psi, v)
    dp2, dv2 = rhs.rhs(alpha * psi, alpha * v)
    sl = g.interior
    assert np.allclose(dp2[sl], alpha * dp1[sl], rtol=1e-10)
    assert np.allclose(dv2[sl], alpha * dv1[sl], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
