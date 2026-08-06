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
from pyteukolsky.evolve import Evolution, SnapshotWriter, n_evolve_steps


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
# set_state: seed (psi, v) directly, including psi=0 with a nonzero v,
# which set_initial_data cannot express.
# ---------------------------------------------------------------------------

def test_set_state_array():
    evo, g, _ = make_evo()
    psi_arr = gaussian_psi(g, r0=10.0)
    v_arr   = gaussian_psi(g, r0=12.0)
    evo.set_state(psi_arr, v_arr, t=5.0)
    assert np.allclose(evo.psi, psi_arr)
    assert np.allclose(evo.v,   v_arr)
    assert evo.t == 5.0


def test_set_state_default_t_is_zero():
    evo, g, _ = make_evo()
    evo.t = 99.0
    psi_arr = gaussian_psi(g, r0=10.0)
    evo.set_state(psi_arr, np.zeros(g.shape, dtype=complex))
    assert evo.t == 0.0


def test_set_state_callable():
    evo, g, _ = make_evo()
    f_psi = lambda R, MU: np.exp(-((R - 10.0) / 2.0)**2).astype(complex)
    f_v   = lambda R, MU: np.zeros(R.shape, dtype=complex)
    evo.set_state(f_psi, f_v)
    assert np.allclose(evo.psi, f_psi(g.R, g.MU))
    assert np.allclose(evo.v,   0.0)


def test_set_state_v_only_family():
    """The v-only basis family: psi=0, v=bump. set_initial_data cannot
    express this cleanly (v = (psi1-psi0)/dt_init requires two psi slices);
    set_state sets v directly."""
    evo, g, _ = make_evo()
    v_bump = gaussian_psi(g, r0=10.0, sigma=1.5)
    evo.set_state(np.zeros(g.shape, dtype=complex), v_bump)
    assert np.allclose(evo.psi, 0.0)
    assert np.allclose(evo.v,   v_bump)


def test_set_state_dtype_is_complex():
    evo, g, _ = make_evo()
    psi_real = np.exp(-((g.R.real - 10.0))**2)   # plain real array
    v_real   = np.zeros(g.shape)
    evo.set_state(psi_real, v_real)
    assert evo.psi.dtype == np.complex128
    assert evo.v.dtype   == np.complex128


def test_set_state_then_step_evolves():
    """A run seeded via set_state should evolve just like set_initial_data."""
    evo, g, _ = make_evo()
    v_bump = gaussian_psi(g, r0=10.0, sigma=1.5)
    evo.set_state(np.zeros(g.shape, dtype=complex), v_bump)
    psi_before = evo.psi.copy()
    dt = evo.cfl_dt(cfl=0.3)
    evo.step(dt)
    assert not np.allclose(evo.psi, psi_before)


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
# provenance() (§4.4)
# ---------------------------------------------------------------------------

def test_provenance_core_fields():
    evo, g, rhs = make_evo(a=0.3, m=2)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, cfl=0.3, dt=dt, record_every=2)
    prov = evo.provenance()
    for key, expected in [
        ('M', rhs.M), ('a', rhs.a), ('m', rhs.m),
        ('Nr', g.Nr), ('Nmu', g.Nmu), ('rmin', g.rmin), ('rmax', g.rmax),
        ('order', g.order), ('ghost', g.ghost),
        ('dissipation', rhs.dissipation),
        ('one_sided_horizon', rhs.one_sided_horizon),
        ('cfl', 0.3), ('dt', dt), ('t_final', 3 * dt), ('record_every', 2),
    ]:
        assert key in prov, f"missing provenance field {key!r}"
        assert prov[key] == expected, f"{key}: {prov[key]!r} != {expected!r}"
    assert 'git_commit' in prov
    assert isinstance(prov['git_commit'], str) and len(prov['git_commit']) > 0


def test_provenance_before_evolve_omits_run_fields():
    """cfl/dt/t_final/record_every are omitted if evolve() hasn't run yet
    (rather than crashing or emitting bogus values)."""
    evo, g, _ = make_evo()
    prov = evo.provenance()
    for key in ('cfl', 'dt', 't_final', 'record_every'):
        assert key not in prov
    assert 'M' in prov and 'git_commit' in prov


def test_provenance_extra_kwargs_merged():
    """Basis descriptors (family, bump_index, x_center, ...) are merged in
    and take precedence over auto-populated fields."""
    evo, g, _ = make_evo()
    prov = evo.provenance(family="v", bump_index=7, x_center=1.23,
                           r_center=3.4, sigma_x=0.05, l_seed=[2, 3, 4, 5, 6],
                           M=999.0)   # deliberately override an auto field
    assert prov['family'] == "v"
    assert prov['bump_index'] == 7
    assert prov['x_center'] == 1.23
    assert prov['l_seed'] == [2, 3, 4, 5, 6]
    assert prov['M'] == 999.0


def test_save_waveforms_backward_compatible_without_provenance():
    """save_waveforms(path) with no provenance arg saves exactly the same
    keys as before this feature was added."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, dt=dt)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "wf_compat")
        evo.save_waveforms(path)
        loaded = np.load(path + ".npz")
        assert set(loaded.keys()) == {
            'times', 'mu_grid', 'Nmu', 'M', 'a', 'm', 't_current',
            'waveform_15.000000',
        }


def test_save_waveforms_with_provenance():
    evo, g, _ = make_evo(a=0.2)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    evo.add_detector(15.0)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=3 * dt, dt=dt)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "wf_prov")
        evo.save_waveforms(path, provenance=evo.provenance(family="psi",
                                                             bump_index=2))
        loaded = np.load(path + ".npz")
        assert 'rmin' in loaded and 'rmax' in loaded and 'order' in loaded
        assert 'git_commit' in loaded
        assert loaded['family'] == "psi"
        assert loaded['bump_index'] == 2


def test_save_snapshots_backward_compatible_without_provenance():
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    evo.evolve(t_final=4 * dt, dt=dt, snapshot_every=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "snap_compat")
        evo.save_snapshots(path)
        loaded = np.load(path + ".npz")
        assert set(loaded.keys()) == {
            'times_snap', 'psi', 'r_grid', 'mu_grid', 'Nr', 'Nmu', 'ghost',
            'M', 'a', 'm',
        }


# ---------------------------------------------------------------------------
# n_evolve_steps / SnapshotWriter (§4.4 streaming snapshots)
# ---------------------------------------------------------------------------

def test_n_evolve_steps_matches_evolve_loop():
    """n_evolve_steps must exactly predict how many (t, psi) pairs a plain
    (non-streaming) evolve() run with snapshot_every=1 records."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 7 * dt + 0.37 * dt   # deliberately not an exact multiple of dt
    evo.evolve(t_final=t_final, dt=dt, snapshot_every=1)
    assert len(evo.snapshots) == n_evolve_steps(t_final, dt)


def test_snapshot_writer_count_snapshots_matches_ram_path():
    """SnapshotWriter.count_snapshots must match len(evo.snapshots) from
    the equivalent non-streaming run for the same (t_final, dt, every)."""
    evo, g, _ = make_evo()
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 9 * dt
    evo.evolve(t_final=t_final, dt=dt, snapshot_every=3)
    n_expected = SnapshotWriter.count_snapshots(t_final, dt, 3)
    assert n_expected == len(evo.snapshots)


def test_snapshot_writer_streams_to_disk_not_ram():
    """When snapshot_writer is given, evo.snapshots must stay empty (data
    goes to disk instead of accumulating in RAM)."""
    evo, g, _ = make_evo(Nr=20, Nmu=16)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 6 * dt
    n_snap = SnapshotWriter.count_snapshots(t_final, dt, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SnapshotWriter(os.path.join(tmpdir, "stream"), g, n_snap)
        evo.evolve(t_final=t_final, dt=dt, snapshot_every=2,
                   snapshot_writer=writer)
        writer.close()
        assert evo.snapshots == []
        psi_arr = np.load(os.path.join(tmpdir, "stream_psi.npy"))
        assert psi_arr.shape == (n_snap, g.Nmu, g.Nr)   # interior-only default


def test_snapshot_writer_matches_ram_values():
    """The streamed snapshots must equal (up to the complex64 downcast) the
    interior psi values that the RAM-accumulating path would have stored."""
    evo_ram, g, _ = make_evo(Nr=20, Nmu=16)
    psi = gaussian_psi(g)
    evo_ram.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo_ram.cfl_dt(cfl=0.3)
    t_final = 6 * dt
    evo_ram.evolve(t_final=t_final, dt=dt, snapshot_every=2)

    evo_stream, g2, _ = make_evo(Nr=20, Nmu=16)
    evo_stream.set_initial_data(psi, psi, dt_init=1e-3)
    n_snap = SnapshotWriter.count_snapshots(t_final, dt, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SnapshotWriter(os.path.join(tmpdir, "stream2"), g2, n_snap,
                                 interior_only=True, dtype=np.complex64)
        evo_stream.evolve(t_final=t_final, dt=dt, snapshot_every=2,
                           snapshot_writer=writer)
        writer.close()
        psi_arr = np.load(os.path.join(tmpdir, "stream2_psi.npy"))

    gh = g.ghost
    for k, (t, psi_full) in enumerate(evo_ram.snapshots):
        expected = psi_full[gh:gh + g.Nmu, gh:gh + g.Nr].astype(np.complex64)
        assert np.allclose(psi_arr[k], expected, rtol=1e-6, atol=1e-6)


def test_snapshot_writer_radial_clip():
    """r_save_max clips the streamed snapshot's radial extent."""
    evo, g, _ = make_evo(Nr=40, Nmu=16, M=1.0)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 4 * dt
    n_snap = SnapshotWriter.count_snapshots(t_final, dt, 2)
    r_int = g.r[g.ghost:g.ghost + g.Nr]
    r_save_max = float(r_int[len(r_int) // 2])   # keep roughly the inner half
    n_expected_cols = int(np.sum(r_int <= r_save_max))
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SnapshotWriter(os.path.join(tmpdir, "clip"), g, n_snap,
                                 interior_only=True, r_save_max=r_save_max)
        evo.evolve(t_final=t_final, dt=dt, snapshot_every=2,
                   snapshot_writer=writer)
        writer.close()
        psi_arr = np.load(os.path.join(tmpdir, "clip_psi.npy"))
        assert psi_arr.shape[2] == n_expected_cols
        assert psi_arr.shape[2] < g.Nr   # actually clipped, not a no-op


def test_snapshot_writer_dtype_downcast():
    evo, g, _ = make_evo(Nr=20, Nmu=16)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 4 * dt
    n_snap = SnapshotWriter.count_snapshots(t_final, dt, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SnapshotWriter(os.path.join(tmpdir, "dc"), g, n_snap,
                                 dtype=np.complex64)
        evo.evolve(t_final=t_final, dt=dt, snapshot_every=2,
                   snapshot_writer=writer)
        writer.close()
        psi_arr = np.load(os.path.join(tmpdir, "dc_psi.npy"))
        assert psi_arr.dtype == np.complex64


def test_snapshot_writer_meta_and_provenance():
    evo, g, _ = make_evo(Nr=20, Nmu=16)
    psi = gaussian_psi(g)
    evo.set_initial_data(psi, psi, dt_init=1e-3)
    dt = evo.cfl_dt(cfl=0.3)
    t_final = 4 * dt
    n_snap = SnapshotWriter.count_snapshots(t_final, dt, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta")
        writer = SnapshotWriter(path, g, n_snap)
        evo.evolve(t_final=t_final, dt=dt, snapshot_every=2,
                   snapshot_writer=writer)
        writer.close(provenance=evo.provenance(family="psi", bump_index=1))
        meta = np.load(path + "_meta.npz", allow_pickle=True)
        assert 'times_snap' in meta and len(meta['times_snap']) == n_snap
        assert 'r_grid' in meta and 'mu_grid' in meta
        assert meta['family'] == "psi"
        assert meta['bump_index'] == 1
        assert 'git_commit' in meta


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
