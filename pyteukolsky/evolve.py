"""
Evolution: RK4 time integrator for the Teukolsky mode equation.

State vector: (psi, v) where v = d_t psi (first-order-in-time reduction).
Time integration uses classic explicit RK4 (method of lines).
The time step is set by a CFL condition derived from the principal-part
characteristic speeds (README §1.5).

Sommerfeld outgoing BC at the outer radial boundary is enforced inside
TeukolskyRHS.rhs(); the inner (excision) and pole BCs are also handled there.

Usage
-----
    evo = Evolution(rhs)
    evo.set_initial_data(psi0, psi1, dt_init)
    evo.add_detector(r_extract=100.0)
    evo.evolve(t_final=300.0, cfl=0.5)
    evo.save("waveforms.npz")
"""

import os
import subprocess

import numpy as np


def _git_commit_hash():
    """Return the current git commit hash (with a '-dirty' suffix if the
    working tree has uncommitted changes), or 'unknown' if git or the repo
    is unavailable (e.g. a pip-installed copy with no .git directory)."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=repo_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def n_evolve_steps(t_final, dt, t0=0.0):
    """Number of RK4 steps Evolution.evolve() will take to reach t_final
    from t0 at fixed step dt -- mirrors evolve()'s loop condition exactly
    (including the final, possibly-shorter step), so callers can size a
    streaming SnapshotWriter (or otherwise pre-allocate) before the loop
    runs.
    """
    remaining = t_final - t0
    t = 0.0
    n = 0
    while t < remaining - 1e-12 * dt:
        t += min(dt, remaining - t)
        n += 1
    return n


class SnapshotWriter:
    """Incrementally writes 2D psi snapshots to disk during evolve(), so
    they never accumulate in RAM. At large-scale run resolutions, full
    complex128 snapshots can be ~1.4 GB/run; this writer instead streams
    directly into a preallocated, disk-backed array (a plain numpy .npy
    memmap -- no extra dependencies) and supports interior-only selection,
    radial clipping, and float32/complex64 downcast to shrink the
    footprint further.

    Two files are produced by close():
      <path>_psi.npy   -- disk-backed array, shape (n_snap, n_mu, n_r)
      <path>_meta.npz  -- times_snap, r_grid, mu_grid, n_snap, and any
                           provenance passed to close()

    Usage
    -----
        n_snap = SnapshotWriter.count_snapshots(t_final, dt, snapshot_every)
        writer = SnapshotWriter(path, grid, n_snap, r_save_max=120.0)
        evo.evolve(t_final, dt=dt, snapshot_every=snapshot_every,
                   snapshot_writer=writer)
        writer.close(provenance=evo.provenance(family="psi", bump_index=3))
    """

    def __init__(self, path, grid, n_snap, interior_only=True,
                 r_save_max=None, dtype=np.complex64):
        g = self.grid = grid
        if interior_only:
            mu0, mu1 = g.ghost, g.ghost + g.Nmu
            r0, r1 = g.ghost, g.ghost + g.Nr
        else:
            mu0, mu1 = 0, g.shape[0]
            r0, r1 = 0, g.shape[1]
        if r_save_max is not None:
            r_vals = g.r[r0:r1].real
            n_keep = int(np.searchsorted(r_vals, r_save_max, side="right"))
            n_keep = int(np.clip(n_keep, 1, r1 - r0))
            r1 = r0 + n_keep

        self.mu_sl = slice(mu0, mu1)
        self.r_sl  = slice(r0, r1)
        self.dtype = dtype
        self.n_snap = int(n_snap)
        self.path = path

        shape = (self.n_snap, mu1 - mu0, r1 - r0)
        self._psi_path = path + "_psi.npy"
        self.psi = np.lib.format.open_memmap(
            self._psi_path, mode="w+", dtype=dtype, shape=shape)
        self.times = np.empty(self.n_snap, dtype=float)
        self._i = 0

    @staticmethod
    def count_snapshots(t_final, dt, snapshot_every, t0=0.0):
        """Exact number of snapshots evolve() will produce for the given
        (t_final, dt, snapshot_every) -- use this to size n_snap up front."""
        return n_evolve_steps(t_final, dt, t0) // snapshot_every

    def write(self, t, psi_full):
        """Write one snapshot slice of the full-grid psi array. Must be
        called at most n_snap times (evolve() enforces this by construction
        when n_snap comes from count_snapshots)."""
        if self._i >= self.n_snap:
            raise IndexError(
                f"SnapshotWriter received more than the preallocated "
                f"n_snap={self.n_snap} writes; recompute n_snap with "
                f"count_snapshots() for the actual (t_final, dt, "
                f"snapshot_every)."
            )
        self.psi[self._i] = psi_full[self.mu_sl, self.r_sl].astype(self.dtype)
        self.times[self._i] = t
        self._i += 1

    def close(self, provenance=None):
        """Flush the memmap and write the companion metadata .npz."""
        self.psi.flush()
        g = self.grid
        meta = dict(
            times_snap=self.times[:self._i],
            r_grid=g.r[self.r_sl],
            mu_grid=g._mu[self.mu_sl],
            n_snap=np.array(self._i),
            psi_path=os.path.basename(self._psi_path),
        )
        if provenance:
            meta.update(provenance)
        np.savez(self.path + "_meta.npz", **meta)


class Evolution:
    """RK4 driver for a single Teukolsky azimuthal mode.

    Parameters
    ----------
    rhs : TeukolskyRHS
        Precomputed right-hand-side object (owns the Grid and physics).
    """

    def __init__(self, rhs):
        self.rhs_obj = rhs
        self.grid = rhs.grid
        self.psi = np.zeros(self.grid.shape, dtype=complex)
        self.v   = np.zeros(self.grid.shape, dtype=complex)
        self.t   = 0.0
        self._detectors = []   # list of (r_ext, abs_i0, abs_i1, w0, w1)
        self.times     = np.array([], dtype=float)
        self.waveforms = {}    # {r_ext: ndarray shape (Nt, Nmu)} after evolve()
        self.snapshots = []    # list of (t, psi_copy) if snapshot_every is used

    # ------------------------------------------------------------------
    # Initial data
    # ------------------------------------------------------------------

    def set_initial_data(self, psi0, psi1, dt_init):
        """Seed state from two time slices separated by dt_init.

        Parameters
        ----------
        psi0, psi1 : array_like or callable
            Field values at t=0 and t=dt_init.  Callables are called as
            f(R, MU) where R, MU are 2D meshes of shape grid.shape.
        dt_init : float
            Time separation used to compute v = (psi1 - psi0) / dt_init.
        """
        g = self.grid
        if callable(psi0):
            psi0 = psi0(g.R, g.MU)
        if callable(psi1):
            psi1 = psi1(g.R, g.MU)
        psi0 = np.asarray(psi0, dtype=complex)
        psi1 = np.asarray(psi1, dtype=complex)
        self.psi[:] = psi1
        self.v[:]   = (psi1 - psi0) / dt_init
        self.t      = 0.0

    def set_state(self, psi, v, t=0.0):
        """Seed (psi, v) directly. Callables are called as f(R, MU).

        Unlike set_initial_data (which derives v from two closely-spaced
        time slices), this sets psi and v independently -- needed for
        initial data with psi=0 and a nonzero v, which set_initial_data
        cannot express cleanly.

        Parameters
        ----------
        psi, v : array_like or callable
            Field values. Callables are called as f(R, MU) where R, MU are
            2D meshes of shape grid.shape.
        t : float, optional
            Initial time (default 0.0).
        """
        g = self.grid
        if callable(psi):
            psi = psi(g.R, g.MU)
        if callable(v):
            v = v(g.R, g.MU)
        psi = np.asarray(psi, dtype=complex)
        v   = np.asarray(v, dtype=complex)
        self.psi[:] = psi
        self.v[:]   = v
        self.t      = t

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def add_detector(self, r_extract):
        """Register an extraction radius for waveform recording.

        Linear interpolation in r between the two nearest interior grid
        points.  The recorded quantity is psi(t, r_extract, mu_j) for
        all interior angular cells j.

        Parameters
        ----------
        r_extract : float
            Physical extraction radius (must be inside the grid).
        """
        g = self.grid
        r_int = g.r[g.ghost:g.ghost + g.Nr]  # interior r, shape (Nr,)
        idx = int(np.searchsorted(r_int, r_extract))
        idx = int(np.clip(idx, 1, g.Nr - 1))
        i0, i1 = idx - 1, idx
        r0, r1 = r_int[i0], r_int[i1]
        w1 = float((r_extract - r0) / (r1 - r0))
        w0 = 1.0 - w1
        self._detectors.append((r_extract, g.ghost + i0, g.ghost + i1, w0, w1))
        self.waveforms[r_extract] = np.empty((0, g.Nmu), dtype=complex)

    # ------------------------------------------------------------------
    # CFL time step
    # ------------------------------------------------------------------

    def cfl_dt(self, cfl=0.5):
        """CFL-limited time step (README §1.5).

        dt = cfl * min over interior grid of min(dr_local/c_r, dmu/c_mu)

        where c_r = sqrt(Delta/A), c_mu = sqrt((1-mu^2)/A),
        dr_local = r * dx  (physical radial cell size on the log grid).
        Inside the horizon Delta < 0; those cells contribute no radial
        CFL constraint (excision).
        """
        g  = self.grid
        eq = self.rhs_obj

        A_int     = np.real(eq.A[g.interior])           # (Nmu, Nr)
        Delta_int = np.real(eq.Delta[g.interior])
        MU_int    = g.MU[g.interior]                     # real

        # Inside the horizon Delta < 0 → no outgoing radial characteristics
        Delta_eff      = np.maximum(Delta_int, 0.0)
        one_minus_mu2  = 1.0 - MU_int**2                # positive on interior

        dr_local = g.dr_cell   # (Nr,) — cell widths at interior points

        with np.errstate(divide='ignore', invalid='ignore'):
            c_r2  = Delta_eff / A_int
            c_mu2 = one_minus_mu2 / A_int
            dt_r  = np.where(c_r2  > 0,
                             dr_local[np.newaxis, :] / np.sqrt(c_r2),
                             np.inf)
            dt_mu = np.where(c_mu2 > 0,
                             g.dmu / np.sqrt(c_mu2),
                             np.inf)

        return cfl * float(np.minimum(dt_r, dt_mu).min())

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, dt):
        """Advance state (psi, v) by one RK4 step of size dt."""
        rhs = self.rhs_obj.rhs

        k1p, k1v = rhs(self.psi, self.v)
        k2p, k2v = rhs(self.psi + 0.5*dt*k1p, self.v + 0.5*dt*k1v)
        k3p, k3v = rhs(self.psi + 0.5*dt*k2p, self.v + 0.5*dt*k2v)
        k4p, k4v = rhs(self.psi +     dt*k3p, self.v +     dt*k3v)

        self.psi += (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        self.v   += (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        self.t   += dt

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self, t_final, cfl=0.5, dt=None, record_every=1,
               snapshot_every=None, on_step=None, snapshot_writer=None):
        """March the state to t_final, recording detector waveforms.

        Parameters
        ----------
        t_final      : float
            Target final time.
        cfl          : float
            CFL factor (used when dt is None).
        dt           : float or None
            Fixed time step.  If None, computed from CFL.
        record_every : int
            Record detector data every this many steps.
        snapshot_every : int or None
            Store full-grid (psi) snapshots every this many steps.
        on_step : callable or None
            If given, called with no arguments after every completed RK4
            step.  Intended for progress reporting (e.g. a tqdm update).
        snapshot_writer : SnapshotWriter or None
            If given (and snapshot_every is not None), each snapshot is
            streamed straight to disk via writer.write(t, psi) instead of
            being appended to self.snapshots -- avoids ever holding more
            than one full-grid snapshot in RAM at a time (see SnapshotWriter
            docstring). self.snapshots stays empty ([]) in this mode. If
            None (default), behaviour is unchanged: snapshots accumulate
            in self.snapshots as before.

        After returning, results are in self.times (shape (Nt,)) and
        self.waveforms ({r_ext: array shape (Nt, Nmu)}).
        """
        if dt is None:
            dt = self.cfl_dt(cfl)

        # Remembered for provenance() -- see save_waveforms/save_snapshots.
        self._last_cfl = cfl
        self._last_dt = dt
        self._last_t_final = t_final
        self._last_record_every = record_every

        g = self.grid
        mu_sl = slice(g.ghost, g.ghost + g.Nmu)

        t_list = []
        w_lists = {r: [] for r, *_ in self._detectors}

        step_count = 0
        while self.t < t_final - 1e-12 * dt:
            dt_this = min(dt, t_final - self.t)
            self.step(dt_this)
            step_count += 1

            if step_count % record_every == 0:
                t_list.append(self.t)
                for r_ext, ai0, ai1, w0, w1 in self._detectors:
                    psi_mu = (w0 * self.psi[mu_sl, ai0]
                            + w1 * self.psi[mu_sl, ai1])
                    w_lists[r_ext].append(psi_mu.copy())

            if snapshot_every is not None and step_count % snapshot_every == 0:
                if snapshot_writer is not None:
                    snapshot_writer.write(self.t, self.psi)
                else:
                    self.snapshots.append((self.t, self.psi.copy()))

            if on_step is not None:
                on_step()

        self.times = np.array(t_list, dtype=float)
        for r_ext, *_ in self._detectors:
            lst = w_lists[r_ext]
            if lst:
                self.waveforms[r_ext] = np.array(lst)      # (Nt, Nmu)
            else:
                self.waveforms[r_ext] = np.empty((0, g.Nmu), dtype=complex)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def provenance(self, **extra):
        """Assemble the full provenance dict for this run's outputs: M, a,
        m, Nr, Nmu, rmin, rmax, order, ghost, dissipation,
        one_sided_horizon, cfl, dt, t_final, record_every, and the git
        commit hash. cfl/dt/t_final/record_every are picked up
        automatically from the most recent evolve() call (and omitted if
        evolve() hasn't run yet). Any keyword arguments passed here (e.g.
        basis descriptors) are merged in and take precedence over the
        auto-populated fields.
        """
        g, eq = self.grid, self.rhs_obj
        prov = dict(
            M=eq.M, a=eq.a, m=eq.m,
            Nr=g.Nr, Nmu=g.Nmu, rmin=g.rmin, rmax=g.rmax,
            order=g.order, ghost=g.ghost,
            dissipation=eq.dissipation,
            one_sided_horizon=eq.one_sided_horizon,
            cfl=getattr(self, '_last_cfl', None),
            dt=getattr(self, '_last_dt', None),
            t_final=getattr(self, '_last_t_final', None),
            record_every=getattr(self, '_last_record_every', None),
            git_commit=_git_commit_hash(),
        )
        prov = {k: v for k, v in prov.items() if v is not None}
        prov.update(extra)
        return prov

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_waveforms(self, path, provenance=None):
        """Save detector time series and grid metadata to a .npz file.

        Waveforms are small and long-lived; this file is intended to be
        kept for post-processing and ringdown analysis.

        Parameters
        ----------
        path : str
            Output path (.npz extension added if absent).
        provenance : dict or None
            Extra key/value pairs merged into the saved file (see
            Evolution.provenance()) -- e.g. run parameters, basis
            descriptors, git commit hash. Backward compatible: omitting
            this (the default) saves exactly the same keys as before.
        """
        g  = self.grid
        eq = self.rhs_obj
        data = {
            'times':   np.asarray(self.times),
            'mu_grid': g._mu[g.ghost:g.ghost + g.Nmu],  # interior mu only
            'Nmu':     np.array(g.Nmu),
            'M':       np.array(eq.M),
            'a':       np.array(eq.a),
            'm':       np.array(eq.m),
            't_current': np.array(self.t),
        }
        for r_ext, *_ in self._detectors:
            data[f'waveform_{r_ext:.6f}'] = np.asarray(self.waveforms[r_ext])
        if provenance:
            data.update(provenance)
        np.savez(path, **data)

    def save_snapshots(self, path, provenance=None):
        """Save full-grid (psi) snapshots accumulated during evolve() to a .npz.

        Snapshots are large; this file is intended for checkpointing or
        restart and may be overwritten between runs. For large batches of
        runs, prefer streaming snapshots directly to disk via
        SnapshotWriter (passed to evolve() as snapshot_writer) instead of
        accumulating them in self.snapshots and saving them here.

        Parameters
        ----------
        path : str
            Output path (.npz extension added if absent).
        provenance : dict or None
            Extra key/value pairs merged into the saved file (see
            Evolution.provenance()). Backward compatible: omitting this
            (the default) saves exactly the same keys as before.
        """
        g  = self.grid
        eq = self.rhs_obj
        times_snap = np.array([t for t, _ in self.snapshots])
        # Stack into shape (Nsnap, Nmu_full, Nr_full)
        psi_stack  = np.array([p for _, p in self.snapshots]) if self.snapshots else \
                     np.empty((0,) + g.shape, dtype=complex)
        data = {
            'times_snap': times_snap,
            'psi':        psi_stack,
            'r_grid':     g.r,
            'mu_grid':    g._mu,
            'Nr':         np.array(g.Nr),
            'Nmu':        np.array(g.Nmu),
            'ghost':      np.array(g.ghost),
            'M':          np.array(eq.M),
            'a':          np.array(eq.a),
            'm':          np.array(eq.m),
        }
        if provenance:
            data.update(provenance)
        np.savez(path, **data)
