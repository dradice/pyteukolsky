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

import numpy as np


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
               snapshot_every=None, on_step=None):
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

        After returning, results are in self.times (shape (Nt,)) and
        self.waveforms ({r_ext: array shape (Nt, Nmu)}).
        """
        if dt is None:
            dt = self.cfl_dt(cfl)

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
    # I/O
    # ------------------------------------------------------------------

    def save_waveforms(self, path):
        """Save detector time series and grid metadata to a .npz file.

        Waveforms are small and long-lived; this file is intended to be
        kept for post-processing and ringdown analysis.

        Parameters
        ----------
        path : str
            Output path (.npz extension added if absent).
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
        np.savez(path, **data)

    def save_snapshots(self, path):
        """Save full-grid (psi) snapshots accumulated during evolve() to a .npz.

        Snapshots are large; this file is intended for checkpointing or
        restart and may be overwritten between runs.

        Parameters
        ----------
        path : str
            Output path (.npz extension added if absent).
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
        np.savez(path, **data)
