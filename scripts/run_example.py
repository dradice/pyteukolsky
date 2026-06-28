"""
run_example.py — Gaussian pulse evolution with the Teukolsky solver.

Evolves an m=2 (default) time-symmetric Gaussian pulse in Schwarzschild
(a=0, default) on a uniform radial grid and produces three output files:

  <stem>_static.png — multi-time equatorial slice comparison
  <stem>_1d.gif     — animation of r Re[ψ_m] at θ = π/2 vs r
  <stem>_2d.gif     — animation of r Re[ψ_m] on the (r, θ) meridional plane
  <stem>_waveforms.npz — detector time series (see Evolution.save_waveforms)

Usage:
    ~/local/miniforge/bin/python scripts/run_example.py
    ~/local/miniforge/bin/python scripts/run_example.py --Nr 300 --t_final 100
    ~/local/miniforge/bin/python scripts/run_example.py --a 0.9 --stem kerr
"""

import argparse
import os
import sys
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as mplanim
import numpy as np

from pyteukolsky import Evolution, Grid, TeukolskyRHS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Teukolsky Gaussian pulse example")
    p.add_argument("--Nr",       type=int,   default=200,
                   help="radial interior cells  (default 200, uniform grid)")
    p.add_argument("--Nmu",      type=int,   default=32,
                   help="angular interior cells (default 32)")
    p.add_argument("--rmin",     type=float, default=1.5,
                   help="inner radial boundary / M (default 1.5, inside horizon)")
    p.add_argument("--rmax",     type=float, default=100.0,
                   help="outer radial boundary / M (default 100)")
    p.add_argument("--r0",       type=float, default=15.0,
                   help="Gaussian pulse centre / M (default 15)")
    p.add_argument("--sigma",    type=float, default=2.0,
                   help="Gaussian pulse width in r / M (default 2)")
    p.add_argument("--sigma_mu", type=float, default=0.3,
                   help="Gaussian pulse half-width in μ = cos θ (default 0.3)")
    p.add_argument("--a",        type=float, default=0.0,
                   help="Kerr spin parameter |a| < M (default 0)")
    p.add_argument("--m",        type=int,   default=2,
                   help="azimuthal mode number (default 2)")
    p.add_argument("--t_final",  type=float, default=80.0,
                   help="final time / M (default 80)")
    p.add_argument("--cfl",      type=float, default=0.45,
                   help="CFL factor (default 0.45)")
    p.add_argument("--diss",     type=float, default=0.3,
                   help="Kreiss-Oliger dissipation ε (default 0.3)")
    p.add_argument("--n_frames", type=int,   default=120,
                   help="approximate number of animation frames (default 120)")
    p.add_argument("--fps",      type=int,   default=20,
                   help="animation frames per second (default 20)")
    p.add_argument("--stem",     default="pulse_animation",
                   help="output filename stem (default 'pulse_animation')")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(args):
    """Set up grid, initial data, and run the evolution.

    Returns a dict with pre-computed arrays ready for plotting:
      r_int    : interior r values, shape (Nr,)
      mu_int   : interior μ values, shape (Nmu,)
      theta_int: interior θ = arccos(μ) in degrees, shape (Nmu,)
      t_arr    : snapshot times, shape (n_snap,)
      psi_2d   : r·Re[ψ] on interior grid, shape (n_snap, Nmu, Nr)
      r_H      : horizon radius / M
      evo      : Evolution object (for save_waveforms)
    """
    M   = 1.0
    r_H = M + np.sqrt(max(M**2 - args.a**2, 0.0))

    g   = Grid(rmin=args.rmin, rmax=args.rmax, Nr=args.Nr, Nmu=args.Nmu,
               ghost=2, M=M)
    rhs = TeukolskyRHS(g, M=M, a=args.a, m=args.m, dissipation=args.diss)
    evo = Evolution(rhs)

    # Time-symmetric 2D Gaussian: v = 0.
    # Angular envelope centred at μ=0 keeps the field near zero at the poles,
    # where V = (2μ−m)²/(1−μ²)−2 diverges for m ≠ 0.
    psi0 = (np.exp(-((g.R - args.r0)  / args.sigma)    ** 2)
          * np.exp(-(  g.MU           / args.sigma_mu) ** 2)).astype(complex)
    evo.set_initial_data(psi0, psi0, dt_init=1e-3)

    r_out = g.r[g.ghost + g.Nr - 1]
    for r_ext in [20.0, 40.0, 70.0]:
        if r_ext < 0.95 * r_out:
            evo.add_detector(r_ext)

    r_int     = g.r[g.ghost : g.ghost + g.Nr]
    mu_int    = g._mu[g.ghost : g.ghost + g.Nmu]
    theta_int = np.degrees(np.arccos(mu_int))

    dt         = evo.cfl_dt(cfl=args.cfl)
    n_steps    = int(np.ceil(args.t_final / dt))
    snap_every = max(1, n_steps // args.n_frames)

    dr_mean = (g.r[g.ghost + g.Nr - 1] - g.r[g.ghost]) / max(g.Nr - 1, 1)
    print(f"Grid   : Nr={args.Nr} (mean dr≈{dr_mean:.3f} M), Nmu={args.Nmu}")
    print(f"         r ∈ [{r_int[0]:.2f}, {r_int[-1]:.2f}] M")
    print(f"Physics: M={M}, a={args.a}, m={args.m}, r_H={r_H:.3f} M")
    print(f"Pulse  : r0={args.r0} M, σ_r={args.sigma} M, σ_μ={args.sigma_mu}")
    print(f"Time   : t_final={args.t_final} M, dt≈{dt:.4f} M, "
          f"~{n_steps} steps, ~{n_steps // snap_every} frames")

    evo.evolve(args.t_final, dt=dt, snapshot_every=snap_every)
    print(f"Collected {len(evo.snapshots)} snapshots.")

    gs     = g.ghost
    t_arr  = np.array([t for t, _ in evo.snapshots])
    psi_2d = np.array([
        r_int[np.newaxis, :] * psi[gs:gs + g.Nmu, gs:gs + g.Nr].real
        for _, psi in evo.snapshots
    ])   # shape (n_snap, Nmu, Nr)

    return dict(r_int=r_int, mu_int=mu_int, theta_int=theta_int,
                t_arr=t_arr, psi_2d=psi_2d, r_H=r_H, M=M, evo=evo)


# ---------------------------------------------------------------------------
# Static figure: equatorial slices at several times
# ---------------------------------------------------------------------------

def make_static_figure(data, path, n_show=8):
    """Save a PNG comparing r·Re[ψ] at θ=π/2 across multiple times."""
    r_int     = data['r_int']
    mu_int    = data['mu_int']
    t_arr     = data['t_arr']
    psi_2d    = data['psi_2d']
    r_H       = data['r_H']

    i_eq   = int(np.argmin(np.abs(mu_int)))
    psi_eq = psi_2d[:, i_eq, :]            # (n_snap, Nr)

    n_show = min(n_show, len(t_arr))
    idxs   = np.linspace(0, len(t_arr) - 1, n_show, dtype=int)
    cmap   = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel(r"$r / M$", fontsize=12)
    ax.set_ylabel(r"$r\,\mathrm{Re}[\psi_m]$ at $\theta = \pi/2$", fontsize=12)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(r_H, color="r", lw=1, ls="--", label="Horizon")
    for k, idx in enumerate(idxs):
        alpha = 0.45 + 0.55 * k / max(n_show - 1, 1)
        ax.plot(r_int, psi_eq[idx],
                color=cmap(k / max(n_show - 1, 1)), alpha=alpha, lw=1.2,
                label=rf"$t = {t_arr[idx]:.0f}\,M$")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Static figure    → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1D animation: equatorial slice
# ---------------------------------------------------------------------------

def make_animation_1d(data, path, fps=20):
    """Animate r·Re[ψ_m] at θ=π/2 as a function of r."""
    r_int  = data['r_int']
    mu_int = data['mu_int']
    t_arr  = data['t_arr']
    psi_2d = data['psi_2d']
    r_H    = data['r_H']

    i_eq   = int(np.argmin(np.abs(mu_int)))
    psi_eq = psi_2d[:, i_eq, :]            # (n_snap, Nr)
    ymax   = max(np.abs(psi_eq).max() * 1.15, 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel(r"$r / M$", fontsize=12)
    ax.set_ylabel(r"$r\,\mathrm{Re}[\psi_m]$ at $\theta = \pi/2$", fontsize=12)
    ax.set_xlim(r_int[0], r_int[-1])
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(r_H, color="r", lw=1, ls="--",
               label=rf"Horizon $r_H = {r_H:.2f}\,M$")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    (line,) = ax.plot([], [], "steelblue", lw=1.5)
    tlabel  = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10)

    def _init():
        line.set_data([], [])
        tlabel.set_text("")
        return line, tlabel

    def _frame(i):
        line.set_data(r_int, psi_eq[i])
        tlabel.set_text(rf"$t = {t_arr[i]:.1f}\,M$")
        return line, tlabel

    ani = mplanim.FuncAnimation(fig, _frame, frames=len(t_arr),
                                 init_func=_init, blit=True, interval=1000 // fps)
    ani.save(path, writer="pillow", fps=fps, dpi=120)
    print(f"1D animation     → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D animation: meridional (r, θ) plane
# ---------------------------------------------------------------------------

def make_animation_2d(data, path, fps=20):
    """Animate r·Re[ψ_m] on the (r, θ) meridional plane."""
    r_int     = data['r_int']
    theta_int = data['theta_int']   # degrees, shape (Nmu,)
    t_arr     = data['t_arr']
    psi_2d    = data['psi_2d']      # (n_snap, Nmu, Nr)
    r_H       = data['r_H']

    R_2d, TH_2d = np.meshgrid(r_int, theta_int)   # both (Nmu, Nr)

    # Color limits: 99th percentile outside the near-horizon region
    r_mask = r_int > 5.0
    vmax   = max(np.percentile(np.abs(psi_2d[:, :, r_mask]), 99), 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlabel(r"$r / M$", fontsize=12)
    ax.set_ylabel(r"$\theta$ (deg)", fontsize=12)
    ax.set_xlim(r_int[0], r_int[-1])
    ax.set_ylim(180, 0)          # θ = 0° (north pole) at top
    ax.axvline(r_H, color="k", lw=1, ls="--", label="Horizon")
    ax.axhline(90,  color="k", lw=0.5, ls=":")
    ax.legend(loc="upper right", fontsize=9)

    mesh = ax.pcolormesh(R_2d, TH_2d, psi_2d[0],
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="nearest")
    fig.colorbar(mesh, ax=ax, label=r"$r\,\mathrm{Re}[\psi_m]$")
    tlabel = ax.text(0.5, 1.02, rf"$t = {t_arr[0]:.1f}\,M$",
                     transform=ax.transAxes, ha="center", fontsize=11)
    fig.tight_layout()

    def _frame(i):
        mesh.set_array(psi_2d[i].ravel())
        tlabel.set_text(rf"$t = {t_arr[i]:.1f}\,M$")
        return mesh, tlabel

    ani = mplanim.FuncAnimation(fig, _frame, frames=len(t_arr),
                                 blit=True, interval=1000 // fps)
    ani.save(path, writer="pillow", fps=fps, dpi=100)
    print(f"2D animation     → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data = run_simulation(args)
    stem = args.stem

    make_static_figure(data, stem + "_static.png")
    make_animation_1d( data, stem + "_1d.gif",   fps=args.fps)
    make_animation_2d( data, stem + "_2d.gif",   fps=args.fps)

    evo = cast(Evolution, data['evo'])
    evo.save_waveforms(stem + "_waveforms.npz")
    print(f"Waveforms        → {stem}_waveforms.npz")


if __name__ == "__main__":
    main()
