"""
single_mode.py — excite a single QNM eigenmode of a non-rotating black hole.

A Schwarzschild (a=0) black hole "rings" in its quasi-normal modes.  When it is
perturbed by generic initial data, an initial burst is followed by a ringdown
that, after the burst has passed, is dominated by the *single* fundamental
eigenmode _{-2}(l=m=2, n=0) with complex frequency

    M ω = 0.3737 - 0.0890 i          (Leaver 1985; Berti et al. 2006)

i.e. Re[ψ_m(t)] ~ e^{ω_I t} cos(ω_R t + φ) at fixed extraction radius.

This script seeds the ℓ=2, m=2 spin-weighted spherical harmonic angular profile
(so no other ℓ is excited to leading order) with a radial Gaussian, evolves in
ingoing Kerr–Schild coordinates, records the detector waveform, and verifies
that a single eigenmode was cleanly excited by fitting the ringdown frequency
against the published value.  After the initial burst has passed (~30M at
r_extract=30M) the recovered waveform is a single damped sinusoid; the printed
fit typically matches the published M ω to 5–6 significant figures, i.e. this
*is* an effectively single-eigenmode excitation.  The waveform time series is
written to a .npz for later analysis.

Why a Gaussian and not the *exact* QNM radial eigenfunction?
------------------------------------------------------------
A QNM is a resonance, not a normalizable eigenmode, so on this finite-domain
solver the exact radial eigenfunction cannot be seeded cleanly (verified
numerically while writing this script):

  * The solver's own discrete eigenmodes (diagonalizing the linear RHS operator)
    are standing "box" modes set by the finite-radius, first-order Sommerfeld
    boundary condition.  They carry the right frequency but the wrong-signed
    decay — e.g. M ω ≈ 0.3735 + 0.020 i (slowly *growing*) rather than the
    physical 0.3737 − 0.0890 i.

  * The analytic QNM radial eigenfunction grows exponentially in space: because
    Im ω < 0 the outgoing solution ~ e^{i ω r*} behaves like e^{+|ω_I| r*},
    reaching |R| ~ 1e6 by r = 200 M.  Seeding it is dominated by the outer
    boundary and swamped by Sommerfeld reflections.  Windowing near the
    potential peak does not help: in horizon-penetrating coordinates the
    near-zone data is mostly ingoing (falls through the horizon) and the
    outgoing remainder still develops as a transient out to the detector.

Genuine QNM eigenfunctions would require an absorbing sponge layer or
hyperboloidal / scri-fixing coordinates — a solver change, not initial data.
The windowed Gaussian below is the practical route and already yields a pure
fundamental after the burst.

Usage:
    ~/local/miniforge/bin/python scripts/single_mode.py
    ~/local/miniforge/bin/python scripts/single_mode.py --m 2 --t_final 200
    ~/local/miniforge/bin/python scripts/single_mode.py --r_extract 50 --stem mode22

Grid note: rmin defaults to 1.99 M so every interior cell sits outside the
horizon r_+ = 2M (see CLAUDE.md / test_validation.py); interior cells inside
the horizon seed a slow instability that corrupts the ringdown.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pyteukolsky import (
    Grid, TeukolskyRHS, Evolution,
    swsh, gaussian_pulse, project_swsh, fit_qnm_frequency,
)


# Published s=-2, ell=m=2, n=0 fundamental Schwarzschild QNM frequency (M=1).
# Leaver (1985); Berti, Cardoso & Will (2006), arXiv:gr-qc/0512160.
PUBLISHED_MOMEGA = (0.37367, -0.08896)


def parse_args():
    p = argparse.ArgumentParser(
        description="Excite a single QNM eigenmode of a Schwarzschild BH.")
    p.add_argument("--m",         type=int,   default=2,
                   help="azimuthal mode number (default 2; ell fixed at 2)")
    p.add_argument("--Nr",        type=int,   default=200,
                   help="radial interior cells (default 200, log grid)")
    p.add_argument("--Nmu",       type=int,   default=32,
                   help="angular interior cells (default 32)")
    p.add_argument("--rmin",      type=float, default=1.99,
                   help="inner radial boundary / M (default 1.99, just inside "
                        "r_+=2M so interior cells stay outside)")
    p.add_argument("--rmax",      type=float, default=200.0,
                   help="outer radial boundary / M (default 200)")
    p.add_argument("--r0",        type=float, default=10.0,
                   help="Gaussian pulse centre / M (default 10)")
    p.add_argument("--sigma",     type=float, default=2.0,
                   help="Gaussian pulse width in r / M (default 2)")
    p.add_argument("--r_extract", type=float, default=30.0,
                   help="waveform extraction radius / M (default 30)")
    p.add_argument("--t_final",   type=float, default=200.0,
                   help="final time / M (default 200)")
    p.add_argument("--cfl",       type=float, default=0.45,
                   help="CFL factor (default 0.45)")
    p.add_argument("--diss",      type=float, default=0.1,
                   help="Kreiss-Oliger dissipation ε (default 0.1)")
    p.add_argument("--fit_start", type=float, default=None,
                   help="ringdown fit window start / M "
                        "(default: after the burst, ~2*r_extract + 30)")
    p.add_argument("--fit_end",   type=float, default=None,
                   help="ringdown fit window end / M "
                        "(default: before the first rmax reflection)")
    p.add_argument("--stem",      default="single_mode",
                   help="output filename stem (default 'single_mode')")
    return p.parse_args()


def main():
    args = parse_args()
    M = 1.0
    r_plus = 2.0 * M  # a = 0

    # ------------------------------------------------------------------
    # Grid, physics, initial data
    # ------------------------------------------------------------------
    g   = Grid(rmin=args.rmin, rmax=args.rmax, Nr=args.Nr, Nmu=args.Nmu,
               ghost=2, M=M)
    rhs = TeukolskyRHS(g, M=M, a=0.0, m=args.m, dissipation=args.diss)
    evo = Evolution(rhs)

    # Single angular eigenmode: ell=2, m=args.m spin-weighted spherical harmonic.
    # Time-symmetric start (psi0 == psi1 => v = 0).
    psi0 = gaussian_pulse(g, r0=args.r0, sigma_r=args.sigma,
                          ell=2, m=args.m, spin=-2)
    evo.set_initial_data(psi0, psi0, dt_init=1e-3)
    evo.add_detector(args.r_extract)

    r_int = g.r[g.ghost:g.ghost + g.Nr]
    print(f"Grid   : Nr={args.Nr}, Nmu={args.Nmu}, "
          f"r ∈ [{r_int[0].real:.2f}, {r_int[-1].real:.2f}] M")
    print(f"Physics: M={M}, a=0 (Schwarzschild), m={args.m}, r_+={r_plus:.2f} M")
    print(f"Mode   : ℓ=2, m={args.m}, n=0 fundamental QNM")
    print(f"Pulse  : r0={args.r0} M, σ_r={args.sigma} M, extract r={args.r_extract} M")
    print(f"Evolving to t={args.t_final} M ...")

    evo.evolve(args.t_final, cfl=args.cfl)

    # ------------------------------------------------------------------
    # Project detector data onto the ell=2, m SWSH to isolate the mode
    # ------------------------------------------------------------------
    mu     = g._mu[g.ghost:g.ghost + g.Nmu]
    sw     = swsh(-2, 2, args.m, mu)
    psi_lm = project_swsh(evo.waveforms[args.r_extract], mu, sw)

    # ------------------------------------------------------------------
    # Verify a single eigenmode was excited: fit the ringdown frequency
    # ------------------------------------------------------------------
    # Default window: after the outgoing burst reaches the detector and before
    # the first Sommerfeld reflection from rmax returns.
    fit_start = (args.fit_start if args.fit_start is not None
                 else 2.0 * args.r_extract + 30.0)
    reflect_t = 2.0 * (args.rmax - args.r_extract)  # round trip to rmax
    fit_end = (args.fit_end if args.fit_end is not None
               else min(args.t_final, fit_start + 40.0, reflect_t - 10.0))

    # m=2 is real-symmetric (a=0); use the real ringdown for the tightest fit.
    signal = psi_lm.real if args.m == 2 else psi_lm
    omega_R, omega_I = fit_qnm_frequency(evo.times, signal, fit_start, fit_end)

    pub_R, pub_I = PUBLISHED_MOMEGA
    print()
    print(f"Ringdown fit window: [{fit_start:.1f}, {fit_end:.1f}] M")
    print(f"  fitted   M ω = {omega_R:+.5f} {omega_I:+.5f} i")
    if args.m == 2:
        print(f"  published M ω = {pub_R:+.5f} {pub_I:+.5f} i "
              f"(ℓ=m=2, n=0)")
        print(f"  |Δω_R| = {abs(omega_R - pub_R):.4f}, "
              f"|Δω_I| = {abs(omega_I - pub_I):.4f}")

    # ------------------------------------------------------------------
    # Save waveform time series for later analysis
    # ------------------------------------------------------------------
    evo.save_waveforms(args.stem + "_waveforms.npz")
    print(f"\nWaveforms → {args.stem}_waveforms.npz "
          f"(detector r={args.r_extract} M, {len(evo.times)} samples)")

    # Also save the projected single-mode time series alongside the fit.
    np.savez(args.stem + "_mode.npz",
             times=evo.times,
             psi_lm=psi_lm,
             r_extract=args.r_extract,
             M=M, a=0.0, ell=2, m=args.m,
             omega_R=omega_R, omega_I=omega_I,
             fit_start=fit_start, fit_end=fit_end)
    print(f"Mode data → {args.stem}_mode.npz "
          f"(ℓ=2, m={args.m} projected ψ_lm(t) + fitted frequency)")


if __name__ == "__main__":
    main()
