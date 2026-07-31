"""
stability_excision_test.py -- long-duration stability comparison of the
centered vs. one-sided ("excision") derivative stencils at the horizon.

Background
----------
The solver's inner radial boundary sits *inside* the horizon (`rmin < r_+`),
relying on the coordinates being horizon-penetrating so that, in principle,
no physical boundary condition is needed there (README, "Inner (excision)").
In practice this only holds if every *interior* grid cell also lands at
`r > r_+`; if `rmin` is set more aggressively, interior cells with `r < r_+`
have a positive-real `Cv/A` coefficient -- an intrinsically unstable local
ODE term -- and that growth leaks into the physically valid exterior through
the ordinary centered-difference coupling between neighboring cells
(CLAUDE.md, "Working with QNM runs (gotcha)").

`TeukolskyRHS(..., one_sided_horizon=True)` fixes this properly: it finds
`i_exc`, the first interior column with `r > r_+`, and overrides the radial
derivative stencils there with a one-sided (outward-only) form, so that
column -- and hence the whole exterior domain -- never reads a value from
inside the horizon, no matter how badly that excised region behaves.

This script runs a 2x2 comparison (safe/aggressive `rmin` x
centered/one-sided) for Schwarzschild ell=m=2, evolved far beyond the
~130-140M window used by the standard demo/tests, and tracks:

  good_max(t) = max|psi_m| over the "good" columns [i_exc:]      -- this is
      the number that should differ sharply between the two stencils on the
      aggressive grid.
  bad_max(t)  = max|psi_m| over the excised columns [ghost:i_exc) -- expected
      to blow up in *both* cases on the aggressive grid: excision correctly
      *contains* the instability rather than trying to fix it.

The aggressive `rmin=1.5` default below matches scripts/run_example.py's own
shipped default, which is inside the Schwarzschild horizon r_+=2M.

Usage:
    ~/local/miniforge/bin/python scripts/stability_excision_test.py
    ~/local/miniforge/bin/python scripts/stability_excision_test.py --t_final 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        description="Compare centered vs. one-sided horizon derivative "
                     "stencils for long-duration stability (Schwarzschild, "
                     "ell=m=2 Gaussian pulse).")
    p.add_argument("--Nr",             type=int,   default=200)
    p.add_argument("--Nmu",            type=int,   default=32)
    p.add_argument("--rmax",           type=float, default=100.0)
    p.add_argument("--rmin_safe",       type=float, default=1.99,
                   help="rmin for the 'safe' grid (default 1.99, matches "
                        "test_validation.py / single_mode.py)")
    p.add_argument("--rmin_aggressive", type=float, default=1.5,
                   help="rmin for the 'aggressive' grid (default 1.5, "
                        "matches run_example.py's own shipped default)")
    p.add_argument("--r0",             type=float, default=10.0)
    p.add_argument("--sigma",          type=float, default=2.0)
    p.add_argument("--r_extract",      type=float, default=30.0)
    p.add_argument("--t_final",        type=float, default=600.0,
                   help="final time / M (default 600, ~4.5x the standard "
                        "~130-140M demo/test window)")
    p.add_argument("--cfl",            type=float, default=0.45)
    p.add_argument("--diss",           type=float, default=0.1)
    p.add_argument("--sample_every",   type=int,   default=20,
                   help="record good_max/bad_max diagnostics every this "
                        "many RK4 steps (default 20)")
    p.add_argument("--fit_start",      type=float, default=90.0)
    p.add_argument("--fit_end",        type=float, default=130.0)
    p.add_argument("--stem",           default="stability_excision")
    return p.parse_args()


def run_case(M, a, m, rmin, rmax, Nr, Nmu, r0, sigma, r_extract, t_final,
             cfl, diss, one_sided_horizon, sample_every):
    """Evolve one (rmin, one_sided_horizon) configuration and return
    diagnostics + the extracted waveform."""
    g   = Grid(rmin=rmin, rmax=rmax, Nr=Nr, Nmu=Nmu, ghost=2, M=M)
    rhs = TeukolskyRHS(g, M=M, a=a, m=m, dissipation=diss,
                        one_sided_horizon=one_sided_horizon)
    evo = Evolution(rhs)

    # Time-symmetric ell=2 Gaussian pulse (v=0 initially), same convention as
    # tests/test_validation.py::run_schwarz and scripts/single_mode.py.
    psi0 = gaussian_pulse(g, r0=r0, sigma_r=sigma, ell=2, m=m, spin=-2)
    evo.set_initial_data(psi0, psi0, dt_init=1e-3)
    evo.add_detector(r_extract)

    i_exc = rhs.i_exc
    gh    = g.ghost
    mu_sl = slice(gh, gh + g.Nmu)

    diag_t, diag_good, diag_bad = [], [], []
    step_count = [0]

    def on_step():
        step_count[0] += 1
        if step_count[0] % sample_every != 0:
            return
        diag_t.append(evo.t)
        good_block = evo.psi[mu_sl, i_exc:gh + g.Nr]
        diag_good.append(float(np.max(np.abs(good_block))))
        if i_exc > gh:
            bad_block = evo.psi[mu_sl, gh:i_exc]
            diag_bad.append(float(np.max(np.abs(bad_block))))
        else:
            diag_bad.append(0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        evo.evolve(t_final, cfl=cfl, on_step=on_step)

        mu     = g._mu[gh:gh + g.Nmu]
        sw     = swsh(-2, 2, m, mu)
        psi_22 = project_swsh(evo.waveforms[r_extract], mu, sw)

    return dict(
        times=evo.times, psi_22=psi_22,
        diag_t=np.array(diag_t), diag_good=np.array(diag_good),
        diag_bad=np.array(diag_bad),
        i_exc=i_exc, r_plus=rhs.r_plus, r_first=float(g.r[gh].real),
    )


def summarize_and_fit(label, result, fit_start, fit_end):
    print(f"\n--- {label} ---")
    print(f"  i_exc={result['i_exc']}, r_plus={result['r_plus']:.4f} M, "
          f"first interior r={result['r_first']:.4f} M")

    good = result["diag_good"]
    finite = np.isfinite(good)
    if finite.all() and good.max() < 1e6:
        print(f"  good_max(t): bounded, max={good.max():.3e}, "
              f"final={good[-1]:.3e}")
    else:
        last_finite = good[finite][-1] if finite.any() else float("nan")
        print(f"  good_max(t): DIVERGED (last finite value {last_finite:.3e}, "
              f"{np.count_nonzero(~finite)} non-finite samples)")

    with np.errstate(over="ignore", invalid="ignore"):
        try:
            omega_R, omega_I = fit_qnm_frequency(
                result["times"], result["psi_22"].real, fit_start, fit_end)
            pub_R, pub_I = PUBLISHED_MOMEGA
            print(f"  QNM fit: M*omega = {omega_R:+.5f} {omega_I:+.5f}i "
                  f"(published {pub_R:+.5f} {pub_I:+.5f}i, "
                  f"|dR|={abs(omega_R - pub_R):.4f}, "
                  f"|dI|={abs(omega_I - pub_I):.4f})")
        except Exception as exc:
            print(f"  QNM fit failed: {exc}")


def main():
    args = parse_args()
    M, a, m = 1.0, 0.0, 2

    configs = [
        ("safe_centered",        args.rmin_safe,       False),
        ("safe_one_sided",       args.rmin_safe,       True),
        ("aggressive_centered",  args.rmin_aggressive, False),
        ("aggressive_one_sided", args.rmin_aggressive, True),
    ]

    results = {}
    for label, rmin, osh in configs:
        print(f"Running {label}: rmin={rmin}, one_sided_horizon={osh}, "
              f"t_final={args.t_final} M ...")
        results[label] = run_case(
            M=M, a=a, m=m, rmin=rmin, rmax=args.rmax, Nr=args.Nr, Nmu=args.Nmu,
            r0=args.r0, sigma=args.sigma, r_extract=args.r_extract,
            t_final=args.t_final, cfl=args.cfl, diss=args.diss,
            one_sided_horizon=osh, sample_every=args.sample_every,
        )
        summarize_and_fit(label, results[label], args.fit_start, args.fit_end)

    # ------------------------------------------------------------------
    # Plot log10(good_max) vs t for all four runs
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, _, _ in configs:
        r = results[label]
        good = np.clip(r["diag_good"], 1e-300, None)
        with np.errstate(over="ignore", invalid="ignore"):
            ax.plot(r["diag_t"], np.log10(good), label=label, lw=1.5)
    ax.set_xlabel(r"$t / M$")
    ax.set_ylabel(r"$\log_{10}\,\max|\psi_m|$  (good region, $r>r_+$)")
    ax.set_title("Long-duration stability: centered vs. one-sided horizon stencil")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = args.stem + "_good_max.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot         -> {plot_path}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save raw diagnostics
    # ------------------------------------------------------------------
    npz_data = dict(t_final=args.t_final, Nr=args.Nr, Nmu=args.Nmu)
    for label, _, _ in configs:
        r = results[label]
        npz_data[f"{label}_t"]    = r["diag_t"]
        npz_data[f"{label}_good"] = r["diag_good"]
        npz_data[f"{label}_bad"]  = r["diag_bad"]
    npz_path = args.stem + "_diagnostics.npz"
    np.savez(npz_path, **npz_data)
    print(f"Diagnostics  -> {npz_path}")


if __name__ == "__main__":
    main()
