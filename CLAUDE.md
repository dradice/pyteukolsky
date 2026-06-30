# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyTeukolsky is a pure-`numpy` finite-difference, time-domain solver for a single
azimuthal mode `psi_m(t, r, theta)` of the Teukolsky equation in ingoing
Kerr–Schild (horizon-penetrating) coordinates, following Campanelli et al.
(2001), gr-qc/0010034v2. The user supplies `(M, a)`, the azimuthal mode number
`m`, and initial data; the solver marches forward in time and records `psi_m` at
extraction radii for waveform analysis. It reproduces the Schwarzschild and Kerr
`l=m=2` fundamental QNM frequencies to ~1%.

`README.md` is the long-form reference (full formalism, the mode equation,
numerical method, validation tables). Consult it for physics/math detail; this
file is the orientation map.

## Commands

Plain `python`/`python3` are **not** on PATH. Always use the user's interpreter,
which has numpy/scipy/sympy/matplotlib/pytest:

```bash
~/local/miniforge/bin/python scripts/check_equations.py   # symbolic verification — expect two "IDENTICAL" reports
~/local/miniforge/bin/python scripts/run_example.py       # end-to-end demo (writes gitignored *.gif/*.npz/*.png)
~/local/miniforge/bin/python -m pytest                    # full suite
~/local/miniforge/bin/python -m pytest tests/test_kerr.py::test_name   # one test
```

`scripts/check_equations.py` is the **source of truth for every equation**. When
you change any coefficient or operator in the solver, update the corresponding
`def` there and re-run it; both checks must report `IDENTICAL -> difference
simplifies to 0`.

## Architecture

The solver is a three-layer stack, each owning the one below; the public API is
re-exported from `pyteukolsky/__init__.py`.

1. **`Grid` (`grid.py`)** — coordinates, FD operators, ghost fills. No physics.
2. **`TeukolskyRHS` (`equation.py`)** — precomputes the PDE coefficient arrays
   from `(M, a, m)` and exposes `rhs(psi, v)`. Owns a `Grid`.
3. **`Evolution` (`evolve.py`)** — RK4 method-of-lines time driver, CFL timestep,
   detectors, snapshot/waveform I/O. Owns a `TeukolskyRHS`.

Plus two leaf modules: `initialdata.py` (`swsh`, `gaussian_pulse`) and
`diagnostics.py` (`project_swsh`, `fit_qnm_frequency`).

### Key design decisions (the non-obvious parts)

- **First-order-in-time reduction.** The state is the pair `(psi, v)` with
  `v = d_t psi`. This turns the mixed `psi_tr` term into an explicit spatial
  derivative `v_r`, so the whole system is method-of-lines + RK4. `d_t psi = v`,
  `d_t v = invA * (L[psi] + B*v_r + Cv*v)` where
  `L[psi] = Delta*psi_rr + Cr*psi_r + angular(psi) - V*psi`.

- **The field is complex everywhere.** `Cv = 4r + 4i*a*mu + 6M` and
  `Cr = 2i*a*m + 6r - 6M` have imaginary parts, so all field/coefficient arrays
  are `complex128` even for `a=0`. The full Kerr `a`-dependence already lives in
  the coefficients (`Sigma = r^2 + a^2*mu^2`, `Delta`, `A`, `Cv`, `Cr`); there is
  **no separate Kerr code path** — a 2D `(r, mu)` run captures the spheroidal
  angular structure automatically. For `a!=0` use the complex branch of
  `fit_qnm_frequency` (it auto-detects real vs. complex input).

- **Logarithmic radial grid.** Uniform in `x` with `r = M*exp(x)`, so
  `drdx = d2rdx2 = r` and the chain rule for `dr`/`drr` is trivial. Fine near the
  horizon, coarse far away. `dr_cell = r*dx` is the physical cell width used by
  CFL.

- **Angular variable is `mu = cos(theta)`, staggered.** Cells at
  `mu_j = -1 + (j-1/2)*dmu` so no point lands exactly on a pole. The angular
  operator is the Legendre operator `d/dmu[(1-mu^2) d/dmu]` (flux form, faces at
  `mu_j + dmu/2`) and the potential is rational: `V = (2mu - m)^2/(1-mu^2) - 2`.
  No trig anywhere. Verified by CHECK 3.

- **Ghost cells (width 2 each side).** Radial inner/outer use 2nd-order
  extrapolation (`fill_ghosts_r`); angular uses pole reflection with sign
  `parity = (-1)**m` (`fill_ghosts_mu`). The parity sign is physically required —
  forcing the wrong sign shifts the recovered QNM frequency measurably.

- **Sommerfeld outgoing BC is applied inside `TeukolskyRHS.rhs`**, not in the
  ghost fill: at the outermost interior cell it overrides `d_t psi = -d_r psi -
  psi/r` (same for `v`).

### Equation hierarchy

Three nested forms, all defined and cross-checked in `check_equations.py`:

- **BIG** — full Teukolsky equation for `psi_4`.
- **PTC** — the 3D equation actually evolved, for `psi = zeta^4 * psi_4` with
  `zeta = r - i a cos(theta)`. (`PTC = 2*Sigma*zeta^4 * BIG`.)
- **MODE** — the per-`m` 2D reduction `psi = sum_m psi_m exp(i m phi)`; this is
  what the solver integrates.

CHECK 1: MODE is the exact reduction of PTC. CHECK 2: PTC descends from BIG.
CHECK 3: the angular sector reduces to the rational-in-`mu` Legendre form.

## Working with QNM runs (gotcha)

Set `rmin` just **inside** the horizon `r_+ = M + sqrt(M^2 - a^2)` (e.g.
`rmin = 1.99` for Schwarzschild `r_+ = 2M`, or `0.99*r_+` for Kerr) so every
interior cell sits **outside** `r_+`. Interior cells inside the horizon land where
`Cv/A` is positive-real, producing slow exponential growth that leaks through the
centered-difference stencil. Fit the ringdown after the initial burst and before
the first Sommerfeld reflection from `rmax` (for the standard demo: window
`[90, 130]M`).

## Repository notes

- **Never commit `ks6.tex`** (the source paper). It was downloaded from arXiv,
  we have no redistribution rights, and it is gitignored. If present it is local
  only.
- `scripts/run_example.py` outputs (`*.gif`, `*.npz`, `*_static.png`) are
  gitignored.
