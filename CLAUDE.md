# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

PyTeukolsky is intended to be a finite-difference time-domain solver for the
Teukolsky equation in (ingoing/outgoing) Kerr–Schild horizon-penetrating
coordinates, following Campanelli et al. (2001), gr-qc/0010034v2.

The repository contains:
- `ks6.tex` — the source paper deriving the formalism (equations, initial-data
  prescription). This is the **authoritative reference for every equation**;
  code should cite line ranges of this file (e.g. "lines 431-449").
  **Never commit `ks6.tex` to the repository.** It was downloaded from arXiv
  and we do not have copyright permission to redistribute it. Keep it local
  only (e.g. add it to `.gitignore`); never `git add` it.
- `scripts/check_equations.py` — symbolic verification of the equations.
- `README.md` — formalism summary, the mode equation, and the full
  implementation plan. **Consult it before starting any new milestone.**
- `pyteukolsky/` — the solver package (milestones 1–2 complete; see below).
- `tests/` — pytest unit tests for every completed milestone.

The chosen scheme is method-of-lines (first-order-in-time reduction with
`v = psi_t`, so the mixed `psi_tr` term becomes an explicit spatial derivative
`v_r`) integrated with RK4, on a log-stretched radial grid and a staggered
grid in `mu = cos(theta)`. In `mu` the angular operator is the Legendre
operator `d/dmu[(1-mu^2) d/dmu]` and the potential is
`(2 mu - m)^2/(1-mu^2) - 2`; all coefficients become rational in `mu` (no trig).
This substitution is verified by CHECK 3 in scripts/check_equations.py.

## Implementation status

### Done

**Milestone 1 — `pyteukolsky/grid.py` (`Grid`)**
Coordinates, finite-difference operators, and ghost fills.
- Log-stretched radial grid `r = M exp(x)`; staggered angular grid `mu_j = -1 + (j-½)Δμ`.
- `dr(f)`, `drr(f)` — 2nd-order centered FD in `x`, chain-ruled to `r`.
- `angular(f)` — Legendre operator `d/dmu[(1-mu^2) d/dmu f]` in flux form.
- `fill_ghosts_r(f)` — 2nd-order extrapolation at inner (excision) and outer boundaries.
- `fill_ghosts_mu(f, parity)` — pole reflection with sign `(-1)**m`.
- `ko_dissipation_r/mu(f, epsilon)` — Kreiss–Oliger 4th-difference dissipation.
- Tests: `tests/test_grid.py` (10 tests, all 2nd-order convergence verified).

**Milestone 2 — `pyteukolsky/equation.py` (`TeukolskyRHS`)**
Precomputed PDE coefficients and the `rhs(psi, v)` method.
- Coefficient arrays (complex128, full 2D grid): `Sigma`, `Delta`, `A`, `invA`,
  `Cv = 4r + 4i·a·μ + 6M`, `Cr = 2i·a·m + 6r − 6M`, `B = 4Mr`,
  `V = (2μ − m)²/(1 − μ²) − 2`.
- `rhs(psi, v)` fills ghosts, computes `L[psi]`, returns `(v, invA*(L + B*v_r + Cv*v))`.
- Tests: `tests/test_equation.py` (16 tests; coefficients cross-checked against
  `scripts/check_equations.py` at sample interior points).

### Pending

- **Milestone 3** — `pyteukolsky/evolve.py` (`Evolution`): RK4 driver, CFL timestep,
  Sommerfeld outer BC, detector registration.
- **Milestone 4** — Validation: Schwarzschild (`a=0`) `ℓ=m=2` ringdown, QNM frequency
  `Mω ≈ 0.3737 − 0.0890i`, self-convergence tests.
- **Milestone 5** — Kerr (`a≠0`): QNM vs. published tables, validate pole parity.
- **Milestone 6** — Polish: snapshot I/O, `run_example.py`, docs.

## Commands

The user's Python interpreter is `~/local/miniforge/bin/python` (has sympy
1.14). Plain `python`/`python3` are not on PATH — always use the miniforge path.

Verify the equations:
```
~/local/miniforge/bin/python scripts/check_equations.py
```
Expected output: both checks report `IDENTICAL -> difference simplifies to 0`.

## Equation hierarchy

The physics is organized as three nested forms, all defined in
`scripts/check_equations.py` and tied to `ks6.tex`:

- **BIG** (ks6.tex lines 354-391): the full Teukolsky equation for `psi_4`.
- **PTC** (lines 402-424): the 3D equation actually evolved, for
  `psi = zeta^4 * psi_4` where `zeta = r - i a cos(theta)`. Related to BIG by
  `PTC[psi] = 2*Sigma*zeta^4 * BIG[zeta^-4 psi]`.
- **MODE** (lines 431-449): the angular reduction
  `psi = sum_m psi_m exp(i m phi)`. This per-`m` 2D equation (in t, r, theta) is
  what a solver integrates.

`check_equations.py` verifies MODE is the exact reduction of PTC (CHECK 1) and
PTC is the exact rescaling of BIG (CHECK 2). When changing any equation,
update both the relevant `def` in the script and the corresponding lines in
`ks6.tex`, then re-run the check.

Key symbols: `Sigma = r^2 + a^2 cos^2(theta)`, `Delta = r^2 - 2 M r + a^2`,
`(M, a)` the Kerr mass and spin parameters, `m` the azimuthal mode number.
