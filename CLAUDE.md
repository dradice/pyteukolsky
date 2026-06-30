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
- `pyteukolsky/` — the solver package (milestones 1–5 complete; see below).
- `scripts/run_example.py` — end-to-end pulse demo (produces static PNG, 1D/2D GIFs, waveform .npz).
- `tests/` — pytest unit tests for every completed milestone.

The chosen scheme is method-of-lines (first-order-in-time reduction with
`v = psi_t`, so the mixed `psi_tr` term becomes an explicit spatial derivative
`v_r`) integrated with RK4, on a logarithmically stretched radial grid
(uniform in `x` with `r = M exp(x)`) and a staggered grid in `mu = cos(theta)`. In `mu`
the angular operator is the Legendre operator `d/dmu[(1-mu^2) d/dmu]` and the
potential is `(2 mu - m)^2/(1-mu^2) - 2`; all coefficients become rational in
`mu` (no trig). This substitution is verified by CHECK 3 in
scripts/check_equations.py.

## Implementation status

### Done

**Milestone 1 — `pyteukolsky/grid.py` (`Grid`)**
Coordinates, finite-difference operators, and ghost fills.
- Radial grid: uniform in `x` with the log map `r = M exp(x)` (`rmin`, `rmax`,
  `Nr`), giving geometric stretching in `r`. Exposes `dx`, `drdx = r`,
  `d2rdx2 = r`. Staggered angular grid `mu_j = -1 + (j-½)Δμ`.
- `dr(f)`, `drr(f)` — 2nd-order centered stencils in `x` mapped to `r` via the
  chain rule (`d/dr = (1/r) d/dx`).
- `angular(f)` — Legendre operator `d/dmu[(1-mu^2) d/dmu f]` in flux form.
- `fill_ghosts_r(f)` — 2nd-order extrapolation at inner (excision) and outer boundaries.
- `fill_ghosts_mu(f, parity)` — pole reflection with sign `(-1)**m`.
- `ko_dissipation_r/mu(f, epsilon)` — Kreiss–Oliger 4th-difference dissipation
  (radial version divides by `dx`).
- `dr_cell` — physical radial cell width `r·dx`, shape `(Nr,)`, used by the CFL condition.
- Tests: `tests/test_grid.py` (11 tests, all 2nd-order convergence verified).

**Milestone 2 — `pyteukolsky/equation.py` (`TeukolskyRHS`)**
Precomputed PDE coefficients and the `rhs(psi, v)` method.
- Coefficient arrays (complex128, full 2D grid): `Sigma`, `Delta`, `A`, `invA`,
  `Cv = 4r + 4i·a·μ + 6M`, `Cr = 2i·a·m + 6r − 6M`, `B = 4Mr`,
  `V = (2μ − m)²/(1 − μ²) − 2`.
- `rhs(psi, v)` fills ghosts, computes `L[psi]`, returns `(v, invA*(L + B*v_r + Cv*v))`.
- Tests: `tests/test_equation.py` (16 tests; coefficients cross-checked against
  `scripts/check_equations.py` at sample interior points).

**Milestone 3 — `pyteukolsky/evolve.py` (`Evolution`)**
RK4 time driver, CFL timestep, detector waveform extraction, snapshot I/O.
- `cfl_dt(cfl)` — CFL-limited timestep using `g.dr_cell` for radial width and
  `sqrt(Delta/A)`, `sqrt((1-mu^2)/A)` for characteristic speeds.
- `step(dt)` — one RK4 step of `(psi, v)`.
- `evolve(t_final, dt, snapshot_every)` — main loop with waveform recording.
- `add_detector(r_extract)` — linear interpolation to extraction radius.
- `save_waveforms(path)` / `save_snapshots(path)` — separate `.npz` files
  (waveforms are small/long-lived; snapshots are large/checkpoint-style).
- Tests: `tests/test_evolve.py` (30 tests).

**`scripts/run_example.py`**
End-to-end demonstration: time-symmetric 2D Gaussian pulse on the log r
grid, evolved and saved as static figure + 1D/2D GIF animations + waveforms.
- `run_simulation(args)` — sets up grid, initial data `(psi0, v=0)`, evolves,
  returns pre-computed `r·Re[ψ_m]` snapshots.
- `make_static_figure`, `make_animation_1d`, `make_animation_2d` — separate
  visualization functions outside `main()`.
- Initial data: 2D Gaussian `exp(-((r-r0)/σ_r)²) * exp(-(μ/σ_μ)²)` with
  `σ_μ=0.3` to suppress the near-pole singularity of `V = (2μ-m)²/(1-μ²)-2`.
- Plots `r·Re[ψ_m]` (not `ψ_m`) to remove the 1/r fall-off.

**Milestone 4 — `pyteukolsky/initialdata.py`, `pyteukolsky/diagnostics.py`**
Schwarzschild validation, SWSH initial data, QNM frequency extraction.
- `pyteukolsky/initialdata.py`: `swsh(spin, ell, m, mu)` — analytic _{-2}Y_{2m}(μ) at φ=0
  using Wigner d-matrix elements; `gaussian_pulse(grid, r0, sigma_r, ...)` — Gaussian
  shell × SWSH angular profile.
- `pyteukolsky/diagnostics.py`: `project_swsh(psi_mu, mu, swsh_profile)` — midpoint-rule
  SWSH projection; `fit_qnm_frequency(times, psi_t, t_start, t_end)` — extracts (ω_R, ω_I)
  via damped-cosine `curve_fit` (real signals) or log-amplitude/phase linear fit (complex).
- Key finding: for QNM runs use `rmin ≈ 1.99M` (Schwarzschild r₊ = 2M) so all interior
  cells are outside the horizon; with `rmin = 1.5M` several interior cells sit inside the
  horizon where the Cv/A coefficient is positive-real → exponential growth leaking through
  the FD stencil. The transient wave burst at r_ext=30M peaks at t ≈ 60-80M; fit the QNM
  in [90, 130]M to avoid the burst.
- QNM result: Schwarzschild ℓ=m=2 fundamental mode extracted as Mω_R = 0.3717,
  Mω_I = −0.0904 (known: 0.3737 − 0.0890i) with Nr=100, Nmu=16.
- Tests: `tests/test_validation.py` (12 tests: SWSH normalization, pulse shape,
  projection, synthetic QNM fit, Schwarzschild QNM frequency, Nr self-convergence).

**Milestone 5 — Kerr validation (`tests/test_kerr.py`)**
Kerr (`a≠0`) ℓ=m=2 QNM frequencies vs. published tables; pole-parity check.
- No new solver code: `TeukolskyRHS` coefficients (`Sigma = r²+a²μ²`, `Delta`,
  `A`, `Cv`, `Cr`) already carry full `a`-dependence, so the 2D `(r,μ)` evolution
  captures the spin-weighted *spheroidal* angular structure automatically (the
  `a²ω²μ²` coupling enters through `A` acting on `d_t v`). Projecting onto the
  spherical `_{-2}Y_{2m}` still isolates the dominant ℓ=2 content.
- For `a≠0` the field is genuinely complex (`Cv`, `Cr` have imaginary parts), so
  use the **complex code path** of `fit_qnm_frequency` (log-amp + unwrapped-phase
  linear fits). The real-part fit is noticeably noisier for Kerr.
- Grid: `rmin = 0.99·r₊` with `r₊ = M + √(M²−a²)` keeps all interior cells
  outside the horizon (same instability avoidance as Schwarzschild). Window
  `[100,150]M` (rmax=120M, t_final=160M) is clean of burst and reflections.
- Results (Nr=120, Nmu=24): code vs. published Mω:
  a=0.5 → (0.4597, −0.0878) vs (0.4641, −0.0846);
  a=0.9 → (0.6687, −0.0643) vs (0.6716, −0.0649). All ω_R within ~1%.
  Reference values: Berti, Cardoso & Will, PRD 73, 064030 (2006),
  arXiv:gr-qc/0512160 (tables at https://pages.jh.edu/eberti2/ringdown/),
  via Leaver's method, Proc. R. Soc. Lond. A 402, 285 (1985).
- Pole parity: the correct `(−1)^m = +1` parity for m=2 recovers the published
  QNM; forcing parity `−1` shifts Mω_R to ~0.51 (a ~0.05 error), confirming the
  ghost-fill sign matters. Tested by overriding `rhs.parity`.
- Tests: `tests/test_kerr.py` (5 tests: interior-cell placement, a=0.5 & a=0.9
  QNM, prograde spin trend, pole parity).

### Pending

- **Milestone 6** — Polish: docs.

## Commands

The user's Python interpreter is `~/local/miniforge/bin/python` (has sympy 1.14,
scipy 1.18, numpy, matplotlib). Plain `python`/`python3` are not on PATH — always
use the miniforge path.

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
