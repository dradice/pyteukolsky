# PyTeukolsky: Teukolsky equation in Kerr-Schild coordinates

This code uses a finite-differencing method to solve the Teukolsky equation following the approach of [Campanelli et al., (2001)](https://arxiv.org/abs/gr-qc/0010034v2).

## Formalism

The Teukolsky equation is decomposed into angular modes as $\psi=\sum_m \psi_{m} e^{im {\tilde \varphi}}$. Each mode evolves according to

```math
\begin{align*}
(\Sigma + 2Mr){{\partial^2 {\psi_m}}\over 
{\partial t^2}} -&\Delta {{\partial^2 {\psi_m}}\over 
{\partial r^2}} - (2\, a\, i\, m + 6\, r - 6\, M){{\partial {\psi_m}}\over
{\partial r}} \\
&-{{1}\over {\sin \theta}}{{\partial}\over {\partial \theta}} \left (
\sin \theta {{\partial {\psi_m}}\over {\partial \theta}}\right ) 
-4\, M\, r{{\partial^2 {\psi_m}}\over {\partial t\, \partial r}} \\
& - \left ({4\, r+4\, i\, a\, \cos\theta+6\, M}\right ) {{\partial {\psi_m}}\over {\partial
 t}} \\
&+ (4\cot^{2}\theta-2+{m^2}\csc^{2}\theta-4\, m\, \cot\theta\csc\theta){\psi_m} 
= 0.
\end{align*}
```
Here $(t,r,\theta,\varphi)$ are the Kerr-Schild coordinates,
```math
\Sigma = r^2 + a^2\, c^2, \qquad
\Delta = r^2 - 2\, M\, r + a^2
```
and $(M,a)$ are the mass and Kerr parameter for the black-hole.

The script `scripts/check_equations.py` verifies that the equations are correct.
It checks (CHECK 1) that the mode equation above is the exact $e^{im\varphi}$
reduction of the 3D equation the code conceptually evolves, and (CHECK 2) that
the latter descends from the full $\psi_4$ Teukolsky equation. **Treat this
script as the source of truth for every coefficient**; the symbols below match
its `MODE` function line-for-line.

---

# Implementation plan

A `numpy`-based, finite-difference, time-domain solver for a single azimuthal
mode $\psi_m(t,r,\theta)$ of the Teukolsky equation in ingoing Kerr–Schild
coordinates. The user supplies $(M, a, m)$ and the initial $\psi_m$ on two
nearby time slices; the solver marches forward to a requested final time and
records $\psi_m$ at a set of extraction radii for waveform analysis.

$\psi_m$ is **complex** everywhere (the coefficients contain $i\,a\,m$ and
$i\,a\,c$), so all field and coefficient arrays are `complex128`.

## 1. Numerical method

### 1.1 First-order-in-time reduction (method of lines)

The mode equation contains a *mixed* derivative $\psi_{tr}$. In a pure
second-order-in-time update this term couples the new time level across radial
neighbours, making the update implicit. We avoid this by reducing to first
order in time. Introduce the velocity $v \equiv \partial_t\psi$; then
$\psi_{tr}=\partial_r v$ is an ordinary *spatial* derivative of a known field
and the scheme stays fully explicit. Solving the mode equation for the highest
time derivative:

```math
\begin{aligned}
\partial_t \psi &= v,\\[4pt]
\partial_t v &= \frac{1}{A}\Big[\, \mathcal{L}[\psi] \;+\; 4Mr\,\partial_r v \;+\; C_v\, v \,\Big],
\end{aligned}
```

with the angular variable $\mu \equiv \cos\theta$ (see §1.2),

```math
\begin{aligned}
A      &= \Sigma + 2Mr,\qquad \Sigma = r^2 + a^2\mu^2,\qquad C_v = 4r + 4\,i\,a\,\mu + 6M,\\
\mathcal{L}[\psi] &= \Delta\,\partial_r^2\psi
   + (2\,i\,a\,m + 6r - 6M)\,\partial_r\psi
   + \partial_\mu\!\big[(1-\mu^2)\,\partial_\mu\psi\big]
   - V\,\psi,\\
V &= \frac{(2\mu - m)^2}{1-\mu^2} - 2.
\end{aligned}
```

The angular sector is written in $\mu=\cos\theta$: the operator
$\tfrac{1}{\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta)$ becomes the
Legendre operator $\partial_\mu[(1-\mu^2)\partial_\mu] = (1-\mu^2)\partial_\mu^2 - 2\mu\,\partial_\mu$,
and the potential collapses to the perfect square above (note
$(2\mu-m)^2 = (m+s\mu)^2$ with spin weight $s=-2$, so $\mathcal{L}$'s angular
part is exactly the spin-weighted spherical-harmonic operator). Every
coefficient of the full equation is then **rational in $\mu$** — no
transcendental functions appear. This identity is verified symbolically by
CHECK 3 of `scripts/check_equations.py`.

The state vector is the pair $(\psi, v)$ on the 2D $(r,\mu)$ grid. The two
initial time slices supplied by the user are used only to *seed* $(\psi, v)$
(see §3), after which time integration is by Runge–Kutta.

### 1.2 Spatial grid

- **Radial.** A uniform grid in $x$ mapped to $r$ by the logarithmic map
  $r = M\,e^{x}$, so cells are stretched geometrically in $r$: fine near the
  horizon, coarse in the far zone.  The grid is built from $r_\min$, $r_\max$,
  $N_r$; it stores $\Delta x$ and the map derivatives $r'(x)=r''(x)=r$.  The
  domain runs from $r_\min$ *inside* the horizon $r_+ = M+\sqrt{M^2-a^2}$ to a
  large $r_\max$.  Ghost cells at both ends extend the uniform-$x$ array.
  Radial derivatives are taken with 2nd-order centered stencils in $x$ and
  mapped to $r$ by the chain rule:

  ```math
  \partial_r f = \frac{1}{r'(x)}\,\partial_x f,
  \qquad
  \partial_x f\big|_i = \frac{f_{i+1}-f_{i-1}}{2\,\Delta x}.
  ```

- **Angular.** A uniform, *staggered* grid in $\mu=\cos\theta$:
  $\mu_j = -1 + (j-\tfrac12)\,\Delta\mu$, $j=1,\dots,N_\mu$,
  $\Delta\mu = 2/N_\mu$, covering $\mu\in(-1,1)$. Staggering keeps the grid off
  the poles $\mu=\pm1$, where the $1/(1-\mu^2)$ potential is singular;
  regularity there is imposed through ghost cells (§1.4). Working in $\mu$ makes
  the angular operator polynomial (Legendre form) and the poles standard
  regular-singular points.

### 1.3 Finite differences

Second-order centered stencils in both $x$ and $\mu$, applied with `numpy`
array slicing (no Python loops over grid points). The radial stencils are taken
in the uniform coordinate $x$ and mapped to $r$ by the chain rule (§1.2). The
angular operator is discretized in flux form,
$\partial_\mu[(1-\mu^2)\partial_\mu\psi]$, evaluating $(1-\mu^2)$ at cell faces
$\mu_{j\pm1/2}$ for a compact, conservative stencil. Ghost width = **2** cells
per side per direction (one is needed for the 2nd-order stencils, two to
support Kreiss–Oliger dissipation, §1.6).

### 1.4 Boundary conditions

- **Inner (excision).** Because Kerr–Schild coordinates are horizon
  penetrating and $r_\min < r_+$, all characteristics at the inner edge point
  into the hole — no physical boundary data is required. Inner radial ghost
  cells are filled by 2nd-order extrapolation ("excision").
- **Outer.** A simple outgoing/Sommerfeld condition applied via the outer
  radial ghost cells (e.g. $\partial_t\psi \approx -\partial_r\psi - \psi/r$),
  with $r_\max$ chosen far enough that boundary reflections arrive after the
  physics of interest. (Exact at large $r$ only; documented as approximate.)
- **Poles ($\mu=\pm1$).** Fill angular ghost cells by reflection across the
  pole with a parity factor $p$: $\psi(\mu_{\rm ghost}) = p\,\psi(\mu_{\rm mirror})$.
  For a spin-weight $s=-2$, azimuthal-$m$ field the parity is $p=(-1)^{m}$;
  the staggered grid makes the choice a property of the ghost fill only.
  This parity is confirmed correct for Schwarzschild $\ell=m=2$ (Milestone 4).

### 1.5 Time integration & stability

Classic explicit **RK4** in time (method of lines). The time step follows from
the local characteristic speeds of the principal part (the coefficients of the
second derivatives in $\partial_t v$): $c_r=\sqrt{\Delta/A}$ in $r$ and
$c_\mu=\sqrt{(1-\mu^2)/A}$ in $\mu$. With $\Delta r_{\rm local}=r'(x)\,\Delta x$,

```math
\Delta t = \mathrm{CFL}\cdot \min_{\rm grid}\;\min\!\Big(\frac{\Delta r_{\rm local}}{c_r},\; \frac{\Delta\mu}{c_\mu}\Big),
\qquad \mathrm{CFL}\lesssim 0.5 .
```

The radial cell width $\Delta r_{\rm local}=r'(x)\,\Delta x = r\,\Delta x$ is
stored in `Grid.dr_cell`.  The angular bound is tightest
near the equator ($\mu=0$, where $c_\mu$ is largest).

### 1.6 Dissipation (optional)

Kreiss–Oliger dissipation to suppress high-frequency grid noise. For the
2nd-order scheme, add to each RHS the 4th-difference operator (per direction)

```math
Q f_i = -\frac{\varepsilon}{16}\,\big(f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}\big),
\qquad \varepsilon\in[0,1),
```

scaled appropriately by the grid spacing. Defaults to $\varepsilon=0$ (off).

## 2. Code structure

A small package (it can collapse to a single `teukolsky.py` if preferred):

```
pyteukolsky/
  __init__.py        # exports Grid, TeukolskyRHS, Evolution,
                     #         swsh, gaussian_pulse, project_swsh, fit_qnm_frequency
  grid.py            # Grid: coordinates, FD operators, ghost fills
  equation.py        # TeukolskyRHS: precomputed coefficients + rhs()
  evolve.py          # Evolution: state, time stepping, run loop
  initialdata.py     # swsh(), gaussian_pulse()
  diagnostics.py     # project_swsh(), fit_qnm_frequency()
scripts/
  check_equations.py # (exists) symbolic verification — source of truth
  run_example.py     # end-to-end demo: ringdown of a Gaussian pulse
tests/
  test_grid.py       # 11 tests: FD operators, 2nd-order convergence
  test_equation.py   # 16 tests: coefficient arrays, rhs() linearity
  test_evolve.py     # 30 tests: RK4 driver, CFL, detectors, I/O
  test_validation.py # 12 tests: SWSH normalization, QNM frequency, self-convergence
```

### 2.1 `Grid` (`grid.py`)

Owns the discretization; knows nothing about the physics.

- `__init__(self, rmin, rmax, Nmu, Nr, ghost=2, M=1.0)`
  - builds the radial cell array `r` from the log map $r = M\,e^{x}$ (uniform in
    $x$) with ghost-cell extensions; builds staggered `mu`
    (in $[-1,1]$, with ghosts); stores `r`, `dx`, `drdx`, `d2rdx2`, `dr_cell`,
    `dmu`; builds 2D meshes `R`, `MU`.
- `dr(f)`, `drr(f)` — first/second radial derivatives (apply the $r(x)$ map).
- `angular(f)` — the Legendre operator $\partial_\mu[(1-\mu^2)\partial_\mu f]$
  (flux form, $(1-\mu^2)$ at cell faces).
- `fill_ghosts_r(f, outer="sommerfeld", ...)` — extrapolate inner, radiate outer.
- `fill_ghosts_mu(f, parity)` — pole reflection with sign `parity`.
- properties: `shape`, `interior` (slice selecting non-ghost cells).

### 2.2 `TeukolskyRHS` (`equation.py`)

Encapsulates the PDE for fixed $(M,a,m)$; **the heart of the code**.

- `__init__(self, grid, M, a, m, dissipation=0.0)`
  - precomputes coefficient arrays on the grid: `Sigma`, `Delta`, `A`,
    `invA = 1/A`, `Cv`, the $\psi_r$ coefficient `Cr = 2iam+6r-6M`, `B = 4Mr`,
    and `V = (2*MU - m)**2/(1 - MU**2) - 2`; precomputes the $\mu$ parity factor
    from `m`. With $\mu$, `Sigma = R**2 + a**2*MU**2` and `Cv = 4*R + 4i*a*MU + 6M`
    are plain polynomials — no trig.
- `rhs(self, psi, v) -> (dpsi_dt, dv_dt)`
  1. fill radial + angular ghosts of `psi` and `v`;
  2. form spatial derivatives (`drr(psi)`, `dr(psi)`, `angular(psi)`, `dr(v)`);
  3. assemble $\mathcal{L}[\psi]$ and `dv_dt = invA*(L + B*dr(v) + Cv*v)`,
     `dpsi_dt = v`;
  4. add Kreiss–Oliger dissipation if `dissipation > 0`.

### 2.3 `Evolution` (`evolve.py`)

The user-facing driver.

- `__init__(self, rhs)` — holds the `TeukolskyRHS`, grid, and state `(psi, v)`, `t`.
- `set_initial_data(self, psi0, psi1, dt_init)` — seed
  `psi = psi1`, `v = (psi1 - psi0)/dt_init`. Overloads accept a
  callable `f(r, mu)` or a prebuilt array.
- `add_detector(self, r_extract)` — register an extraction radius; stores the
  nearest radial index / interpolation weights.
- `step(self, dt)` — one RK4 update of `(psi, v)`.
- `evolve(self, t_final, cfl=0.5, dt=None, record_every=1, snapshot_every=None)`
  — main loop: compute `dt` from CFL if not given, step to `t_final`, record
  detector time series each `record_every` steps, optionally store full-grid
  snapshots.
- results held as attributes: `times`, `waveforms` (dict
  `{r_extract: array shape (Nt, Nmu)}`), optional `snapshots`.
- `save_waveforms(self, path)` — dump detector time series + metadata to a small `.npz` (kept long-term).
- `save_snapshots(self, path)` — dump full-grid psi snapshots accumulated during `evolve()` (large, for checkpointing).

### 2.4 Initial-data helpers (`initialdata.py`)

- `swsh(spin, ell, m, mu)` — spin-weighted spherical harmonic $_{s}Y_{\ell m}(\mu)$
  at $\varphi=0$, implemented analytically for $s=-2$, $\ell=2$ using Wigner
  $d$-matrix elements:
  $_{-2}Y_{2m}(\theta) = \sqrt{5/4\pi}\,d^2_{2,m}(\theta)$.
  Returns a real array of the same shape as `mu`.
- `gaussian_pulse(grid, r0, sigma_r, ell=2, m=2, spin=-2, sigma_mu=None, amplitude=1.0)` —
  $\psi = A\,\exp\!\big(-((r-r_0)/\sigma_r)^2\big)\,{_{-2}Y_{\ell m}}(\mu)$.
  For a time-symmetric start pass the result as both `psi0` and `psi1`
  to `Evolution.set_initial_data` ($v=0$ exactly).
  Optional `sigma_mu` adds $\exp(-(\mu/\sigma_\mu)^2)$ to suppress the field
  near poles where $V=(2\mu-m)^2/(1-\mu^2)-2$ is large.

### 2.5 Diagnostics (`diagnostics.py`)

- `project_swsh(psi_mu, mu, swsh_profile)` — project the $\mu$-profile of the
  waveform at a detector onto a SWSH via midpoint-rule quadrature:
  $\psi_{\ell m}(t) = \int_{-1}^{1} \psi_m(t,\mu)\,\overline{Y(\mu)}\,d\mu$.
  Handles shape `(Nmu,)` or `(Nt, Nmu)` input; returns shape `()` or `(Nt,)`.
- `fit_qnm_frequency(times, psi_t, t_start, t_end)` — extract $({\omega_R}, {\omega_I})$
  from a waveform slice. For **real** signals (Schwarzschild) fits a damped
  cosine $A\,e^{\omega_I t}\cos(\omega_R t + \phi)$ via `scipy.optimize.curve_fit`
  with zero-crossing-rate and log-slope initial guesses. For **complex** signals
  (Kerr / complex initial data) uses linear regression on $\log|\psi|$ and the
  unwrapped phase. Returns $(\omega_R>0,\, \omega_I<0)$.

## 3. Public API and usage

```python
from pyteukolsky import Grid, TeukolskyRHS, Evolution
from pyteukolsky import swsh, gaussian_pulse, project_swsh, fit_qnm_frequency

M, a, m = 1.0, 0.0, 2
# For Schwarzschild: rmin just inside r_+ = 2M to avoid inside-horizon instability
grid = Grid(rmin=1.99, rmax=100.0, Nmu=32, Nr=200, M=M)
rhs  = TeukolskyRHS(grid, M, a, m, dissipation=0.1)
sim  = Evolution(rhs)

psi0 = gaussian_pulse(grid, r0=10.0, sigma_r=2.0, ell=2, m=m)
sim.set_initial_data(psi0=psi0, psi1=psi0, dt_init=1e-3)  # time-symmetric

sim.add_detector(30.0)
sim.evolve(t_final=140.0, cfl=0.45)
sim.save_waveforms("ringdown.npz")
```

### 3.1 Waveform extraction and QNM fitting

At each detector the solver records the full $\mu$ profile $\psi_m(t,\mu)$.
Project onto $_{-2}Y_{\ell m}$ to collapse to a scalar time series, then fit
the ringdown:

```python
mu  = grid._mu[grid.ghost : grid.ghost + grid.Nmu]
sw  = swsh(-2, 2, m, mu)

# psi_22 shape: (Nt,)
psi_22 = project_swsh(sim.waveforms[30.0], mu, sw)

# Fit in window after transient burst has passed and before boundary reflections
omega_R, omega_I = fit_qnm_frequency(sim.times, psi_22.real,
                                     t_start=90.0, t_end=130.0)
print(f"Mω = {M*omega_R:.4f} + {M*omega_I:.4f}i")
# Schwarzschild ℓ=m=2: Mω ≈ 0.3737 - 0.0890i
```

**Timing note for Schwarzschild ($a=0$):** the outgoing burst from $r_0=10\,M$ peaks
at $r_{\rm ext}=30\,M$ around $t\approx60\text{–}80\,M$.  Start the fit at
$t\approx90\,M$ (burst gone) and stop before $t\approx160\,M$ (first Sommerfeld
reflection from $r_{\rm max}=100\,M$).

**rmin guideline:** with the log grid and $r_+=2M$ (Schwarzschild), choosing
$r_{\rm min}\approx1.99\,M$ ensures all interior cells sit outside the horizon
($r>r_+$).  Choosing $r_{\rm min}=1.5\,M$ places several interior cells inside
the horizon where the $C_v/A$ coefficient is positive-real, causing slow
exponential growth that leaks through the centered-difference stencil.  For
Kerr use $r_{\rm min}$ just inside $r_+=M+\sqrt{M^2-a^2}$.

## 4. Validation & roadmap

Milestones, in order:

1. ✅ **Grid + operators** — `Grid` with derivative and ghost-fill methods;
   unit-test FD operators against analytic functions (2nd-order convergence
   verified; 11 tests in `tests/test_grid.py`).
2. ✅ **RHS** — `TeukolskyRHS.rhs`; coefficient arrays cross-checked against
   `scripts/check_equations.py` at sample points (16 tests in
   `tests/test_equation.py`).
3. ✅ **Evolution** — RK4 driver + CFL + Sommerfeld/excision/pole BCs; pulse
   propagation verified via `scripts/run_example.py` (1D and 2D animations,
   waveform extraction; 30 tests in `tests/test_evolve.py`).
4. ✅ **Validation** — Schwarzschild ($a=0$) $\ell=m=2$ ringdown with SWSH
   initial data (`initialdata.py`), SWSH projection and damped-cosine fitting
   (`diagnostics.py`). Extracted $M\omega_R=0.3717$, $M\omega_I=-0.0904$ vs.
   known $0.3737-0.0890\,i$ (<2% error at $N_r=100$, $N_\mu=16$).
   Self-convergence ratio $>3$ confirmed (2nd-order in $N_r$).
   Key finding: use $r_{\rm min}\approx r_+$ to keep interior cells outside
   the horizon (see §3.1). 12 tests in `tests/test_validation.py`.
5. **Kerr** — repeat for $a\neq0$; confirm QNM frequencies vs. published tables;
   validate the pole-parity factor (§1.4) here.
6. **Polish** — docs.

## 5. Notes

- `ks6.tex` is a local-only arXiv copy (not redistributable); it is gitignored.
  Keep equation cross-references pointing at `scripts/check_equations.py`, which
  is the committed source of truth.