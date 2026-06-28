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

- **Radial.** The grid stores cell-centre positions `r` directly.  By default
  they are uniformly spaced from $r_\min$ to $r_\max$ with $N_r$ cells; an
  arbitrary monotone sequence may be supplied via `r_array`.  The domain runs
  from $r_\min$ *inside* the horizon $r_+ = M+\sqrt{M^2-a^2}$ to a large
  $r_\max$.  Ghost cells at both ends extend the array using the local boundary
  spacing, keeping FD stencils well-conditioned.  Radial FD operators use
  2nd-order formulas for general non-uniform spacing:

  ```math
  \partial_r f\big|_i = \frac{h_-^2\,f_{i+1} - (h_+^2 - h_-^2)\,f_i - h_+^2\,f_{i-1}}
                             {h_+\,h_-\,(h_+ + h_-)},
  ```

  with $h_+ = r_{i+1}-r_i$, $h_- = r_i-r_{i-1}$; this reduces to
  $(f_{i+1}-f_{i-1})/(2\Delta r)$ for a uniform grid.

- **Angular.** A uniform, *staggered* grid in $\mu=\cos\theta$:
  $\mu_j = -1 + (j-\tfrac12)\,\Delta\mu$, $j=1,\dots,N_\mu$,
  $\Delta\mu = 2/N_\mu$, covering $\mu\in(-1,1)$. Staggering keeps the grid off
  the poles $\mu=\pm1$, where the $1/(1-\mu^2)$ potential is singular;
  regularity there is imposed through ghost cells (§1.4). Working in $\mu$ makes
  the angular operator polynomial (Legendre form) and the poles standard
  regular-singular points.

### 1.3 Finite differences

Second-order centered stencils in both $r$ and $\mu$, applied with `numpy`
array slicing (no Python loops over grid points). The radial stencils use the
non-uniform Lagrange formula (§1.2), which reduces to the standard form on
a uniform grid. The angular operator is discretized in flux form,
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
  For a spin-weight $s=-2$, azimuthal-$m$ field the parity is
  $p=(-1)^{m}$ — **to be validated** against a known solution / the regularity
  exponent of $_{-2}Y_{\ell m}$; the staggered grid makes the choice a property
  of the ghost fill only.

### 1.5 Time integration & stability

Classic explicit **RK4** in time (method of lines). The time step follows from
the local characteristic speeds of the principal part (the coefficients of the
second derivatives in $\partial_t v$): $c_r=\sqrt{\Delta/A}$ in $r$ and
$c_\mu=\sqrt{(1-\mu^2)/A}$ in $\mu$. With $\Delta r_{\rm local}=r'(x)\,\Delta x$,

```math
\Delta t = \mathrm{CFL}\cdot \min_{\rm grid}\;\min\!\Big(\frac{\Delta r_{\rm local}}{c_r},\; \frac{\Delta\mu}{c_\mu}\Big),
\qquad \mathrm{CFL}\lesssim 0.5 .
```

The radial cell width $\Delta r_{\rm local}$ is the average of the forward
and backward spacing stored in `Grid.dr_cell`.  The angular bound is tightest
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
  __init__.py        # exports Grid, TeukolskyRHS, Evolution
  grid.py            # Grid: coordinates, FD operators, ghost fills
  equation.py        # TeukolskyRHS: precomputed coefficients + rhs()
  evolve.py          # Evolution: state, time stepping, run loop
  diagnostics.py     # Detector: extraction-radius time series, SWSH projection
  initialdata.py     # helpers: gaussian_pulse, swsh, seed-from-two-slices
scripts/
  check_equations.py # (exists) symbolic verification — source of truth
  run_example.py     # end-to-end demo: ringdown of a Gaussian pulse
tests/
  test_*.py          # convergence, QNM frequency, regression
```

### 2.1 `Grid` (`grid.py`)

Owns the discretization; knows nothing about the physics.

- `__init__(self, rmin=None, rmax=None, Nmu=None, Nr=None, ghost=2, M=1.0, r_array=None)`
  - builds the radial cell array `r` (uniform by default; user-supplied if
    `r_array` is given) with ghost-cell extensions; builds staggered `mu`
    (in $[-1,1]$, with ghosts); stores `r`, `dr_cell`, `dmu`; builds 2D meshes
    `R`, `MU`.
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

### 2.4 `Detector` / diagnostics (`diagnostics.py`)

- records $\psi_m(t, r_{\rm ext}, \mu)$ at each registered radius (radial
  interpolation to $r_{\rm ext}$);
- `project_swsh(mu_profile, ell, m, spin=-2)` — optionally collapse the
  $\mu$ profile onto a spin-weighted spherical harmonic $_{-2}Y_{\ell m}$ via
  Gauss–Legendre-style quadrature in $\mu$ to produce a single complex waveform
  $\psi_{\ell m}(t)$ for ringdown analysis.

### 2.5 Initial-data helpers (`initialdata.py`)

- `gaussian_pulse(grid, r0, sigma, ell=2, m=..., spin=-2, amplitude=1.0)` —
  a Gaussian shell in $r$ times $_{-2}Y_{\ell m}(\mu)$; convenient default test data.
- `swsh(spin, ell, m, mu)` — spin-weighted spherical harmonics as functions of
  $\mu$ (for data construction and projection).
- time-symmetric option: `psi0 = psi1` (zero initial velocity).

## 3. Public API and usage

```python
from pyteukolsky import Grid, TeukolskyRHS, Evolution
from pyteukolsky.initialdata import gaussian_pulse

M, a, m = 1.0, 0.9, 2
grid = Grid(rmin=0.8, rmax=400.0, Nmu=64, Nr=4000)      # rmin < r_+ (excision)
rhs  = TeukolskyRHS(grid, M, a, m, dissipation=0.1)
sim  = Evolution(rhs)

psi1 = gaussian_pulse(grid, r0=20.0, sigma=2.0, ell=2, m=m)
sim.set_initial_data(psi0=psi1, psi1=psi1, dt_init=1e-3)  # time-symmetric

for r_ext in (50.0, 100.0, 200.0):
    sim.add_detector(r_ext)

sim.evolve(t_final=500.0, cfl=0.5)
sim.save("ringdown.npz")            # times + psi_m(t, mu) at each detector
```

### 3.1 Waveform extraction

At each detector radius the solver stores the full $\mu$ profile of
$\psi_m(t,\mu)$. For ringdown studies, project onto $_{-2}Y_{\ell m}$ to get
$\psi_{\ell m}(t)$, then fit damped sinusoids to read off quasinormal-mode
frequencies. The dominant test case is $\ell=m=2$.

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
4. **Validation** — Schwarzschild ($a=0$) ringdown: extract $\ell=m=2$ QNM
   frequency and compare to the known value ($M\omega \approx 0.3737 - 0.0890\,i$);
   self-convergence test in $N_r$, $N_\mu$.
5. **Kerr** — repeat for $a\neq0$; confirm QNM frequencies vs. published tables;
   validate the pole-parity factor (§1.4) here.
6. **Polish** — snapshot I/O, docs.

## 5. Notes

- `ks6.tex` is a local-only arXiv copy (not redistributable); it is gitignored.
  Keep equation cross-references pointing at `scripts/check_equations.py`, which
  is the committed source of truth.