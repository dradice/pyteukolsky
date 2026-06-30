# PyTeukolsky: Teukolsky equation in Kerr–Schild coordinates

PyTeukolsky is a `numpy`-based, finite-difference, time-domain solver for a
single azimuthal mode $\psi_m(t,r,\theta)$ of the Teukolsky equation in
ingoing Kerr–Schild (horizon-penetrating) coordinates, following the approach
of [Campanelli et al. (2001)](https://arxiv.org/abs/gr-qc/0010034v2).

The user supplies the black-hole parameters $(M, a)$, the azimuthal mode number
$m$, and initial data for $\psi_m$; the solver marches forward in time and
records $\psi_m$ at chosen extraction radii for waveform analysis. It reproduces
the Schwarzschild and Kerr $\ell=m=2$ fundamental quasi-normal-mode (QNM)
frequencies to ~1% (see [Validation](#validation)).

## Formalism

The Teukolsky equation is decomposed into azimuthal modes,
$\psi=\sum_m \psi_{m} e^{im{\tilde\varphi}}$. Each mode evolves according to

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

Here $(t,r,\theta,\varphi)$ are the Kerr–Schild coordinates,

```math
\Sigma = r^2 + a^2\, \cos^2\theta, \qquad
\Delta = r^2 - 2\, M\, r + a^2,
```

and $(M,a)$ are the mass and Kerr spin parameter of the black hole. The field
$\psi_m$ is **complex** everywhere — the coefficients contain $i\,a\,m$ and
$i\,a\cos\theta$ — so all field and coefficient arrays are `complex128`.

The script `scripts/check_equations.py` symbolically verifies the equations. It
checks (CHECK 1) that the mode equation above is the exact $e^{im\varphi}$
reduction of the 3D equation the code evolves, (CHECK 2) that the latter
descends from the full $\psi_4$ Teukolsky equation, and (CHECK 3) that the
angular operator and potential reduce to the rational-in-$\mu$ forms used in the
code (below). **Treat this script as the source of truth for every
coefficient.**

## Installation & requirements

Pure Python; no build step. Dependencies:

- `numpy` (arrays, FD operators, time stepping)
- `scipy` (QNM curve fitting in `diagnostics.py`)
- `matplotlib` (only for the `scripts/run_example.py` animations)
- `sympy` (only for `scripts/check_equations.py`)
- `pytest` (to run the test suite)

Put the repository root on `PYTHONPATH` (or run from it) and import the package:

```python
from pyteukolsky import Grid, TeukolskyRHS, Evolution
from pyteukolsky import swsh, gaussian_pulse, project_swsh, fit_qnm_frequency
```

## Quick start

A Schwarzschild ($a=0$) $\ell=m=2$ ringdown:

```python
from pyteukolsky import Grid, TeukolskyRHS, Evolution
from pyteukolsky import swsh, gaussian_pulse, project_swsh, fit_qnm_frequency

M, a, m = 1.0, 0.0, 2
# rmin just inside r_+ = 2M to keep all interior cells outside the horizon
grid = Grid(rmin=1.99, rmax=100.0, Nmu=32, Nr=200, M=M)
rhs  = TeukolskyRHS(grid, M, a, m, dissipation=0.1)
sim  = Evolution(rhs)

# Time-symmetric Gaussian pulse: pass it as both slices so v = 0 exactly
psi0 = gaussian_pulse(grid, r0=10.0, sigma_r=2.0, ell=2, m=m)
sim.set_initial_data(psi0=psi0, psi1=psi0, dt_init=1e-3)

sim.add_detector(30.0)
sim.evolve(t_final=140.0, cfl=0.45)

# Project the detector mu-profile onto _{-2}Y_{2m}, then fit the ringdown
mu = grid._mu[grid.ghost : grid.ghost + grid.Nmu]
psi_22 = project_swsh(sim.waveforms[30.0], mu, swsh(-2, 2, m, mu))
omega_R, omega_I = fit_qnm_frequency(sim.times, psi_22.real,
                                     t_start=90.0, t_end=130.0)
print(f"Mω = {M*omega_R:.4f} + {M*omega_I:.4f}i")
# Schwarzschild ℓ=m=2: Mω ≈ 0.3737 - 0.0890i
```

**Timing note ($a=0$).** The outgoing burst from $r_0=10\,M$ peaks at
$r_{\rm ext}=30\,M$ around $t\approx60\text{–}80\,M$. Fit *after* the burst
($t\gtrsim90\,M$) and *before* the first Sommerfeld reflection from
$r_{\rm max}=100\,M$ ($t\approx160\,M$).

**Choosing `rmin`.** With the logarithmic grid, choose $r_{\rm min}$ just inside
the horizon $r_+=M+\sqrt{M^2-a^2}$ ($r_+=2M$ for Schwarzschild) so that *every
interior cell* sits outside $r_+$. Placing interior cells inside the horizon
(e.g. $r_{\rm min}=1.5\,M$) lands them where the $C_v/A$ coefficient is
positive-real, producing slow exponential growth that leaks through the
centered-difference stencil.

## Running the code

```bash
# Symbolic equation check (source of truth) — expect two "IDENTICAL" reports
python scripts/check_equations.py

# End-to-end demo: ringdown of a Gaussian pulse, with figures and animations
python scripts/run_example.py

# Test suite (73 tests)
pytest
```

`scripts/run_example.py` evolves a time-symmetric 2D Gaussian pulse on the log-$r$
grid and writes a static figure, 1D and 2D GIF animations, and a waveform
`.npz`. It plots $r\,\mathrm{Re}[\psi_m]$ (not $\psi_m$) to remove the $1/r$
fall-off. All of its outputs (`*.gif`, `*.npz`, `*_static.png`) are gitignored.

## Numerical method

### Method of lines (first-order-in-time reduction)

The mode equation contains a *mixed* derivative $\psi_{tr}$, which in a
second-order-in-time update would couple the new time level across radial
neighbours and make the scheme implicit. The code reduces to first order in
time instead: introduce the velocity $v\equiv\partial_t\psi$, so $\psi_{tr}=
\partial_r v$ becomes an ordinary *spatial* derivative of a known field and the
update stays fully explicit. Solving the mode equation for the highest time
derivative gives the evolution system

```math
\begin{aligned}
\partial_t \psi &= v,\\[4pt]
\partial_t v &= \frac{1}{A}\Big[\, \mathcal{L}[\psi] \;+\; 4Mr\,\partial_r v \;+\; C_v\, v \,\Big],
\end{aligned}
```

with the angular variable $\mu\equiv\cos\theta$,

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

The state vector is the pair $(\psi, v)$ on the 2D $(r,\mu)$ grid; RK4 advances
it in time. The two initial time slices supplied by the user only *seed*
$(\psi, v)$.

Working in $\mu=\cos\theta$ turns the angular operator
$\tfrac{1}{\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta)$ into the
Legendre operator
$\partial_\mu[(1-\mu^2)\partial_\mu]=(1-\mu^2)\partial_\mu^2-2\mu\,\partial_\mu$,
and the potential collapses to the perfect square above (note
$(2\mu-m)^2=(m+s\mu)^2$ for spin weight $s=-2$, so $\mathcal{L}$'s angular part
is exactly the spin-weighted spherical-harmonic operator). Every coefficient is
then **rational in $\mu$** — no transcendental functions appear (CHECK 3).

### Spatial grid

- **Radial.** A uniform grid in $x$ mapped to $r$ by the logarithmic map
  $r=M\,e^{x}$, so cells stretch geometrically in $r$: fine near the horizon,
  coarse in the far zone. Built from $r_\min$, $r_\max$, $N_r$; the map
  derivatives satisfy $r'(x)=r''(x)=r$. The domain runs from $r_\min$ *inside*
  the horizon to a large $r_\max$, with ghost cells extending the uniform-$x$
  array at both ends. Radial derivatives use 2nd-order centered stencils in $x$
  mapped to $r$ by the chain rule, $\partial_r f = r'(x)^{-1}\partial_x f$.

- **Angular.** A uniform, *staggered* grid in $\mu$:
  $\mu_j=-1+(j-\tfrac12)\,\Delta\mu$, $\Delta\mu=2/N_\mu$, covering
  $\mu\in(-1,1)$. Staggering keeps the grid off the poles $\mu=\pm1$ where the
  $1/(1-\mu^2)$ potential is singular; regularity there is imposed through ghost
  cells (below). The angular operator is discretized in flux form, evaluating
  $(1-\mu^2)$ at cell faces $\mu_{j\pm1/2}$ for a compact, conservative stencil.

Ghost width is **2** cells per side per direction (one for the 2nd-order
stencils, two to support Kreiss–Oliger dissipation).

### Boundary conditions

- **Inner (excision).** Because the coordinates are horizon-penetrating and
  $r_\min<r_+$, all characteristics at the inner edge point into the hole and no
  physical boundary data is required. Inner radial ghosts are filled by
  2nd-order extrapolation.
- **Outer (Sommerfeld).** An outgoing/radiative condition
  ($\partial_t\psi\approx-\partial_r\psi-\psi/r$) is applied via the outer radial
  ghosts. It is exact only as $r\to\infty$; choose $r_\max$ so reflections arrive
  after the physics of interest.
- **Poles ($\mu=\pm1$).** Angular ghosts are filled by reflection across the
  pole with a parity factor $p$: $\psi(\mu_{\rm ghost})=p\,\psi(\mu_{\rm mirror})$.
  For a spin-weight $s=-2$, azimuthal-$m$ field the parity is $p=(-1)^{m}$. This
  is confirmed correct for $\ell=m=2$: the Schwarzschild and Kerr QNM frequencies
  are recovered only with $p=+1$ ([Validation](#validation)); forcing $p=-1$
  shifts the frequency out of tolerance.

### Time integration & stability

Classic explicit **RK4**. The time step follows from the local characteristic
speeds of the principal part, $c_r=\sqrt{\Delta/A}$ and
$c_\mu=\sqrt{(1-\mu^2)/A}$, with radial cell width
$\Delta r_{\rm local}=r'(x)\,\Delta x=r\,\Delta x$ (stored as `Grid.dr_cell`):

```math
\Delta t = \mathrm{CFL}\cdot \min_{\rm grid}\;\min\!\Big(\frac{\Delta r_{\rm local}}{c_r},\; \frac{\Delta\mu}{c_\mu}\Big),
\qquad \mathrm{CFL}\lesssim 0.5 .
```

The angular bound is tightest near the equator ($\mu=0$).

### Dissipation (optional)

Kreiss–Oliger dissipation suppresses high-frequency grid noise. For the
2nd-order scheme, each RHS gets the 4th-difference operator (per direction)

```math
Q f_i = -\frac{\varepsilon}{16}\,\big(f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}\big),
\qquad \varepsilon\in[0,1),
```

scaled by the grid spacing; $\varepsilon=0$ disables it (default).

## Code structure

```
pyteukolsky/
  __init__.py        # public API exports
  grid.py            # Grid: coordinates, FD operators, ghost fills
  equation.py        # TeukolskyRHS: precomputed coefficients + rhs()
  evolve.py          # Evolution: state, RK4 stepping, run loop, detectors, I/O
  initialdata.py     # swsh(), gaussian_pulse()
  diagnostics.py     # project_swsh(), fit_qnm_frequency()
scripts/
  check_equations.py # symbolic verification — source of truth for coefficients
  run_example.py     # end-to-end demo: ringdown of a Gaussian pulse
tests/
  test_grid.py       # 10 tests: FD operators, 2nd-order convergence
  test_equation.py   # 16 tests: coefficient arrays, rhs() linearity
  test_evolve.py     # 30 tests: RK4 driver, CFL, detectors, I/O
  test_validation.py # 12 tests: SWSH, Schwarzschild QNM, self-convergence
  test_kerr.py       #  5 tests: Kerr QNM vs tables, spin trend, pole parity
```

### `Grid` (`grid.py`)

Owns the discretization; knows nothing about the physics.

- `__init__(rmin, rmax, Nmu, Nr, ghost=2, M=1.0)` — builds the radial cell array
  `r` from the log map and the staggered `mu` array (both with ghosts); stores
  `dx`, `drdx`, `d2rdx2`, `dr_cell`, `dmu`, and 2D meshes `R`, `MU`.
- `dr(f)`, `drr(f)` — first/second radial derivatives (chain-ruled through the
  $r(x)$ map).
- `angular(f)` — the Legendre operator $\partial_\mu[(1-\mu^2)\partial_\mu f]$ in
  flux form.
- `fill_ghosts_r(f, outer="sommerfeld", dt=None)` — extrapolate inner, radiate
  outer.
- `fill_ghosts_mu(f, parity)` — pole reflection with sign `parity`.
- `ko_dissipation_r(f, epsilon)`, `ko_dissipation_mu(f, epsilon)` — KO 4th
  differences.

### `TeukolskyRHS` (`equation.py`)

Encapsulates the PDE for fixed $(M,a,m)$ — the heart of the solver.

- `__init__(grid, M, a, m, dissipation=0.0)` — precomputes the coefficient
  arrays `Sigma = R**2 + a**2*MU**2`, `Delta`, `A`, `invA = 1/A`,
  `Cv = 4*R + 4i*a*MU + 6M`, the $\psi_r$ coefficient `Cr = 2iam+6r-6M`,
  `B = 4Mr`, and `V = (2*MU - m)**2/(1 - MU**2) - 2`; sets the pole parity
  `self.parity = (-1)**m`. All coefficients are polynomial/rational in $\mu$.
- `rhs(psi, v) -> (dpsi_dt, dv_dt)` — fills ghosts, forms spatial derivatives,
  assembles `dv_dt = invA*(L + B*dr(v) + Cv*v)` and `dpsi_dt = v`, and adds KO
  dissipation if enabled.

### `Evolution` (`evolve.py`)

The user-facing driver.

- `__init__(rhs)` — holds the `TeukolskyRHS`, grid, and state `(psi, v, t)`.
- `set_initial_data(psi0, psi1, dt_init)` — seed `psi = psi1`,
  `v = (psi1 - psi0)/dt_init` (pass the same array twice for $v=0$).
- `add_detector(r_extract)` — register an extraction radius (stores interpolation
  weights).
- `cfl_dt(cfl=0.5)` — the CFL-limited time step.
- `step(dt)` — one RK4 update.
- `evolve(t_final, cfl=0.5, dt=None, record_every=1, snapshot_every=None)` — main
  loop: records detector time series and, optionally, full-grid snapshots.
- Results: `times`, `waveforms` (dict `{r_extract: array (Nt, Nmu)}`), optional
  `snapshots`.
- `save_waveforms(path)` / `save_snapshots(path)` — separate `.npz` files
  (waveforms small/long-lived; snapshots large/checkpoint-style).

### Initial data (`initialdata.py`)

- `swsh(spin, ell, m, mu)` — spin-weighted spherical harmonic
  $_sY_{\ell m}(\mu)$ at $\varphi=0$, implemented analytically for $s=-2$,
  $\ell=2$ via Wigner $d$-matrix elements,
  $_{-2}Y_{2m}(\theta)=\sqrt{5/4\pi}\,d^2_{2,m}(\theta)$. Returns a real array.
- `gaussian_pulse(grid, r0, sigma_r, ell=2, m=2, spin=-2, sigma_mu=None, amplitude=1.0)`
  — $\psi=A\exp(-((r-r_0)/\sigma_r)^2)\,{_{-2}Y_{\ell m}}(\mu)$. Optional
  `sigma_mu` adds $\exp(-(\mu/\sigma_\mu)^2)$ to suppress the field near the poles
  where $V$ is large.

### Diagnostics (`diagnostics.py`)

- `project_swsh(psi_mu, mu, swsh_profile)` — project a detector's $\mu$-profile
  onto a SWSH by midpoint-rule quadrature,
  $\psi_{\ell m}(t)=\int_{-1}^{1}\psi_m(t,\mu)\,\overline{Y(\mu)}\,d\mu$. Handles
  shape `(Nmu,)` or `(Nt, Nmu)`.
- `fit_qnm_frequency(times, psi_t, t_start, t_end)` — extract $(\omega_R,
  \omega_I)$ from a waveform slice. For **real** signals (Schwarzschild) it fits
  a damped cosine via `scipy.optimize.curve_fit`; for **complex** signals
  (Kerr) it does linear regression on $\log|\psi|$ and the unwrapped phase.
  Returns $(\omega_R>0,\ \omega_I<0)$.

## Validation

The solver is tested at every layer (73 tests total; run with `pytest`):

| Layer | File | What it checks |
|---|---|---|
| Grid & operators | `test_grid.py` | FD operators reach 2nd-order convergence against analytic functions |
| RHS | `test_equation.py` | coefficient arrays match `check_equations.py`; `rhs()` linearity |
| Evolution | `test_evolve.py` | RK4 driver, CFL, detectors, snapshot/waveform I/O |
| Schwarzschild | `test_validation.py` | SWSH normalization, $a=0$ QNM, $N_r$ self-convergence |
| Kerr | `test_kerr.py` | $a\neq0$ QNM vs. tables, prograde-spin trend, pole parity |

**Schwarzschild $\ell=m=2$.** Extracted $M\omega = 0.3717 - 0.0904\,i$ vs. the
known $0.3737 - 0.0890\,i$ (<2% error at $N_r=100$, $N_\mu=16$); the $N_r$
self-convergence ratio exceeds 3 (2nd order).

**Kerr $\ell=m=2$.** No new solver code is needed — `TeukolskyRHS` already
carries the full $a$-dependence, so the 2D $(r,\mu)$ evolution captures the
spin-weighted *spheroidal* angular structure automatically (the $a^2\omega^2\mu^2$
coupling enters through $A$ acting on $\partial_t v$). Projecting onto the
spherical $_{-2}Y_{2m}$ still isolates the dominant $\ell=2$ content. Because the
Kerr field is genuinely complex, the complex-path fit is used. At $N_r=120$,
$N_\mu=24$:

| $a/M$ | code $M\omega$ | published $M\omega$ | $\omega_R$ error |
|---|---|---|---|
| 0.5 | $0.4597 - 0.0878\,i$ | $0.4641 - 0.0846\,i$ | ~0.9% |
| 0.9 | $0.6687 - 0.0643\,i$ | $0.6716 - 0.0649\,i$ | ~0.4% |

The pole-parity factor $(-1)^m$ is validated explicitly: forcing the wrong sign
shifts $M\omega_R$ by ~0.05, out of tolerance.

## References

- M. Campanelli, G. Khanna, P. Laguna, J. Pullin & M. P. Ryan,
  *Phys. Rev. D* (2001), [arXiv:gr-qc/0010034v2](https://arxiv.org/abs/gr-qc/0010034v2)
  — the formalism and initial-data prescription this code follows.
- S. A. Teukolsky, *Astrophys. J.* **185**, 635 (1973) — the original
  perturbation equation.
- E. Berti, V. Cardoso & C. M. Will, *Phys. Rev. D* **73**, 064030 (2006),
  [arXiv:gr-qc/0512160](https://arxiv.org/abs/gr-qc/0512160)
  ([tabulated QNM data](https://pages.jh.edu/eberti2/ringdown/)) — reference QNM
  frequencies.
- E. W. Leaver, *Proc. R. Soc. Lond. A* **402**, 285 (1985) — the
  continued-fraction method used to compute those frequencies.

## Notes

- `scripts/check_equations.py` is the committed source of truth for every
  coefficient; keep equation cross-references pointing at it.
- `scripts/run_example.py` output (`*.gif`, `*.npz`, `*_static.png`) is gitignored.
