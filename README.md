# PyTeukolsky: Teukolsky equation in Kerr-Schild coordinates

This code uses a finite-differencing method to solve the Teukolsky equation following the approach of [Campanelli et al., (2001)](https://arxiv.org/abs/gr-qc/0010034v2).

## Formalism

The Teukolsky equation is decomposed into angular modes as $\psi=\sum_m \psi_{m} e^{im {\tilde \varphi}}$. Each mode evolves according to

```math
\begin{align*}
(\Sigma + 2Mr){{\partial^2 {\psi_m}}\over 
{\partial t^2}} -&\triangle {{\partial^2 {\psi_m}}\over 
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