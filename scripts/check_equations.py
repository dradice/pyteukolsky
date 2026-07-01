"""
Verify the mode equation ('MODE') the code evolves, following Campanelli et al.
(2001), gr-qc/0010034.

Two checks:
  CHECK 1 (the direct question):
    The mode equation must be the angular reduction psi = psi_m * exp(i m phi)
    of the 3D 'PTC' equation that the code evolves.

  CHECK 2 (sanity of the parent):
    The PTC equation should follow from the 'big psi_4' Teukolsky equation
    via psi = (r - i a cos theta)^4 psi_4, up to an overall
    nonzero factor (which the second-derivative coefficients fix to 2*Sigma*zeta^4).

  CHECK 3 (numerics: angular variable mu = cos theta):
    Confirms the substitution used by the solver. The angular sector of the
    mode equation, written with mu = cos theta, becomes the Legendre-type
    operator  -d/dmu[(1-mu^2) d/dmu]  with potential  (2 mu - m)^2/(1-mu^2) - 2.

  CHECK 4 (waveform extraction: psi_4 from psi_m, and the tetrad rescaling):
    The evolved field is psi = zeta^4 psi_4 with zeta = r - i a cos theta, so
    the horizon-penetrating (Kerr-Schild) Weyl scalar is
        psi_4^KS  = psi_m e^{i m phi} / zeta^4.
    Its outgoing radial solution falls as 1/r (verified here by the large-r
    indicial exponent of the mode radial operator), so psi_4^KS ~ 1/r^5 and does
    NOT peel. The Kinnersley / radiation-frame scalar is recovered by undoing the
    l -> Delta l horizon-regularising boost (psi_4 ~ n^2, n -> n/Delta):
        psi_4^Kin = Delta^2 psi_4^KS,
    whose radial part zeta^4 psi_4^Kin = Delta^2 psi_m ~ r^3, matching the
    standard s=-2 outgoing Teukolsky peeling r^{-1-2s_spin} = r^3.
    (Implemented as pyteukolsky.diagnostics.psi4_kinnersley.)
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta varphi', real=True)
M, a, m = sp.symbols('M a m', real=True)
I = sp.I
c, s = sp.cos(th), sp.sin(th)

Sigma = r**2 + a**2*c**2
Delta = r**2 - 2*M*r + a**2
zeta  = r - I*a*c          # psi = zeta^4 * psi_4

# ----------------------------------------------------------------------
# helper: extract the coefficient of every derivative of a function f
# ----------------------------------------------------------------------
def op_coeffs(expr, f, varlist):
    """Return dict {derivative_descriptor : coefficient} for a linear
    differential operator 'expr' acting on Function f(*varlist)."""
    expr = sp.expand(expr)
    derivs = expr.atoms(sp.Derivative) | {f}
    out = {}
    for d in derivs:
        co = expr.coeff(d)
        if co != 0:
            out[d] = sp.simplify(co)
    return out

def compare(opA, opB, f, name):
    diff = sp.expand(opA - opB)
    diff = sp.simplify(diff)
    print(f"\n=== {name} ===")
    if diff == 0:
        print("  IDENTICAL  ->  difference simplifies to 0")
        return
    # not zero: report residual term by term
    print("  NOT identical. Residual collected by derivative:")
    da = op_coeffs(diff, f, [t, r, th, ph])
    for d, co in da.items():
        cco = sp.simplify(co)
        if cco != 0:
            print(f"    coeff of {d}:  {cco}")

# ======================================================================
# CHECK 1 : mode eq  vs  angular reduction of the PTC eq
# ======================================================================
print("#"*70)
print("CHECK 1 : mode equation  vs  reduction of the 3D PTC equation")
print("#"*70)

psi   = sp.Function('psi')(t, r, th, ph)
psi_m = sp.Function('psi_m')(t, r, th)

def PTC(P):
    # 3D equation for psi = zeta^4 psi_4 that the code evolves
    return ( (Sigma + 2*M*r)*sp.diff(P, t, 2)
             - Delta*sp.diff(P, r, 2)
             - 6*(r - M)*sp.diff(P, r)
             - (1/s)*sp.diff(s*sp.diff(P, th), th)
             - (1/s**2)*sp.diff(P, ph, 2)
             - 4*M*r*sp.diff(P, t, r)
             - 2*a*sp.diff(P, r, ph)
             + (4*I*c/s**2)*sp.diff(P, ph)          # 4 i cot/ sin = 4 i cos / sin^2
             - (4*r + 4*I*a*c + 6*M)*sp.diff(P, t)
             + 2*(3*sp.cot(th)**2 - 1/s**2)*P )

def MODE(F):
    # per-m angular reduction (the paper's claimed result)
    return ( (Sigma + 2*M*r)*sp.diff(F, t, 2)
             - Delta*sp.diff(F, r, 2)
             - (2*a*I*m + 6*r - 6*M)*sp.diff(F, r)
             - (1/s)*sp.diff(s*sp.diff(F, th), th)
             - 4*M*r*sp.diff(F, t, r)
             - (4*r + 4*I*a*c + 6*M)*sp.diff(F, t)
             + (4*sp.cot(th)**2 - 2 + m**2/s**2 - 4*m*sp.cot(th)/s)*F )

# substitute the single mode psi = psi_m * exp(i m phi) into PTC and divide it out
reduced = sp.expand( PTC(psi_m*sp.exp(I*m*ph)) / sp.exp(I*m*ph) )
claimed = MODE(psi_m)
compare(reduced, claimed, psi_m, "PTC reduced to mode m   vs   paper's mode equation")

# ======================================================================
# CHECK 2 : PTC eq  vs  (big psi_4 eq) under psi = zeta^4 psi_4
# ======================================================================
print()
print("#"*70)
print("CHECK 2 : PTC equation  vs  big psi_4 equation  (psi = zeta^4 psi_4)")
print("#"*70)

def BIG(P4):
    # full psi_4 Teukolsky equation, acting on psi_4
    secn = ( sp.Rational(1,2)*(Sigma + 2*M*r)/Sigma*sp.diff(P4, t, 2)
             - sp.Rational(1,2)*Delta/Sigma*sp.diff(P4, r, 2)
             - 2*M*r/Sigma*sp.diff(P4, r, t)
             - a/Sigma*sp.diff(P4, r, ph)
             - sp.Rational(1,2)*(1/Sigma)*sp.diff(P4, th, 2)
             - sp.Rational(1,2)*(1/(s**2*Sigma))*sp.diff(P4, ph, 2) )
    Cr = -( 3*a**2*(r-M)*c**2 + r*(7*r**2 + 4*a**2 - 11*M*r) + 4*I*a*c*Delta )/Sigma**2
    Ct = -( r**2*(2*r+11*M) + a**2*(2*r-3*M)*c**2
            - 2*I*a*(r**2 + 7*M*r + a**2*c**2)*c )/(Sigma*(r - I*a*c)**2)
    Cth= -( (r**2 + 8*a**2 - 9*a**2*c**2)*c + 2*I*a*r*(4 - 5*c**2) )/(2*Sigma*(r - I*a*c)**2*s)
    Cph= -( 4*a*r*(s**2 - c**2) - 2*I*(r**2 + 2*a**2 - 3*a**2*c**2)*c )/(Sigma*(r - I*a*c)**2*s**2)
    C0 = ( 24*M*r*s**2 - 19*r**2 + (21*r**2 - 9*a**2 + 7*a**2*c**2)*c**2
           - 2*I*a*((7*r - 6*M)*c**2 - (5*r - 6*M))*c )/(Sigma*(r - I*a*c)**2*s**2)
    return ( secn + Cr*sp.diff(P4, r) + Ct*sp.diff(P4, t)
             + Cth*sp.diff(P4, th) + Cph*sp.diff(P4, ph) + C0*P4 )

# big eq written for psi_4 = zeta^-4 psi, multiplied by 2*Sigma*zeta^4
big_in_psi = sp.expand( BIG(zeta**(-4)*psi) * (2*Sigma*zeta**4) )
ptc        = PTC(psi)
compare(big_in_psi, ptc, psi, "2*Sigma*zeta^4 * BIG[zeta^-4 psi]   vs   PTC[psi]")

# ======================================================================
# CHECK 3 : angular sector under the substitution mu = cos(theta)
# ======================================================================
print()
print("#"*70)
print("CHECK 3 : angular sector rewritten with mu = cos(theta)")
print("#"*70)

mu = sp.Symbol('mu', real=True)

def to_mu(expr):
    """Replace cos(theta) -> mu and sin(theta) -> sqrt(1-mu^2)."""
    return expr.subs({c: mu, s: sp.sqrt(1 - mu**2)})

# --- 3a: the potential V collapses to (2 mu - m)^2/(1-mu^2) - 2 ---
# V written purely with c = cos(theta), s = sin(theta) (cot = c/s).
V_theta = 4*(c/s)**2 - 2 + m**2/s**2 - 4*m*(c/s)/s
V_mu    = (2*mu - m)**2/(1 - mu**2) - 2
res_V   = sp.simplify(to_mu(V_theta) - V_mu)
print("\n--- 3a: potential V(theta) -> (2 mu - m)^2/(1-mu^2) - 2 ---")
print("  residual =", res_V, " ->", "OK" if res_V == 0 else "MISMATCH")

# --- 3b: full angular contribution to the mode equation ---
# angular part of MODE acting on F:        -(1/s) d/dth( s dF/dth ) + V F
# claim (with F = g(mu), mu = cos theta):  -d/dmu[(1-mu^2) dF/dmu] + V_mu F
a_   = sp.symbols('a0:6')                      # generic quintic test function
g_mu = sum(a_[k]*mu**k for k in range(6))
g_th = g_mu.subs(mu, c)                        # the same F as a function of theta

ang_theta = -(1/s)*sp.diff(s*sp.diff(g_th, th), th) + V_theta*g_th
ang_mu    = -sp.diff((1 - mu**2)*sp.diff(g_mu, mu), mu) + V_mu*g_mu

res_ang = sp.simplify(ang_theta - ang_mu.subs(mu, c))
print("\n--- 3b: full angular operator (generic quintic test function) ---")
print("  [theta-form] - [mu-form] =", res_ang,
      " ->", "OK" if res_ang == 0 else "MISMATCH")

# ======================================================================
# CHECK 4 : psi_4 extraction and the Kerr-Schild -> Kinnersley rescaling
# ======================================================================
print()
print("#"*70)
print("CHECK 4 : psi_4 = psi_m e^{i m phi}/zeta^4  and  psi_4^Kin = Delta^2 psi_4^KS")
print("#"*70)

# Large-r indicial exponent of the mode RADIAL operator (Schwarzschild, a=0):
# with psi_m = e^{-i omega t} R(r), the radial ODE is
#     Delta R'' + (Cr - i omega B) R' + (A omega^2 - i omega Cv + lam) R = 0,
# and the OUTGOING solution behaves as R ~ e^{i omega r} r^s at large r.
Mr, wr = sp.symbols('M omega', real=True, positive=True)
rr, ss, lam = sp.symbols('r s lambda')
Delta0 = rr**2 - 2*Mr*rr        # a = 0
A0     = rr**2 + 2*Mr*rr
Cr0    = 6*rr - 6*Mr
B0     = 4*Mr*rr
Cv0    = 4*rr + 6*Mr

def outgoing_exponent(Delta, A, Cr, B, Cv):
    """Solve the leading large-r balance for s in R = e^{i omega r} r^s."""
    P = Cr - sp.I*wr*B
    Q = A*wr**2 - sp.I*wr*Cv + lam
    # For R = e^{i w r} r^s:  R'/R = i w + s/r,  R''/R = (i w + s/r)^2 - s/r^2
    L_over_R = sp.expand(Delta*((sp.I*wr + ss/rr)**2 - ss/rr**2)
                         + P*(sp.I*wr + ss/rr) + Q)
    pmax = max(int(t.as_independent(rr)[1].as_base_exp()[1]) if t.has(rr) else 0
               for t in L_over_R.as_ordered_terms())
    lead = sp.simplify(L_over_R.coeff(rr, pmax))
    return sp.simplify(sp.solve(sp.Eq(lead, 0), ss)[0])

s_ks = outgoing_exponent(Delta0, A0, Cr0, B0, Cv0)
re_ks = sp.re(s_ks)
print("\n--- 4a: outgoing radial exponent of psi_m = zeta^4 psi_4^KS ---")
print(f"  s = {s_ks}   ->   |psi_m| ~ r^(Re s) = r^{re_ks}")
print("  ->", "OK (falls as 1/r; hence psi_4^KS = psi_m/zeta^4 ~ 1/r^5)"
      if re_ks == -1 else "UNEXPECTED")

# Kinnersley radial part is Delta^2 * (zeta^4 psi_4^KS) = Delta^2 * R.
# Delta^2 ~ r^4 at large r, so the magnitude exponent shifts by +4.
re_kin      = re_ks + 4
s_spin      = -2
peel_expect = -1 - 2*s_spin       # standard s=-2 outgoing Teukolsky: r^{-1-2s}
print("\n--- 4b: Kinnersley psi_4^Kin = Delta^2 psi_4^KS restores peeling ---")
print(f"  zeta^4 psi_4^Kin = Delta^2 psi_m ~ r^{re_kin}   "
      f"(standard s=-2 outgoing r^{{-1-2s}} = r^{peel_expect})")
print("  ->", "OK -> psi_4^Kin ~ 1/r (peels)" if re_kin == peel_expect
      else "MISMATCH")
