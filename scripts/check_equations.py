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
