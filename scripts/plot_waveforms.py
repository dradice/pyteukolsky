"""
plot_waveforms.py — plot the detector waveforms saved by run_example.py.

Reads a ``*_waveforms.npz`` file (see Evolution.save_waveforms) and produces a
multi-panel figure for one extraction radius:

  * the asymptotic waveform r·ψ₄ (real and imaginary part), OR r·h with the
    strain h computed from ψ₄ by fixed-frequency integration (FFI, Reisswig &
    Pollney 2011);
  * the amplitude |·| (optionally on a log scale);
  * the instantaneous frequency M·ω = -d(arg)/dt.

The detector waveform is stored on the full (t, μ) grid; it is reduced to a
single complex time series by projecting onto the spin-weighted spherical
harmonic ₋₂Y_{ℓm}(μ) via diagnostics.project_swsh.

Usage:
    ~/local/miniforge/bin/python scripts/plot_waveforms.py --list
    ~/local/miniforge/bin/python scripts/plot_waveforms.py --radius 40
    ~/local/miniforge/bin/python scripts/plot_waveforms.py -r 40 --quantity strain --log
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from pyteukolsky.diagnostics import project_swsh
from pyteukolsky.initialdata import swsh


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_waveforms(path):
    """Load a waveforms .npz and return (radii, data-dict).

    Returns
    -------
    radii : list[float]      sorted extraction radii
    npz   : dict-like        the loaded archive (keeps 'times', 'mu_grid', …)

    The per-detector arrays live under keys ``waveform_<r>`` of shape
    (Nt, Nmu); ``radii`` maps each parsed radius back to its key via
    ``waveform_key``.
    """
    npz = np.load(path)
    radii = sorted(float(k[len("waveform_"):])
                   for k in npz.files if k.startswith("waveform_"))
    return radii, npz


def waveform_key(radii_keys, r):
    """Return the .npz key for the stored radius nearest to ``r``."""
    nearest = min(radii_keys, key=lambda rr: abs(rr - r))
    return nearest, f"waveform_{nearest:.6f}"


# ---------------------------------------------------------------------------
# Reduction: (Nt, Nmu) → (Nt,) complex via SWSH projection
# ---------------------------------------------------------------------------

def project_mode(psi_tmu, mu, spin, ell, m):
    """Project a (Nt, Nmu) waveform onto ₋₂Y_{ℓm}(μ), returning (Nt,) complex.

    Falls back to the equatorial slice (μ closest to 0) if the analytic SWSH
    is not available for the requested (spin, ℓ).
    """
    try:
        profile = swsh(spin, ell, m, mu)
    except (NotImplementedError, ValueError) as exc:
        i_eq = int(np.argmin(np.abs(mu)))
        print(f"SWSH projection unavailable ({exc}); using equatorial slice "
              f"μ={mu[i_eq]:+.3f}.")
        return psi_tmu[:, i_eq]
    return project_swsh(psi_tmu, mu, profile)


# ---------------------------------------------------------------------------
# Fixed-frequency integration: ψ₄ → strain h
# ---------------------------------------------------------------------------

def fixed_frequency_integration(t, psi4, omega0, taper=0.05):
    """Recover strain h from ψ₄ = ḧ by fixed-frequency integration.

    In Fourier space ψ̂₄(ω) = -ω² ĥ(ω), so ĥ = -ψ̂₄/ω².  FFI regularises the
    ω→0 divergence by clamping |ω| to a floor ``omega0`` (Reisswig & Pollney,
    Class. Quantum Grav. 28 (2011) 195015).

    Parameters
    ----------
    t      : (Nt,) uniform time samples.
    psi4   : (Nt,) complex ψ₄ time series.
    omega0 : float, high-pass cutoff frequency (1/M).
    taper  : float, Tukey taper fraction applied at each end before the FFT
             to suppress spectral leakage (0 disables).

    Returns
    -------
    (Nt,) complex strain h(t).
    """
    n  = len(t)
    dt = t[1] - t[0]

    y = psi4 * _tukey(n, taper) if taper > 0 else psi4

    omega     = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    omega_eff = np.where(np.abs(omega) < omega0,
                         np.sign(omega) * omega0, omega)
    omega_eff[omega_eff == 0.0] = omega0          # sign(0)=0 → use the floor

    h_hat = -np.fft.fft(y) / omega_eff**2
    return np.fft.ifft(h_hat)


def _tukey(n, alpha):
    """Tukey (tapered-cosine) window of length ``n`` with taper fraction alpha."""
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    k = np.arange(edge + 1)
    ramp = 0.5 * (1.0 + np.cos(np.pi * (2.0 * k / (alpha * (n - 1)) - 1.0)))
    w[:edge + 1]  = ramp
    w[n - edge - 1:] = ramp[::-1]
    return w


# ---------------------------------------------------------------------------
# Analytic signal, envelope and instantaneous frequency
# ---------------------------------------------------------------------------

def _is_real(z):
    """True if z is effectively real (as in diagnostics.fit_qnm_frequency)."""
    peak = np.max(np.abs(z))
    return peak == 0.0 or np.max(np.abs(z.imag)) < 1e-10 * peak


def analytic_signal(z):
    """Return a complex analytic signal for envelope/frequency diagnostics.

    A genuinely complex field (e.g. Kerr) already carries phase information and
    is returned unchanged.  A real field (Schwarzschild with real initial data)
    has an ill-defined phase, so its analytic signal is built from the Hilbert
    transform of the real part.
    """
    return hilbert(z.real) if _is_real(z) else np.asarray(z)


def instantaneous_frequency(t, z):
    """M·ω_GW(t) = |d/dt arg(z_a)|, from the analytic signal z_a of z.

    The magnitude is reported so the result is a positive GW angular frequency
    directly comparable to a QNM ω_R, regardless of the exp(±iωt) convention.
    """
    za = analytic_signal(z)
    return np.abs(np.gradient(np.unwrap(np.angle(za)), t))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(t, z, label, log_amp, path, meta):
    """Three-panel figure: (Re, Im); amplitude; instantaneous frequency."""
    amp   = np.abs(analytic_signal(z))     # envelope (robust for real signals)
    omega = instantaneous_frequency(t, z)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    title = (rf"$M={meta['M']:g}$, $a={meta['a']:g}$, $m={meta['m']:d}$, "
             rf"$r_{{\rm ext}}={meta['r']:.2f}\,M$  (mode $\ell={meta['ell']}$)")
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    ax.plot(t, z.real, "steelblue", lw=1.3, label=rf"$\mathrm{{Re}}\,{label}$")
    ax.plot(t, z.imag, "indianred", lw=1.0, ls="--",
            label=rf"$\mathrm{{Im}}\,{label}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(rf"${label}$", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(t, amp, "k", lw=1.3)
    ax.set_ylabel(rf"$|{label}|$ (envelope)", fontsize=12)
    if log_amp:
        ax.set_yscale("log")
        floor = amp[amp > 0].min() if np.any(amp > 0) else 1e-30
        ax.set_ylim(max(floor, amp.max() * 1e-6), amp.max() * 2)

    ax = axes[2]
    ax.plot(t, omega, "seagreen", lw=1.3)
    ax.set_ylabel(r"$M\,\omega_{\rm GW} = |\mathrm{d}\,\arg/\mathrm{d}t|$",
                  fontsize=12)
    ax.set_xlabel(r"$t / M$", fontsize=12)
    ax.axhline(0, color="k", lw=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print(f"Figure → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", nargs="?", default="pulse_animation_waveforms.npz",
                   help="waveforms .npz written by run_example.py "
                        "(default pulse_animation_waveforms.npz)")
    p.add_argument("--list", action="store_true",
                   help="list the available extraction radii and exit")
    p.add_argument("-r", "--radius", type=float, default=None,
                   help="extraction radius / M to plot (nearest available is "
                        "used; default: the first radius)")
    p.add_argument("--quantity", choices=["psi4", "strain"], default="psi4",
                   help="plot ψ₄ directly, or the strain h from ψ₄ via "
                        "fixed-frequency integration (default psi4)")
    p.add_argument("--ell", type=int, default=2,
                   help="spherical-harmonic ℓ to project onto (default 2)")
    p.add_argument("--log", action="store_true",
                   help="plot the amplitude panel on a log scale")
    p.add_argument("--omega0", type=float, default=None,
                   help="FFI high-pass cutoff M·ω0 (default: half the median "
                        "instantaneous frequency of ψ₄)")
    p.add_argument("--taper", type=float, default=0.05,
                   help="Tukey taper fraction for FFI (default 0.05)")
    p.add_argument("-o", "--out", default=None,
                   help="output PNG path (default derived from npz + radius)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.npz):
        sys.exit(f"No such file: {args.npz}")

    radii, npz = load_waveforms(args.npz)
    if not radii:
        sys.exit(f"{args.npz} contains no 'waveform_*' detector arrays.")

    if args.list:
        print(f"{args.npz}: {len(radii)} extraction radius/radii")
        for rr in radii:
            print(f"  r = {rr:8.3f} M")
        return

    r_req = radii[0] if args.radius is None else args.radius
    r_ext, key = waveform_key(radii, r_req)
    if args.radius is not None and abs(r_ext - args.radius) > 1e-6:
        print(f"Requested r={args.radius:.3f} M → nearest available "
              f"r={r_ext:.3f} M.")

    t   = np.asarray(npz["times"], dtype=float)
    mu  = np.asarray(npz["mu_grid"], dtype=float)
    m   = int(npz["m"])
    M   = float(npz["M"])
    a   = float(npz["a"])
    psi_tmu = np.asarray(npz[key])                    # (Nt, Nmu) complex

    if psi_tmu.shape[0] != t.size:
        sys.exit(f"Waveform/time length mismatch: {psi_tmu.shape[0]} vs {t.size}.")

    psi4 = project_mode(psi_tmu, mu, spin=-2, ell=args.ell, m=m)   # (Nt,)

    if args.quantity == "psi4":
        z, label = psi4, r"r\,\psi_4"
    else:
        omega0 = args.omega0
        if omega0 is None:
            inst = instantaneous_frequency(t, psi4)
            amp  = np.abs(analytic_signal(psi4))
            sig  = amp > 0.1 * amp.max()          # where the signal is present
            omega0 = 0.5 * float(np.median(inst[sig])) if np.any(sig) else 0.0
            print(f"FFI cutoff (auto): M·ω0 = {omega0:.4f}")
        if not omega0 > 0.0:
            sys.exit("FFI cutoff M·ω0 must be positive; pass --omega0 "
                     "(e.g. half the dominant M·ω).")
        z = fixed_frequency_integration(t, psi4, omega0, taper=args.taper)
        label = r"r\,h"

    # Plot the asymptotic waveform r·(quantity): the field falls off as 1/r,
    # so scaling by the extraction radius gives an r-independent amplitude.
    z = r_ext * z

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.npz))[0]
        args.out = f"{stem}_{args.quantity}_r{r_ext:.0f}.png"

    meta = dict(M=M, a=a, m=m, r=r_ext, ell=args.ell)
    make_figure(t, z, label, args.log, args.out, meta)


if __name__ == "__main__":
    main()
