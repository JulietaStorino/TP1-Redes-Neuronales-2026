import os
from bisect import bisect_right

import numpy as np
import matplotlib.pyplot as plt

from .ode import rk4, solve_ode
from .izhikevich_model import IzhikevichParams, izhikevich_model, reset


def piecewise_current(breaks, values):
    """
    Builds a piecewise-constant input current.

    Parameters
    ----------
    breaks : list of float
        Increasing times [ms] at which the current changes.
    values : list of float
        Current on each interval, len(values) == len(breaks) + 1. values[k]
        applies for breaks[k-1] <= t < breaks[k].

    Returns
    -------
    I : callable
        Input current I(t).
    """
    def I(t):
        return values[bisect_right(breaks, t)]

    return I


def paper_params(a, b, c, d, I):
    """
    Builds IzhikevichParams from the a, b, c, d notation of Izhikevich (2003).

    Parameters
    ----------
    a, b, c, d : float
        Parameters as named in the paper and in the TP1 table.
    I : callable
        Input current I(t).

    Returns
    -------
    p : IzhikevichParams
        The same model in the notation of guide 6.
    """
    return IzhikevichParams(
        I=I,
        g=np.array([140, 5, 0.04]),
        v_minus=c,
        v_plus=30,
        b=a,
        c=b,
        delta_u=d,
    )


def simulate(p, v0, t_end, h=0.1):
    """
    Integrates the Izhikevich model with RK4 on [0, t_end], starting from
    v(0) = v0 and u(0) = b v(0), and records the spike times.

    Parameters
    ----------
    p : IzhikevichParams
        Model parameters.
    v0 : float
        Initial membrane potential [mV].
    t_end : float
        Final time [ms].
    h : float, optional
        Time step [ms].

    Returns
    -------
    t : ndarray
        Time points [ms].
    v : ndarray
        Membrane potential [mV]. At a spike step it already holds the reset
        value c.
    u : ndarray
        Recovery variable.
    spikes : ndarray
        Times [ms] of the steps at which v >= 30 mV was detected.
    """
    spikes = []

    def reset_and_record(x, t, p):
        if x[0] >= p.v_plus:
            spikes.append(t)
        return reset(x, t, p)

    xa = np.array([v0, p.c * v0], dtype=np.double)
    k = round(t_end / h)

    t, w = solve_ode(rk4, izhikevich_model, xa, 0, t_end, k, p, c=reset_and_record)

    return t, w[0], w[1], np.array(spikes)


def with_peaks(t, v, spikes, v_peak=30):
    """
    Returns a copy of v with the value at each spike step replaced by the
    30 mV peak, so the plotted spikes reach their apex before the reset.
    """
    v = v.copy()
    v[np.isin(t, spikes)] = v_peak
    return v


# ------------------------------------------------------------------
# Exercise 3: the eight panels of Fig. 2 in Izhikevich (2003).
# Each case is (label, a, b, c, d, I, v0, t_end).
# ------------------------------------------------------------------

I1 = piecewise_current([20], [0, 10])
I2 = piecewise_current([20], [0, 1])
I3 = piecewise_current([20, 80], [0, -25, 0])
I4 = piecewise_current([40, 180, 181], [0, 0.2, 2.2, 0.2])

CASES = [
    ("regular spiking (RS)", 0.02, 0.20, -65, 8, I1, -65, 230),
    ("intrinsically bursting (IB)", 0.02, 0.20, -55, 4, I1, -65, 230),
    ("chattering (CH)", 0.02, 0.20, -50, 2, I1, -65, 230),
    ("fast spiking (FS)", 0.10, 0.20, -65, 2, I1, -65, 230),
    ("thalamo-cortical (TC1)", 0.02, 0.25, -65, 0.05, I2, -63, 230),
    ("thalamo-cortical (TC2)", 0.02, 0.25, -65, 0.05, I3, -65, 230),
    ("resonator (RZ)", 0.10, 0.26, -65, 2, I4, -62.5, 300),
    ("low-threshold spiking (LTS)", 0.02, 0.25, -65, 2, I1, -65, 230),
]


# ------------------------------------------------------------------
# Figures, sized for the two-column REVTeX template of the report.
# ------------------------------------------------------------------

COLUMN_WIDTH = 3.4  # in, \columnwidth
TEXT_WIDTH = 7.0  # in, \textwidth

INK = "#0b0b0b"
MUTED = "#52514e"

STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "lines.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}

STEP_SIZES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
H_REF = 0.0005


def comma(x):
    """Formats a number with a decimal comma and a minus sign, as in the report."""
    return f"{x:g}".replace(".", ",").replace("-", "\u2212")


def exercise_1_neuron():
    """Parameters and initial condition of exercises 1 and 2."""
    return paper_params(0.02, 0.2, -65, 2, piecewise_current([10], [0, 10])), -70


def plot_exercise_1(path):
    """
    Exercises 1 and 2: v(t), u(t) and I(t) on [0, 100] ms, marking the
    stimulus onset and the spike times.

    Returns
    -------
    spikes : ndarray
        Spike times [ms].
    """
    p, v0 = exercise_1_neuron()
    t, v, u, spikes = simulate(p, v0, 100)

    fig, axes = plt.subplots(
        3, 1, figsize=(COLUMN_WIDTH, 2.7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.2]}
    )

    for ax in axes:
        ax.axvline(10, color=MUTED, linestyle="--", linewidth=0.6)

    ax = axes[0]

    ax.plot(t, with_peaks(t, v, spikes), color=INK)
    ax.plot(spikes, np.full(len(spikes), 40), "v", color=MUTED, markersize=3)
    ax.set_ylim(-80, 45)
    ax.set_ylabel(r"$v$ (mV)")

    ax = axes[1]

    ax.plot(t, u, color=INK)
    ax.set_yticks([-14, -10, -6], labels=[comma(x) for x in [-14, -10, -6]])
    ax.set_ylabel(r"$u$")

    ax = axes[2]

    ax.plot(t, [p.I(tj) for tj in t], color=INK)
    ax.text(11.5, 5, "estímulo", color=MUTED, fontsize=7, va="center")
    ax.set_yticks([0, 10])
    ax.set_ylabel(r"$I$")
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"$t$ (ms)")

    fig.align_ylabels(axes)
    fig.savefig(path)
    plt.close(fig)

    return spikes


def plot_firing_patterns(path):
    """
    Exercise 3: v(t) and I(t) for the eight cases of Fig. 2 in
    Izhikevich (2003), with an inset zooming into the subthreshold
    oscillations of the resonator.

    Returns
    -------
    spikes : dict
        Spike times [ms] for each case label.
    """
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.15))
    outer = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.32)

    spikes = {}

    for n, (label, a, b, c, d, I, v0, t_end) in enumerate(CASES):
        t, v, u, spikes[label] = simulate(paper_params(a, b, c, d, I), v0, t_end)
        current = [I(tj) for tj in t]

        inner = outer[n // 4, n % 4].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.12)
        ax_v = fig.add_subplot(inner[0])
        ax_I = fig.add_subplot(inner[1], sharex=ax_v)

        ax = ax_v

        ax.plot(t, with_peaks(t, v, spikes[label]), color=INK)
        ax.set_title(label, loc="left")
        ax.set_ylim(-95, 40)
        ax.tick_params(labelbottom=False)

        if n % 4 == 0:
            ax.set_ylabel(r"$v$ (mV)")
        else:
            ax.tick_params(labelleft=False)

        if "RZ" in label:
            m = (t >= 40) & (t < 180)

            inset = ax.inset_axes([0.04, 0.5, 0.5, 0.42])
            inset.plot(t[m], v[m], color=INK, linewidth=0.6)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.spines[["top", "right"]].set_visible(True)

        ax = ax_I

        ax.plot(t, current, color=INK)
        levels = sorted({min(current), max(current)})
        ax.set_yticks(levels, labels=[comma(x) for x in levels])
        ax.set_xlim(0, t_end)

        if n % 4 == 0:
            ax.set_ylabel(r"$I$")

        if n // 4 == 1:
            ax.set_xlabel(r"$t$ (ms)")

    fig.savefig(path)
    plt.close(fig)

    return spikes


def plot_step_size(path):
    """
    Effect of the time step h on the neuron of exercise 1: spike rasters
    for several h, and the largest spike-time error as a function of h,
    measured against a run with h = H_REF.

    Returns
    -------
    errors : dict
        Largest spike-time error [ms] for each h in STEP_SIZES.
    """
    p, v0 = exercise_1_neuron()
    ref = simulate(p, v0, 100, h=H_REF)[3]

    fig, axes = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH, 2.8),
        gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.55}
    )

    ax = axes[0]

    rows = [H_REF, 0.1, 0.5, 1.0]

    for s in ref:
        ax.axvline(s, color=MUTED, linestyle=":", linewidth=0.5)

    for i, h in enumerate(rows):
        spikes = simulate(p, v0, 100, h=h)[3]
        ax.plot(spikes, np.full(len(spikes), i), "|", color=INK, markersize=7, markeredgewidth=1)

    ax.set_yticks(range(len(rows)), labels=[comma(h) for h in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_ylabel(r"$h$ (ms)")
    ax.set_xlabel(r"$t$ (ms)")

    ax = axes[1]

    errors = {}

    for h in STEP_SIZES:
        spikes = simulate(p, v0, 100, h=h)[3]
        assert len(spikes) == len(ref)
        errors[h] = np.abs(spikes - ref).max()

    hs = np.array(STEP_SIZES)
    e0 = errors[hs[0]]

    ax.loglog(hs, e0 * hs / hs[0], linestyle="--", color=MUTED, linewidth=0.6)
    ax.loglog(hs, e0 * (hs / hs[0]) ** 4, linestyle=":", color=MUTED, linewidth=0.8)
    ax.loglog(hs, list(errors.values()), "o-", color=INK, markersize=3)

    ax.text(hs[-1] * 1.08, e0 * hs[-1] / hs[0], r"$\propto h$", color=MUTED, fontsize=7, va="center")
    ax.text(hs[0] * (10 / e0) ** 0.25 * 0.92, 10, r"$\propto h^4$", color=MUTED, fontsize=7, ha="right", va="center")

    ax.minorticks_off()
    ax.set_xticks(hs, labels=[comma(h) for h in hs])
    ax.set_yticks([0.1, 1, 10], labels=[comma(e) for e in [0.1, 1, 10]])
    ax.set_ylim(e0 / 3, max(errors.values()) * 3)
    ax.set_xlabel(r"$h$ (ms)")
    ax.set_ylabel(r"máx. $|\Delta t_\mathrm{disparo}|$ (ms)")

    fig.savefig(path)
    plt.close(fig)

    return errors


def main(output_dir="plots", fmt="pdf"):
    os.makedirs(output_dir, exist_ok=True)

    with plt.rc_context(STYLE):
        spikes = plot_exercise_1(f"{output_dir}/part1_exercise_1.{fmt}")
        print("Exercise 1 spike times [ms]:", np.round(spikes, 1))

        for label, s in plot_firing_patterns(f"{output_dir}/part1_firing_patterns.{fmt}").items():
            print(f"{label:30s} {len(s):3d} spikes, ISIs [ms]: {np.round(np.diff(s), 1)}")

        for h, e in plot_step_size(f"{output_dir}/part1_step_size.{fmt}").items():
            print(f"h = {h:<5} ms: largest spike-time error {e:.3f} ms")


if __name__ == "__main__":
    main()
