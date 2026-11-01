"""Generate every Matplotlib figure of the book into fig/ as PDF.

One function per figure, named chNN_name(), which calls
save(fig, "chNN-name"). Functions named chNN_* run automatically;
`python3 gen_fig.py ch01` runs only the matching ones.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "fig"
sys.path.insert(0, str(Path(__file__).parent / "code"))

# Colours follow preamble.tex.
BLUE = "#1B3B6F"
GREEN = "#1E6F5C"
ORANGE = "#B85C00"
RED = "#9B1B30"
GREY = "#5A6472"
GREY_RULE = "#C9CED6"



plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.edgecolor": GREY_RULE,
    "axes.labelcolor": GREY,
    "axes.titlesize": 9,
    "axes.titlecolor": BLUE,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "text.color": GREY,
    "grid.color": GREY_RULE,
    "legend.frameon": False,
    "figure.dpi": 300,  # size text 8-10 pt (\\scriptsize to \\small): no scaling in \\figmpl
})




def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)



def ch00_projection():
    """A vector, its projection on a direction, and the residual."""
    x = np.array([4.0, 3.0])
    d = np.array([1.0, 2.0])
    u = d / np.linalg.norm(d)
    p = (x @ u) * u
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    t = np.array([-0.5, 3.2])
    ax.plot(t * d[0], t * d[1], color=GREY_RULE, lw=0.8)
    arrow = dict(arrowstyle="-|>", lw=1.2, shrinkA=0, shrinkB=0)
    ax.annotate("", x, (0, 0), arrowprops={**arrow, "color": BLUE})
    ax.annotate("", p, (0, 0), arrowprops={**arrow, "color": GREEN})
    ax.annotate("", d, (0, 0), arrowprops={**arrow, "color": ORANGE})
    ax.plot([x[0], p[0]], [x[1], p[1]], color=RED, lw=0.9, ls="--")
    n = (x - p) / np.linalg.norm(x - p)
    c = np.array([p - 0.25 * u, p - 0.25 * u + 0.25 * n, p + 0.25 * n])
    ax.plot(*c.T, color=GREY, lw=0.6)
    ax.text(4.1, 3.05, r"$\boldsymbol{x}$", color=BLUE)
    ax.text(2.2, 4.1, r"$(\boldsymbol{u}^{\top}\boldsymbol{x})\,\boldsymbol{u}$",
            color=GREEN)
    ax.text(0.15, 1.75, r"$\boldsymbol{d}$", color=ORANGE)
    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(-0.6, 5.2)
    ax.set_aspect("equal")
    ax.grid(lw=0.3)
    save(fig, "ch00-projection")


def ch00_convex():
    """A chord stays above a convex curve; a non-convex curve has two valleys."""
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.9), sharey=True)
    w = np.linspace(-2, 2, 200)
    f = lambda w: 0.5 * w ** 2 + 0.2
    g = lambda w: 0.5 * (w ** 2 - 2) ** 2 / 1.6 + 0.15 * w + 0.3
    for ax, h, title in ((axes[0], f, "convex"), (axes[1], g, "not convex")):
        ax.plot(w, h(w), color=BLUE, lw=1.4)
        ax.set_title(title)
        ax.set_xlabel(r"$w$")
        ax.grid(lw=0.3)
        ax.set_ylim(0, 2.6)
    a, b = -1.6, 1.2
    axes[0].plot([a, b], [f(a), f(b)], color=ORANGE, lw=1, marker="o", ms=3)
    a, b = -1.4, 1.4
    axes[1].plot([a, b], [g(a), g(b)], color=ORANGE, lw=1, marker="o", ms=3)
    save(fig, "ch00-convex")


def ch00_lagrange():
    """Circular contours touch the constraint line at w = (1, 2)."""
    fig, ax = plt.subplots(figsize=(2.8, 2.5))
    t = np.linspace(0, 2 * np.pi, 200)
    for r2 in (1.25, 2.5, 5.0, 8.0):
        r = np.sqrt(r2)
        col = GREEN if r2 == 5.0 else GREY_RULE
        ax.plot(r * np.cos(t), r * np.sin(t), color=col, lw=1)
    x = np.linspace(-1, 5.5, 2)
    ax.plot(x, (5 - x) / 2, color=BLUE, lw=1.3)
    ax.annotate("", (1.5, 3.0), (1, 2), arrowprops=dict(
        arrowstyle="-|>", color=ORANGE, lw=1.1, shrinkA=0, shrinkB=0))
    ax.annotate("", (0.5, 1.0), (1, 2), arrowprops=dict(
        arrowstyle="-|>", color=RED, lw=1.1, shrinkA=0, shrinkB=0))
    ax.plot([1], [2], "o", color=RED, ms=3.5)
    ax.text(2.6, 1.7, r"$w_1 + 2w_2 = 5$", color=BLUE)
    ax.text(1.65, 2.95, r"$\nabla h$", color=ORANGE)
    ax.text(-0.25, 0.55, r"$-\nabla f$", color=RED)
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.grid(lw=0.3)
    save(fig, "ch00-lagrange")


def ch00_machines():
    """The eight machines and the line x1 + 2 x2 = 11."""
    from machines import X, y
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for s, col, mark, lab in ((-1, BLUE, "o", "healthy"),
                              (1, ORANGE, "s", "needs service")):
        ax.scatter(*X[y == s].T, color=col, marker=mark, s=16, label=lab,
                   zorder=3)
    for i, p in enumerate(X, 1):
        dy = -0.5 if i == 4 else 0.12
        ax.text(p[0] + 0.12, p[1] + dy, str(i), fontsize=8)
    x = np.array([0, 8.0])
    ax.plot(x, (11 - x) / 2, color=GREEN, lw=1.1)
    ax.text(0.2, 5.75, r"$x_1 + 2x_2 = 11$", color=GREEN)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"vibration $x_1$ (mm/s)")
    ax.set_ylabel(r"temperature rise $x_2$")
    ax.grid(lw=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "ch00-machines")


def _plot_machines(ax, label=True, s=14):
    """Scatter the eight machines on ax, optionally with row numbers."""
    from machines import X, y
    for c, col, mark in ((-1, BLUE, "o"), (1, ORANGE, "s")):
        ax.scatter(*X[y == c].T, color=col, marker=mark, s=s, zorder=3)
    if label:
        for i, p in enumerate(X, 1):
            dy = -0.55 if i in (4, 7) else 0.12
            ax.text(p[0] + 0.12, p[1] + dy, str(i), fontsize=8, zorder=4)


def ch01_lines():
    """Three separating lines, each with its own street."""
    from machines import X, y
    fig, axes = plt.subplots(1, 3, figsize=(4.75, 2.05), sharex=True,
                             sharey=True)
    cases = (("line A", (1, 1), -7.5), ("line B", (1, 3), -14.0),
             ("line C", (2, 1), -11.5))
    t = np.array([-9.0, 16.0])
    for ax, (name, w, b) in zip(axes, cases):
        w = np.array(w, dtype=float)
        s = X @ w + b
        gap = s[y > 0].min()
        n = w / (w @ w)
        v = np.array([-w[1], w[0]])
        lo, hi = -gap * n, gap * n
        x0 = -b * n
        band = np.array([x0 + lo + t[0] * v, x0 + lo + t[1] * v,
                         x0 + hi + t[1] * v, x0 + hi + t[0] * v])
        ax.fill(*band.T, color=GREEN, alpha=0.15, lw=0)
        ax.plot(*np.array([x0 + t[0] * v, x0 + t[1] * v]).T, color=GREEN,
                lw=1)
        _plot_machines(ax, label=False, s=9)
        width = 2 * gap / np.linalg.norm(w)
        ax.set_title(f"{name}, width {width:.3f}")
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 8)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$x_2$")
    save(fig, "ch01-lines")


def ch01_sweep():
    """Street width against the direction of the normal vector."""
    from machines import X, y
    deg = np.linspace(0, 90, 1801)
    th = np.radians(deg)
    u = np.column_stack([np.cos(th), np.sin(th)])
    s = u @ X.T
    pos, neg = np.where(y > 0)[0], np.where(y < 0)[0]
    ip, im = pos[s[:, pos].argmin(1)], neg[s[:, neg].argmax(1)]
    width = s[np.arange(len(th)), ip] - s[np.arange(len(th)), im]
    fig, ax = plt.subplots(figsize=(3.7, 2.05))
    ax.plot(deg, width, color=BLUE, lw=1.3)
    ax.text(19, 0.2, "nearest pair: 4 and 5", fontsize=8)
    ax.text(78, 0.2, "4 and 7", fontsize=8, ha="right")
    top = np.degrees(np.arctan2(2, 1))
    ax.plot([top], [2 / np.sqrt(5)], "o", color=RED, ms=4)
    ax.annotate(r"$63.4^\circ$, width $0.894$", (top, 2 / np.sqrt(5)),
                (6, 0.88), color=RED, va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel(r"angle of the normal vector $\boldsymbol{w}$ (degrees)")
    ax.set_ylabel("street width")
    ax.grid(lw=0.3)
    save(fig, "ch01-sweep")


def ch01_bridge():
    """The two convex hulls, the shortest bridge, and its bisector."""
    from machines import X
    neg, pos = X[[1, 0, 3, 2]], X[[6, 7, 5, 4]]
    p = np.array([4.4, 3.8])
    m = (X[3] + p) / 2
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.5),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax in axes:
        ax.fill(*neg.T, color=BLUE, alpha=0.13, lw=0.7)
        ax.fill(*pos.T, color=ORANGE, alpha=0.13, lw=0.7)
        x = np.array([0, 8.0])
        ax.plot(x, (11 - x) / 2, color=GREEN, lw=1)
        ax.plot(*np.array([X[3], p]).T, color=RED, lw=1.5, zorder=2)
        ax.set_aspect("equal")
        ax.grid(lw=0.3)
    _plot_machines(axes[0], s=11)
    axes[0].set_xlim(0, 7.5)
    axes[0].set_ylim(0, 8)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].plot(*np.array([[3.4, 2.6], [5.2, 2.6], [5.2, 4.4], [3.4, 4.4],
                            [3.4, 2.6]]).T, color=GREY, lw=0.6)
    ax = axes[1]
    _plot_machines(ax, label=False, s=16)
    ax.plot(*p, "D", color=RED, ms=3, zorder=4)
    ax.plot(*m, "o", mfc="white", mec=GREEN, ms=4, zorder=4)
    ax.text(4.0, 2.76, r"$\boldsymbol{x}_4$", color=BLUE, ha="center")
    ax.text(3.82, 4.09, r"$\boldsymbol{x}_5$", color=ORANGE)
    ax.text(5.03, 2.72, r"to $\boldsymbol{x}_7$", color=ORANGE, ha="center")
    ax.text(4.5, 3.9, r"$\boldsymbol{p}$", color=RED)
    ax.text(3.45, 3.24, "midpoint", color=GREEN)
    ax.annotate("", (4.17, 3.38), (4.0, 3.32), arrowprops=dict(
        arrowstyle="-", color=GREEN, lw=0.5, shrinkA=0, shrinkB=2))
    ax.set_xlim(3.4, 5.2)
    ax.set_ylim(2.6, 4.4)
    ax.set_xticks([3.5, 4, 4.5, 5])
    ax.set_yticks([3, 3.5, 4])
    ax.set_xlabel(r"$x_1$ (zoom)")
    save(fig, "ch01-bridge")


def ch01_street():
    """The finished SVM: street, support vectors, new machines."""
    from machines import X, alpha
    fig, ax = plt.subplots(figsize=(3.6, 3.15))
    x = np.array([0, 8.0])
    ax.fill_between(x, (10 - x) / 2, (12 - x) / 2, color=GREEN, alpha=0.13,
                    lw=0)
    ax.plot(x, (11 - x) / 2, color=GREEN, lw=1.2)
    for c, ls in ((10, "--"), (12, "--")):
        ax.plot(x, (c - x) / 2, color=GREEN, lw=0.7, ls=ls)
    sv = alpha > 0
    ax.scatter(*X[sv].T, s=62, facecolor="none", edgecolor=RED, lw=0.8,
               zorder=3)
    _plot_machines(ax, label=False)
    for i, dx, dy in ((1, 0.14, 0.12), (2, 0.14, 0.12), (3, 0.14, 0.12),
                      (6, 0.14, 0.12), (8, 0.14, 0.12)):
        ax.text(X[i - 1, 0] + dx, X[i - 1, 1] + dy, str(i), fontsize=8)
    ax.text(3.62, 3.0, r"4: $\alpha = 5/2$", color=RED, ha="right",
            va="center")
    ax.text(4.32, 4.02, r"5: $\alpha = 2$", color=RED, va="bottom")
    ax.text(6.1, 3.3, r"7: $\alpha = 1/2$", color=RED, va="bottom")
    new = np.array([[2, 2], [5, 3.2]])
    ax.scatter(*new.T, marker="*", s=42, color=GREEN, zorder=4)
    ax.text(2.15, 1.55, r"$\boldsymbol{a}$", color=GREEN)
    ax.text(5.15, 3.42, r"$\boldsymbol{c}$", color=GREEN)
    ax.text(0.2, 6.15, r"$f(\boldsymbol{x}) = +1$", color=GREEN, fontsize=8)
    ax.text(4.5, 1.5, r"$f(\boldsymbol{x}) = -1$", color=GREEN, fontsize=8)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"vibration $x_1$ (mm/s)")
    ax.set_ylabel(r"temperature rise $x_2$")
    ax.grid(lw=0.3)
    save(fig, "ch01-street")


def ch01_balance():
    """Tip-to-tail sum of the signed support vectors gives w."""
    from machines import X
    steps = (-2.5 * X[3], 2.0 * X[4], 0.5 * X[6])
    tips = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    fig, ax = plt.subplots(figsize=(3.0, 2.75))
    for k, col in enumerate((BLUE, ORANGE, ORANGE)):
        ax.annotate("", tips[k + 1], tips[k], arrowprops=dict(
            arrowstyle="-|>", color=col, lw=1.2, shrinkA=0, shrinkB=0))
    ax.annotate("", tips[3], tips[0], arrowprops=dict(
        arrowstyle="-|>", color=GREEN, lw=1.6, shrinkA=0, shrinkB=0))
    ax.plot(0, 0, "o", color=GREY, ms=2.5)
    ax.text(-4.1, -4.9, r"$-\frac{5}{2}\boldsymbol{x}_4$", color=BLUE)
    ax.text(-8.6, -2.6, r"$+2\boldsymbol{x}_5$", color=ORANGE)
    ax.text(-2.1, 2.1, r"$+\frac{1}{2}\boldsymbol{x}_7$", color=ORANGE)
    ax.text(1.5, 0.6, r"$\boldsymbol{w}$", color=GREEN)
    ax.set_xlim(-11, 3)
    ax.set_ylim(-8.5, 4)
    ax.set_xticks([-10, -5, 0])
    ax.set_aspect("equal")
    ax.set_xlabel(r"first entry")
    ax.set_ylabel(r"second entry")
    ax.grid(lw=0.3)
    save(fig, "ch01-balance")


def _hull(P):
    """Corners of the convex hull of P, in order around it."""
    from scipy.spatial import ConvexHull
    return P[ConvexHull(P).vertices]


def _gap(P, c, deg):
    """Street width of each direction; negative when the classes mix."""
    th = np.radians(deg)
    s = np.column_stack([np.cos(th), np.sin(th)]) @ P.T
    return s[:, c > 0].min(1) - s[:, c < 0].max(1)


def ch02_landscape():
    """Level lines of f, the normal w, and a drop to the line."""
    from machines import X
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    g = np.linspace(0, 8, 200)
    A, B = np.meshgrid(g, g)
    F = A + 2 * B - 11
    ax.contourf(A, B, F, levels=[-20, 0, 20], colors=[BLUE, ORANGE], alpha=0.08)
    for c in range(-8, 13, 4):
        if c:
            ax.plot(g, (c + 11 - g) / 2, color=GREY_RULE, lw=0.7)
    for c, xt in ((-8, 1.5), (-4, 4.6), (4, 6.2), (8, 6.5)):
        ax.text(xt, (c + 11 - xt) / 2 + 0.08, f"$f = {c}$", rotation=-26.6,
                rotation_mode="anchor", fontsize=8)
    ax.plot(g, (11 - g) / 2, color=GREEN, lw=1.3)
    ax.text(0.15, 5.5, r"$f = 0$", rotation=-26.6, rotation_mode="anchor",
            color=GREEN, fontsize=8)
    _plot_machines(ax, label=False)
    for i, p in enumerate(X, 1):
        dx, dy = {6: (-0.3, 0.12), 4: (0.12, -0.55), 7: (0.12, -0.55)}.get(
            i, (0.12, 0.12))
        ax.text(p[0] + dx, p[1] + dy, str(i), fontsize=8, zorder=4)
    q = X[7]
    h = q - 8 / 5 * np.array([1, 2])
    ax.plot(*np.array([q, h]).T, color=RED, lw=0.9, ls="--", zorder=2)
    u = np.array([1, 2]) / np.sqrt(5)
    v = np.array([2, -1]) / np.sqrt(5)
    c = np.array([h + 0.22 * v, h + 0.22 * (v + u), h + 0.22 * u])
    ax.plot(*c.T, color=RED, lw=0.6)
    ax.text(5.0, 5.3, r"distance $8/\sqrt{5}$", color=RED,
            bbox=dict(fc="white", ec="none", pad=1))
    ax.annotate("", (4.3, 5.6), (5.0, 5.42), arrowprops=dict(
        arrowstyle="-", color=RED, lw=0.5, shrinkA=0, shrinkB=0))
    o = np.array([1.5, 4.75])
    ax.annotate("", o + np.array([1, 2]), o, arrowprops=dict(
        arrowstyle="-|>", color=GREEN, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text(2.6, 6.45, r"$\boldsymbol{w}$", color=GREEN)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    save(fig, "ch02-landscape")


def ch02_knobs():
    """What a turn of w, a change of b, and a common scale each do."""
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.0), sharex=True, sharey=True)
    x = np.array([0, 8.0])
    line = lambda w, c: (c - w[0] * x) / w[1]
    for ax in axes:
        _plot_machines(ax, label=False, s=7)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0, 4, 8])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    ax = axes[0]
    ax.plot(x, line((1, 2), 11), color=GREEN, lw=1.1)
    ax.plot(x, line((1, 1), 7.6), color=ORANGE, lw=1, ls="--")
    ax.plot(x, line((1, 4), 17.8), color=RED, lw=1, ls="--")
    ax.plot(4.2, 3.4, "o", mfc="white", mec=GREY, ms=3.5, zorder=4)
    ax.set_title(r"turn $\boldsymbol{w}$")
    ax.set_ylabel(r"$x_2$")
    ax = axes[1]
    ax.plot(x, line((1, 2), 11), color=GREEN, lw=1.1)
    ax.plot(x, line((1, 2), 7), color=ORANGE, lw=1, ls="--")
    ax.plot(x, line((1, 2), 15), color=RED, lw=1, ls="--")
    ax.text(1.5, 0.3, r"$b = -7$", color=ORANGE, fontsize=8)
    ax.text(4.5, 5.85, r"$b = -15$", color=RED, fontsize=8)
    ax.set_title(r"change $b$")
    ax = axes[2]
    ax.fill_between(x, line((1, 2), 10), line((1, 2), 12), color=GREEN,
                    alpha=0.12, lw=0)
    ax.fill_between(x, line((1, 2), 10.5), line((1, 2), 11.5), color=GREEN,
                    alpha=0.28, lw=0)
    ax.plot(x, line((1, 2), 11), color=GREEN, lw=1.1)
    ax.text(0.3, 6.3, r"pale: $k = 1$" "\n" r"dark: $k = 2$", fontsize=8)
    ax.set_title(r"scale $(\boldsymbol{w}, b)$ by $k$")
    save(fig, "ch02-knobs")


def ch02_separable():
    """Convex hulls of one separable data set and two that are not."""
    from machines import X, y
    Xr = np.vstack([X, [2, 2]])
    yr = np.append(y, 1.0)
    Xx = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    yx = np.array([-1, -1, 1, 1], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.05))
    data = (("eight machines", X, y), (r"add $\boldsymbol{z}$", Xr, yr))
    for ax, (title, P, c) in zip(axes, data):
        ax.fill(*_hull(P[c < 0]).T, color=BLUE, alpha=0.15, lw=0.7)
        ax.fill(*_hull(P[c > 0]).T, color=ORANGE, alpha=0.15, lw=0.7)
        ax.scatter(*P[c < 0].T, color=BLUE, marker="o", s=8, zorder=3)
        ax.scatter(*P[c > 0].T, color=ORANGE, marker="s", s=8, zorder=3)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0, 4, 8])
        ax.set_title(title)
    axes[1].scatter([2], [2], s=45, facecolor="none", edgecolor=RED, lw=0.8,
                    zorder=4)
    axes[1].text(1.3, 0.3, r"$\boldsymbol{z}$", color=RED)
    axes[1].annotate("", (1.9, 1.75), (1.55, 0.8), arrowprops=dict(
        arrowstyle="-", color=RED, lw=0.5, shrinkA=0, shrinkB=0))
    x = np.array([0, 8.0])
    axes[0].plot(x, (11 - x) / 2, color=GREEN, lw=1)
    ax = axes[2]
    ax.plot(*Xx[:2].T, color=BLUE, lw=1.2, alpha=0.6)
    ax.plot(*Xx[2:].T, color=ORANGE, lw=1.2, alpha=0.6)
    ax.scatter(*Xx[:2].T, color=BLUE, marker="o", s=12, zorder=3)
    ax.scatter(*Xx[2:].T, color=ORANGE, marker="s", s=12, zorder=3)
    ax.plot(0.5, 0.5, "o", mfc="none", mec=RED, ms=6, mew=0.8, zorder=4)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title("XOR")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch02-separable")


def ch02_gap():
    """The gap of each direction for the three data sets."""
    from machines import X, y
    deg = np.linspace(-180, 180, 3601)
    Xr = np.vstack([X, [2, 2]])
    yr = np.append(y, 1.0)
    Xx = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    yx = np.array([-1, -1, 1, 1], dtype=float)
    fig, ax = plt.subplots(figsize=(4.3, 2.3))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axhspan(0, 1.5, color=GREEN, alpha=0.07, lw=0)
    for P, c, col, ls in ((X, y, BLUE, "-"), (Xr, yr, RED, "--"),
                          (Xx, yx, ORANGE, "-.")):
        ax.plot(deg, _gap(P, c, deg), color=col, lw=1.2, ls=ls)
    ax.plot([63.43], [2 / np.sqrt(5)], "o", color=BLUE, ms=3.5)
    ax.text(72, 0.98, r"$63.4^\circ$, $0.894$", color=BLUE)
    ax.text(-68, -5.6, "eight machines", color=BLUE)
    ax.text(-172, -0.62, "XOR", color=ORANGE)
    ax.text(12, -3.1, r"with $\boldsymbol{z}$", color=RED)
    ax.text(-172, 0.3, "separating directions", color=GREEN, fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-7.2, 1.5)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_xlabel(r"angle of the direction $\boldsymbol{u}$ (degrees)")
    ax.set_ylabel(r"gap$(\boldsymbol{u})$")
    ax.grid(lw=0.3)
    save(fig, "ch02-gap")


def ch02_units():
    """Machines 4 and 8 alone: the bisector turns when the units change."""
    from machines import X
    fig, ax = plt.subplots(figsize=(3.2, 2.75))
    y0 = np.array([0, 8.0])
    x = np.array([0, 8.0])
    ax.plot(x, (24.5 - x) / 4, color=GREEN, lw=1.2)
    ax.plot(x, (2004.5 - x) / 400, color=ORANGE, lw=1.1, ls="--")
    ax.plot(4.7 - 0.04 * y0, y0, color=RED, lw=1.1, ls="-.")
    ax.plot(*X[[3, 7]].T, color=GREY, lw=0.6, ls=":")
    ax.scatter(*X[3], color=BLUE, s=16, zorder=3)
    ax.scatter(*X[7], color=ORANGE, marker="s", s=16, zorder=3)
    ax.text(3.45, 2.3, "4", fontsize=8)
    ax.text(5.15, 7.1, "8", fontsize=8)
    ax.text(0.2, 6.25, "same units", color=GREEN)
    ax.text(5.5, 5.2, r"$x_2 \times 10$", color=ORANGE)
    ax.text(4.8, 0.4, r"$x_1 \times 10$", color=RED)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ (original units)")
    ax.set_ylabel(r"$x_2$ (original units)")
    ax.grid(lw=0.3)
    save(fig, "ch02-units")


def ch03_update():
    """One update as vector addition and as a moved line."""
    from ch03_perceptron import perceptron
    from machines import X, y
    log = perceptron(X, y)[4]
    _, i, w1, b1 = log[-1]
    _, _, w0, b0 = log[-2]
    p, c = X[i - 1], y[i - 1]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    style = dict(arrowstyle="-|>", lw=1, shrinkA=0, shrinkB=0,
                 mutation_scale=7)
    o = np.zeros(2)
    ax.annotate("", w0, o, arrowprops=dict(color=GREY, **style))
    ax.annotate("", w1, w0, arrowprops=dict(color=RED, **style))
    ax.annotate("", w1, o, arrowprops=dict(color=GREEN, **style))
    ax.text(2.2, 1.0, r"old $\boldsymbol{w}$", color=GREY)
    ax.text(0.4, 3.9, r"$y_i\boldsymbol{x}_i$", color=RED)
    ax.text(-3.5, 1.2, r"new $\boldsymbol{w}$", color=GREEN)
    ax.scatter(0, 0, color=GREY, s=6)
    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_yticks([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title("weight space")
    ax = axes[1]
    x = np.array([0, 8.0])
    for v, q, col, ls, lab in ((w0, b0, GREY, "--", "before"),
                               (w1, b1, GREEN, "-", "after")):
        ax.plot(x, -(q + v[0] * x) / v[1], color=col, lw=1, ls=ls, label=lab)
    _plot_machines(ax, label=False)
    shift = {1: (0.3, -0.45), 2: (-0.45, 0.2), 4: (0.15, -0.55),
             7: (0.15, -0.55)}
    for j, q in enumerate(X, 1):
        dx, dy = shift.get(j, (0.15, 0.12))
        ax.text(q[0] + dx, q[1] + dy, str(j), fontsize=8, zorder=4)
    ax.scatter(*p, s=60, facecolor="none", edgecolor=RED, lw=0.8, zorder=4)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("data space")
    ax.legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch03-update")


def ch03_path():
    """Lines visited by the raw run, and mistakes in each pass."""
    from ch03_perceptron import perceptron
    from machines import X, y
    log = perceptron(X, y)[4]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    x = np.array([0, 8.0])
    ends = {}
    for e, _, v, c in log:
        ends[e] = (v, c)
    last = max(ends)
    for e, (v, c) in ends.items():
        s = e / last
        if e < last and abs(v[1]) > 1e-9:
            ax.plot(x, -(c + v[0] * x) / v[1], color=GREY, lw=0.5,
                    alpha=0.25 + 0.5 * s)
    v, c = ends[last]
    ax.plot(x, -(c + v[0] * x) / v[1], color=GREEN, lw=1.2)
    _plot_machines(ax, label=False, s=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("line after each pass")
    ax.grid(lw=0.3)
    ax = axes[1]
    n = np.bincount([e for e, _, _, _ in log], minlength=last + 2)[1:]
    ax.bar(np.arange(1, last + 2), n, color=BLUE, width=0.7)
    ax.set_xlabel("pass through the data")
    ax.set_ylabel("mistakes")
    ax.set_xticks([1, 6, 12, 18])
    ax.set_yticks(range(0, int(n.max()) + 1))
    ax.set_title("mistakes in each pass")
    ax.grid(lw=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1)
    save(fig, "ch03-path")


def ch03_squeeze():
    """The two inequalities of the proof, measured and extended."""
    from ch03_perceptron import lift, perceptron
    from machines import X, y
    log = perceptron(X, y)[4]
    u = np.array([1, 2, -11.0])
    u /= np.linalg.norm(u)
    Z = lift(X)
    R = np.sqrt((Z ** 2).sum(axis=1).max())
    g = (y * (Z @ u)).min()
    W = np.array([np.r_[v, c] for _, _, v, c in log])
    k = np.arange(1, len(W) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
    ax = axes[0]
    ax.plot(k, np.linalg.norm(W, axis=1), color=ORANGE, lw=0.9,
            label="length")
    ax.plot(k, W @ u, color=BLUE, lw=0.9, label="progress")
    ax.plot(k, np.sqrt(k) * R, color=ORANGE, lw=0.9, ls="--")
    ax.plot(k, k * g, color=BLUE, lw=0.9, ls="--")
    ax.set_xlabel("mistakes $k$")
    ax.set_title("the first 43 mistakes")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(lw=0.3)
    ax = axes[1]
    K = np.logspace(0, 4.3, 50)
    ax.loglog(K, np.sqrt(K) * R, color=ORANGE, lw=1, ls="--")
    ax.loglog(K, K * g, color=BLUE, lw=1, ls="--")
    lim = (R / g) ** 2
    ax.scatter([lim], [lim * g], color=RED, s=12, zorder=4)
    ax.text(lim / 1.6, lim * g * 1.6, r"$k = R^2/\gamma^2$", color=RED,
            ha="right")
    ax.text(3, 40, r"$\sqrt{k}\,R$", color=ORANGE)
    ax.text(90, 1.6, r"$k\gamma$", color=BLUE)
    ax.axvline(43, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("mistakes $k$")
    ax.set_title("the two limits meet")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch03-squeeze")


def ch03_fan():
    """Every line from every order, and the spread of their widths."""
    from ch03_perceptron import survey
    from machines import X, y
    T = survey(X, y)
    lines = np.unique(T[:, :3], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    x = np.array([0, 8.0])
    for a, c, d in lines:
        if abs(c) < 1e-9:
            ax.plot([-d / a] * 2, [0, 8], color=GREY, lw=0.35, alpha=0.6)
        else:
            ax.plot(x, -(d + a * x) / c, color=GREY, lw=0.35, alpha=0.6)
    ax.plot(x, (11 - x) / 2, color=GREEN, lw=1.2)
    _plot_machines(ax, label=False, s=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("115 lines")
    ax.grid(lw=0.3)
    ax = axes[1]
    bins = np.linspace(0.1, 0.9, 17)
    ax.hist(T[:, 5] * (1 - 1e-9), bins=bins, color=BLUE, alpha=0.75,
            weights=np.full(len(T), 100 / len(T)), label="as found")
    ax.hist(T[:, 6] * (1 - 1e-9), bins=bins, histtype="step", color=ORANGE,
            lw=1, weights=np.full(len(T), 100 / len(T)),
            label="moved to the middle")
    ax.axvline(2 / np.sqrt(5), color=GREEN, lw=1)
    ax.text(0.875, 25.5, "0.894", color=GREEN, ha="right")
    ax.set_ylim(0, 44)
    ax.set_xlim(0.05, 0.95)
    ax.set_xlabel("street width")
    ax.set_ylabel("percent of orders")
    ax.set_title("widths over 40320 orders")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(lw=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1)
    save(fig, "ch03-fan")


def _street(ax, w, b, col=GREEN, xlim=(0, 8)):
    """Draw the line f = 0 and shade the street of (w, b)."""
    x = np.array(xlim, dtype=float)
    if abs(w[1]) < 1e-9:
        return
    for lvl, ls in ((-1, "--"), (0, "-"), (1, "--")):
        ax.plot(x, (lvl - b - w[0] * x) / w[1], color=col, lw=0.9, ls=ls)
    ax.fill_between(x, (-1 - b - w[0] * x) / w[1], (1 - b - w[0] * x) / w[1],
                    color=col, alpha=0.14, lw=0)


def ch04_maxmin():
    """The margin over direction and offset: a tent with creases."""
    from machines import X, y
    th = np.radians(np.linspace(0, 90, 361))
    c = np.linspace(3.2, 6, 561)
    T, C = np.meshgrid(th, c)
    U = np.stack([np.cos(T), np.sin(T)], axis=-1)
    rho = (y * (U @ X.T - C[..., None])).min(axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25))
    ax = axes[0]
    lv = [0.01, 0.1, 0.2, 0.3, 0.4]
    ax.contourf(np.degrees(T), C, rho, levels=[0.01, 0.1, 0.2, 0.3, 0.4, 0.46],
                cmap="Blues", alpha=0.85)
    ax.contour(np.degrees(T), C, rho, levels=lv, colors=BLUE, linewidths=0.5)
    ax.text(70, 5.35, "0.447", color=RED)
    t0, c0 = np.degrees(np.arctan2(2, 1)), 11 / np.sqrt(5)
    ax.scatter([t0], [c0], color=RED, s=14, zorder=4)
    ax.set_xlabel(r"direction angle $\theta$ (degrees)")
    ax.set_ylabel(r"offset $c$")
    ax.set_title("smallest distance")
    ax.grid(lw=0.3)
    ax = axes[1]
    u = np.array([1, 2]) / np.sqrt(5)
    c = np.linspace(4.2, 5.7, 50)
    for i, col in ((3, BLUE), (4, ORANGE)):
        ax.plot(c, y[i] * (X[i] @ u - c), color=col, lw=0.9)
    r = np.min([y[i] * (X[i] @ u - c) for i in range(8)], axis=0)
    ax.plot(c, r, color=RED, lw=1.8, alpha=0.55)
    ax.text(5.15, 0.4, "machine 4", color=BLUE)
    ax.text(4.65, 0.4, "5 and 7", color=ORANGE, ha="right")
    ax.text(4.92, 0.18, "0.447", color=RED, ha="center")
    ax.set_ylim(-0.1, 0.8)
    ax.set_xlabel(r"offset $c$, at the best $\theta$")
    ax.set_title("the minimum has a peak")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch04-maxmin")


def ch04_slices():
    """Feasible weights for three fixed offsets, with the shortest one."""
    from ch04_margin import problem
    from machines import X, y
    A = problem(X, y)[2]
    g = np.linspace(0, 4, 801)
    W1, W2 = np.meshgrid(g, g)
    t = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.0), sharex=True,
                             sharey=True)
    for ax, b in zip(axes, (-10.0, -11.0, -15.0)):
        ok = (A[:, 0, None, None] * W1 + A[:, 1, None, None] * W2
              + A[:, 2, None, None] * b - 1).min(axis=0) >= -1e-9
        ax.contourf(W1, W2, ok, levels=[0.5, 1.5], colors=[BLUE], alpha=0.25)
        if ok.any():
            n = np.hypot(W1[ok], W2[ok])
            j = n.argmin()
            r = n[j]
            ax.plot(r * np.cos(t), r * np.sin(t), color=GREEN, lw=0.9)
            ax.scatter([W1[ok][j]], [W2[ok][j]], color=RED, s=10, zorder=4)
            ax.text(0.15, 0.25, rf"$\Vert\boldsymbol{{w}}\Vert = {r:.2f}$",
                    color=GREEN)
        else:
            ax.text(2, 2, "empty", color=GREY, ha="center")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_title(rf"$b = {b:.0f}$")
        ax.set_xlabel(r"$w_1$")
        ax.set_aspect("equal")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$w_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch04-slices")


def ch04_unique():
    """Midpoint for w; the wedge of feasible offsets for b."""
    from machines import X, y
    g = np.linspace(-1, 4, 801)
    W1, W2 = np.meshgrid(g, g)
    W = np.stack([W1, W2], axis=-1)
    s = W @ X.T
    gap = s[..., y > 0].min(axis=-1) - s[..., y < 0].max(axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
    ax = axes[0]
    ax.contourf(W1, W2, gap, levels=[2, 1e3], colors=[BLUE], alpha=0.22)
    ax.contour(W1, W2, gap, levels=[2], colors=[BLUE], linewidths=0.8)
    t = np.linspace(0, 2 * np.pi, 300)
    r0 = np.sqrt(5)
    ax.plot(r0 * np.cos(t), r0 * np.sin(t), color=GREEN, lw=1)
    r = 3.0
    pts = np.array([[r * np.cos(a), r * np.sin(a)]
                    for a in np.radians((46, 64))])
    ax.plot(r * np.cos(t), r * np.sin(t), color=GREY, lw=0.6, ls="--")
    ax.plot(*pts.T, color=ORANGE, lw=0.9, marker="o", ms=2.5)
    m = pts.mean(axis=0)
    ax.scatter(*m, color=ORANGE, s=10, zorder=4)
    ax.text(m[0] + 0.2, m[1] + 0.45, "midpoint", color=ORANGE)
    ax.scatter([1], [2], color=RED, s=12, zorder=4)
    ax.text(-0.85, 1.15, r"$\boldsymbol{w}^{\ast}$", color=RED)
    ax.annotate("", (0.97, 1.97), (-0.5, 1.35),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title("weights that allow some $b$")
    ax.grid(lw=0.3)
    ax = axes[1]
    k = np.linspace(0.7, 1.6, 50)
    lo, hi = 1 - 12 * k, -1 - 10 * k
    ax.plot(k, lo, color=ORANGE, lw=1)
    ax.plot(k, hi, color=BLUE, lw=1)
    ax.fill_between(k, lo, hi, where=k >= 1, color=BLUE, alpha=0.22, lw=0)
    ax.scatter([1], [-11], color=RED, s=12, zorder=4)
    ax.text(1.04, -10.3, r"$b^{\ast} = -11$", color=RED)
    ax.text(0.72, -13.6, "no $b$ fits", color=GREY)
    ax.annotate("many $b$ fit", (1.45, -16.1), (1.02, -17.3), color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel(r"scale $k$ of $k\,\boldsymbol{w}^{\ast}$")
    ax.set_ylabel(r"feasible $b$")
    ax.set_title("offsets for each scale")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch04-unique")


def ch04_rounds():
    """Constraint generation: two rounds on the eight machines."""
    from ch04_margin import generate
    from machines import X, y
    rounds = generate(X, y, [3, 4])
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    for ax, (keep, w, b, s), name in zip(axes, rounds, ("first", "second")):
        _street(ax, w, b)
        _plot_machines(ax)
        on = X[keep]
        ax.scatter(*on.T, s=46, facecolors="none", edgecolors=RED, lw=0.8,
                   zorder=5)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(f"{name} round: {len(keep)} machines kept")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=1)
    save(fig, "ch04-rounds")


def ch04_fragile():
    """A ninth machine walks to the healthy hull and the street collapses."""
    from ch04_margin import best_gap, generate
    from machines import X, y
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
    ax = axes[0]
    for t, col in ((0.6, GREEN), (0.8, ORANGE)):
        z = np.array([5, 5.0]) - 2 * t
        Xn, yn = np.vstack([X, z]), np.append(y, 1.0)
        _, w, b, _ = generate(Xn, yn, [3, 4])[-1]
        _street(ax, w, b, col)
        ax.scatter(*z, color=col, marker="D", s=14, zorder=5)
    _plot_machines(ax, label=False, s=10)
    ax.plot([5, 3], [5, 3], color=GREY, lw=0.6, ls=":")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("two positions of machine 9")
    ax.grid(lw=0.3)
    ax = axes[1]
    T = np.linspace(0, 0.995, 80)
    G = [best_gap(np.vstack([X, np.array([5, 5.0]) - 2 * t]),
                  np.append(y, 1.0), 20001) for t in T]
    ax.plot(T, G, color=BLUE, lw=1.2)
    ax.axvspan(1, 1.25, color=RED, alpha=0.12, lw=0)
    ax.text(1.125, 0.45, "infeasible", color=RED, rotation=90, ha="center",
            va="center")
    for t, col in ((0.6, GREEN), (0.8, ORANGE)):
        ax.axvline(t, color=col, lw=0.6, ls=":")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("walk $t$")
    ax.set_ylabel("widest street")
    ax.set_title("the width as it walks")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch04-fragile")


def ch05_cases():
    """One inequality: the solution ignores it, or leans on it."""
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35), sharey=True)
    t = np.linspace(0, 2 * np.pi, 200)
    u = np.linspace(-1.5, 4.5, 2)
    for k, ax in enumerate(axes):
        for r2 in (1.25, 2.5, 5.0, 8.0):
            r = np.sqrt(r2)
            on = k == 1 and r2 == 5.0
            ax.plot(r * np.cos(t), r * np.sin(t),
                    color=GREEN if on else GREY_RULE, lw=1)
        top = (5 - u) / 2
        ax.fill_between(u, -2, top, color=BLUE, alpha=0.10, lw=0) if k == 0 \
            else ax.fill_between(u, top, 4.5, color=BLUE, alpha=0.10, lw=0)
        ax.plot(u, top, color=BLUE, lw=1.3)
        ax.set_xlim(-1.5, 4.5)
        ax.set_ylim(-2, 4.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$w_1$")
        ax.grid(lw=0.3)
    ax = axes[0]
    ax.plot([0], [0], "o", color=RED, ms=3.5)
    ax.text(0.15, -0.6, r"$\boldsymbol{w}^{\ast}$, $\alpha = 0$", color=RED)
    ax.text(2.45, -1.2, "feasible", color=BLUE)
    ax.set_title("inactive: constraint ignored")
    ax.set_ylabel(r"$w_2$")
    ax = axes[1]
    ax.annotate("", (0.45, 0.9), (1, 2), arrowprops=dict(
        arrowstyle="-|>", color=RED, lw=1.1, shrinkA=0, shrinkB=0))
    ax.plot([1], [2], "o", color=RED, ms=3.5)
    ax.text(1.15, 2.35, r"$\boldsymbol{w}^{\ast}$, $\alpha = 1$", color=RED)
    ax.text(-0.55, 1.25, r"$-\nabla f$", color=RED)
    ax.text(2.3, 3.5, "feasible", color=BLUE)
    ax.set_title("active: constraint pushes back")
    fig.tight_layout(w_pad=1)
    save(fig, "ch05-cases")


def ch05_cone():
    """At a corner, the KKT test asks if -grad f lies in a cone."""
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    z = np.array([1.0, 2.0])
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, c, sol in zip(axes, (np.array([3.0, 3]), np.array([0.0, 4])),
                          (z, np.array([-0.5, 3.5]))):
        xs = np.array([-1.5, 1.0])
        ax.fill_between(xs, -0.5, 3 - xs, color=BLUE, alpha=0.10, lw=0)
        ax.plot([-1.5, 1], [4.5, 2], color=BLUE, lw=1.2)
        ax.plot([1, 1], [2, -0.5], color=BLUE, lw=1.2)
        ax.plot([1, 3.2], [2, 2], color=ORANGE, lw=0.5, ls="--")
        ax.plot([1, 3.0], [2, 4.0], color=ORANGE, lw=0.5, ls="--")
        ax.fill([1, 3.2, 3.2, 3.0], [2, 2, 4.2, 4.0], color=ORANGE,
                alpha=0.14, lw=0)
        r = np.linalg.norm(sol - c)
        ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=GREEN,
                lw=0.9)
        ax.plot(*c, "x", color=GREY, ms=4)
        for d, col in ((c - z, RED),):
            ax.annotate("", z + 0.75 * d / np.linalg.norm(d) * 1.3, z,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2,
                                        shrinkA=0, shrinkB=0))
        ax.plot(*sol, "o", color=RED, ms=3.5)
        ax.set_xlim(-1.5, 3.2)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$z_1$")
        ax.grid(lw=0.3)
    ax = axes[0]
    ax.text(-1.3, 0.55, "feasible", color=BLUE)
    ax.text(2.6, 2.2, "cone", color=ORANGE, ha="center")
    ax.text(2.88, 3.08, r"$\boldsymbol{c}$", color=GREY, ha="right")
    ax.text(2.06, 2.6, r"$-\nabla f$", color=RED)
    ax.text(0.92, 1.65, r"$\boldsymbol{z}^{\ast}$", color=RED, ha="right")
    ax.set_title(r"$-\nabla f$ inside the cone: stop")
    ax.set_ylabel(r"$z_2$")
    ax = axes[1]
    ax.annotate("", (0.15, 2.85), (1, 2), arrowprops=dict(
        arrowstyle="-|>", color=GREEN, lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(2.6, 2.2, "cone", color=ORANGE, ha="center")
    ax.text(1.05, 1.62, "corner", color=GREY, ha="left")
    ax.text(0.72, 2.92, r"$-\nabla f$", color=RED, ha="left")
    ax.text(-1.32, 2.1, "descent\nalong edge", color=GREEN, ha="left",
            va="center")
    ax.text(0.14, 4.0, r"$\boldsymbol{c}$", color=GREY)
    ax.set_title(r"$-\nabla f$ outside: move on")
    fig.tight_layout(w_pad=1)
    save(fig, "ch05-cone")


def ch05_corner():
    """The eight machines in the plane of the weights."""
    from machines import X, y
    d = (X[y > 0][:, None] - X[y < 0][None, :]).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    g1, g2 = np.meshgrid(np.linspace(-0.5, 3.5, 600),
                         np.linspace(-0.5, 3.5, 600))
    ok = (d @ np.array([g1.ravel(), g2.ravel()])).min(axis=0) >= 2
    ax.contourf(g1, g2, ok.reshape(g1.shape), levels=[0.5, 1.5],
                colors=[BLUE], alpha=0.12)
    u = np.linspace(-0.5, 3.5, 2)
    for e in d:
        if abs(e[1]) > 1e-9:
            ax.plot(u, (2 - e[0] * u) / e[1], color=GREY_RULE, lw=0.4)
        else:
            ax.axvline(2 / e[0], color=GREY_RULE, lw=0.4)
    ax.plot(u, [2, 2], color=ORANGE, lw=1.2)
    ax.plot([1, 1], [-0.5, 3.5], color=ORANGE, lw=1.2)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(5 ** 0.5 * np.cos(t), 5 ** 0.5 * np.sin(t), color=GREEN, lw=0.9)
    for tip, col in (((1, 0), ORANGE), ((0, 2), ORANGE), ((1, 2), RED)):
        ax.annotate("", tip, (0, 0), arrowprops=dict(
            arrowstyle="-|>", color=col, lw=1.2, shrinkA=0, shrinkB=0))
    ax.plot([1], [2], "o", color=RED, ms=3.5)
    ax.text(1.08, 0.12, r"$(\boldsymbol{x}_7-\boldsymbol{x}_4)/2$",
            color=ORANGE)
    ax.text(-0.42, 1.0, r"$2(\boldsymbol{x}_5-\boldsymbol{x}_4)$",
            color=ORANGE, va="center", rotation=90)
    ax.text(0.34, 1.35, r"$\boldsymbol{w}^{\ast}$", color=RED)
    ax.text(2.0, 2.9, "feasible", color=BLUE, ha="center")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.grid(lw=0.3)
    save(fig, "ch05-corner")


def ch05_price():
    """Raising one machine's requirement: the slope at 0 is its alpha."""
    from ch05_kkt import price
    from machines import alpha
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    E = np.linspace(-0.4, 2.6, 61)
    for i, col in ((3, RED), (4, ORANGE), (6, BLUE), (5, GREEN)):
        ax.plot(E, [price(i, e) for e in E], color=col, lw=1.1)
        ax.plot(E, 2.5 + alpha[i] * E, color=col, lw=0.5, ls=":")
    ax.scatter([0], [2.5], color=GREY, s=10, zorder=5)
    ax.text(1.1, 7.75, "machine 4, slope 5/2", color=RED,
            ha="right")
    ax.text(2.56, 5.75, r"5, slope 2", color=ORANGE, ha="right")
    ax.text(1.95, 4.05, "7, slope 1/2", color=BLUE, ha="right")
    ax.text(0.5, 1.95, r"6, slope 0 until $\epsilon = 2$", color=GREEN)
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(1, 9)
    ax.set_xlabel(r"extra requirement $\epsilon$")
    ax.set_ylabel(r"best $\Vert\boldsymbol{w}\Vert^2/2$")
    ax.grid(lw=0.3)
    save(fig, "ch05-price")


def ch05_walk():
    """A ninth machine crosses towards the street: slack, then alpha."""
    from ch04_margin import hard_margin, slacks
    from ch05_kkt import multipliers
    from machines import X, y
    path = lambda t: np.array([5, 3.5]) + (1 - t) * np.array([1.0, 2.0])
    T = np.linspace(0, 1.3, 79)
    S, A = [], []
    for t in T:
        Xn, yn = np.vstack([X, path(t)]), np.append(y, 1.0)
        w, b, _ = hard_margin(Xn, yn)
        al, _ = multipliers(w, b, Xn, yn)
        S.append(max(slacks(w, b, Xn, yn)[8], 0))
        A.append(al[8])
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35))
    ax = axes[0]
    _street(ax, np.array([1.0, 2]), -11)
    _plot_machines(ax, label=False, s=10)
    p, q = path(0), path(1.3)
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=RED,
                lw=1, shrinkA=0, shrinkB=0))
    ax.scatter(*path(0), color=RED, marker="D", s=12, zorder=5)
    ax.text(6.2, 5.6, r"$t = 0$", color=RED)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("the path of machine 9")
    ax.grid(lw=0.3)
    ax = axes[1]
    ax.plot(T, S, color=BLUE, lw=1.2)
    ax.plot(T, A, color=ORANGE, lw=1.2)
    ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.text(0.5, 3.1, "slack", color=BLUE)
    ax.text(0.97, 1.6, r"$\alpha_9$", color=ORANGE, ha="right")
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 5.3)
    ax.set_xlabel("walk $t$")
    ax.set_title("one of the two is zero")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch05-walk")


def ch06_matrix():
    """The matrix Q, with the block of the three support vectors."""
    from ch06_dual import dual_matrix
    Q = dual_matrix()
    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    ax.imshow(np.sign(Q) * np.sqrt(np.abs(Q)), cmap="RdBu", vmin=-22,
              vmax=22, aspect="equal")
    S = (3, 4, 6)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{Q[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color=BLUE if i in S and j in S else GREY,
                    fontweight="bold" if i in S and j in S else "normal")
            if i in S and j in S:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, ec=ORANGE, lw=1.1))
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(8), range(1, 9))
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_xlabel("machine $j$", labelpad=4)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("machine $i$")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "ch06-matrix")


def ch06_gap():
    """Primal values sit above dual values, and they meet at 5/2."""
    from ch06_dual import dual_matrix, dual_value
    from machines import X, alpha, b, w, y
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4), sharey=True)
    ax = axes[0]
    T = np.linspace(0, 2, 101)
    ax.plot(T, 5 * T - 2.5 * T**2, color=GREEN, lw=1.2)
    Tp = np.linspace(1, 1.5, 41)
    ax.plot(Tp, 2.5 * Tp**2, color=BLUE, lw=1.2)
    ax.axhline(2.5, color=GREY, lw=0.6, ls=":")
    ax.scatter([1], [2.5], color=ORANGE, s=16, zorder=5)
    ax.text(1.27, 5.2, r"primal, $(t\boldsymbol{w}^*\!,\ tb^*)$",
            color=BLUE, ha="right")
    ax.text(0.62, 0.55, r"dual, $t\boldsymbol{\alpha}^*$", color=GREEN,
            ha="left")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 6)
    ax.set_xlabel("scale $t$")
    ax.set_ylabel("objective value")
    ax.set_title("along one ray")
    ax.grid(lw=0.3)
    ax = axes[1]
    rng = np.random.default_rng(20261101)
    Q = dual_matrix()
    D = []
    while len(D) < 400:
        a = alpha + rng.exponential(0.25, size=8) * rng.uniform(0, 1)
        a[y > 0] *= a[y < 0].sum() / a[y > 0].sum()
        a *= rng.uniform(0.3, 1.2)
        D.append(dual_value(a, Q))
    P = []
    while len(P) < 400:
        v = w + rng.normal(size=2) * rng.uniform(0, 0.8)
        c = -11 + rng.normal() * 2
        m = (y * (X @ v + c)).min()
        if m > 0:
            P.append((v @ v) / (2 * m * m))
    ax.scatter(rng.uniform(0, 1, 400), P, s=2, color=BLUE, alpha=0.6, lw=0)
    D = np.array(D)
    ax.scatter(rng.uniform(0, 1, 400), D, s=2, color=GREEN, alpha=0.6, lw=0)
    ax.axhline(2.5, color=ORANGE, lw=0.9)
    ax.text(1.02, 2.5, "5/2", color=ORANGE, va="center")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("random feasible points")
    fig.tight_layout(w_pad=1)
    save(fig, "ch06-gap")


def ch06_reduced():
    """The dual after the balance condition removes alpha_4."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    s, t = np.meshgrid(np.linspace(0, 4, 301), np.linspace(0, 2, 301))
    D = 2 * (s + t) - (4 * t**2 + s**2) / 2
    lev = [0, 1, 1.5, 2, 2.25, 2.4]
    c = ax.contour(s, t, D, levels=lev, colors=BLUE, linewidths=0.7)
    ax.clabel(c, fmt=lambda v: f"{v:g}", fontsize=8)
    ax.scatter([2], [0.5], color=ORANGE, s=18, zorder=5)
    ax.annotate(r"$\alpha_5 = 2,\ \alpha_7 = \frac{1}{2}$, value $\frac{5}{2}$",
                (2, 0.5), (1.15, 1.72), color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
    ax.scatter([0], [0], color=GREY, s=12, zorder=5, clip_on=False)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$\alpha_5$")
    ax.set_ylabel(r"$\alpha_7$")
    ax.grid(lw=0.3)
    save(fig, "ch06-reduced")


def ch06_hull():
    """Multipliers as weights of the bridge; the dual peaks at 2/delta^2."""
    from machines import X, y
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
    ax = axes[0]
    for s, col, idx in ((1, ORANGE, [4, 6, 7, 5]), (-1, BLUE, [1, 0, 3, 2])):
        P = X[idx]
        ax.fill(P[:, 0], P[:, 1], color=col, alpha=0.10, lw=0)
    ax.scatter(*X[y > 0].T, s=6, color=ORANGE, marker="s", zorder=3, lw=0)
    ax.scatter(*X[y < 0].T, s=5, color=BLUE, zorder=3, lw=0)
    for i, lam, col in ((3, 1.0, BLUE), (4, 0.8, ORANGE), (6, 0.2, ORANGE)):
        ax.scatter(*X[i], s=90 * lam, facecolor="none", edgecolor=col,
                   lw=0.9, zorder=4)
    p, q = np.array([4.4, 3.8]), X[3]
    ax.plot(*zip(q, p), color=RED, lw=1.3, zorder=5)
    ax.scatter(*p, color=RED, s=9, zorder=6)
    ax.annotate(r"$\boldsymbol{p}$", p, (5.1, 5.0), color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
    ax.text(3.3, 4.75, "4/5", color=ORANGE, ha="center")
    ax.annotate("", (4, 4.12), (3.5, 4.65),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.4))
    ax.text(6.4, 3.0, "1/5", color=ORANGE, ha="left", va="center")
    ax.text(4.2, 1.95, r"$\boldsymbol{q}$, weight 1", color=BLUE, ha="left")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("the ends of the bridge")
    ax.grid(lw=0.3)
    ax = axes[1]
    A = np.linspace(0, 4, 201)
    for d2, col, name, xy in ((0.8, RED, "4 to bridge end", (1.5, 2.72)),
                              (1.0, GREEN, "4 to 5", (0.72, 2.25)),
                              (4.0, BLUE, "4 to 7", (0.65, 0.72))):
        ax.plot(A, 2 * A - d2 * A**2 / 2, color=col, lw=1.1)
        pk = 2 / d2
        ax.scatter([pk], [2 * pk - d2 * pk**2 / 2], color=col, s=9, zorder=5)
        ax.text(*xy, name, color=col, ha="left" if col != GREEN else "left")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("scale $A$")
    ax.set_ylabel("dual value")
    ax.set_title(r"peak $2/\delta^2$")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch06-hull")


def ch06_degenerate():
    """A ninth machine on the edge: one line, a segment of multipliers."""
    from machines import X
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4))
    ax = axes[0]
    _street(ax, np.array([1.0, 2.0]), -11.0, BLUE)
    _plot_machines(ax, label=False, s=10)
    ax.scatter(2, 5, color=ORANGE, marker="D", s=18, zorder=5)
    ax.text(1.9, 5.45, "9", color=ORANGE, ha="center")
    for i, dx, dy in ((3, 0.15, -0.95), (4, 0.2, 0.2), (6, 0.2, 0.2)):
        ax.text(X[i, 0] + dx, X[i, 1] + dy, str(i + 1), color=GREY)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("one street")
    ax.grid(lw=0.3)
    ax = axes[1]
    T = np.array([0, 1.0])
    ax.plot(T, 0 * T + 2.5, color=BLUE, lw=1.2)
    ax.plot(T, 2 - 2 * T, color=RED, lw=1.2)
    ax.plot(T, 0.5 + T, color=GREEN, lw=1.2)
    ax.plot(T, T, color=ORANGE, lw=1.2)
    ax.text(0.5, 2.6, r"$\alpha_4$", color=BLUE, ha="center")
    ax.text(0.12, 1.85, r"$\alpha_5$", color=RED, ha="left")
    ax.text(0.97, 1.62, r"$\alpha_7$", color=GREEN, ha="right")
    ax.text(0.82, 0.52, r"$\alpha_9$", color=ORANGE, ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r"position on the segment, $\alpha_9$")
    ax.set_ylabel("multiplier")
    ax.set_title("many certificates")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch06-degenerate")


def _nine(ax, s=12):
    """Scatter the nine machines, with the rogue as a diamond."""
    from machines import X, y
    for c, col, mark in ((-1, BLUE, "o"), (1, ORANGE, "s")):
        ax.scatter(*X[y == c].T, color=col, marker=mark, s=s, zorder=3)
    ax.scatter(2, 2, color=ORANGE, marker="D", s=s + 4, zorder=4)


def ch07_three():
    """Three kinds of point at C = 1/2, with the slacks drawn."""
    from ch07_soft import X9, y9, kinds, soft_line
    C = 0.5
    w, b, xi, a = soft_line(C, X9, y9)
    k = kinds(a, C)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    _street(ax, w, b, GREEN)
    _nine(ax, 14)
    u = w @ w
    for i, p in enumerate(X9):
        if xi[i] > 1e-6:
            q = p + y9[i] * xi[i] * w / u   # foot on the own edge
            ax.plot(*zip(p, q), color=RED, lw=1.1, zorder=2)
    for i, ki in enumerate(k):
        if ki != "0":
            ax.scatter(*X9[i], s=62, facecolor="none", lw=0.9, zorder=5,
                       edgecolor=RED if ki == "C" else GREEN)
    off = {0: (0.22, -0.32), 1: (0.2, -0.1), 2: (0.2, 0.2),
           3: (0.25, -0.34), 4: (0.25, -0.38), 5: (0.2, 0.12),
           6: (0.22, 0.14), 7: (0.2, 0.0), 8: (-0.14, -0.5)}
    for i, (dx, dy) in off.items():
        ax.text(X9[i, 0] + dx, X9[i, 1] + dy, str(i + 1), color=GREY)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(lw=0.3)
    h = [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=RED, ms=7),
         plt.Line2D([], [], ls="", marker="o", mfc="none", mec=GREEN, ms=7),
         plt.Line2D([], [], color=RED, lw=1.1)]
    ax.legend(h, [r"$\alpha_i = C$", r"$0 < \alpha_i < C$", "slack"],
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              handletextpad=0.5)
    save(fig, "ch07-three")


def ch07_move():
    """The street of the nine machines at three values of C."""
    from ch07_soft import X9, y9, soft_line
    fig, axes = plt.subplots(1, 3, figsize=(4.75, 1.95), sharey=True)
    for ax, C in zip(axes, (0.1, 0.5, 2.0)):
        w, b, xi, _ = soft_line(C, X9, y9)
        _street(ax, w, b, GREEN)
        _nine(ax, 9)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(rf"$C = {C:g}$, width {2 / np.linalg.norm(w):.2f}")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch07-move")


def ch07_path():
    """Width and total slack as C grows, with and without the rogue."""
    from ch07_soft import X9, y9, soft_line
    from machines import X, y
    Cs = np.exp(np.linspace(np.log(0.02), np.log(20), 160))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.25), sharex=True)
    for Xd, yd, col, ls in ((X, y, BLUE, "-"), (X9, y9, RED, "--")):
        rows = [soft_line(C, Xd, yd) for C in Cs]
        axes[0].plot(Cs, [2 / np.linalg.norm(r[0]) for r in rows],
                     color=col, lw=1.2, ls=ls)
        axes[1].plot(Cs, [r[2].sum() for r in rows], color=col, lw=1.2,
                     ls=ls)
    ax = axes[0]
    ax.axvline(2.5, color=GREY, lw=0.5, ls=":")
    ax.axvline(0.8, color=GREY, lw=0.5, ls=":")
    ax.text(0.72, 7.2, r"$C = 4/5$", ha="right")
    ax.text(2.8, 7.2, r"$5/2$", ha="left")
    ax.text(5.0, 2.55, "nine", color=RED, ha="center")
    ax.text(5.0, 0.15, "eight", color=BLUE, ha="center")
    ax.set_ylim(0, 8)
    ax.set_ylabel("street width")
    ax = axes[1]
    ax.axvline(2.5, color=GREY, lw=0.5, ls=":")
    ax.axvline(0.8, color=GREY, lw=0.5, ls=":")
    ax.text(5.0, 3.85, "nine", color=RED, ha="center")
    ax.text(0.12, 0.6, "eight", color=BLUE, ha="center")
    ax.set_ylim(0, 6)
    ax.set_ylabel("total slack")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"penalty $C$")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch07-path")


def ch07_box():
    """The dual of machines 4 and 7: a cap clips the best point."""
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
    ax = axes[0]
    g = np.linspace(0, 1.1, 161)
    A, B = np.meshgrid(g, g)
    D = A + B - (25 * A**2 - 66 * A * B + 45 * B**2) / 2
    ax.contour(A, B, D, levels=[-2, -1, -0.3, 0, 0.3, 0.5],
               colors=GREY_RULE, linewidths=0.5, linestyles="solid")
    ax.plot([0, 1.1], [0, 1.1], color=BLUE, lw=1.1)
    ax.add_patch(plt.Rectangle((0, 0), 0.3, 0.3, fill=False, ec=RED, lw=1.1))
    ax.scatter([0.5], [0.5], color=BLUE, s=14, zorder=5)
    ax.scatter([0.3], [0.3], color=RED, s=14, zorder=5)
    ax.text(0.33, 0.14, r"box, $C = 0.3$", color=RED)
    ax.text(0.56, 0.42, "free best", color=BLUE)
    ax.text(0.68, 0.87, "balance", color=BLUE, ha="right")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha_4$")
    ax.set_ylabel(r"$\alpha_7$")
    ax.grid(lw=0.3)
    ax = axes[1]
    T = np.linspace(0, 1, 101)
    ax.plot(T, 2 * T - 2 * T**2, color=BLUE, lw=1.2)
    ax.axvline(0.3, color=RED, lw=0.9, ls="--")
    ax.scatter([0.5], [0.5], color=BLUE, s=14, zorder=5)
    ax.scatter([0.3], [0.42], color=RED, s=14, zorder=5)
    ax.text(0.33, 0.06, r"cap $C = 0.3$", color=RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r"$a = \alpha_4 = \alpha_7$")
    ax.set_ylabel(r"dual value $D$")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch07-box")


def ch07_offset():
    """The objective against b: a flat bottom or a single point."""
    from ch07_soft import X9, y9, soft_line
    from machines import X, y

    def pen(Xd, yd, w, b, C):
        return C * np.maximum(0, 1 - yd * (Xd @ w + b)).sum() + w @ w / 2
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    ax = axes[0]
    w = np.array([0.3, 0.2])
    B = np.linspace(-2.6, -0.9, 341)
    ax.plot(B, [pen(X, y, w, b, 0.05) for b in B], color=BLUE, lw=1.2)
    ax.axvspan(-1.9, -1.5, color=GREEN, alpha=0.15, lw=0)
    ax.text(-1.7, 0.2285, "any $b$ here", color=GREEN, ha="center")
    ax.set_title(r"eight machines, $C = 1/20$")
    ax.set_ylim(0.22, 0.27)
    ax.set_ylabel("objective with $\\boldsymbol{w}$ fixed")
    ax = axes[1]
    w = soft_line(0.5, X9, y9)[0]
    B = np.linspace(-3.6, -2.2, 281)
    ax.plot(B, [pen(X9, y9, w, b, 0.5) for b in B], color=BLUE, lw=1.2)
    ax.scatter([-2.9], [pen(X9, y9, w, -2.9, 0.5)], color=GREEN, s=14,
               zorder=5)
    ax.text(-2.86, 2.09, "$b = -2.9$", color=GREEN, ha="center")
    ax.set_ylim(2.03, 2.6)
    ax.set_title(r"nine machines, $C = 1/2$")
    for ax in axes:
        ax.set_xlabel("offset $b$")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1)
    save(fig, "ch07-offset")


def _capped_hull(P, cap):
    """Corners of the capped hull of the rows of P, by a sweep."""
    pts = []
    for t in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        d = np.array([np.cos(t), np.sin(t)])
        left, c = 1.0, np.zeros(2)
        for i in np.argsort(-(P @ d)):
            u = min(cap, left)
            c = c + u * P[i]
            left -= u
        pts.append(c)
    return np.array(pts)


def ch07_reduced():
    """The bridge of C = 1/2 joins the capped hulls of the nine."""
    from scipy.spatial import ConvexHull
    from ch07_soft import X9, y9, reduced_bridge
    cap, p, q, _ = reduced_bridge(0.5)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for c, col in ((-1, BLUE), (1, ORANGE)):
        P = X9[y9 == c]
        v = ConvexHull(P).vertices
        ax.plot(*P[np.r_[v, v[0]]].T, color=col, lw=0.6, ls="--")
        ax.fill(*_capped_hull(P, cap).T, color=col, alpha=0.3, lw=0.8)
    _nine(ax, 10)
    ax.plot(*zip(p, q), color=RED, lw=1.5, zorder=6)
    ax.annotate("bridge", (3.3, 2.8), (1.2, 4.6), color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
    ax.text(0.5, 7.3, r"cap $\mu = 0.424$", color=GREY)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(lw=0.3)
    save(fig, "ch07-reduced")


def ch08_losses():
    """Four losses against t = y f(x), and the pull of each."""
    from ch08_hinge import hinge, logistic, squared
    t = np.linspace(-1.5, 3, 451)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.7, 2.35))
    a1.plot(t, np.where(t <= 0, 1, 0), color=GREY, lw=1, ls=":")
    a1.plot(t, hinge(t), color=BLUE, lw=1.5)
    a1.plot(t, logistic(t), color=ORANGE, lw=1.2)
    a1.plot(t, squared(t), color=GREEN, lw=1.2, ls="--")
    a1.set_ylim(0, 4)
    a1.set_xlabel(r"$t = y\,f(\boldsymbol{x})$")
    a1.set_title("loss")
    a2.axhline(0, color=GREY_RULE, lw=0.6)
    a2.plot(t, (t < 1) * 1.0, color=BLUE, lw=1.5)
    a2.plot(t, 1 / (1 + np.exp(t)), color=ORANGE, lw=1.2)
    a2.plot(t, 2 * (1 - t), color=GREEN, lw=1.2, ls="--")
    a2.set_ylim(-2, 3)
    a2.set_xlabel(r"$t = y\,f(\boldsymbol{x})$")
    a2.set_title(r"pull $-\phi^{\prime}(t)$")
    for ax in (a1, a2):
        ax.axvline(1, color=GREY_RULE, lw=0.6)
        ax.set_xlim(-1.5, 3)
        ax.grid(lw=0.3)
    h = [plt.Line2D([], [], color=c, lw=1.3, ls=s) for c, s in
         ((BLUE, "-"), (ORANGE, "-"), (GREEN, "--"), (GREY, ":"))]
    fig.legend(h, ["hinge", "logistic", "squared", "0-1"], ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch08-losses")


def ch08_fan():
    """Subgradients at a kink, and a nonzero one at the minimiser."""
    from ch08_hinge import objective, subgradient
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.7, 2.3))
    t = np.linspace(-0.5, 2.5, 301)
    a1.plot(t, np.maximum(0, 1 - t), color=BLUE, lw=1.6, zorder=3)
    for s, c in ((0, GREEN), (-0.25, GREY), (-0.5, ORANGE), (-0.75, GREY),
                 (-1, RED)):
        a1.plot(t, s * (t - 1), color=c, lw=0.8, ls="--")
    a1.scatter([1], [0], s=16, color=BLUE, zorder=4)
    a1.set_xlim(-0.5, 2.5)
    a1.set_ylim(-1, 1.6)
    a1.set_xlabel(r"$t$")
    a1.set_title(r"$h(t)$, slopes in $[-1, 0]$ at $t = 1$")
    w, b, C = np.array([0.4, 0.5]), -2.9, 0.5
    gw, gb, _ = subgradient(w, b, C)
    s = np.linspace(-0.2, 0.2, 401)
    J = [objective(w - u * gw, b - u * gb, C) for u in s]
    a2.plot(s, J, color=BLUE, lw=1.6)
    a2.scatter([0], [objective(w, b, C)], s=16, color=BLUE, zorder=4)
    a2.set_xlim(-0.2, 0.2)
    a2.set_xlabel(r"step length $\eta$")
    a2.set_title(r"$J$ along $-\boldsymbol{g}$ from the minimiser")
    for ax in (a1, a2):
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch08-fan")


def ch08_descent():
    """The path over the level sets, and the gap against k."""
    from ch08_hinge import X9, descend, hinge, subgradient, y9
    C = 0.5
    w, b, path = np.zeros(2), 0.0, []
    for k in range(3000):
        path.append(w.copy())
        gw, gb, _ = subgradient(w, b, C)
        step = 0.2 / np.sqrt(k + 1)
        w, b = w - step * gw, b - step * gb
    path = np.array(path)
    u = v = np.linspace(0, 0.8, 201)
    Z = np.zeros((len(v), len(u)))
    for i, vi in enumerate(v):
        for j, ui in enumerate(u):
            s = X9 @ np.array([ui, vi])
            bs = y9 - s                    # J(b) is lowest at a kink
            r = hinge(y9[:, None] * (s[:, None] + bs)).sum(axis=0)
            Z[i, j] = (ui ** 2 + vi ** 2) / 2 + C * r.min()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.7, 2.5))
    a1.contour(u, v, Z, levels=[2.2, 2.4, 2.7, 3.1, 3.6],
               colors=GREY, linewidths=0.5)
    a1.plot(*path.T, color=BLUE, lw=0.35)
    a1.scatter([0.4], [0.5], s=26, color=RED, marker="x", zorder=5,
               lw=1.1)
    a1.set_xlim(0, 0.8)
    a1.set_ylim(0, 0.8)
    a1.set_xlabel(r"$w_1$")
    a1.set_ylabel(r"$w_2$")
    a1.set_aspect("equal")
    a1.set_title("first 3000 steps")
    marks = np.unique(np.logspace(0, 5, 41).astype(int))
    _, _, trace = descend(C, steps=100_000, marks=set(marks))
    k = np.array([r[0] for r in trace])
    last = np.array([r[1] for r in trace]) - 2.155
    top_ = np.array([r[2] for r in trace]) - 2.155
    a2.loglog(k, np.maximum(last, 1e-6), color=GREY_RULE, lw=0.8)
    a2.loglog(k, top_, color=BLUE, lw=1.4)
    a2.loglog(k, 3 / np.sqrt(k), color=ORANGE, lw=0.8, ls="--")
    a2.set_xlabel(r"step $k$")
    a2.set_title(r"gap $J - J^{\ast}$")
    a2.grid(lw=0.3)
    h = [plt.Line2D([], [], color=c, lw=1.1, ls=s) for c, s in
         ((GREY_RULE, "-"), (BLUE, "-"), (ORANGE, "--"))]
    fig.legend(h, ["current", "best so far", r"$3/\sqrt{k}$"], ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch08-descent")


def _bound(ax, w, b, x, **kw):
    ax.plot(x, -(b + w[0] * x) / w[1], **kw)


def ch08_lines():
    """Three lines on the nine; a far machine moves only one of them."""
    from ch08_hinge import (X, X9, fit_logistic, fit_squared, soft_line,
                            y, y9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.7, 2.75),
                                 gridspec_kw={"width_ratios": [1, 1.45]})
    x = np.linspace(-1, 18, 2)
    _nine(a1, 12)
    w, b = soft_line(0.5, X9, y9)[:2]
    _bound(a1, w, b, x, color=BLUE, lw=1.4)
    _bound(a1, *fit_logistic(0.5), x, color=ORANGE, lw=1.1)
    _bound(a1, *fit_squared(0.5), x, color=GREEN, lw=1.1, ls="--")
    a1.set_xlim(0, 7)
    a1.set_ylim(0, 8)
    a1.set_title(r"nine, $C = 1/2$")
    far = np.array([16.0, 6.0])
    Xo, yo = np.vstack([X, far]), np.append(y, 1.0)
    a2.scatter(*X[y < 0].T, s=12, color=BLUE, zorder=4)
    a2.scatter(*Xo[yo > 0].T, s=12, color=ORANGE, marker="s", zorder=4)
    w, b = soft_line(1, Xo, yo)[:2]
    _bound(a2, w, b, x, color=BLUE, lw=1.4)
    _bound(a2, *fit_logistic(1, Xo, yo), x, color=ORANGE, lw=1.1)
    _bound(a2, *fit_squared(1, Xo, yo), x, color=GREEN, lw=1.1, ls="--")
    _bound(a2, *fit_squared(1, X, y), x, color=GREEN, lw=0.7, ls=":")
    a2.annotate("far machine", far, (10.2, 7.4), color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    a2.set_xlim(0, 17)
    a2.set_ylim(0, 8)
    a2.set_title(r"eight and one far, $C = 1$")
    for ax in (a1, a2):
        ax.set_xlabel(r"$x_1$")
        ax.grid(lw=0.3)
    a1.set_ylabel(r"$x_2$")
    h = [plt.Line2D([], [], color=c, lw=1.2, ls=s) for c, s in
         ((BLUE, "-"), (ORANGE, "-"), (GREEN, "--"), (GREEN, ":"))]
    fig.legend(h, ["hinge", "logistic", "squared",
                   "squared without far"], ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0,
               handlelength=1.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch08-lines")


def ch08_sweep():
    """Training and test error against C on small overlapping samples."""
    from ch08_hinge import sweep
    Cs = np.logspace(-2, 3, 11)
    train, test = sweep(Cs)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.semilogx(Cs, 100 * test, color=RED, lw=1.4, marker="o", ms=2.5)
    ax.semilogx(Cs, 100 * train, color=BLUE, lw=1.4, marker="o", ms=2.5)
    ax.text(0.4, 100 * test[6] + 1.2, "test", color=RED)
    ax.text(0.4, 100 * train[6] - 2.6, "training", color=BLUE)
    ax.set_ylim(0, 26)
    ax.set_xlabel(r"$C$")
    ax.set_ylabel("error (%)")
    ax.grid(lw=0.3)
    save(fig, "ch08-sweep")


def _grid(xlim, ylim, n=301):
    u, v = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    return u, v, np.column_stack([u.ravel(), v.ravel()])


def _classes(ax, X, y, s=12):
    ax.scatter(*X[y < 0].T, s=s, color=BLUE, zorder=4)
    ax.scatter(*X[y > 0].T, s=s, color=ORANGE, marker="s", zorder=4)


def ch09_lift():
    """XOR lifted by x1 x2, and rings straightened by squares."""
    from ch09_kernel import XOR, y_xor, rings
    fig = plt.figure(figsize=(4.6, 1.95))
    ax = fig.add_axes([-0.02, 0.03, 0.36, 0.95], projection="3d")
    lift = np.column_stack([XOR, XOR[:, 0] * XOR[:, 1]])
    u, v = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    ax.plot_surface(u, v, (u + v - 0.5) / 2, color=GREEN, alpha=0.25,
                    lw=0, shade=False)
    ax.scatter(*lift[y_xor < 0].T, s=14, color=BLUE, depthshade=False)
    ax.scatter(*lift[y_xor > 0].T, s=14, color=ORANGE, marker="s",
               depthshade=False)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.tick_params(labelsize=8, pad=-3)
    ax.set_xlabel(r"$x_1$", labelpad=-9)
    ax.set_ylabel(r"$x_2$", labelpad=-9)
    ax.view_init(18, -62)
    ax.set_title(r"XOR, height $x_1x_2$", pad=-2)
    A, t = rings()
    a1 = fig.add_axes([0.39, 0.20, 0.25, 0.66])
    _classes(a1, A, t, 6)
    th = np.linspace(0, 2 * np.pi, 200)
    a1.plot(1.5 * np.cos(th), 1.5 * np.sin(th), color=GREEN, lw=1.1)
    a1.set_aspect("equal")
    a1.set_xticks([-2, 0, 2])
    a1.set_yticks([-2, 0, 2])
    a1.set_xlabel(r"$x_1$")
    a1.set_ylabel(r"$x_2$", labelpad=-3)
    a1.set_title("rings")
    a2 = fig.add_axes([0.745, 0.20, 0.25, 0.66])
    _classes(a2, A ** 2, t, 6)
    a2.plot([0, 2.25], [2.25, 0], color=GREEN, lw=1.1)
    a2.set_xlim(0, 4.6)
    a2.set_ylim(0, 4.6)
    a2.set_aspect("equal")
    a2.set_xlabel(r"$x_1^2$")
    a2.set_ylabel(r"$x_2^2$", labelpad=1)
    a2.set_title("rings, squared")
    for ax2 in (a1, a2):
        ax2.grid(lw=0.3)
    save(fig, "ch09-lift")


def ch09_square():
    """Two routes from a pair of points to one number."""
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.35", fc="white", ec=GREY_RULE, lw=0.8)
    hot = dict(boxstyle="round,pad=0.35", fc="#F6EBDD", ec=ORANGE, lw=0.8)
    kw = dict(ha="center", va="center", fontsize=9)
    ax.text(1.7, 3.3, r"$\boldsymbol{x}$, $\boldsymbol{z}$ in $\mathbb{R}^2$"
            "\n" r"$(1,\,2)$, $(3,\,1)$", bbox=box, **kw)
    ax.text(1.7, 0.75, r"$\boldsymbol{\phi}(\boldsymbol{x})$, $\boldsymbol{\phi}(\boldsymbol{z})$"
            r" in $\mathbb{R}^3$" "\n"
            r"$(1,\,2\sqrt{2},\,4)$, $(9,\,3\sqrt{2},\,1)$", bbox=box, **kw)
    ax.text(8.3, 3.3, r"$\boldsymbol{x}^{\top}\boldsymbol{z} = 5$",
            bbox=box, **kw)
    ax.text(8.3, 0.75, r"$K = 25$", bbox=hot, **kw)
    arrow = dict(arrowstyle="-|>", color=GREY, lw=0.9, shrinkA=0, shrinkB=0)
    ax.annotate("", (6.95, 3.3), (3.15, 3.3), arrowprops=arrow)
    ax.annotate("", (7.35, 0.75), (4.3, 0.75), arrowprops=arrow)
    ax.annotate("", (1.7, 1.45), (1.7, 2.55), arrowprops=arrow)
    ax.annotate("", (8.3, 1.2), (8.3, 2.85), arrowprops=arrow)
    ax.text(5.05, 3.6, "inner product, 2 products", ha="center",
            fontsize=8.5, color=GREY)
    ax.text(5.8, 1.05, "inner product,", ha="center", fontsize=8.5)
    ax.text(5.8, 0.42, "3 products", ha="center", fontsize=8.5)
    ax.text(1.9, 2.0, r"map $\boldsymbol{\phi}$, 3 entries each", ha="left", fontsize=8.5)
    ax.text(8.15, 2.0, "square,\n1 product", ha="right", va="center",
            fontsize=8.5)
    save(fig, "ch09-square")


def ch09_cost():
    """Length of the explicit map against the cost of the kernel."""
    from math import comb
    ns = np.unique(np.logspace(0.31, 4, 40).astype(int))
    fig, ax = plt.subplots(figsize=(3.6, 2.35))
    for d, c in ((5, RED), (3, ORANGE), (2, GREEN)):
        ax.loglog(ns, [comb(int(n) + d, d) for n in ns], color=c, lw=1.3)
        ax.text(1.2e4, comb(10000 + d, d), rf"$d = {d}$", color=c,
                va="center")
    ax.loglog(ns, ns + 3, color=BLUE, lw=1.3)
    ax.text(1.2e4, 1.1e4, "kernel", color=BLUE, va="center")
    ax.set_xlim(2, 1e4)
    ax.set_ylim(1, 1e19)
    ax.set_yticks([1e0, 1e6, 1e12, 1e18])
    ax.set_xlabel(r"input features $n$")
    ax.set_ylabel("products per pair")
    ax.grid(lw=0.3)
    save(fig, "ch09-cost")


def ch09_xor():
    """The XOR solution in the two placements of the square."""
    from ch09_kernel import XOR, y_xor, fit_kernel, quad1, score
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.45))
    for ax, P, lim in ((axes[0], XOR, (-0.6, 1.6)),
                       (axes[1], 2 * XOR - 1, (-2.2, 2.2))):
        a, b = fit_kernel(quad1(P, P), y_xor)
        u, v, G = _grid(lim, lim)
        F = score(G, P, y_xor, a, b).reshape(u.shape)
        ax.contourf(u, v, F, levels=[-1e9, 0, 1e9],
                    colors=["#E3E8F0", "#F6EBDD"])
        ax.contour(u, v, F, levels=[0], colors=GREEN, linewidths=1.4)
        ax.contour(u, v, F, levels=[-1, 1], colors=GREEN, linewidths=0.7,
                   linestyles="--")
        _classes(ax, P, y_xor, 20)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title("corners 0 and 1")
    axes[1].set_title(r"corners $-1$ and $1$")
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch09-xor")


def ch09_nine():
    """Surfaces of growing degree on the nine machines."""
    from ch07_soft import X9, y9
    from ch09_kernel import poly_svm
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.95), sharey=True)
    u, v, G = _grid((0, 8), (0, 8))
    for ax, d in zip(axes, (1, 2, 3)):
        c = poly_svm(X9 / 10, y9, d)
        F = c.decision_function(G / 10).reshape(u.shape)
        ax.contourf(u, v, F, levels=[-1e12, 0, 1e12],
                    colors=["#E3E8F0", "#F6EBDD"])
        ax.contour(u, v, F, levels=[0], colors=GREEN, linewidths=1.3)
        _classes(ax, X9, y9, 11)
        ax.plot(2, 2, "s", ms=6.5, mfc="none", mec=RED, mew=0.9, zorder=5)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0, 4, 8])
        ax.set_xlabel(r"$x_1$")
        ax.set_title(rf"degree {d}")
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch09-nine")


def ch09_perceptron():
    """The kernel perceptron on the rings after a growing mistake count."""
    from ch09_kernel import rings, quad1, kernel_perceptron, poly_svm
    A, t = rings()
    K = quad1(A, A)
    log = kernel_perceptron(K, t)[2]
    u, v, G = _grid((-2.6, 2.6), (-2.6, 2.6))
    KG = quad1(G, A)
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.95), sharey=True)
    for ax, k in zip(axes, (5, 8, len(log))):
        a = np.bincount(log[:k], minlength=len(t)).astype(float)
        b = sum(t[i] for i in log[:k])
        F = (KG @ (a * t) + b).reshape(u.shape)
        ax.contourf(u, v, F, levels=[-1e12, 0, 1e12],
                    colors=["#E3E8F0", "#F6EBDD"])
        ax.contour(u, v, F, levels=[0], colors=GREEN, linewidths=1.3)
        _classes(ax, A, t, 4)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r"$x_1$")
        ax.set_title(f"{k} mistakes")
    c = poly_svm(A, t, 2)
    F = c.decision_function(G).reshape(u.shape)
    axes[2].contour(u, v, F, levels=[0], colors=RED, linewidths=0.9,
                    linestyles="--")
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch09-perceptron")


def ch10_spectra():
    """Eigenvalues of four Gram matrices on the eight machines."""
    from ch10_mercer import linear, near, sigmoid, quad1
    from machines import X
    cases = [(r"$\boldsymbol{x}^{\top}\boldsymbol{z}$", linear(X, X)),
             (r"$(1+\boldsymbol{x}^{\top}\boldsymbol{z})^2/100$",
              quad1(X, X) / 100),
             (r"near, $r=2$", near(X, X, 2.0)),
             (r"$\tanh(0.1\,\boldsymbol{x}^{\top}\boldsymbol{z})$",
              sigmoid(X, X, 0.1, 0.0))]
    fig, axes = plt.subplots(1, 4, figsize=(4.9, 1.75))
    for ax, (name, K) in zip(axes, cases):
        e = np.linalg.eigvalsh(K)
        s = np.sign(e) * np.abs(e) ** 0.5
        ax.bar(range(1, 9), s, color=[RED if v < -1e-9 else BLUE for v in e],
               width=0.7)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_title(name)
        ax.set_xticks([1, 8])
        ax.set_xlabel("rank")
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.15 * hi), hi)
    axes[0].set_ylabel(r"sign$\cdot\sqrt{|\lambda|}$")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch10-spectra")


def ch10_saddle():
    """The dual on three points: a bowl, then a saddle."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3), sharey=True)
    u, v = np.meshgrid(np.linspace(0, 4.5, 301), np.linspace(0, 4.5, 301))
    p = u + v
    bowl = 2 * p - (u**2 + v**2) / 2
    sad = 2 * p + u * v
    for ax, D, name, lv, spots in [(axes[0], bowl, r"$\rho=\frac{1}{2}$: a peak",
                             [1, 2, 3], [(3.73, 0.27), (3.41, 0.59), (3.0, 1.0)]),
                            (axes[1], sad, r"$\rho=1$: no ceiling",
                             [4, 10, 16, 22, 28],
                             [(1.4, 0.2), (2.33, 1.13), (3.12, 1.92), (3.8, 2.6),
                              (4.1, 3.2)])]:
        cs = ax.contour(u, v, D, levels=lv, colors=BLUE, linewidths=0.8)
        ax.clabel(cs, fontsize=8, fmt="%g", inline_spacing=2, manual=spots)
        ax.set_aspect("equal")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 4.5)
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([0, 2, 4])
        ax.set_xlabel(r"$\alpha_1$")
        ax.set_title(name)
    axes[0].plot(2, 2, "o", ms=5, color=ORANGE, zorder=5)
    axes[0].set_ylabel(r"$\alpha_3$")
    axes[1].annotate("", xy=(4.3, 4.3), xytext=(0.3, 0.3),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch10-saddle")


def ch10_build():
    """Ramps, their sum, and the value read as an area under the slope."""
    t = np.linspace(0, 1, 801)
    p = np.array([0.2, 0.5, 0.8])
    c = np.array([8 / 3, -2.0, 4 / 3])
    r = np.minimum.outer(t, p)
    f = r @ c
    s = np.array([(c * (u < p)).sum() for u in t])
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.9))
    for k, col in enumerate((BLUE, ORANGE, GREEN)):
        axes[0].plot(t, r[:, k], color=col, lw=1.2)
        axes[0].text(0.98, p[k] + 0.02, rf"$x_i={p[k]:g}$",
                     color=col, fontsize=8, va="bottom", ha="right")
    axes[0].set_title(r"the ramps $K(\cdot, x_i)$")
    axes[0].set_ylim(0, 1)
    axes[1].plot(t, f, color=GREEN, lw=1.4)
    x0 = 0.6
    y0 = np.interp(x0, t, f)
    axes[1].plot([x0, x0], [0, y0], color=GREY, lw=0.7, ls="--")
    axes[1].plot(x0, y0, "o", ms=4, color=ORANGE)
    axes[1].text(0.03, 0.66, rf"$f({x0})={y0:.2f}$", fontsize=8.5,
                 color=GREY, va="top")
    axes[1].set_ylim(0, 0.7)
    axes[1].set_title(r"their combination $f$")
    axes[2].step(t, s, where="post", color=BLUE, lw=1.2)
    m = t <= x0
    axes[2].fill_between(t[m], 0, s[m], color=ORANGE, alpha=0.3, lw=0)
    axes[2].axhline(0, color=GREY, lw=0.6)
    axes[2].set_title(r"the slope $f\,'$")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"$x$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch10-build")


def ch10_mercer():
    """Eigenfunctions of min(x, z), partial sums, and Gram eigenvalues."""
    from ch10_mercer import low, low_terms
    t = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 3, figsize=(4.9, 1.95))
    for k, col in zip((1, 2, 3), (BLUE, ORANGE, GREEN)):
        axes[0].plot(t, np.sqrt(2) * np.sin((k - 0.5) * np.pi * t),
                     color=col, lw=1.1, label=rf"$e_{k}$")
    axes[0].axhline(0, color=GREY, lw=0.6)
    axes[0].set_title("eigenfunctions")
    axes[0].set_xlabel(r"$x$")
    axes[0].legend(fontsize=8, loc="lower left", handlelength=1.2,
                   borderpad=0.1, labelspacing=0.15)
    axes[0].set_ylim(-2.6, 1.7)
    z = np.array([0.5])
    axes[1].plot(t, low(t, z), color=GREY, lw=2.2, alpha=0.45)
    for k, col, ls in ((1, BLUE, "-"), (3, ORANGE, "-"), (10, GREEN, "-")):
        axes[1].plot(t, low_terms(t, z, k), color=col, lw=0.9, ls=ls,
                     label=f"{k}")
    axes[1].set_title(r"$K(\cdot, \frac{1}{2})$ from terms")
    axes[1].set_xlabel(r"$x$")
    axes[1].legend(title="terms", fontsize=8, title_fontsize=8, loc="lower right",
                   handlelength=1.2, borderpad=0.1, labelspacing=0.15)
    axes[1].set_ylim(0, 0.62)
    kk = np.arange(1, 31)
    exact = 1 / ((kk - 0.5) * np.pi) ** 2
    axes[2].loglog(kk, exact, color=GREY, lw=1.1, label="exact")
    for m, col, mk in ((20, BLUE, "o"), (200, ORANGE, "s")):
        g = (np.arange(m) + 0.5) / m
        e = np.linalg.eigvalsh(low(g, g))[::-1][:30] / m
        axes[2].loglog(np.arange(1, len(e) + 1), e, mk, ms=2.4, color=col,
                       label=f"$m={m}$")
    axes[2].set_title(r"$\lambda_k$ and Gram$/m$")
    axes[2].set_xlabel(r"$k$")
    axes[2].legend(fontsize=8, loc="lower left", handlelength=1.0,
                   borderpad=0.1, labelspacing=0.15)
    for ax in axes[:2]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch10-mercer")


def ch10_sigmoid():
    """How far below zero the sigmoid Gram matrix reaches."""
    from ch10_mercer import sigmoid
    from machines import X
    us = np.linspace(-3, 0.5, 141)
    cs = np.linspace(-2, 2, 161)
    R = np.empty((len(cs), len(us)))
    for i, c in enumerate(cs):
        for j, u in enumerate(us):
            e = np.linalg.eigvalsh(sigmoid(X, X, 10.0**u, c))
            R[i, j] = np.log10(-e[0] / np.abs(e).max())
    fig, ax = plt.subplots(figsize=(3.9, 2.25))
    im = ax.pcolormesh(10.0**us, cs, R, cmap="YlOrRd", shading="auto",
                       vmin=-6, vmax=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$c$")
    ax.plot([0.1, 0.03], [0, -0.5], "o", ms=4, mfc="white", mec=BLUE, mew=0.9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ of $|\lambda_{\min}|/|\lambda|_{\max}$", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "ch10-sigmoid")


def ch10_representer():
    """Projection onto the span, and the interpolant of least norm."""
    from ch10_mercer import tent
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.1),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.fill([0.1, 2.9, 3.6, 0.8], [0.1, 0.1, 1.0, 1.0], color="#E3E8F0", lw=0)
    ax.text(0.42, 0.2, r"span of the $K(\cdot, x_i)$", fontsize=8.5, color=BLUE,
            va="bottom")
    o = np.array([1.9, 0.62])
    top = np.array([1.9, 2.05])
    ax.annotate("", xy=o, xytext=(0.75, 0.45),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.annotate("", xy=top, xytext=(0.75, 0.45),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.plot([o[0], top[0]], [o[1], top[1]], color=RED, lw=1.1, ls="--")
    ax.plot([o[0], o[0] + 0.13, o[0] + 0.13], [o[1] + 0.13, o[1] + 0.13, o[1]],
            color=GREY, lw=0.5)
    ax.text(0.95, 1.4, r"$f$", color=ORANGE, fontsize=10)
    ax.text(1.4, 0.72, r"$f_{\parallel}$", color=GREEN, fontsize=10)
    ax.text(2.0, 1.35, r"$f_{\perp}$", color=RED, fontsize=10)
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax = axes[1]
    t = np.linspace(0, 1, 601)
    f, g = tent(t)
    ax.plot(t, f + g, color=ORANGE, lw=1.1, label=r"$\Vert f\Vert^2=1.80$")
    ax.plot(t, f, color=GREEN, lw=1.4, label=r"$\Vert f\Vert^2=1.47$")
    ax.plot([0.2, 0.5, 0.8], [0.4, 0.2, 0.6], "o", ms=4, color=BLUE, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"$x$")
    ax.legend(fontsize=8, loc="upper left", handlelength=1.3, borderpad=0.1)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch10-representer")


def ch11_terms():
    """Share of each degree in the polynomial kernel, and its boundary."""
    from ch11_rbf import term_weights
    from ch09_kernel import rings
    from sklearn.svm import SVC
    fig, axes = plt.subplots(1, 4, figsize=(4.9, 1.7),
                             gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    ax = axes[0]
    d = 4
    for j, (c, col) in enumerate(((0.25, ORANGE), (1, GREEN), (4, BLUE))):
        w = np.array(term_weights(c, d), float)
        ax.bar(np.arange(d + 1) + (j - 1) * 0.27, w / w.sum(), width=0.25,
               color=col, label=f"$c={c}$")
    ax.set_xticks(range(d + 1))
    ax.set_xlabel("degree $k$")
    ax.set_ylabel("share")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8, loc="upper center", ncol=1, handlelength=0.9,
              borderpad=0.1, labelspacing=0.15, columnspacing=0.5)
    A, t = rings()
    Z = (A + [1.0, 0.5]) / 3
    u, v, G = _grid((-0.5, 1.5), (-0.8, 1.2))
    for ax, c in zip(axes[1:], (0, 1, 100)):
        s = SVC(kernel="poly", degree=4, gamma=1, coef0=c, C=100, tol=1e-4,
                max_iter=3000000).fit(Z, t)
        ax.contourf(u, v, s.decision_function(G).reshape(u.shape), levels=[-1e9, 0, 1e9],
                    colors=["#E3E9F2", "#F6E6D6"])
        ax.contour(u, v, s.decision_function(G).reshape(u.shape), levels=[0],
                   colors=GREEN, linewidths=1.1)
        _classes(ax, Z, t, s=4)
        ax.set_title(f"$c={c}$")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch11-terms")


def ch11_series():
    """Entries of the Gaussian map, and partial sums of the kernel."""
    from ch11_rbf import phi_k, series
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 1.85))
    x = np.linspace(-4, 4, 801)
    ax = axes[0]
    for k, col in ((0, BLUE), (1, GREEN), (2, ORANGE), (6, RED)):
        ax.plot(x, phi_k(x, k), color=col, lw=1.1, label=f"$k={k}$")
    ax.set_xlabel(r"$x$")
    ax.set_title(r"entries $\phi_k(x)$")
    ax.set_ylim(-0.75, 1.75)
    ax.legend(fontsize=8, loc="upper center", ncol=2, handlelength=1.1,
              borderpad=0.1, labelspacing=0.15, columnspacing=0.8)
    ax = axes[1]
    ax.plot(x, np.exp(-0.5 * (x - 1) ** 2), color=GREY, lw=2.2, alpha=0.55,
            label="exact")
    for t, col, ls in ((2, RED, "--"), (4, ORANGE, "--"), (8, BLUE, "-")):
        ax.plot(x, series(1.0, x, 0.5, t), color=col, lw=1, ls=ls,
                label=f"{t} terms")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel(r"$z$")
    ax.set_title(r"$K(1,z)$ from its series")
    ax.legend(fontsize=8, loc="upper center", ncol=2, handlelength=1.5,
              borderpad=0.1, labelspacing=0.15, columnspacing=0.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch11-series")


def ch11_hills():
    """Signed hills adding up to the score, wide and narrow."""
    from math import log
    from ch11_rbf import f_hand, fit_rbf, weights
    x = np.linspace(-3.2, 3.2, 801)
    pts = np.array([-1.0, 0.0, 1.0])
    t = np.array([-1.0, 1.0, -1.0])
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.0), sharey=True)
    for ax, g, name in zip(axes, (log(2), 6.0),
                           (r"$\gamma=\ln 2$", r"$\gamma=6$")):
        s = fit_rbf(pts[:, None], t, g)
        a, b = weights(s), s.intercept_[0]
        f = b + 0
        for p, ai, ti in zip(pts, a, t):
            h = ai * ti * np.exp(-g * (x - p) ** 2)
            f = f + h
            ax.plot(x, h, color=ORANGE if ti > 0 else BLUE, lw=0.9, ls="--")
        ax.plot(x, f, color=GREEN, lw=1.5)
        ax.axhline(b, color=GREY, lw=0.6, ls=":")
        ax.axhline(0, color=GREY_RULE, lw=0.6)
        ax.plot(pts, t, "o", ms=0)
        ax.scatter(pts[t < 0], t[t < 0], s=14, color=BLUE, zorder=4)
        ax.scatter(pts[t > 0], t[t > 0], s=14, color=ORANGE, marker="s",
                   zorder=4)
        ax.set_title(name)
        ax.set_xlabel(r"$x$")
        ax.set_xlim(-3.2, 3.2)
        ax.text(2.6, b + (0.13 if g < 1 else -0.62), r"$b$", fontsize=9)
    axes[0].text(-0.8, 0.35, r"$f$", color=GREEN, fontsize=10)
    axes[0].set_ylim(-2.1, 4.1)
    axes[0].set_ylabel("score")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch11-hills")


def ch11_gamma():
    """The eight machines at four widths, hard margin."""
    from machines import X, y
    from ch11_rbf import fit_rbf
    fig, axes = plt.subplots(1, 4, figsize=(4.9, 1.6), sharex=True,
                             sharey=True)
    u, v, G = _grid((0, 8), (0, 8), 401)
    for ax, g in zip(axes, (0.01, 0.1, 1, 10)):
        s = fit_rbf(X, y, g, 1000)
        f = s.decision_function(G).reshape(u.shape)
        ax.contourf(u, v, f, levels=[-1e9, 0, 1e9],
                    colors=["#E3E9F2", "#F6E6D6"])
        ax.contour(u, v, f, levels=[0], colors=GREEN, linewidths=1.1)
        _classes(ax, X, y, s=8)
        ax.scatter(*X[s.support_].T, s=34, facecolors="none", edgecolors=RED,
                   linewidths=0.7, zorder=5)
        ax.set_title(rf"$\gamma={g}$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch11-gamma")


def ch11_extremes():
    """Spectrum of the Gram matrix, and accuracy along gamma."""
    from ch11_rbf import gauss, noisy
    from ch09_kernel import rings
    from sklearn.svm import SVC
    A, t = noisy()
    Ate, tte = rings(4000, 7, noise=0.35)
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 1.9))
    ax = axes[0]
    for g, col in ((0.01, BLUE), (1, GREEN), (100, ORANGE)):
        e = np.linalg.eigvalsh(gauss(A, A, g))[::-1]
        ax.semilogy(range(1, 121), np.maximum(e, 1e-16), color=col, lw=1.2)
    ax.text(70, 40, r"$\gamma=100$", color=ORANGE, fontsize=8.5)
    ax.text(82, 1e-3, r"$\gamma=1$", color=GREEN, fontsize=8.5)
    ax.text(36, 3e-12, r"$\gamma=0.01$", color=BLUE, fontsize=8.5)
    ax.set_ylim(1e-16, 1e4)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
    ax.set_xlabel("rank")
    ax.set_title("Gram eigenvalues")
    ax = axes[1]
    gs = np.logspace(-3, 3, 25)
    tr, te = [], []
    for g in gs:
        s = SVC(gamma=g, C=1).fit(A, t)
        tr.append(s.score(A, t))
        te.append(s.score(Ate, tte))
    ax.semilogx(gs, tr, color=ORANGE, lw=1.2, label="train")
    ax.semilogx(gs, te, color=BLUE, lw=1.2, label="test")
    ax.set_xlabel(r"$\gamma$")
    ax.set_title(r"accuracy at $C=1$")
    ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
    ax.set_ylim(0.45, 1.04)
    ax.legend(fontsize=8, loc="lower center", handlelength=1.3, borderpad=0.1,
              labelspacing=0.15)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch11-extremes")


def ch11_grid():
    """Boundaries on the noisy rings across gamma and C."""
    from ch11_rbf import noisy
    from sklearn.svm import SVC
    A, t = noisy()
    u, v, G = _grid((-3.2, 3.2), (-3.2, 3.2), 251)
    gs, Cs = (0.01, 1, 100), (0.1, 10, 1000)
    fig, axes = plt.subplots(3, 3, figsize=(4.3, 4.3), sharex=True,
                             sharey=True)
    for i, g in enumerate(gs):
        for j, C in enumerate(Cs):
            ax = axes[i, j]
            s = SVC(gamma=g, C=C).fit(A, t)
            f = s.decision_function(G).reshape(u.shape)
            lo, hi = min(f.min(), -1e-6), max(f.max(), 1e-6)
            ax.contourf(u, v, f, levels=[lo, 0, hi],
                        colors=["#E3E9F2", "#F6E6D6"])
            if lo < -1e-5 and hi > 1e-5:
                ax.contour(u, v, f, levels=[0], colors=GREEN, linewidths=0.9)
            _classes(ax, A, t, s=3)
            ax.text(-3.0, 2.45, f"{len(s.support_)} sv", fontsize=8,
                    color=GREY, bbox=dict(fc="white", ec="none", pad=0.8,
                                          alpha=0.85))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(f"$C={C}$")
            if j == 0:
                ax.set_ylabel(rf"$\gamma={g}$", color=BLUE, fontsize=9)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    save(fig, "ch11-grid")


def ch11_ridge():
    """Test accuracy over gamma and C on the noisy rings."""
    from ch11_rbf import noisy, surface, GAMMAS, CS
    from ch09_kernel import rings
    A, t = noisy()
    S = surface(A, t, *rings(4000, 7, noise=0.35))
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.imshow(S, cmap="Blues", vmin=0.45, vmax=1.0, aspect="auto")
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, f"{S[i, j]:.2f}"[1:], ha="center", va="center",
                    fontsize=8, color="white" if S[i, j] > 0.78 else BLUE)
    ax.set_xticks(range(len(CS)))
    ax.set_xticklabels([f"{c:g}" for c in CS])
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([f"{g:g}" for g in GAMMAS])
    ax.set_xlabel(r"$C$")
    ax.set_ylabel(r"$\gamma$")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, "ch11-ridge")


def ch12_segment():
    """The two segments in the square, and D along step 5 of the hand run."""
    fig, axes = plt.subplots(1, 3, figsize=(4.7, 1.75),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    C = 1.0
    for ax, title, ai, aj, sgn in ((axes[0], r"$y_i \neq y_j$", 0.3, 0.55, 1),
                                   (axes[1], r"$y_i = y_j$", 0.65, 0.65, -1)):
        ax.add_patch(plt.Rectangle((0, 0), C, C, fc="#E8EDF5", ec=BLUE, lw=0.9))
        if sgn > 0:
            d = aj - ai
            p, q = (0, d), (C - d, C)
        else:
            s = ai + aj
            p, q = (s - C, C), (C, s - C)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, lw=1.6)
        ax.plot([ai], [aj], "o", color=BLUE, ms=3.5)
        ax.plot([p[0]] * 2, [0, p[1]], color=GREY, lw=0.5, ls=":")
        ax.plot([q[0]] * 2, [0, q[1]], color=GREY, lw=0.5, ls=":")
        lo, hi = sorted((p[1], q[1]))
        ax.annotate(r"$L$", (-0.02, lo), xytext=(-0.17, lo), va="center",
                    color=GREEN, annotation_clip=False, fontsize=8.5)
        ax.annotate(r"$H$", (-0.02, hi), xytext=(-0.17, hi), va="center",
                    color=GREEN, annotation_clip=False, fontsize=8.5)
        ax.plot([0, 0], [lo, hi], color=GREEN, lw=2.2, clip_on=False)
        ax.set_xlim(0, C)
        ax.set_ylim(0, C)
        ax.set_aspect("equal")
        ax.set_xticks([0, C], ["0", r"$C$"])
        ax.set_yticks([])
        ax.set_xlabel(r"$\alpha_i$", labelpad=-5)
        ax.set_title(title, pad=3)
        ax.tick_params(length=0)
    ax = axes[2]
    t = np.linspace(-0.02, 0.2, 200)
    f = 1.2 * t - 4.5 * t ** 2
    ax.plot(t, f, color=BLUE, lw=1.3)
    ax.axvline(2 / 25, color=GREEN, lw=0.9)
    ax.axvspan(2 / 25, 0.2, color=GREY_RULE, alpha=0.45, lw=0)
    ax.plot([2 / 15], [1.2 * 2 / 15 - 4.5 * (2 / 15) ** 2], "o", mfc="white",
            mec=RED, ms=4)
    ax.plot([2 / 25], [1.2 * 2 / 25 - 4.5 * (2 / 25) ** 2], "o", color=GREEN,
            ms=4)
    ax.text(0.138, 0.033, "free", color=RED, ha="left", fontsize=8.5)
    ax.text(0.074, 0.145, "taken", color=GREEN, ha="right", fontsize=8.5)
    ax.set_xlim(-0.02, 0.2)
    ax.set_ylim(-0.03, 0.2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"gain in $D$", labelpad=1)
    ax.set_title("step 5", pad=3)
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.1)
    save(fig, "ch12-segment")


def ch12_gap():
    """Scores of the points on one line: the violating pair and the stop."""
    from machines import X, y
    from ch12_smo import exact_run
    a, F, rows = exact_run("first")
    st = [[0.0] * 8] + [[float(v) for v in r[6]] for r in rows]
    G = X @ X.T
    fig, axes = plt.subplots(3, 1, figsize=(4.4, 2.5))
    for ax, k in zip(axes, (0, 2, 6)):
        al = np.array(st[k])
        F2 = (al * y) @ G - y
        up = (y > 0) | (al > 0)
        low = (y < 0) | (al > 0)
        lo, hi = F2[up].min(), F2[low].max()
        pad = 0.3 * max(hi - lo, 0.5) + 0.6
        ax.axhline(0, color=GREY_RULE, lw=0.8)
        for u, off, c, mk in ((up, 0.32, GREEN, "^"), (low, -0.32, ORANGE, "v")):
            ax.plot(F2[u], np.full(u.sum(), off), mk, color=c, ms=3.8, ls="")
        if hi > lo:
            ax.axvspan(lo, hi, color=RED, alpha=0.13, lw=0)
            ax.text((lo + hi) / 2, 0.62, f"gap {hi - lo:g}",
                    ha="center", color=RED, fontsize=8.5)
        else:
            ax.axvline(lo, color=BLUE, lw=0.9)
            ax.text(lo + 0.12, 0.62, r"gap 0, $-b = 11$", color=BLUE, fontsize=8.5)
        ax.set_ylim(-0.8, 1.15)
        ax.set_yticks([])
        ax.set_ylabel(f"step {k}", rotation=0, ha="right", va="center", labelpad=4)
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
    axes[0].set_xlim(-1.6, 1.6)
    axes[1].set_xlim(-0.2, 2.6)
    axes[2].set_xlim(3, 19)
    axes[2].set_xlabel(r"$F_k$  (triangles up: can rise, $U$; triangles down: can fall, $W$)")
    fig.tight_layout(h_pad=0.25)
    save(fig, "ch12-gap")


def ch12_path():
    """Lines after each hand step, and the dual value by step."""
    from machines import X, y
    from ch12_smo import exact_run
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    G = X @ X.T
    a, F, rows = exact_run("first")
    ax.scatter(*X[y < 0].T, marker="o", s=14, color=BLUE, zorder=3)
    ax.scatter(*X[y > 0].T, marker="s", s=14, color=ORANGE, zorder=3)
    xs = np.array([0, 7.5])
    for k, r in enumerate(rows, 1):
        al = np.array([float(v) for v in r[6]])
        w = (al * y) @ X
        Fk = (al * y) @ G - y
        up, low = (y > 0) | (al > 0), (y < 0) | (al > 0)
        b = -(Fk[up].min() + Fk[low].max()) / 2
        c = GREEN if k == 6 else GREY_RULE
        if abs(w[1]) > 1e-9:
            ax.plot(xs, (-b - w[0] * xs) / w[1], color=c,
                    lw=1.4 if k == 6 else 0.8, zorder=2 if k == 6 else 1)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", labelpad=1)
    ax.grid(lw=0.3)
    ax.set_aspect("equal")
    ax = axes[1]
    for rule, c, mk, lab in (("first", BLUE, "o", "first order"),
                             ("second", ORANGE, "s", "second order")):
        _, _, rows = exact_run(rule)
        D = [0.0]
        for r in rows:
            al = np.array([float(v) for v in r[6]])
            v = al * y
            D.append(al.sum() - v @ G @ v / 2)
        ax.plot(range(len(D)), D, color=c, marker=mk, ms=3, lw=1.1, label=lab)
    ax.axhline(2.5, color=GREEN, lw=0.8, ls="--")
    ax.text(3.6, 2.57, r"$D^* = 2.5$", color=GREEN, ha="right", fontsize=8.5)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$D$", labelpad=1)
    ax.set_xticks(range(0, 7))
    ax.set_ylim(0, 2.9)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.4, borderaxespad=0.1)
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch12-path")


def ch12_converge():
    """Gap by step on the clouds: first order against second order."""
    from ch12_smo import clouds, gaussian, smo
    A, t = clouds(1000)
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    for rule, c, lab in (("first", BLUE, "first order"),
                         ("second", ORANGE, "second order")):
        tr = []
        smo(A, t, 1.0, gaussian(0.5), rule, tol=1e-3, trace=tr)
        g = np.array([r[2] for r in tr])
        keep = np.minimum.accumulate(g)
        ax.plot(np.arange(1, len(g) + 1), g, color=c, lw=0.3, alpha=0.35)
        ax.plot(np.arange(1, len(g) + 1), keep, color=c, lw=1.2, label=lab)
    ax.axhline(1e-3, color=GREEN, lw=0.8, ls="--")
    ax.text(3500, 2.3e-4, r"$\tau = 10^{-3}$", color=GREEN, ha="right", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(1.6e-4, 8)
    ax.set_xlabel("step")
    ax.set_ylabel("gap")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(lw=0.3)
    fig.tight_layout()
    save(fig, "ch12-converge")


def ch12_scale():
    """Measured seconds against m, on logarithmic axes."""
    import json
    res = json.loads((Path(__file__).parent / "code/ch12_timing.json").read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    for key, c, mk, lab, up in (("slsqp", RED, "^", "SLSQP", 1.0),
                                ("smo", BLUE, "o", "our SMO", 1.6),
                                ("sklearn", GREEN, "s", "SVC", 0.6)):
        r = np.array(res[key])
        ax.loglog(r[:, 0], r[:, 1], color=c, marker=mk, ms=3.5, lw=1.1)
        ax.text(r[-1, 0] * 1.3, r[-1, 1] * up, lab, color=c, va="center",
                fontsize=8.5)
    ax.set_xlim(80, 30000)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("seconds")
    ax.grid(lw=0.3, which="major")
    fig.tight_layout()
    save(fig, "ch12-scale")


def ch13_step():
    """The gain along one coordinate, and the clip to [0, C]."""
    cases = ((-4 / 7, 3, 0.0, 0.5, r"(a) inside $[0, C]$"),
             (2 / 3, 11, 0.0, 0.5, "(b) below 0"),
             (-59 / 14, 33, 0.0, 0.1, r"(c) above $C$"))
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.75), sharey=False)
    for ax, (g, q, a0, C, lab) in zip(axes, cases):
        peak = a0 - g / q
        lo, hi = min(-0.12, peak - 0.08), max(C, peak) + 0.1
        d = np.linspace(lo, hi, 200)
        gain = -g * (d - a0) - q * (d - a0) ** 2 / 2
        ax.plot(d, gain, color=GREY_RULE, lw=1.0)
        inside = (d >= 0) & (d <= C)
        ax.plot(d[inside], gain[inside], color=BLUE, lw=1.5)
        ax.axvline(0, color=GREY, lw=0.5, ls=":")
        ax.axvline(C, color=GREY, lw=0.5, ls=":")
        new = min(max(peak, 0), C)
        ax.plot([new], [-g * (new - a0) - q * (new - a0) ** 2 / 2], "o",
                color=RED, ms=4, zorder=4)
        ax.plot([peak], [-g * (peak - a0) - q * (peak - a0) ** 2 / 2],
                "o", mfc="white", mec=GREY, ms=3.5, zorder=3)
        top = g * g / (2 * q)
        ax.set_ylim(-0.75 * top, 1.25 * top)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel(r"$\alpha_i$")
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"gain in $D$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch13-step")


def ch13_paths():
    """Both solvers on the centred machines, where w has two parts."""
    from ch13_linear import dcd, pegasos, primal, rows_of
    from machines import X, y
    A = X - X.mean(axis=0)
    C, lam = 0.5, 1 / 4
    r = rows_of(A)
    u = np.linspace(-1.05, 2.2, 160)
    v = np.linspace(-1.05, 2.2, 160)
    J = np.array([[primal(np.array([a, b]), A, y, C) for a in u] for b in v])
    trace = []
    a, ws, _, _ = dcd(r, y, 2, C, tol=1e-6, epochs=4000, trace=trace)
    marks = list(range(1, 401))
    _, saved = pegasos(r, y, 2, lam, 400, project=True, marks=marks)
    P = np.array([s[1] for s in saved])
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.35), sharey=True)
    for ax in axes:
        ax.contour(u, v, J, levels=[3.2, 3.6, 4.5, 6, 8, 11], colors=GREY_RULE,
                   linewidths=0.6)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(2 * np.cos(th), 2 * np.sin(th), color=ORANGE, lw=0.8, ls="--")
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 2.2)
        ax.set_ylim(-1.05, 2.2)
        ax.set_xlabel(r"$w_1$")
        ax.tick_params(labelsize=8)
    axes[0].plot(P[:, 0], P[:, 1], color=BLUE, lw=0.5, alpha=0.75)
    axes[0].plot(*ws, "*", color=RED, ms=7, zorder=5)
    axes[0].set_title("Pegasos, 400 steps", fontsize=8.5)
    axes[0].set_ylabel(r"$w_2$")
    W = np.array([np.zeros(2)] + [t[2] for t in trace[:12]])
    axes[1].plot(W[:, 0], W[:, 1], color=GREEN, lw=1.0, marker="o", ms=2.2)
    axes[1].plot(*ws, "*", color=RED, ms=7, zorder=5)
    axes[1].set_title("coordinate descent, by pass", fontsize=8.5)
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch13-paths")


def ch13_rate():
    """Excess objective against visits to a point, on logarithmic axes."""
    from ch13_linear import dcd, pegasos, primal, rows_of, with_offset
    from ch07_soft import X9, y9
    Z = with_offset(X9)
    r = rows_of(Z)
    C, lam = 0.5, 2 / 9
    trace = []
    _, ws, _, _ = dcd(r, y9, 3, C, tol=1e-11, epochs=4000, trace=trace)
    Js = primal(ws, Z, y9, C)
    marks = [int(t) for t in np.unique(np.logspace(1, 5, 25).astype(int))]
    runs = [pegasos(r, y9, 3, lam, marks[-1], seed=k, marks=marks)[1]
            for k in range(20)]
    gap = np.mean([[primal(w, Z, y9, C) - Js for _, w in run]
                   for run in runs], axis=0)
    T = np.array(marks, dtype=float)
    Rr = np.sqrt((Z ** 2).sum(axis=1).max())
    bound = 9 * C * (2 * Rr) ** 2 * (1 + np.log(T)) / (2 * lam * T)
    d = np.array([(9 * e, primal(w, Z, y9, C) - Js) for e, _, w in trace])
    d = d[d[:, 1] > 1e-9]
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    ax.loglog(T, bound, color=GREY, lw=0.9, ls="--")
    ax.loglog(T, gap, color=BLUE, lw=1.2)
    ax.loglog(d[:, 0], d[:, 1], color=GREEN, lw=1.2)
    ax.text(T[10], bound[10] * 3, "bound on the mean", color=GREY,
            fontsize=8.5, ha="left")
    ax.text(T[-1] * 0.85, gap[-1] * 12, "Pegasos", color=BLUE, fontsize=8.5,
            ha="right")
    ax.text(d[-1, 0] * 1.2, d[-1, 1] * 1.6, "coordinate descent", color=GREEN,
            fontsize=8.5, ha="left")
    ax.set_xlabel("points visited")
    ax.set_ylabel(r"$J(\boldsymbol{w}) - J^\ast$")
    ax.grid(lw=0.3, which="major")
    fig.tight_layout()
    save(fig, "ch13-rate")


def ch13_balance():
    """The offset feature s: closer to the true line, and slower."""
    from ch13_linear import dcd, rows_of, with_offset
    from ch07_soft import X9, y9
    S = np.array([1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300.0])
    res = []
    for s in S:
        Z = with_offset(X9, s)
        a, w, ep, _ = dcd(rows_of(Z), y9, 3, 0.5, tol=1e-6, epochs=200000)
        res.append((np.linalg.norm(np.r_[w[:2] - [0.4, 0.5],
                                         s * w[2] + 2.9]), ep))
    res = np.array(res)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0))
    axes[0].loglog(S, res[:, 0], color=BLUE, marker="o", ms=2.5, lw=1)
    axes[0].set_ylabel("distance to Chapter 7")
    axes[1].loglog(S, res[:, 1], color=RED, marker="o", ms=2.5, lw=1)
    axes[1].set_ylabel("passes")
    for ax in axes:
        ax.set_xlabel(r"$s$")
        ax.grid(lw=0.3, which="major")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "ch13-balance")


def ch13_scale():
    """Measured seconds against m on the sparse data."""
    import json
    res = json.loads((Path(__file__).parent / "code/ch13_timing.json").read_text())
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    for key, c, mk, lab, up in (("smo", RED, "^", "our SMO", 1.0),
                                ("svc", ORANGE, "D", "SVC", 1.0),
                                ("dcd", BLUE, "o", "our coordinate", 1.5),
                                ("pegasos", GREEN, "v", "our Pegasos", 0.6),
                                ("liblinear", GREY, "s", "LinearSVC", 1.0)):
        r = np.array(res[key])
        ax.loglog(r[:, 0], r[:, 1], color=c, marker=mk, ms=3, lw=1.1)
        ax.text(r[-1, 0] * 1.3, r[-1, 1] * up, lab, color=c, va="center",
                fontsize=8.5)
    ax.set_xlim(800, 2.4e6)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("seconds")
    ax.grid(lw=0.3, which="major")
    fig.tight_layout()
    save(fig, "ch13-scale")


def ch14_units():
    """One Gaussian SVM in three sets of units, drawn in the raw axes."""
    from sklearn.svm import SVC
    from machines import X, y
    fig, axes = plt.subplots(1, 3, figsize=(4.7, 1.9), sharey=True)
    a, b = np.meshgrid(np.linspace(0, 7.5, 300), np.linspace(0, 8, 300))
    G = np.c_[a.ravel(), b.ravel()]
    for ax, u, title in zip(axes, ([1, 1], [1, 10], 1 / X.std(0)),
                            ("raw units", "degrees", "standardised")):
        s = SVC(C=1e6, gamma=0.5).fit(X * u, y)
        f = s.decision_function(G * u).reshape(a.shape)
        ax.contourf(a, b, f, levels=[-1e9, 0, 1e9],
                    colors=["#E8EDF4", "#F7E9DC"])
        ax.contour(a, b, f, levels=[0], colors=[GREY], linewidths=0.9)
        ax.scatter(*X[y < 0].T, s=13, color=BLUE, zorder=3)
        ax.scatter(*X[y > 0].T, s=13, color=ORANGE, marker="s", zorder=3)
        ax.set_title(title)
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch14-units")


def ch14_loo():
    """The three lines that appear when an essential point leaves."""
    from ch14_tuning import loo_line
    from machines import X, y
    fig, axes = plt.subplots(1, 3, figsize=(4.7, 1.9), sharey=True)
    t = np.array([-1.0, 8.5])
    for ax, i in zip(axes, (3, 4, 6)):
        w, b, s = loo_line(i)
        keep = np.arange(8) != i
        ax.plot(t, (11 - t) / 2, color=GREY_RULE, lw=0.9, ls="--")
        if abs(w[1]) > 1e-9:
            ax.plot(t, -(b + w[0] * t) / w[1], color=GREEN, lw=1.2)
        for sign, c, mk in ((-1, BLUE, "o"), (1, ORANGE, "s")):
            q = keep & (y == sign)
            ax.scatter(*X[q].T, s=13, color=c, marker=mk, zorder=3)
            if y[i] == sign:
                ax.scatter(*X[i], s=34, facecolors="white", edgecolors=c,
                           marker=mk, linewidths=1.1, zorder=4)
        ax.set_title(f"without machine {i + 1}")
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch14-loo")


def ch14_folds():
    """Five folds, and the same plan with a search inside each."""
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 1.9))
    def row(ax, y0, x0, w, k, mark, h, fc=BLUE, pale="#DDE3EC"):
        for j in range(k):
            ax.add_patch(plt.Rectangle(
                (x0 + j * w, y0), w, h, ec="white", lw=0.8,
                fc=fc if j == mark else pale))
    ax = axes[0]
    for r in range(5):
        row(ax, 4 - r, 0, 1, 5, r, 0.72, ORANGE)
        ax.text(5.15, 4.36 - r, f"score {r + 1}", va="center", fontsize=8.5)
    ax.set_title("k-fold validation")
    ax.set_xlim(0, 6.7)
    ax.set_ylim(-0.1, 4.9)
    ax = axes[1]
    row(ax, 4, 0, 1, 5, 4, 0.72, RED)
    ax.text(5.15, 4.36, "outer", va="center", fontsize=8.5)
    ax.plot([0, 0, 4, 4], [3.92, 3.45, 3.45, 3.92], color=GREY, lw=0.7)
    for r in range(3):
        row(ax, 2.5 - r, 0, 0.8, 5, r, 0.72, ORANGE)
    ax.text(4.15, 1.86, "inner:\nsearch\nC and $\\gamma$", va="center",
            fontsize=8.5)
    ax.text(0, 0.05, "outer fold scores the winner", fontsize=8.5,
            va="center", color=RED)
    ax.set_title("nested validation")
    ax.set_xlim(0, 6.7)
    ax.set_ylim(-0.1, 4.9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch14-folds")


def ch14_grid():
    """Cross-validated accuracy on the log grid; a ridge and two cliffs."""
    from ch14_tuning import CS, GAMMAS, cancer, folds, grid_search, surface_of
    from sklearn.model_selection import train_test_split
    from matplotlib.colors import LinearSegmentedColormap
    A, t, _ = cancer()
    Xtr, _, ttr, _ = train_test_split(A, t, test_size=0.25, stratify=t,
                                      random_state=20261101)
    gs = grid_search(Xtr, ttr, folds())
    S = surface_of(gs)
    cmap = LinearSegmentedColormap.from_list("b", ["#F4F6F9", "#7C93B8", BLUE])
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    ax.imshow(S, origin="lower", cmap=cmap, vmin=0.6, vmax=1.0, aspect="auto")
    i, j = np.unravel_index(S.argmax(), S.shape)
    for r in range(S.shape[0]):
        for c in range(S.shape[1]):
            txt = f"{S[r, c]:.3f}"[1:]
            ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                    color="white" if S[r, c] > 0.82 else GREY)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               ec=ORANGE, lw=1.6))
    ax.set_xticks(range(len(CS)), [f"$10^{{{int(np.log10(c))}}}$" for c in CS])
    ax.set_yticks(range(len(GAMMAS)),
                  [f"$10^{{{int(np.log10(g))}}}$" for g in GAMMAS])
    ax.set_xlabel("$C$")
    ax.set_ylabel(r"$\gamma$")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, "ch14-grid")


def ch14_nested():
    """On random labels the best cell looks better than chance."""
    from ch14_tuning import nested_noise
    best, outer = nested_noise()
    fig, ax = plt.subplots(figsize=(3.9, 1.9))
    bins = np.arange(0.2, 0.85, 0.04)
    ax.hist(outer, bins=bins, color=GREEN, alpha=0.75, label="nested")
    ax.hist(best, bins=bins, color=ORANGE, alpha=0.75, label="best cell")
    ax.axvline(0.5, color=GREY, lw=0.9, ls="--")
    ax.set_ylim(0, 14)
    ax.text(0.51, 13, "chance", fontsize=8.5, va="center")
    ax.set_xlabel("estimated accuracy on random labels")
    ax.set_ylabel("data sets")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "ch14-nested")


def ch14_imbalance():
    """Two lines with one direction, and the ROC curve they share."""
    from ch14_tuning import SEED, imbalanced, lin_pipe
    from sklearn.metrics import roc_curve
    A, t = imbalanced(2000)
    Ate, tte = imbalanced(20000, seed=SEED + 1)
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.3))
    ax = axes[0]
    ax.scatter(*A[t < 0].T, s=1.5, color=BLUE, alpha=0.35, linewidths=0)
    ax.scatter(*A[t > 0].T, s=7, color=ORANGE, marker="s", linewidths=0)
    u = np.array([-3.4, 5.2])
    for kw, c, lab, ha, dx in (({}, GREY, "plain", "left", 0.2),
                               ({"class_weight": "balanced"}, GREEN,
                                "balanced", "right", -0.2)):
        p = lin_pipe(**kw).fit(A, t)
        sc, s = p[0], p[-1]
        w = s.coef_[0] / sc.scale_
        b = s.intercept_[0] - w @ sc.mean_
        ax.plot(u, -(b + w[0] * u) / w[1], color=c, lw=1.2)
        ax.text(-(b + w[1] * -2.9) / w[0] + dx, -2.9, lab, color=c,
                fontsize=8.5, va="center", ha=ha)
    ax.set_xlim(-3.4, 5.2)
    ax.set_ylim(-3.4, 4.8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax = axes[1]
    p = lin_pipe().fit(A, t)
    s = p.decision_function(Ate)
    fpr, tpr, _ = roc_curve(tte, s)
    ax.plot(fpr, tpr, color=BLUE, lw=1.2)
    ax.plot([0, 1], [0, 1], color=GREY_RULE, lw=0.8, ls="--")
    for q, c, lab, dx, dy in ((p, GREY, "plain", 0.05, -0.09),
                              (lin_pipe(class_weight="balanced").fit(A, t),
                               GREEN, "balanced", 0.05, -0.09)):
        f = q.decision_function(Ate) > 0
        x0, y0 = np.mean(f[tte < 0]), np.mean(f[tte > 0])
        ax.scatter(x0, y0, s=22, color=c, zorder=3)
        ax.text(x0 + dx, y0 + dy, lab, color=c, fontsize=8.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("recall")
    ax.grid(lw=0.3)
    fig.tight_layout(w_pad=2.2)
    save(fig, "ch14-imbalance")


def ch15_ovo():
    """Three pair lines, their triangle, and the cycle inside it."""
    from matplotlib.patches import Polygon
    from ch15_multiclass import X3, c3, votes
    colour = {1: BLUE, 2: ORANGE, 3: GREEN}
    mark = {1: "o", 2: "s", 3: "^"}
    tri = np.array([[6.2, 2.4], [7, 2], [5, 0]])
    fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.5),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    t = (np.arange(600) + 0.5) / 600 * 11.5
    G = np.stack(np.meshgrid(t, t * 8.5 / 11.5), axis=-1)
    V = votes(G)
    lab = V.argmax(axis=-1).astype(float) + 1
    lab[(V == 1).all(axis=-1)] = np.nan
    ax.contourf(G[..., 0], G[..., 1], lab, levels=[.5, 1.5, 2.5, 3.5],
                colors=[BLUE, ORANGE, GREEN], alpha=0.10)
    for k in (1, 2, 3):
        ax.scatter(*X3[c3 == k].T, s=14, color=colour[k], marker=mark[k],
                   zorder=3, label=f"class {k}")
    x = np.array([0, 11.5])
    ax.plot(x, (11 - x) / 2, color=GREY, lw=0.9)
    ax.plot(x, 2 * x - 10, color=GREY, lw=0.9, ls="--")
    ax.plot(x, x - 5, color=GREY, lw=0.9, ls=":")
    ax.add_patch(Polygon(tri, color=RED, alpha=0.75, lw=0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.5)
    ax.text(2.3, 4.75, r"$\ell_{12}$", fontsize=8.5)
    ax.text(8.75, 7.4, r"$\ell_{13}$", fontsize=8.5)
    ax.text(10.1, 3.9, r"$\ell_{23}$", fontsize=8.5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="upper left", fontsize=8, handletextpad=0.1,
              borderaxespad=0.1, ncol=1)
    ax = axes[1]
    ax.add_patch(Polygon(tri, color=RED, alpha=0.13, lw=0))
    x = np.array([4.6, 7.5])
    ax.plot(x, (11 - x) / 2, color=GREY, lw=0.9)
    ax.plot(x, 2 * x - 10, color=GREY, lw=0.9, ls="--")
    ax.plot(x, x - 5, color=GREY, lw=0.9, ls=":")
    ax.scatter([6], [1.5], s=14, color=RED, zorder=3)
    ax.annotate(r"$(6,\,1.5)$", (6, 1.5), xytext=(6.6, 0.95), fontsize=8,
                color=RED, arrowprops=dict(arrowstyle="-", lw=0.5, color=RED,
                                           shrinkA=0, shrinkB=2))
    arrow = dict(arrowstyle="-|>", lw=0.6, shrinkA=0, shrinkB=0)
    for text, xy, xyt, col in (("1 beats 2", (6.42, 2.02), (6.45, 2.75), BLUE),
                               ("3 beats 1", (5.75, 1.3), (4.68, 2.15), GREEN),
                               ("2 beats 3", (5.9, 0.95), (6.15, 0.25), ORANGE)):
        ax.annotate(text, xy, xytext=xyt, fontsize=8, color=col,
                    arrowprops={**arrow, "color": col})
    ax.set_xlim(4.6, 7.5)
    ax.set_ylim(-0.2, 3.1)
    ax.set_xlabel(r"$x_1$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch15-ovo")


def ch15_ovr():
    """Who claims a point in one versus rest."""
    from ch15_multiclass import X3, c3, ovr_scores
    colour = {1: BLUE, 2: ORANGE, 3: GREEN}
    mark = {1: "o", 2: "s", 3: "^"}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    t = (np.arange(800) + 0.5) / 800
    G = np.stack(np.meshgrid(t * 11.5, t * 8.5), axis=-1)
    f = ovr_scores(G)
    n = (f > 0).sum(axis=-1)
    only = np.where(n == 1, f.argmax(axis=-1) + 1, 0)
    ax.contourf(G[..., 0], G[..., 1], only, levels=[.5, 1.5, 2.5, 3.5],
                colors=[BLUE, ORANGE, GREEN], alpha=0.10)
    ax.contourf(G[..., 0], G[..., 1], n, levels=[-.5, .5], colors=[GREY],
                alpha=0.35)
    ax.contourf(G[..., 0], G[..., 1], n, levels=[1.5, 3.5], colors=[RED],
                alpha=0.30)
    for k in (1, 2, 3):
        ax.scatter(*X3[c3 == k].T, s=14, color=colour[k], marker=mark[k],
                   zorder=3)
    x = np.array([0, 11.5])
    ax.plot(x, (26 - 3 * x) / 4, color=BLUE, lw=0.8)
    ax.plot(x, (23 - x) / 6, color=ORANGE, lw=0.8)
    ax.plot(x, x - 5, color=GREEN, lw=0.8)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.5)
    ax.annotate("no claim", (6.3, 2.55), xytext=(7.0, 4.3), fontsize=8,
                color=GREY, arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                            color=GREY, shrinkA=1, shrinkB=0))
    ax.text(6.4, 0.25, "two claims", fontsize=8, color=RED, ha="right")
    ax.text(0.5, 7.4, "class 2 only", fontsize=8, color=ORANGE)
    ax.text(0.4, 0.2, "class 1 only", fontsize=8, color=BLUE)
    ax.text(11.2, 0.2, "class 3 only", fontsize=8, color=GREEN, ha="right")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.tight_layout()
    save(fig, "ch15-ovr")


def ch15_resolve():
    """Decision regions of three rules that always give an answer."""
    from ch15_multiclass import (X3, c3, crammer_eleven, margin_sum,
                                 ovr_scores)
    colour = {1: BLUE, 2: ORANGE, 3: GREEN}
    mark = {1: "o", 2: "s", 3: "^"}
    t = (np.arange(500) + 0.5) / 500
    G = np.stack(np.meshgrid(t * 11.5, t * 8.5), axis=-1)
    s, c = crammer_eleven()
    W = s.coef_
    cs = (G - c) @ W.T
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.95), sharey=True)
    for ax, sc, name in zip(axes,
                            (margin_sum(G), ovr_scores(G), cs),
                            ("sum of pair scores", "largest rest score",
                             "Crammer and Singer")):
        lab = sc.argmax(axis=-1) + 1
        ax.contourf(G[..., 0], G[..., 1], lab, levels=[.5, 1.5, 2.5, 3.5],
                    colors=[BLUE, ORANGE, GREEN], alpha=0.12)
        ax.contour(G[..., 0], G[..., 1], lab, levels=[1.5, 2.5], colors=GREY,
                   linewidths=0.7)
        for k in (1, 2, 3):
            ax.scatter(*X3[c3 == k].T, s=7, color=colour[k], marker=mark[k],
                       zorder=3)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlim(0, 11.5)
        ax.set_ylim(0, 8.5)
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch15-resolve")


def ch15_cost():
    """Kernel entries for one versus one and one versus rest on digits."""
    from ch15_multiclass import weighted_steps
    ks = np.arange(2, 11)
    r = np.array([weighted_steps(k) for k in ks], float)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(ks, r[:, 2], color=ORANGE, lw=1.2, marker="o", ms=3)
    ax.plot(ks, r[:, 1], color=BLUE, lw=1.2, marker="o", ms=3)
    ax.text(2.4, 2.5e6, "one versus rest", color=ORANGE, fontsize=8.5)
    ax.text(10, 7e5, "one versus one", color=BLUE, fontsize=8.5, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("classes $k$")
    ax.set_ylabel("steps $\\times$ points in problem")
    ax.grid(lw=0.3)
    fig.tight_layout()
    save(fig, "ch15-cost")


def ch15_scores():
    """Scores on training data, in folds, and on new data."""
    from ch15_multiclass import study
    S = study()
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.95), sharey=True)
    bins = np.linspace(-1.5, 1.5, 31)
    for ax, f, t, name in zip(axes, (S["ftr"], S["fcv"], S["fte"]),
                              (S["t"], S["t"], S["tte"]),
                              ("training scores", "fold scores", "test scores")):
        ax.hist(f[t < 0], bins=bins, color=BLUE, alpha=0.6)
        ax.hist(f[t > 0], bins=bins, color=ORANGE, alpha=0.6)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel(r"$f(\boldsymbol{x})$")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 1000)
    axes[0].set_ylabel("points")
    axes[0].text(-0.8, 150, "even", color=BLUE, fontsize=8)
    axes[0].text(0.8, 150, "odd", color=ORANGE, fontsize=8, ha="right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch15-scores")


def ch15_sigmoid():
    """Observed frequency against score, with the two fitted sigmoids."""
    from ch15_multiclass import sigmoid, study
    S = study()
    f, t = S["fte"], S["tte"]
    edges = np.quantile(f, np.linspace(0, 1, 13))
    g = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, 11)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.scatter([f[g == k].mean() for k in range(12)],
               [np.mean(t[g == k] > 0) for k in range(12)],
               s=12, color=GREY, zorder=3)
    x = np.linspace(-0.6, 0.6, 300)
    ax.plot(x, sigmoid(x, *S["train"][:2]), color=ORANGE, lw=1.2)
    ax.plot(x, sigmoid(x, *S["held"][:2]), color=BLUE, lw=1.2)
    ax.annotate("fit on fold scores", (0.04, 0.72), xytext=(-0.58, 0.8),
                fontsize=8, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE,
                                shrinkA=2, shrinkB=0))
    ax.text(0.16, 0.5, "fit on training scores", fontsize=8, color=ORANGE)
    ax.text(0.16, 0.3, "test groups of 75", fontsize=8, color=GREY)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"score $f(\boldsymbol{x})$")
    ax.set_ylabel(r"$P(y = +1)$")
    ax.grid(lw=0.3)
    fig.tight_layout()
    save(fig, "ch15-sigmoid")


def ch15_reliability():
    """Reliability diagrams for the two fits, with bin counts."""
    from ch15_multiclass import bins, brier, sigmoid, study
    S = study()
    t = S["tte"]
    fig, axes = plt.subplots(2, 2, figsize=(4.7, 2.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.2, 1]))
    for j, (key, name, col) in enumerate((("train", "fit on training scores",
                                           ORANGE),
                                          ("held", "fit on fold scores", BLUE))):
        p = sigmoid(S["fte"], *S[key][:2])
        b = bins(p, t)
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=GREY_RULE, lw=0.8, ls="--")
        q = [r[1] for r in b]
        o = [r[2] for r in b]
        ax.plot(q, o, color=col, lw=1.1, marker="o", ms=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=8.5)
        ax.text(0.04, 0.88, f"Brier {brier(p, t):.3f}", fontsize=8, color=col)
        ax.grid(lw=0.3)
        ax = axes[1, j]
        ax.bar([(r[0] + .5) / 10 for r in b], [r[3] for r in b], width=0.085,
               color=col, alpha=0.6)
        ax.set_ylim(0, 500)
        ax.set_xlabel("forecast probability")
    axes[0, 0].set_ylabel("observed share")
    axes[1, 0].set_ylabel("points")
    fig.tight_layout(w_pad=1.0, h_pad=0.3)
    save(fig, "ch15-reliability")


def ch16_loss():
    """The tube loss beside its neighbours, and the pull of each."""
    from ch16_svr import huber, tube_loss
    r = np.linspace(-3, 3, 601)
    e = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.05))
    ax = axes[0]
    ax.plot(r, r ** 2 / 2, color=GREY, lw=1.0, ls="--", label="squared")
    ax.plot(r, huber(r, 1.0), color=GREEN, lw=1.1, label="Huber")
    ax.plot(r, np.abs(r), color=ORANGE, lw=1.0, ls=":", label="absolute")
    ax.plot(r, tube_loss(r, e), color=BLUE, lw=1.6, label="tube")
    ax.set_ylim(-0.1, 3.2)
    ax.set_xlabel(r"residual $r$")
    ax.set_ylabel("loss")
    ax = axes[1]
    ax.plot(r, r, color=GREY, lw=1.0, ls="--")
    ax.plot(r, np.clip(r, -1, 1), color=GREEN, lw=1.1)
    ax.plot(r, np.sign(r), color=ORANGE, lw=1.0, ls=":")
    ax.plot(r, np.sign(r) * (np.abs(r) > e), color=BLUE, lw=1.6)
    ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel(r"residual $r$")
    ax.set_ylabel(r"pull on $f$")
    for ax in axes:
        ax.grid(lw=0.3)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, ncol=4, loc="upper center", fontsize=8,
               bbox_to_anchor=(0.5, 1.07), handlelength=2.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch16-loss")


def ch16_tube():
    """The worked example: tube, kinds of point, and multipliers."""
    from ch16_svr import HE, HX, HY
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.35),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    t = np.array([-0.5, 4.5])
    f = 0.5 * t + 0.5
    ax.fill_between(t, f - HE, f + HE, color=BLUE, alpha=0.12, lw=0)
    ax.plot(t, f, color=BLUE, lw=1.2)
    ax.plot(t, f + HE, color=BLUE, lw=0.6, ls="--")
    ax.plot(t, f - HE, color=BLUE, lw=0.6, ls="--")
    cols = [ORANGE, GREY, RED, ORANGE]
    ax.scatter(HX, HY, c=cols, s=22, zorder=3)
    ax.annotate("", (2, 2.0), (2, 3), arrowprops=dict(
        arrowstyle="<->", color=RED, lw=0.8, shrinkA=2, shrinkB=0))
    ax.text(2.12, 2.45, r"$\xi_3 = 1$", color=RED, fontsize=8)
    ax.text(0.12, -0.22, "edge", color=ORANGE, fontsize=8)
    ax.text(1.0, 0.45, "inside", color=GREY, fontsize=8, ha="center")
    ax.text(2.12, 3.08, "outside", color=RED, fontsize=8)
    ax.text(4.1, 1.35, "edge", color=ORANGE, fontsize=8, ha="right")
    ax.text(-0.35, 2.85, r"$f(x) = \frac{1}{2}x + \frac{1}{2}$",
            color=BLUE, fontsize=9)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.6, 3.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid(lw=0.3)
    ax = axes[1]
    beta = [-5 / 8, 0, 1, -3 / 8]
    ax.bar(range(1, 5), beta, color=cols, width=0.6)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axhline(1, color=RED, lw=0.7, ls="--")
    ax.axhline(-1, color=RED, lw=0.7, ls="--")
    ax.text(0.55, 0.82, r"cap $C$", color=RED, fontsize=8)
    ax.set_xticks(range(1, 5))
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("point $i$")
    ax.set_ylabel(r"$\beta_i = \alpha_i - \alpha_i^*$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch16-tube")


def ch16_eps():
    """The four points under four widths of tube."""
    from ch16_svr import HX, HY
    from sklearn.svm import SVR
    fig, axes = plt.subplots(1, 4, figsize=(4.85, 1.65), sharey=True)
    t = np.array([-0.6, 4.6])
    for ax, e in zip(axes, (0, 0.5, 1.25, 1.5)):
        m = SVR(kernel="linear", C=100, epsilon=e, tol=1e-8)
        m.fit(HX[:, None], HY)
        f = m.coef_[0, 0] * t + m.intercept_[0]
        ax.fill_between(t, f - e, f + e, color=BLUE, alpha=0.12, lw=0)
        ax.plot(t, f, color=BLUE, lw=1.1)
        keep = np.zeros(4, bool)
        keep[m.support_] = True
        ax.scatter(HX[~keep], HY[~keep], s=14, color=GREY, zorder=3)
        ax.scatter(HX[keep], HY[keep], s=14, color=ORANGE, zorder=3)
        ax.set_title(rf"$\varepsilon = {e:g}$, $w = {m.coef_[0, 0]:.3g}$",
                     fontsize=8)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-1, 4)
        ax.set_xticks([0, 2, 4])
        ax.set_xlabel(r"$x$")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$y$")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch16-eps")


def ch16_outlier():
    """A far label moves least squares; a far input moves both."""
    from ch16_svr import fit_three, line_data
    x, t = line_data()
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.25))
    grid = np.array([0.0, 62.0])
    ax = axes[0]
    xx, tt = np.r_[x, 8], np.r_[t, 25]
    f = fit_three(xx, tt)
    u = np.array([0.0, 10.0])
    ax.scatter(x, t, s=7, color=GREY)
    ax.scatter([8], [25], s=18, color=RED, zorder=3)
    ax.plot(u, f["ls"][0] * u + f["ls"][1], color=ORANGE, lw=1.2)
    ax.plot(u, f["svr"][0] * u + f["svr"][1], color=BLUE, lw=1.2)
    ax.text(7.7, 25, "label raised by 20", color=RED, fontsize=8, ha="right",
            va="center")
    ax.text(5.8, 10.6, "least squares", color=ORANGE, fontsize=8, ha="right")
    ax.text(10, 2.2, "SVR", color=BLUE, fontsize=8, ha="right")
    ax.set_ylim(-2, 27)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax = axes[1]
    xx, tt = np.r_[x, 60], np.r_[t, 0]
    f = fit_three(xx, tt)
    ax.scatter(x, t, s=7, color=GREY)
    ax.scatter([60], [0], s=18, color=RED, zorder=3)
    ax.plot(grid, f["ls"][0] * grid + f["ls"][1], color=ORANGE, lw=1.2)
    ax.plot(grid, f["svr"][0] * grid + f["svr"][1], color=BLUE, lw=1.2)
    ax.plot(grid, 1 + 0.5 * grid, color=GREY_RULE, lw=0.8, ls="--")
    ax.text(57, -1.3, "input at 60", color=RED, fontsize=8, ha="right")
    ax.text(40, 22.5, "the line of the data", color=GREY, fontsize=8, ha="right")
    ax.text(46, 7.5, "SVR", color=BLUE, fontsize=8)
    ax.text(32, 4.3, "least squares", color=ORANGE, fontsize=8, ha="center")
    ax.set_ylim(-2, 32)
    ax.set_xlabel(r"$x$")
    for ax in axes:
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch16-outlier")


def ch16_curve():
    """Gaussian SVR on the noisy sine at three widths."""
    from ch16_svr import GAMMA, GRID, curve
    from sklearn.svm import SVR
    a, t = curve()
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.9), sharey=True)
    g = GRID.ravel()
    for ax, e in zip(axes, (0.05, 0.15, 0.4)):
        m = SVR(C=10, gamma=GAMMA, epsilon=e, tol=1e-5).fit(a, t)
        f = m.predict(GRID)
        ax.fill_between(g, f - e, f + e, color=BLUE, alpha=0.14, lw=0)
        ax.plot(g, f, color=BLUE, lw=1.1)
        keep = np.zeros(len(t), bool)
        keep[m.support_] = True
        ax.scatter(a[~keep], t[~keep], s=7, color=GREY, zorder=3)
        ax.scatter(a[keep], t[keep], s=9, color=ORANGE, zorder=3)
        ax.set_title(rf"$\varepsilon = {e:g}$, {keep.sum()} of {len(t)} kept",
                     fontsize=8)
        ax.set_xlabel(r"$x$")
        ax.grid(lw=0.3)
    axes[0].set_ylabel(r"$y$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch16-curve")


def ch16_sweep():
    """Support vectors and error as the tube widens."""
    from ch16_svr import eps_sweep
    r = np.array(eps_sweep())
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 1.85))
    ax = axes[0]
    ax.plot(r[:, 0], r[:, 1], color=ORANGE, lw=1.1, marker="o", ms=3)
    ax.set_ylabel("support vectors of 60")
    ax = axes[1]
    ax.plot(r[:, 0], r[:, 2], color=BLUE, lw=1.1, marker="o", ms=3)
    ax.axvline(0.2, color=GREY_RULE, lw=0.8, ls="--")
    ax.text(0.23, 0.55, r"noise $\sigma$", color=GREY, fontsize=8)
    ax.set_ylabel("error against the sine")
    for ax in axes:
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "ch16-sweep")


def ch17_origin():
    """A line away from the origin, and the smallest circle."""
    from ch17_oneclass import HEALTHY, SERVICE
    from matplotlib.patches import Circle, Polygon
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.35), sharey=True)
    hull = HEALTHY[[1, 0, 3, 2]]
    ax = axes[0]
    ax.fill([2.7, 7.0, 7.0, -0.5, -0.5],
            [-0.7, -0.7, 7.6, 7.6, 2.5], color=GREEN, alpha=0.09, lw=0)
    ax.plot([-0.5, 2.5], [2.5, -0.5], color=BLUE, lw=1.2)
    ax.annotate("", xy=(1, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1))
    ax.text(0.95, 0.15, r"$\boldsymbol{w}$", color=ORANGE, fontsize=9)
    ax.text(2.9, 0.3, "accepted side", color=GREEN, fontsize=8)
    ax.text(0.3, -0.62, "origin", color=GREY, fontsize=8)
    ax.set_title(r"line: $x_1 + x_2 = 2$")
    ax = axes[1]
    ax.add_patch(Circle((2.5, 2), np.sqrt(3.25), fc=GREEN, alpha=0.09,
                        ec="none"))
    ax.add_patch(Circle((2.5, 2), np.sqrt(3.25), fc="none", ec=BLUE, lw=1.2))
    ax.plot([2.5], [2], marker="+", color=ORANGE, ms=7)
    ax.plot([1, 4], [1, 3], color=ORANGE, lw=0.8, ls="--")
    ax.text(2.72, 1.38, r"$\boldsymbol{c}$", color=ORANGE, fontsize=9)
    ax.set_title(r"circle: centre $\boldsymbol{c}$, $R^2 = 13/4$")
    for ax in axes:
        ax.add_patch(Polygon(hull, fc="none", ec=GREY_RULE, lw=0.8))
        ax.plot(HEALTHY[:, 0], HEALTHY[:, 1], "o", color=BLUE, ms=4.5,
                label="healthy")
        ax.plot(SERVICE[:, 0], SERVICE[:, 1], "s", color=RED, ms=4,
                label="needs service")
        ax.plot([0], [0], "x", color=GREY, ms=5)
        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-0.7, 7.6)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].legend(loc="upper left", fontsize=8, handletextpad=0.2,
                   borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch17-origin")


def ch17_sphere():
    """On a sphere, a plane and a ball cut off the same cap."""
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    t = np.linspace(-0.12, np.pi / 2 + 0.12, 200)
    ax.plot(np.cos(t), np.sin(t), color=GREY, lw=0.9)
    ang = np.radians([14, 22, 33, 41, 49, 58, 75])
    p = np.column_stack([np.cos(ang), np.sin(ang)])
    a0, a1 = np.radians(27), np.radians(63)
    u = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    v = np.array([-u[1], u[0]])
    c = 0.82 * u
    e0 = np.array([np.cos(a0), np.sin(a0)])
    h = e0 @ u
    ends = np.array([h * u + s * v for s in (-0.62, 0.62)])
    ax.plot(ends[:, 0], ends[:, 1], color=BLUE, lw=1.2)
    R = np.linalg.norm(e0 - c)
    b = np.linspace(0, 2 * np.pi, 300)
    ax.plot(c[0] + R * np.cos(b), c[1] + R * np.sin(b), color=GREEN, lw=1.2,
            ls="--")
    tt = np.linspace(a0, a1, 50)
    ax.plot(np.cos(tt), np.sin(tt), color=ORANGE, lw=2.6, alpha=0.6)
    ins = (ang >= a0) & (ang <= a1)
    ax.plot(p[ins, 0], p[ins, 1], "o", color=BLUE, ms=4.5)
    ax.plot(p[~ins, 0], p[~ins, 1], "o", mfc="white", mec=RED, ms=4.5)
    ax.annotate("", xy=c, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.plot([0], [0], "x", color=GREY, ms=5)
    ax.text(0.03, -0.07, "origin", fontsize=8)
    ax.text(0.0, 0.34, r"$\boldsymbol{c} = \boldsymbol{w}$", fontsize=8.5)
    ax.text(*(ends[1] + np.array([-0.17, 0.03])), "plane", color=BLUE,
            fontsize=8.5)
    ax.text(0.52, 0.12, "ball", color=GREEN, fontsize=8.5)
    ax.text(0.90, 0.88, "cap", color=ORANGE, fontsize=8.5)
    ax.text(0.96, 0.05, "unit sphere", fontsize=8)
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(-0.15, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "ch17-sphere")


def ch17_line():
    """The sum of hills on three points, and the level rho."""
    from ch17_oneclass import LG
    pts = np.array([-1.0, 0, 1])
    s = np.linspace(-3, 3, 400)
    H = np.exp(-LG * (s[:, None] - pts) ** 2)
    cases = [(r"$\nu \leq 17/24$", [8 / 17, 1 / 17, 8 / 17], 9 / 17),
             (r"$\nu = 3/4$", [4 / 9, 1 / 9, 4 / 9], 5 / 9),
             (r"$\nu = 1$: mean of hills", [1 / 3] * 3, 2 / 3)]
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.95), sharey=True)
    for ax, (title, a, rho) in zip(axes, cases):
        g = H @ np.array(a)
        ax.plot(s, g, color=BLUE, lw=1.2)
        ax.axhline(rho, color=ORANGE, lw=0.9, ls="--")
        ax.fill_between(s, 0, 1, where=g >= rho - 1e-9, color=GREEN,
                        alpha=0.12, lw=0)
        gp = np.exp(-LG * (pts[:, None] - pts) ** 2) @ np.array(a)
        ax.plot(pts, gp, "o", color=BLUE, ms=3.5)
        ax.text(1.55, rho + 0.03, r"$\rho$", color=ORANGE, fontsize=9)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(r"$x$")
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel(r"$g(x)$")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch17-line")


def grid17(sc, n=220):
    """Grid over the fleet plot, raw and scaled."""
    a, b = np.meshgrid(np.linspace(0, 6, n), np.linspace(0, 5, n))
    P = np.column_stack([a.ravel(), b.ravel()])
    return a, b, sc.transform(P)


def ch17_nu():
    """The boundary tightens as nu grows."""
    from ch17_oneclass import GAMMA, fit, fleet
    tr, te, fa, fb, sc = fleet()
    Z = sc.transform(tr)
    a, b, G = grid17(sc)
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.95), sharey=True)
    for ax, nu in zip(axes, (0.02, 0.2, 0.5)):
        o = fit(Z, nu, GAMMA)
        F = o.decision_function(G).reshape(a.shape)
        ax.contourf(a, b, F, levels=[0, F.max() + 1], colors=[GREEN],
                    alpha=0.13)
        ax.contour(a, b, F, levels=[0], colors=[BLUE], linewidths=1.1)
        s = o.decision_function(Z)
        ax.plot(tr[s >= 0, 0], tr[s >= 0, 1], ".", color=BLUE, ms=1.6)
        ax.plot(tr[s < 0, 0], tr[s < 0, 1], ".", color=ORANGE, ms=2.2)
        ax.set_title(rf"$\nu = {nu}$")
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch17-nu")


def ch17_shares():
    """nu lies between the two shares."""
    from ch17_oneclass import share_rows
    nus = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    r = np.array(share_rows(nus))
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 1.95))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY_RULE, lw=0.9)
    ax.plot(r[:, 0], r[:, 2], color=ORANGE, lw=1.1, marker="o", ms=2.5,
            label="support vectors")
    ax.plot(r[:, 0], r[:, 1], color=BLUE, lw=1.1, marker="s", ms=2.5,
            label="outside")
    ax.legend(loc="upper left", fontsize=8, handletextpad=0.3)
    ax.set_ylabel("share of training points")
    ax = axes[1]
    ax.plot(r[:, 0], r[:, 2] - r[:, 0], color=ORANGE, lw=1.1, marker="o",
            ms=2.5)
    ax.plot(r[:, 0], r[:, 1] - r[:, 0], color=BLUE, lw=1.1, marker="s",
            ms=2.5)
    ax.axhline(0, color=GREY_RULE, lw=0.9)
    ax.set_ylabel(r"share $-\ \nu$")
    for ax in axes:
        ax.set_xlabel(r"$\nu$")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch17-shares")


def ch17_gamma():
    """From a blob to a set of islands."""
    from ch17_oneclass import fit, fleet
    tr, te, fa, fb, sc = fleet()
    Z = sc.transform(tr)
    a, b, G = grid17(sc, 320)
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.95), sharey=True)
    for ax, g in zip(axes, (0.03, 1, 100)):
        o = fit(Z, 0.05, g)
        F = o.decision_function(G).reshape(a.shape)
        ax.contourf(a, b, F, levels=[0, F.max() + 1], colors=[GREEN],
                    alpha=0.13)
        ax.plot(tr[:, 0], tr[:, 1], ".", color=GREY_RULE, ms=1.2)
        ax.contour(a, b, F, levels=[0], colors=[BLUE], linewidths=0.8)
        ax.plot(fa[:, 0], fa[:, 1], ".", color=RED, ms=2.4)
        ax.plot(fb[:, 0], fb[:, 1], ".", color=ORANGE, ms=2.4)
        ax.set_title(rf"$\gamma = {g}$")
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch17-gamma")


def ch17_select():
    """Two curves from healthy data alone."""
    from ch17_oneclass import gamma_rows, pick_gamma
    rows = gamma_rows()
    r = np.array(rows)
    best = pick_gamma(rows)
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 1.95))
    ax = axes[0]
    ax.semilogx(r[:, 0], r[:, 1], color=BLUE, lw=1.1, marker="o", ms=3)
    ax.axhline(0.05, color=ORANGE, lw=0.9, ls="--")
    ax.text(0.03, 0.10, r"$\nu = 0.05$", color=ORANGE, fontsize=8)
    ax.set_ylabel("held-out alarm rate")
    ax = axes[1]
    ax.semilogx(r[:, 0], r[:, 2], color=GREEN, lw=1.1, marker="o", ms=3)
    ax.set_ylabel("accepted share of box")
    for ax in axes:
        ax.axvline(best, color=GREY_RULE, lw=0.9, ls="--")
        ax.set_xlabel(r"$\gamma$")
        ax.grid(lw=0.3)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch17-select")


def ch17_novel():
    """A classifier and a one-class model meet a new fault."""
    from ch17_oneclass import fleet, novelty
    clf, o, sc, res = novelty()
    tr, te, fa, fb, sc = fleet()
    a, b, G = grid17(sc)
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.2), sharey=True)
    Fs = [-clf.decision_function(G).reshape(a.shape),
          o.decision_function(G).reshape(a.shape)]
    for ax, F, title in zip(axes, Fs, ("two-class SVM", "one-class SVM")):
        ax.contourf(a, b, F, levels=[0, F.max() + 1], colors=[GREEN],
                    alpha=0.13)
        ax.contour(a, b, F, levels=[0], colors=[BLUE], linewidths=1.0)
        ax.plot(tr[:, 0], tr[:, 1], ".", color=BLUE, ms=1.4)
        ax.plot(fa[:, 0], fa[:, 1], ".", color=RED, ms=2.6)
        ax.plot(fb[:, 0], fb[:, 1], ".", color=ORANGE, ms=2.6)
        ax.set_title(title + ": green = healthy", fontsize=8.5)
        ax.set_xlabel(r"$x_1$")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
    axes[0].text(0.6, 4.4, "seen fault A", color=RED, fontsize=8)
    axes[0].text(4.0, 0.45, "new fault B", color=ORANGE, fontsize=8)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch17-novel")


def ch18_union():
    """The luckiest of k rules, against the union bound."""
    from ch18_theory import SEED, finite_gap, luckiest
    rng = np.random.default_rng(SEED)
    ks = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    seen = [luckiest(k, 100, 2000, rng) for k in ks]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.semilogx(ks, [finite_gap(100, k) for k in ks], color=ORANGE, lw=1.2)
    ax.semilogx(ks, seen, "o-", color=BLUE, lw=1.0, ms=2.8)
    ax.axhline(0, color=GREY_RULE, lw=0.6)
    ax.text(30, 0.222, "bound, 95%", color=ORANGE, fontsize=8)
    ax.text(250, 0.085, "mean largest gap", color=BLUE, fontsize=8)
    ax.set_xlabel("number of rules $k$")
    ax.set_ylabel("gap")
    ax.set_ylim(-0.01, 0.27)
    fig.tight_layout()
    save(fig, "ch18-union")


def ch18_shatter():
    """Three points in the plane take all labels; four points do not."""
    from sklearn.svm import SVC
    tri = np.array([[0.2, 0.25], [0.8, 0.25], [0.5, 0.8]])
    fig, axes = plt.subplots(2, 5, figsize=(4.6, 1.95))
    lin = np.array([-0.2, 1.2])

    def dots(ax, P, lab):
        for p, l in zip(P, lab):
            ax.plot(*p, "o" if l > 0 else "s", ms=3.6,
                    color=ORANGE if l > 0 else BLUE)

    def frame(ax):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for code, ax in enumerate(axes.flat[:8]):
        lab = np.where([(code >> i) & 1 for i in range(3)], 1.0, -1.0)
        if abs(lab.sum()) < 3:
            s = SVC(kernel="linear", C=1e6).fit(tri, lab)
            w, b = s.coef_[0], s.intercept_[0]
            if abs(w[1]) > 1e-9:
                ax.plot(lin, -(w[0] * lin + b) / w[1], color=GREEN, lw=0.9)
            else:
                ax.axvline(-b / w[0], color=GREEN, lw=0.9)
        else:
            ax.plot(lin, 0.08 + 0 * lin, color=GREEN, lw=0.9)
        dots(ax, tri, lab)
        frame(ax)
    a, b = axes.flat[8:]
    sq = np.array([[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]])
    dots(a, sq, [1, 1, -1, -1])
    a.set_title("no line", color=RED, fontsize=8, pad=2)
    frame(a)
    dots(b, np.vstack([tri, [[0.5, 0.42]]]), [1, 1, 1, -1])
    b.set_title("no line", color=RED, fontsize=8, pad=2)
    frame(b)
    fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.5)
    save(fig, "ch18-shatter")


def ch18_growth():
    """All labellings, Sauer's sum, and Cover's count for lines."""
    from ch18_theory import sauer
    from ch09_kernel import cover
    ms = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    ax.semilogy(ms, [2.0 ** m for m in ms], color=GREY, lw=1.0)
    ax.semilogy(ms, [sauer(m, 3) for m in ms], color=ORANGE, lw=1.1)
    ax.semilogy(ms, [cover(m, 2) for m in ms], color=BLUE, lw=1.1)
    ax.plot([8], [54], "o", color=RED, ms=3.2)
    ax.annotate("eight machines: 54", (8, 54), (9.5, 7),
                color=RED, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
    ax.axvline(3, color=GREY_RULE, lw=0.6, ls="--")
    ax.text(3.3, 1.6e6, "$d = 3$", fontsize=8)
    ax.text(18.5, 1.2e7, "$2^m$", fontsize=8.5)
    ax.text(21.5, 5.2e3, "Sauer", color=ORANGE, fontsize=8)
    ax.text(23.5, 22, "lines", color=BLUE, fontsize=8)
    ax.set_xlabel("points $m$")
    ax.set_ylabel("labellings")
    ax.set_xlim(1, 30)
    ax.set_ylim(1, 2e9)
    fig.tight_layout()
    save(fig, "ch18-growth")


def ch18_sums():
    """Sign sums of three unit vectors, and 1/rho^2 against m."""
    from ch18_theory import norm_growth
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.25),
                             gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    ang = np.deg2rad([90, 210, 330])
    P = np.column_stack([np.cos(ang), np.sin(ang)])
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(3) * np.cos(t), np.sqrt(3) * np.sin(t), color=ORANGE,
            lw=0.9, ls="--")
    for p in P:
        ax.annotate("", p, (0, 0), arrowprops=dict(arrowstyle="->",
                                                   color=BLUE, lw=0.9))
    for code in range(8):
        s = np.array([1 if (code >> i) & 1 else -1 for i in range(3)])
        q = s @ P
        ax.plot(*q, "o", color=GREEN, ms=3)
    ax.text(0.05, -1.45, r"$\sqrt{3}$", fontsize=8, ha="center", color=ORANGE)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    r = np.array(norm_growth())
    ax.loglog(r[:, 0], r[:, 2], "o-", color=RED, lw=1, ms=2.8)
    ax.loglog(r[:, 0], r[:, 1], "o-", color=GREEN, lw=1, ms=2.8)
    ax.loglog(r[:, 0], r[:, 0], color=GREY, lw=0.8, ls="--")
    ax.text(r[3, 0], r[3, 2] * 1.7, "random labels", color=RED, fontsize=8,
            ha="right")
    ax.text(r[-1, 0], r[-1, 1] * 0.42, "true labels", color=GREEN, fontsize=8,
            ha="right")
    ax.text(r[-1, 0], r[-1, 0] * 1.6, r"$1/\rho^2 = m$", color=GREY,
            fontsize=8, ha="right")
    ax.set_xlabel("points $m$")
    ax.set_ylabel(r"$1/\rho^2$")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18-sums")


def ch18_ramp():
    """Zero-one loss, ramp, and hinge."""
    t = np.linspace(-1.5, 2, 701)
    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    ax.plot(t, np.maximum(0, 1 - t), color=GREY, lw=1.0)
    ax.plot(t, np.clip(1 - t, 0, 1), color=ORANGE, lw=1.6)
    ax.plot(t[t < 0], 1 + 0 * t[t < 0], color=BLUE, lw=1.0, ls="--")
    ax.plot(t[t > 0], 0 * t[t > 0], color=BLUE, lw=1.0, ls="--")
    ax.text(-0.35, 1.7, "hinge", fontsize=8)
    ax.text(-0.95, 0.72, "zero-one", color=BLUE, fontsize=8)
    ax.text(0.62, 0.62, "ramp", color=ORANGE, fontsize=8)
    ax.set_xlabel("$t = y f(x)$")
    ax.set_ylabel("loss")
    ax.set_ylim(-0.1, 2.6)
    fig.tight_layout()
    save(fig, "ch18-ramp")


def ch18_dims():
    """More features: test error and bounds with fixed and growing radius."""
    from ch18_theory import dimension_run, vc_gap
    rows = dimension_run()
    ns = sorted({r[0] for r in rows})
    fx = np.array([r[2:] for r in rows if r[1] == "fixed"])
    gr = np.array([r[2:] for r in rows if r[1] == "grow"])
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.05))
    ax = axes[0]
    ax.semilogx(ns, gr[:, 2], "o-", color=RED, lw=1, ms=2.8)
    ax.semilogx(ns, fx[:, 2], "o-", color=GREEN, lw=1, ms=2.8)
    ax.text(380, 0.2, r"$R$ grows", color=RED, fontsize=8)
    ax.text(500, 0.1, r"$R$ fixed", color=GREEN, fontsize=8)
    ax.set_xlabel("features $n$")
    ax.set_ylabel("test error")
    ax.set_ylim(0, 0.55)
    ax = axes[1]
    ax.loglog(ns, gr[:, 3], "o-", color=RED, lw=1, ms=2.8)
    ax.loglog(ns, fx[:, 3], "o-", color=GREEN, lw=1, ms=2.8)
    ax.loglog(ns, [vc_gap(200, n + 1) for n in ns], color=GREY, lw=1,
              ls="--")
    ax.axhline(1, color=GREY_RULE, lw=0.6)
    ax.text(2.2, 7, "margin bound,\n" + r"$R$ grows", color=RED, fontsize=8)
    ax.text(600, 0.72, r"$R$ fixed", color=GREEN, fontsize=8)
    ax.text(2.2, 0.27, "VC term", color=GREY, fontsize=8)
    ax.set_xlabel("features $n$")
    ax.set_ylabel("bound")
    ax.set_ylim(0.2, 25)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18-dims")


def ch18_bounds():
    """Errors and bounds against C, gamma, and m on the rings."""
    from ch18_theory import CS, GAMMA, GS, size_run, sweep
    a = sweep([(GAMMA, C) for C in CS])
    b = sweep([(g, 1.0) for g in GS])
    c = np.array(size_run())
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 2.2), sharey=True)
    runs = ((axes[0], CS, a[:, 0], a[:, 1], a[:, 2], a[:, 5], "$C$"),
            (axes[1], GS, b[:, 0], b[:, 1], b[:, 2], b[:, 5], r"$\gamma$"),
            (axes[2], c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 6], "$m$"))
    for ax, x, tr, ts, sv, bd, name in runs:
        ax.axhline(1, color=GREY_RULE, lw=0.6)
        ax.loglog(x, bd, "o-", color=ORANGE, lw=1, ms=2.4)
        ax.loglog(x, sv, "o-", color=GREEN, lw=1, ms=2.4)
        ax.loglog(x, ts, "o-", color=RED, lw=1, ms=2.4)
        ax.loglog(x, tr, "o-", color=BLUE, lw=1, ms=2.4)
        ax.set_xlabel(name)
        ax.set_ylim(5e-4, 100)
    t = axes[0]
    t.text(0.02, 6, "margin\nbound", color=ORANGE, fontsize=8)
    t.text(30, 0.45, "SV share", color=GREEN, fontsize=8)
    axes[1].text(0.6, 0.012, "train", color=BLUE, fontsize=8)
    axes[1].text(150, 0.07, "test", color=RED, fontsize=8)
    axes[0].set_ylabel("rate or bound")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch18-bounds")


def ch19_flow():
    """The route of the data: what each part may be used for."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(4.85, 2.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, col, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fill, ec=col, lw=0.9,
                                    boxstyle="round,pad=0,rounding_size=1"))
        ax.text(x + w / 2, y + h / 2, text, color=col, ha="center",
                va="center", fontsize=8)

    def arrow(a, b, col=GREY):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(
            arrowstyle="-|>", color=col, lw=0.8, shrinkA=0, shrinkB=0))
    box(0, 15, 18, 14, "178 wines", BLUE)
    box(28, 25, 38, 17, "124 wines for work\nscaler, search, Platt fit,\n"
        "nested comparison", GREEN)
    box(28, 2, 38, 13, "54 test wines, locked\nopened once", RED)
    box(76, 25, 24, 17, "written\nprotocol:\nmodel, floor", ORANGE)
    box(76, 2, 24, 13, "one error\nrate, reported", BLUE)
    arrow((18, 25), (28, 32))
    arrow((18, 19), (28, 9))
    arrow((66, 33.5), (76, 33.5))
    arrow((80, 25), (60, 15), RED)
    arrow((66, 8.5), (76, 8.5), RED)
    ax.text(17.5, 34.5, "stratified", fontsize=8, ha="center")
    ax.text(73, 17, "then", fontsize=8, ha="left")
    save(fig, "ch19-flow")


def ch19_units():
    """Raw distance between wines is the proline difference."""
    from ch19_case import SHORT, load
    X, y, tr, te = load()
    A = X[tr]
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.35),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    o = np.argsort(A.std(0))
    ax.barh(range(13), A.std(0)[o], color=BLUE, height=0.65)
    ax.set_xscale("log")
    ax.set_yticks(range(13))
    ax.set_yticklabels([SHORT[i] for i in o], fontsize=8)
    ax.set_xlabel("standard deviation")
    ax.tick_params(length=2)
    ax = axes[1]
    i, j = np.triu_indices(len(A), 1)
    rng = np.random.default_rng(20261101)
    k = rng.choice(len(i), 1500, replace=False)
    d = ((A[i[k]] - A[j[k]]) ** 2)
    ax.loglog(d[:, 12] + 1, d.sum(1), ".", ms=1.6, color=RED)
    ax.set_xlabel("squared proline gap")
    ax.set_ylabel("squared distance")
    ax.text(25, 2.5, "the two numbers\nnearly coincide", color=RED,
            fontsize=8)
    ax.set_ylim(1, 1e6)
    ax.set_xlim(1, 1e6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch19-units")


def ch19_grid():
    """Fold accuracy over the grid, with the pick of the one-SE rule."""
    from ch19_case import CS, GS, grid_search, load
    X, y, tr, te = load()
    g, M = grid_search(X[tr], y[tr])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    im = ax.imshow(M, cmap="Blues", vmin=0.4, vmax=1.0, origin="lower",
                   aspect="auto")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center",
                    va="center", fontsize=8,
                    color="white" if M[i, j] > 0.75 else GREY)
    ax.add_patch(plt.Rectangle((2.5, 2.5), 1, 1, fill=False, ec=ORANGE,
                               lw=1.6))
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in GS])
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"$10^{{{int(np.log10(v))}}}$" for v in CS])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("$C$")
    ax.tick_params(length=2)
    save(fig, "ch19-grid")


def ch19_plane():
    """Two principal components, for display, with pair lines."""
    from sklearn.svm import SVC
    from ch19_case import load, plane
    X, y, tr, te = load()
    A, c = X[tr], y[tr]
    s, p = plane(A)
    Z = p.transform(s.transform(A))
    m = SVC(kernel="linear", C=0.01).fit(Z, c)
    u, v = np.meshgrid(np.linspace(-4.8, 4.8, 300), np.linspace(-3.6, 4,
                                                                300))
    h = m.predict(np.c_[u.ravel(), v.ravel()]).reshape(u.shape)
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    cols = [BLUE, GREEN, ORANGE]
    ax.contourf(u, v, h, levels=[-0.5, 0.5, 1.5, 2.5], colors=cols,
                alpha=0.09)
    ax.contour(u, v, h, levels=[0.5, 1.5], colors=GREY, linewidths=0.7)
    keep = np.zeros(len(c), bool)
    keep[m.support_] = True
    for k, (col, mk) in enumerate(zip(cols, "o^s")):
        i = (c == k) & keep
        ax.scatter(Z[i, 0], Z[i, 1], s=11, marker=mk, facecolor=col,
                   edgecolor=col, lw=0.4)
        i = (c == k) & ~keep
        ax.scatter(Z[i, 0], Z[i, 1], s=11, marker=mk, facecolor="white",
                   edgecolor=col, lw=0.6)
    ax.text(3.0, -0.6, "class 0", color=BLUE, fontsize=8)
    ax.text(-2.75, -3.3, "class 1", color=GREEN, fontsize=8)
    ax.text(-4.6, 3.4, "class 2", color=ORANGE, fontsize=8)
    ax.set_xlabel("first component")
    ax.set_ylabel("second component")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.6, 4)
    ax.tick_params(length=2)
    save(fig, "ch19-plane")


def ch19_weights():
    """Weights of the three pair machines and their bootstrap signs."""
    from ch19_case import SHORT, linear_fit, load, stability
    X, y, tr, te = load()
    A, c = X[tr], y[tr]
    w = linear_fit(A, c)[1].T
    q = stability(A, c).T
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.75), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    big = np.abs(w).max()
    names = ["0 v 1", "0 v 2", "1 v 2"]
    for ax, M, cmap, lo, hi, fmt, title in (
            (axes[0], w, "RdBu_r", -big, big, "{:+.2f}", "weight"),
            (axes[1], q, "Greys", 0.3, 3.0, "{:.2f}", "sign kept")):
        ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(13):
            for j in range(3):
                dark = cmap == "RdBu_r" and abs(M[i, j]) > 0.5 * big
                ax.text(j, i, fmt.format(M[i, j]), ha="center",
                        va="center", fontsize=8,
                        color="white" if dark else GREY)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=9, pad=3, y=1.08)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(13))
    axes[0].set_yticklabels(SHORT, fontsize=8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch19-weights")


def ch19_floor():
    """Out-of-fold largest probability, with the floor."""
    from ch19_case import FLOOR, load, probabilities
    X, y, tr, te = load()
    A, c = X[tr], y[tr]
    p = probabilities(A, c)
    top, ok = p.max(1), p.argmax(1) == c
    fig, ax = plt.subplots(figsize=(3.9, 1.9))
    b = np.linspace(0.45, 1, 23)
    ax.hist(top[ok], bins=b, color=BLUE, alpha=0.85)
    ax.hist(top[~ok], bins=b, color=RED)
    ax.axvline(FLOOR, color=ORANGE, lw=1, ls="--")
    ax.text(0.885, 25, "floor 0.9", color=ORANGE, fontsize=8, ha="right")
    ax.text(0.5, 12, "right label", color=BLUE, fontsize=8)
    ax.text(0.5, 7.5, "wrong label", color=RED, fontsize=8)
    ax.set_xlabel("largest probability, out of fold")
    ax.set_ylabel("wines")
    ax.tick_params(length=2)
    save(fig, "ch19-floor")


def ch19_choose():
    """Decision map for kernel, solver, and settings."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(4.85, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, fc="white", ec=col, lw=0.9,
                                    boxstyle="round,pad=0,rounding_size=1"))
        ax.text(x + w / 2, y + h / 2, text, color=col, ha="center",
                va="center", fontsize=8)

    def arrow(a, b, text=""):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(
            arrowstyle="-|>", color=GREY, lw=0.8, shrinkA=0, shrinkB=0))
        if text:
            ax.text((a[0] + b[0]) / 2 + 1, (a[1] + b[1]) / 2, text,
                    fontsize=8, va="center")
    box(10, 62, 80, 10, "split, scale in a pipeline, fit a linear model first\n"
        "(Chapter 14)", BLUE)
    box(14, 45, 72, 10, "features at least as many as samples,\n"
        "or text-like sparse features?", ORANGE)
    arrow((50, 62), (50, 55))
    box(0, 24, 30, 13, "stay linear:\ncoordinate descent\n(Chapter 13)", GREEN)
    box(35, 24, 30, 13, "more than about\n$10^5$ samples?", ORANGE)
    box(70, 24, 30, 13, "linear on features\nor random features\n"
        "(Chapter 13)", GREEN)
    box(35, 3, 30, 14, "Gaussian, SMO,\nsearch $C, \\gamma$\n"
        "(Ch. 11, 12, 14)", GREEN)
    box(70, 3, 30, 14, "keep the kernel\nif nested folds\nshow a gain", BLUE)
    arrow((35, 45), (15, 37), "yes")
    arrow((50, 45), (50, 37), "no")
    arrow((65, 30.5), (70, 30.5))
    ax.text(67.5, 39, "yes", fontsize=8, ha="center")
    arrow((50, 24), (50, 17), "no")
    arrow((65, 10), (70, 10))
    save(fig, "ch19-choose")


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in list(globals().items()):
        if re.match(r"ch\d\d_", name) and name.startswith(pick):
            fn()
