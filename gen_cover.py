"""Cover artwork: a Gaussian SVM on the noisy rings of Chapter 11.

The training data, the kernel, and the solver are those of the book
(code/ch11_rbf.py, rings with noise 0.35). Each band of colour is a band
of the decision function f, the heavy white curve is f = 0, the two
lighter curves are the edges of the street f = +1 and f = -1, and the
marked points are the support vectors.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent / "code"))
from ch11_rbf import noisy  # noqa: E402

NX, NY = 2800, 2233          # 148 mm x 118 mm at about 480 dpi
DPI = 480
PALETTE = ["#1B3B6F", "#2E6DA4", "#1E6F5C", "#7FB685", "#F2C14E",
           "#E07A1F", "#B85C00", "#9B1B30"]


def art(out_path, gamma=2.0, C=10.0, half=2.7, centre=(0.0, 0.0)):
    A, t = noisy()
    clf = SVC(kernel="rbf", gamma=gamma, C=C).fit(A, t)
    hy = half * NY / NX
    g1, g2 = np.meshgrid(
        np.linspace(centre[0] - half, centre[0] + half, NX // 2),
        np.linspace(centre[1] - hy, centre[1] + hy, NY // 2))
    F = clf.decision_function(np.column_stack([g1.ravel(), g2.ravel()]))
    F = F.reshape(g1.shape)
    cmap = LinearSegmentedColormap.from_list("cover", PALETTE)
    top = np.percentile(np.abs(F), 99)
    levels = np.linspace(-top, top, 19)
    fig = plt.figure(figsize=(NX / DPI, NY / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(g1, g2, np.clip(F, -top, top), levels=levels, cmap=cmap)
    ax.contour(g1, g2, F, levels=levels, colors="white", linewidths=0.18,
               alpha=0.55)
    ax.contour(g1, g2, F, levels=[-1, 1], colors="white", linewidths=0.6,
               linestyles="solid")
    ax.contour(g1, g2, F, levels=[0], colors="white", linewidths=1.3)
    sv = np.zeros(len(t), bool)
    sv[clf.support_] = True
    for s, face in ((1, "#9B1B30"), (-1, "#1B3B6F")):
        k = t == s
        ax.scatter(A[k & sv, 0], A[k & sv, 1], s=15, c=face,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter(A[k & ~sv, 0], A[k & ~sv, 1], s=7, c="white",
                   edgecolors=face, linewidths=0.4, zorder=3)
    ax.set_xlim(g1.min(), g1.max())
    ax.set_ylim(g2.min(), g2.max())
    ax.axis("off")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(out_path, "SV", int(sv.sum()), "train acc",
          round(clf.score(A, t), 3), "top", round(top, 2))


if __name__ == "__main__":
    art("tikz/cover-art.png")
