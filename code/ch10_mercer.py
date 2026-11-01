"""Chapter 10: which functions are kernels, Mercer, the representer theorem."""
from math import pi, sqrt

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

from ch09_kernel import fit_kernel, quad1
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def linear(A, B):
    """K(x, z) = x'z for all pairs of rows."""
    return A @ B.T


def near(A, B, r=1.0):
    """K(x, z) = 1 when x and z lie within r, else 0."""
    d = np.linalg.norm(A[:, None] - B[None], axis=2)
    return (d <= r + 1e-12).astype(float)


def sigmoid(A, B, kappa=1.0, c=0.0):
    """K(x, z) = tanh(kappa x'z + c)."""
    return np.tanh(kappa * A @ B.T + c)


def low(a, b):
    """K(x, z) = min(x, z) for two arrays of scalars."""
    return np.minimum.outer(a, b)


def passes(K, tol=1e-10):
    """Is K symmetric with no eigenvalue below -tol * largest?"""
    if not np.allclose(K, K.T):
        return False
    e = np.linalg.eigvalsh(K)
    return e[0] >= -tol * max(abs(e[0]), abs(e[-1]))


def gram_map(K, tol=1e-10):
    """Rows R with R R' = K: a feature map on the data alone."""
    e, U = np.linalg.eigh(K)
    keep = e > tol * e[-1]
    return U[:, keep] * np.sqrt(e[keep])


def low_terms(a, b, k):
    """First k Mercer terms of min(x, z) on [0, 1]."""
    s = 0.0
    for j in range(1, k + 1):
        w = (j - 0.5) * pi
        s = s + 2 * np.outer(np.sin(w * a), np.sin(w * b)) / w**2
    return s


def tent(x):
    """The interpolant of least norm, and a wiggle that vanishes at the data."""
    f = np.interp(x, [0, 0.2, 0.5, 0.8, 1], [0, 0.4, 0.2, 0.6, 0.6])
    g = np.where((x > 0.2) & (x < 0.8),
                 0.1 * np.sin(10 * pi * (x - 0.2) / 3), 0.0)
    return f, g


def primal_kernel(K, y, C):
    """Minimise c'Kc/2 + C sum xi over c, b, xi: no dual used."""
    m = len(y)

    def obj(v):
        return v[:m] @ K @ v[:m] / 2 + C * v[m + 1:].sum()

    def grad(v):
        return np.r_[K @ v[:m], 0.0, C * np.ones(m)]

    def con(v):
        return y * (K @ v[:m] + v[m]) - 1 + v[m + 1:]

    def jac(v):
        return np.column_stack([y[:, None] * K, y, np.eye(m)])

    r = minimize(obj, np.zeros(2 * m + 1), jac=grad, method="SLSQP",
                 bounds=[(None, None)] * (m + 1) + [(0, None)] * m,
                 constraints={"type": "ineq", "fun": con, "jac": jac},
                 tol=1e-12, options={"maxiter": 1000})
    return r.x[:m], r.x[m], r


def objective(K, y, c, b, C):
    """J = c'Kc/2 + C sum hinge, the kernel form of Chapter 8."""
    return c @ K @ c / 2 + C * np.maximum(0, 1 - y * (K @ c + b)).sum()


if __name__ == "__main__":
    # Four candidates on the eight machines.
    for name, K in [("linear", linear(X, X)), ("quad1", quad1(X, X)),
                    ("near", near(X, X)), ("sigmoid", sigmoid(X, X))]:
        e = np.linalg.eigvalsh(K)
        print(f"{name:8} least {e[0]:7.3f}  negative "
              f"{int((e < -1e-9).sum())}  passes {passes(K)}")
    print()

    # The three close machines.
    T = X[[3, 4, 5]]
    K = near(T, T)
    print(K)
    print("eigenvalues", np.linalg.eigvalsh(K))
    v = np.array([1, -sqrt(2), 1])
    print("v'Kv", round(v @ K @ v, 3))
    print("distances", round(K[0, 0] + K[1, 1] - 2 * K[0, 1], 3),
          round(K[1, 1] + K[2, 2] - 2 * K[1, 2], 3),
          round(K[0, 0] + K[2, 2] - 2 * K[0, 2], 3))
    print()

    # The dual has no ceiling on an indefinite matrix.
    t3 = np.array([-1.0, 1.0, -1.0])
    Q = np.outer(t3, t3) * K
    r = np.array([1.0, 2.0, 1.0])
    for s in (1, 10, 100):
        print(f"D({s} r) = {s * r.sum() - s**2 * r @ Q @ r / 2:g}")
    print()

    # SVC runs on the sigmoid and raises nothing.
    for kap, cc in [(0.1, 0), (0.03, -0.5)]:
        K = sigmoid(X, X, kap, cc)
        c = SVC(kernel="precomputed", C=1.0).fit(K, y)
        print(f"kappa {kap} c {cc}: least "
              f"{np.linalg.eigvalsh(K)[0]:.3f}, right "
              f"{int(round(c.score(K, y) * 8))} of 8")
    c = SVC(kernel="precomputed", C=1.0).fit(quad1(X, X), y)
    print("quad1: right", int(round(c.score(quad1(X, X), y) * 8)), "of 8")
    print()

    # The sigmoid fails all over a grid of its parameters.
    bad = [not passes(sigmoid(X, X, 10.0**u, v))
           for u in np.linspace(-3, 0.5, 36)
           for v in np.linspace(-2, 2, 41)]
    print("grid failures", sum(bad), "of", len(bad))
    for kap, x in [(1, 1), (1, 2), (0.01, 9)]:
        G = np.tanh(kap * np.outer([x, 2 * x], [x, 2 * x]))
        print(f"kappa {kap}, points {x} and {2 * x}: "
              f"det {np.linalg.det(G):.4f}")
    print()

    # The Gram map on the eight machines.
    K = quad1(X, X)
    R = gram_map(K)
    print("rank", R.shape[1], " worst error",
          round(abs(R @ R.T - K).max(), 9))
    print()

    # The closure rules on random points.
    A = np.random.default_rng(SEED).normal(size=(30, 3))
    K1, K2 = linear(A, A) + quad1(A, A), np.exp(linear(A, A))
    print("least: product", round(np.linalg.eigvalsh(K1 * K2)[0], 3),
          " sum", round(np.linalg.eigvalsh(K1 + K2)[0], 3),
          " difference", round(np.linalg.eigvalsh(K1 - K2)[0], 3))
    print()

    # The min kernel and its Mercer terms.
    t = (np.arange(400) + 0.5) / 400
    K = low(t, t)
    e = np.linalg.eigvalsh(K)[::-1] / 400
    print("Gram/m", e[:4])
    print("exact ", 1 / ((np.arange(1, 5) - 0.5) * pi)**2)
    print("trace", round(e.sum(), 4))
    for k in (1, 3, 10, 100):
        print(f"terms {k:3}  worst error "
              f"{abs(low_terms(t, t, k) - K).max():.4f}")
    print()

    # The interpolant of least norm.
    p = np.array([0.2, 0.5, 0.8])
    v = np.array([0.4, 0.2, 0.6])
    K = low(p, p)
    c = np.linalg.solve(K, v)
    print("c", c, " norm^2", round(v @ c, 4))
    print("wiggle norm^2", round(0.01 * (10 * pi / 3)**2 * 0.3, 4))
    print()

    # Representer: the primal in c, against the dual.
    Z = X / 10
    K = quad1(Z, Z)
    C = 10.0
    a, b = fit_kernel(K, y, C)
    c, b2, _ = primal_kernel(K, y, C)
    print("alpha y", (a * y).round(2))
    print("c      ", c.round(2))
    print("dual  ", K @ (a * y) + b)
    print("primal", K @ c + b2)
    print("J", round(objective(K, y, a * y, b, C), 4),
          round(objective(K, y, c, b2, C), 4))
    print("rank", np.linalg.matrix_rank(K), " difference along null",
          round(abs(K @ (c - a * y)).max(), 4))
