"""Chapter 9: feature maps, the kernel trick, XOR and the kernel perceptron."""
from collections import Counter
from fractions import Fraction as F
from math import comb, sqrt

import numpy as np
from scipy.optimize import linprog
from sklearn.svm import SVC

from ch06_dual import solve_dual
from ch07_soft import X9, y9
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101

XOR = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
y_xor = np.array([-1, -1, 1, 1], dtype=float)


def phi2(x):
    """The map of K(x, z) = (x'z)^2 for two features."""
    return np.array([x[0]**2, sqrt(2) * x[0] * x[1], x[1]**2])


def phi6(x):
    """The map of K(x, z) = (1 + x'z)^2 for two features."""
    r = sqrt(2)
    return np.array([1, r * x[0], r * x[1], x[0]**2,
                     r * x[0] * x[1], x[1]**2])


def quad(A, B):
    """K(x, z) = (x'z)^2 for all pairs of rows."""
    return (A @ B.T) ** 2


def quad1(A, B):
    """K(x, z) = (1 + x'z)^2 for all pairs of rows."""
    return (1 + A @ B.T) ** 2


def separable(X, y):
    """The feasibility test of Chapter 2."""
    A = -y[:, None] * np.column_stack([X, np.ones(len(y))])
    r = linprog(np.zeros(A.shape[1]), A_ub=A,
                b_ub=-np.ones(len(y)), bounds=(None, None))
    return r.status == 0


def count_features(n, d):
    """Number of monomials in n variables of degree at most d."""
    return comb(n + d, d)


def cover(m, p):
    """Labellings of m general points that a plane in R^p separates."""
    return 2 * sum(comb(m - 1, k) for k in range(min(p, m - 1) + 1))


def monomials(X, d):
    """All monomials of degree 1 to d, one column each."""
    cols = [np.ones(len(X))]
    for t in range(1, d + 1):
        for k in range(t + 1):
            cols.append(X[:, 0] ** (t - k) * X[:, 1] ** k)
    return np.column_stack(cols[1:])


def labellings(X, d):
    """How many of the 2^m labellings a degree-d surface separates."""
    m = len(X)
    Z = monomials(X, d)
    Z = (Z - Z.mean(0)) / Z.std(0)
    hits = 0
    for code in range(2 ** m):
        lab = np.where([(code >> i) & 1 for i in range(m)], 1.0, -1.0)
        if abs(lab.sum()) == m:
            hits += 1                       # one class: any far line
        elif separable(Z, lab):
            hits += 1
    return hits


def fit_kernel(K, y=y, C=None):
    """Solve the dual from the Gram matrix alone: alpha and b."""
    Q = np.outer(y, y) * K
    if C is None:
        a = solve_dual(Q, y).x
    else:
        a = SVC(kernel="precomputed", C=C, tol=1e-10).fit(K, y)
        full = np.zeros(len(y))
        full[a.support_] = np.abs(a.dual_coef_[0])
        a = full
    S = a > 1e-7
    free = S if C is None else S & (a < C - 1e-7)
    b = np.mean(y[free] - K[free][:, S] @ (a[S] * y[S]))
    return a, b


def score(A, Xs, ys, a, b, kernel=quad1):
    """f(x) = sum a_i y_i K(x_i, x) + b for each row of A."""
    return kernel(A, Xs) @ (a * ys) + b


def exact_xor():
    """Solve the four KKT equations of XOR in fractions."""
    p, q, t = F(10, 3), F(2), F(8, 3)
    a = [p, q, t, t]
    Xi = [(0, 0), (1, 1), (1, 0), (0, 1)]
    Y = [-1, -1, 1, 1]
    K = [[(1 + u[0]*v[0] + u[1]*v[1]) ** 2 for v in Xi] for u in Xi]
    b = F(-1)
    tight = [Y[i] * (sum(a[j] * Y[j] * K[i][j] for j in range(4)) + b)
             for i in range(4)]
    return a, b, tight, K


def poly_svm(X, y, d, C=1e6):
    """SVC with K(x, z) = (1 + x'z)^d."""
    return SVC(kernel="poly", degree=d, gamma=1, coef0=1, C=C,
               tol=1e-8).fit(X, y)


def distance(x, z, kernel=quad1):
    """Distance between phi(x) and phi(z), from K alone."""
    A = np.array([x, z])
    K = kernel(A, A)
    return sqrt(K[0, 0] - 2 * K[0, 1] + K[1, 1])


def rings(m=60, seed=SEED, noise=0.08):
    """Two noisy rings of radius 1 and 2, half of the points in each."""
    rng = np.random.default_rng(seed)
    k = m // 2
    t = rng.uniform(0, 2 * np.pi, m)
    r = np.r_[np.full(k, 1.0), np.full(m - k, 2.0)]
    r = r + noise * rng.standard_normal(m)
    A = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return A, np.r_[-np.ones(k), np.ones(m - k)]


def kernel_perceptron(K, y, passes=100, order=None):
    """Counts a and offset b; K is the Gram matrix."""
    m = len(y)
    a = np.zeros(m)
    b = 0.0
    log = []
    order = range(m) if order is None else order
    for _ in range(passes):
        wrong = 0
        for i in order:
            if y[i] * (K[i] @ (a * y) + b) <= 0:
                a[i] += 1
                b += y[i]
                wrong += 1
                log.append(i)
        if wrong == 0:
            break
    return a, b, log


def spectrum(s, t, k=2):
    """Inner product of the counts of length-k substrings."""
    def grams(u):
        return Counter(u[i:i + k] for i in range(len(u) - k + 1))
    a, b = grams(s), grams(t)
    return sum(a[g] * b[g] for g in a)


if __name__ == "__main__":
    # The same number by the two routes.
    p, q = np.array([1.0, 2.0]), np.array([3.0, 1.0])
    print("map", round(phi2(p) @ phi2(q), 9), " kernel", (p @ q) ** 2)
    print("map", round(phi6(p) @ phi6(q), 9), " kernel", (1 + p @ q) ** 2)
    print()

    # Size of the explicit map.
    for n, d in ((2, 2), (10, 3), (100, 3), (256, 4), (100, 5), (1000, 5)):
        print(f"n {n:5d}  d {d}  features {count_features(n, d):>18,d}")
    print()

    # Labellings of the eight machines, and of eight random points.
    r = np.random.default_rng(SEED).normal(size=(8, 2))
    for d in (1, 2, 3, 4):
        p = count_features(2, d) - 1
        print(f"degree {d}  features {p:2d}  Cover {cover(8, p):3d}  "
              f"random {labellings(r, d):3d}  machines {labellings(X, d):3d}")
    print()

    # Distance in feature space, from K alone.
    x4, x5 = X[3], X[4]
    print("distance", round(distance(x4, x5), 3), " map",
          round(np.linalg.norm(phi6(x4) - phi6(x5)), 3))
    print()

    # XOR through the dual, from the Gram matrix alone.
    K = quad1(XOR, XOR)
    a, b = fit_kernel(K, y_xor)
    ea, eb, tight, _ = exact_xor()
    print("alpha", a, " b", round(b, 6))
    print("scores", score(XOR, XOR, y_xor, a, b))
    print("exact", [str(v) for v in ea], "b", eb,
          "y f", [str(v) for v in tight])
    Z = np.array([phi6(x) for x in XOR])
    w = (a * y_xor) @ Z
    print("w", w, " |w|^2", round(w @ w, 3))
    print("sum alpha", sum(ea), " width", round(2 / sqrt(sum(ea)), 3))
    c = poly_svm(XOR, y_xor, 2)
    print("SVC alpha", np.abs(c.dual_coef_[0]), " b", c.intercept_.round(3))
    print()

    # The nine machines, in units of ten.
    Z9 = X9 / 10
    for d in (1, 2, 3):
        c = poly_svm(Z9, y9, d)
        lifted = separable(monomials(X9, d), y9)
        print(f"degree {d}  separable {lifted!s:5}  right "
              f"{int(c.score(Z9, y9) * 9)} of 9  "
              f"support {len(c.support_)}")
    print()

    # Rings, centred and shifted; library defaults.
    A, t = rings()
    print("rings: line", separable(A, t), " squares", separable(A ** 2, t))
    B = A + 3
    default = SVC(kernel="poly", degree=2).fit(B, t)
    print(f"shifted: default {default.score(B, t):.2f}  coef0 = 1 "
          f"{poly_svm(B, t, 2, 100).score(B, t):.2f}")
    print()

    # Kernel perceptron.
    a, b, log = kernel_perceptron(K, y_xor)
    print("xor counts", a, " b", b, " mistakes", len(log))
    print("scores", score(XOR, XOR, y_xor, a, b))
    a, b, log = kernel_perceptron(quad1(A, A), t)
    print("rings mistakes", len(log), " points used", int((a > 0).sum()),
          "of", len(t))
    c = poly_svm(A, t, 2)
    print("SVM support vectors", len(c.support_))
    print()

    # Inputs that are not vectors.
    print("spectrum", spectrum("abab", "bab"))
