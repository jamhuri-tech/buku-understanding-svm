"""Chapter 7: the soft margin, its dual, and the role of C.

The nine machines are the eight of machines.py, which stays as it is,
plus the rogue machine z = (2, 2) with label +1 of Chapters 2, 4, and
6. It lies inside the healthy hull, so no line separates the nine.
"""
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.svm import SVC

from ch04_margin import feasible
from ch06_dual import dual_matrix
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
Z_ROGUE = np.array([2.0, 2.0])
X9 = np.vstack([X, Z_ROGUE])
y9 = np.append(y, 1.0)


def solve_soft(C, X=X, y=y):
    """Maximise the dual: 0 <= a <= C, a'y = 0."""
    Q = dual_matrix(X, y)
    return minimize(
        lambda a: a @ Q @ a / 2 - a.sum(), np.zeros(len(y)),
        jac=lambda a: Q @ a - 1, method="SLSQP",
        bounds=[(0, C)] * len(y),
        constraints={"type": "eq", "fun": lambda a: a @ y,
                     "jac": lambda a: y},
        tol=1e-13, options={"maxiter": 2000})


def kinds(a, C, tol=1e-5):
    """One letter per point: 0 idle, f free, C capped."""
    t = tol * max(1, C)
    return "".join("0" if v < t else "C" if v > C - t else "f"
                   for v in a)


def offset_range(w, a, C, X=X, y=y):
    """All offsets that fit the KKT conditions for this a."""
    k = kinds(a, C)
    s = X @ w
    lo, hi = -np.inf, np.inf
    for ki, si, yi in zip(k, s, y):
        if ki == "f":                      # the score is exactly yi
            lo, hi = max(lo, yi - si), min(hi, yi - si)
        elif (ki == "0") == (yi > 0):      # b + si >= yi
            lo = max(lo, yi - si)
        else:                              # b + si <= yi
            hi = min(hi, yi - si)
    return lo, hi


def soft_line(C, X=X, y=y):
    """Line, slacks, and multipliers of the soft margin."""
    a = solve_soft(C, X, y).x
    w = (a * y) @ X
    fr = [i for i, ki in enumerate(kinds(a, C)) if ki == "f"]
    if fr:
        b = np.mean(y[fr] - X[fr] @ w)
    else:
        b = sum(offset_range(w, a, C, X, y)) / 2
    xi = np.maximum(0, 1 - y * (X @ w + b))
    return w, b, xi, a


def primal(w, xi, C):
    """The soft-margin objective."""
    return w @ w / 2 + C * xi.sum()


def kkt_soft(w, b, a, C, X=X, y=y):
    """Largest breach of the soft-margin conditions."""
    s = 1 - y * (X @ w + b)
    xi = np.maximum(0, s)
    return max(np.abs(w - (a * y) @ X).max(), abs(a @ y),
               np.abs(a * (s - xi)).max(),
               np.abs((C - a) * xi).max(), (a - C).max(), (-a).max())


def exact_soft(free, capped, C, X=X, y=y):
    """Solve for the free multipliers and b in fractions."""
    xs = [[F(int(v)) for v in row] for row in X]
    ys = [F(int(v)) for v in y]
    k = len(free)

    def q(i, j):
        return ys[i] * ys[j] * (xs[i][0] * xs[j][0]
                                + xs[i][1] * xs[j][1])
    M = [[q(i, j) for j in free] + [ys[i],
         1 - C * sum(q(i, j) for j in capped)] for i in free]
    M.append([ys[i] for i in free] + [F(0),
             -C * sum(ys[j] for j in capped)])
    for c in range(k + 1):                 # Gauss-Jordan
        p = next(r for r in range(c, k + 1) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(k + 1):
            if r != c:
                M[r] = [u - M[r][c] * v for u, v in zip(M[r], M[c])]
    return [row[-1] for row in M]


def least_slack(X=X9, y=y9):
    """Linear program: the smallest possible total slack."""
    m = len(y)
    A = np.c_[-(y[:, None] * X), -y, -np.eye(m)]
    r = linprog(np.r_[0, 0, 0, np.ones(m)], A_ub=A, b_ub=-np.ones(m),
                bounds=[(None, None)] * 3 + [(0, None)] * m)
    return r.fun


def reduced_bridge(C, X=X9, y=y9):
    """Hull cap C/s, and the bridge ends p and q, of the solution."""
    a = solve_soft(C, X, y).x
    s = a[y > 0].sum()
    p = (a[y > 0] @ X[y > 0]) / s
    q = (a[y < 0] @ X[y < 0]) / s
    return C / s, p, q, s


if __name__ == "__main__":
    # Step 1: the hard margin fails on the nine machines.
    print("nine machines, feasible:", feasible(X9, y9))

    # Step 2: the dual now has a solution.
    r = solve_soft(0.5, X9, y9)
    print("dual: status", r.status, " sum alpha", round(r.x.sum(), 3))
    print("alpha", r.x)

    # Step 3: the worked solution at C = 1/2, in fractions.
    sol = exact_soft([2, 6], [0, 3, 4, 8], F(1, 2), X9, y9)
    print("exact: a3, a7, b =", *sol)
    w, b, xi, a = soft_line(0.5, X9, y9)
    print("w", w, " b", round(b, 3))
    print("xi", xi)
    print("kinds", kinds(a, 0.5), " P", round(primal(w, xi, 0.5), 3),
          " D", round(a.sum() - w @ w / 2, 3),
          " breach", round(kkt_soft(w, b, a, 0.5, X9, y9), 5))

    # Step 4: scikit-learn agrees.
    m = SVC(kernel="linear", C=0.5, tol=1e-8).fit(X9, y9)
    print("SVC: w", m.coef_[0], " b", m.intercept_.round(3))

    # Step 5: the eight machines with C between 1 and 5/2.
    for C in (F(1), F(2), F(5, 2)):
        print("eight, C =", C, ": a5, a7, b =",
              *exact_soft([4, 6], [3], C))

    # Step 6: a sweep over C.
    Cs = (0.05, 0.1, 0.2, 0.5, 1, 2, 2.5, 10, 100)
    for name, Xd, yd in (("eight", X, y), ("nine", X9, y9)):
        print(name)
        for C in Cs:
            w, b, xi, a = soft_line(C, Xd, yd)
            err = int(np.sum(yd * (Xd @ w + b) < 0))
            print(f"C {C:6.2f}  w {w[0]:.3f} {w[1]:.3f}  "
                  f"width {2 / np.linalg.norm(w):.3f}  "
                  f"xi {xi.sum():.3f}  wrong {err}  {kinds(a, C)}")

    # Step 7: the frozen line and the least total slack.
    print("least total slack:", round(least_slack(), 6))
    for C in (1, 10):
        a = np.array([C / 2 + .4, 0, .3 * C, C, C / 2 + .4, 0, .3 * C,
                      0, C])
        wf = (a * y9) @ X9
        print(f"C = {C}: balance {a @ y9 + 0:.1f}  w", wf,
              " breach", round(kkt_soft(wf, -3.8, a, C, X9, y9), 12))

    # Step 8: no free point, so an interval of offsets.
    C = 1 / 20
    a = solve_soft(C).x
    wt = (a * y) @ X
    lo, hi = offset_range(wt, a, C)
    print("tie: kinds", kinds(a, C), " w", wt,
          f" b in [{lo:.3f}, {hi:.3f}]")
    m = SVC(kernel="linear", C=C, tol=1e-8).fit(X, y)
    print("SVC offset", m.intercept_.round(3))

    # Step 9: the walker of Chapter 4, with C = 1.
    for t in (0.6, 0.9, 1.0, 1.1, 1.5):
        zt = np.array([5.0, 5.0]) - 2 * t
        Xw = np.vstack([X, zt])
        w, b, xi, a = soft_line(1, Xw, y9)
        print(f"t {t:.2f}  w {w[0]:5.3f} {w[1]:5.3f}  b {b:6.3f}  "
              f"width {2 / np.linalg.norm(w):5.3f}  "
              f"xi9 {xi[8]:5.3f}  a9 {a[8]:5.3f}")

    # Step 10: units. Scaling x by s and C by 1/s^2 keeps the line.
    w1, b1, _, _ = soft_line(0.5, X9, y9)
    w2, b2, _, _ = soft_line(0.5 / 100, 10 * X9, y9)
    print("scaled: 10 w", 10 * w2, " b", round(b2, 3))

    # Step 11: the bridge with a cap.
    cap, p, q, s = reduced_bridge(0.5)
    print(f"cap {cap:.4f}  s {s:.3f}  p {p}  q {q}")
    print("s (p - q)", s * (p - q))
