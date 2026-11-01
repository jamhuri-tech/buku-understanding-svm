"""Chapter 5: the KKT conditions as a certificate of the solution."""
from fractions import Fraction as F
from itertools import combinations

import numpy as np
from scipy.optimize import nnls

from ch04_margin import TOL, hard_margin, problem, slacks, solve_qp
from machines import X, alpha, y

np.set_printoptions(precision=3, suppress=True)


def kkt_residuals(w, b, a, X=X, y=y):
    """The four KKT conditions: each entry is zero when one holds."""
    s = y * (X @ w + b) - 1                  # slack, s = -g
    return {"stationarity": np.abs(w - (a * y) @ X).max(),
            "balance": abs(a @ y),
            "primal": max(0.0, -s.min()),
            "dual": max(0.0, -a.min()),
            "complementarity": np.abs(a * s).max()}


def multipliers(w, b, X=X, y=y, tol=TOL):
    """The closest alpha >= 0 on the active machines, and its misfit."""
    act = np.where(slacks(w, b, X, y) < tol)[0]
    Z = np.column_stack([y[act, None] * X[act], y[act]])
    a, misfit = nnls(Z.T, np.r_[w, 0.0])
    full = np.zeros(len(y))
    full[act] = a
    return full, misfit


def solve_active(S, X=X, y=y):
    """Treat the machines in S as equalities: the KKT linear system."""
    n, k = X.shape[1], len(S)
    Xs, ys = X[S], y[S]
    K = np.zeros((n + 1 + k, n + 1 + k))
    K[:n, :n] = np.eye(n)
    K[:n, n + 1:] = -(ys[:, None] * Xs).T    # w - sum alpha y x = 0
    K[n, n + 1:] = ys                        # sum alpha y = 0
    K[n + 1:, :n] = ys[:, None] * Xs         # y (w'x + b) = 1
    K[n + 1:, n] = ys
    r = np.r_[np.zeros(n + 1), np.ones(k)]
    z = np.linalg.lstsq(K, r, rcond=None)[0]
    return z[:n], z[n], z[n + 1:], np.abs(K @ z - r).max() < 1e-9


def fate(S, X=X, y=y):
    """Why a guess of the active set fails, or 'pass'."""
    w, b, a, solved = solve_active(list(S), X, y)
    if not solved:
        return "no solution"
    broken = slacks(w, b, X, y).min() < -1e-9
    negative = len(S) > 0 and a.min() < -1e-9
    if broken and negative:
        return "both"
    if broken:
        return "broken"
    return "negative alpha" if negative else "pass"


def price(i, eps):
    """Best objective when machine i must reach 1 + eps."""
    P, q, A, c = problem(X, y)
    c = c.copy()
    c[i] += eps
    z = solve_qp(P, q, A, c, np.array([1.0, 2.0, -11.0])).x
    return z[:2] @ z[:2] / 2


if __name__ == "__main__":
    # Step 1: the exact check on the eight machines, in fractions.
    w, b = (F(1), F(2)), F(-11)
    a = [F(0), F(0), F(0), F(5, 2), F(2), F(0), F(1, 2), F(0)]
    xs = [tuple(F(int(v)) for v in row) for row in X]
    ys = [F(int(v)) for v in y]
    for i in range(8):
        s = ys[i] * (w[0] * xs[i][0] + w[1] * xs[i][1] + b) - 1
        print(f"{i + 1}  slack {s}  alpha {a[i]}  product {a[i] * s}")
    print("sum alpha y x =",
          [str(sum(a[i] * ys[i] * xs[i][j] for i in range(8)))
           for j in range(2)],
          " sum alpha y =", sum(a[i] * ys[i] for i in range(8)))
    print("sum alpha =", sum(a), " |w|^2 =", w[0] ** 2 + w[1] ** 2)

    # Step 2: the certificate of the solver's answer.
    v, c0, _ = hard_margin(X, y)
    al, misfit = multipliers(v, c0)
    print("alpha", al, " misfit", round(misfit, 7))
    for key, val in kkt_residuals(v, c0, al).items():
        print(f"{key:16s}{val:.1e}")

    # Step 3: three candidates that are not the solution.
    for v, c0 in ((np.array([1.0, 3]), -14.0),
                  (np.array([2.0, 4]), -21.0),
                  (np.array([0.0, 2]), -7.0)):
        al, misfit = multipliers(v, c0)
        print("w", v, " b", c0, " worst slack",
              round(slacks(v, c0).min(), 3),
              " active", np.where(np.abs(slacks(v, c0)) < TOL)[0] + 1,
              " misfit", round(misfit, 3))

    # Step 4: every guess of the active set.
    tally, keep = {}, []
    for k in range(9):
        for S in combinations(range(8), k):
            r = fate(S)
            tally[r] = tally.get(r, 0) + 1
            if r == "pass":
                keep.append([i + 1 for i in S])
    print(tally)
    print("passed:", keep)

    # Step 5: the guess {4, 5} alone.
    v, c0, a2, _ = solve_active([3, 4])
    print("guess {4,5}: w", v, " b", round(c0, 3), " alpha", a2,
          " slack of 7:", round(slacks(v, c0)[6], 3))

    # Step 6: a multiplier is a price.
    e = 1e-4
    for i in range(8):
        d = (price(i, e) - price(i, -e)) / (2 * e)
        print(f"machine {i + 1}: slope {d:.3f}")

    # Step 7: a ninth machine on an edge, (2, 5) with label +1.
    Xn, yn = np.vstack([X, [2, 5]]), np.append(y, 1.0)
    v, c0, _ = hard_margin(Xn, yn)
    al, misfit = multipliers(v, c0, Xn, yn)
    print("ninth (2,5): w", v, " b", round(c0, 3), " slack",
          round(slacks(v, c0, Xn, yn)[8], 3), " misfit", round(misfit, 6))
    print("this alpha:", al, " first alpha:", np.r_[alpha, 0.0])
    print(kkt_residuals(v, c0, np.r_[alpha, 0.0], Xn, yn)["stationarity"])
