"""Chapter 12: Sequential Minimal Optimisation, from scratch."""
import json
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from ch07_soft import X9, y9, solve_soft
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def linear(A, B):
    """K(x, z) = x'z for all pairs of rows."""
    return A @ B.T


def gaussian(gamma):
    """K(x, z) = exp(-gamma |x - z|^2) as a function of two arrays."""
    def k(A, B):
        d = ((A[:, None] - B[None]) ** 2).sum(axis=2)
        return np.exp(-gamma * d)
    return k


def limits(ai, aj, yi, yj, C):
    """Ends of the segment for alpha_j."""
    if yi != yj:
        return max(0.0, aj - ai), min(C, C + aj - ai)
    return max(0.0, ai + aj - C), min(C, ai + aj)


def snap(v, C):
    """Put a value within rounding of a bound exactly on the bound."""
    if v < 1e-12:
        return 0.0
    return C if v > C * (1 - 1e-12) else v


def pair_step(ai, aj, yi, yj, Ei, Ej, eta, C):
    """Best pair on the segment. Only Ei - Ej and eta matter."""
    L, H = limits(ai, aj, yi, yj, C)
    new_j = aj + yj * (Ei - Ej) / max(eta, 1e-12)
    new_j = min(H, max(L, new_j))
    return snap(ai + yi * yj * (aj - new_j), C), snap(new_j, C)


class Columns:
    """Kernel columns on demand, keeping the latest `size` columns."""

    def __init__(self, X, kernel, size):
        self.X, self.kernel, self.size = X, kernel, size
        self.saved = OrderedDict()
        self.computed = 0
        self.diag = np.array([kernel(X[[i]], X[[i]])[0, 0]
                              for i in range(len(X))])

    def __call__(self, i):
        if i in self.saved:
            self.saved.move_to_end(i)
            return self.saved[i]
        col = self.kernel(self.X, self.X[[i]])[:, 0]
        self.computed += 1
        self.saved[i] = col
        if len(self.saved) > self.size:
            self.saved.popitem(last=False)
        return col


def up_low(a, y, C):
    """Points whose y_i alpha_i can rise, and those where it can fall."""
    up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
    low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
    return up, low


def choose(F, up, low, cols, rule):
    """Working set: first or second order, with the gap of the pair."""
    iu, il = np.flatnonzero(up), np.flatnonzero(low)
    i = iu[np.argmin(F[iu])]
    j = il[np.argmax(F[il])]
    gap = F[j] - F[i]
    if rule == "second" and gap > 0:
        Ki = cols(i)
        c = il[F[il] > F[i]]
        eta = np.maximum(cols.diag[i] + cols.diag[c] - 2 * Ki[c], 1e-12)
        j = c[np.argmax((F[c] - F[i]) ** 2 / eta)]
    return i, j, gap


def offset(a, F, y, C):
    """Mean over free points, or the middle of the allowed interval."""
    t = 1e-9 * max(1.0, min(C, 1e9))
    free = (a > t) & (a < C - t)
    if free.any():
        return -F[free].mean()
    up, low = up_low(a, y, C)
    return -(F[low].max() + F[up].min()) / 2


def smo(X, y, C=np.inf, kernel=linear, rule="second", tol=1e-3,
        steps=2 * 10 ** 5, cache=200, trace=None):
    """Maximise the dual by pairs. F holds f(x_i) - b - y_i."""
    m = len(y)
    a, F = np.zeros(m), -y.astype(float)
    cols = Columns(X, kernel, cache)
    for k in range(steps):
        up, low = up_low(a, y, C)
        i, j, gap = choose(F, up, low, cols, rule)
        if gap <= tol:
            break
        Ki, Kj = cols(i), cols(j)
        eta = cols.diag[i] + cols.diag[j] - 2 * Ki[j]
        new_i, new_j = pair_step(a[i], a[j], y[i], y[j],
                                 F[i], F[j], eta, C)
        F += (new_i - a[i]) * y[i] * Ki
        F += (new_j - a[j]) * y[j] * Kj
        a[i], a[j] = new_i, new_j
        if trace is not None:
            trace.append((i, j, gap, a.copy()))
    else:
        k = steps
    return a, offset(a, F, y, C), k, cols.computed


def kkt_bad(r, a, C, tol):
    """Platt's test on r = y_i E_i."""
    return (r < -tol and a < C) or (r > tol and a > 0)


def platt(X, y, C=np.inf, kernel=linear, tol=1e-3, steps=2 * 10 ** 5,
          seed=SEED, trace=None):
    """Platt's two loops, error cache, and one threshold b."""
    m = len(y)
    a, b, E = np.zeros(m), 0.0, -y.astype(float)
    Kd = np.array([kernel(X[[i]], X[[i]])[0, 0] for i in range(m)])
    rng = np.random.default_rng(seed)
    done, computed = 0, 0

    def take(i, j):
        nonlocal b, done, computed
        if i == j:
            return False
        kij = kernel(X[[i]], X[[j]])[0, 0]
        L, H = limits(a[i], a[j], y[i], y[j], C)
        if H - L < 1e-12:
            return False
        eta = Kd[i] + Kd[j] - 2 * kij
        new_i, new_j = pair_step(a[i], a[j], y[i], y[j], E[i], E[j],
                                 eta, C)
        if abs(new_j - a[j]) < 1e-5 * (new_j + a[j] + 1e-5):
            return False
        di, dj = (new_i - a[i]) * y[i], (new_j - a[j]) * y[j]
        bi = b - E[i] - di * Kd[i] - dj * kij
        bj = b - E[j] - di * kij - dj * Kd[j]
        if 0 < new_i < C:
            new_b = bi
        elif 0 < new_j < C:
            new_b = bj
        else:
            new_b = (bi + bj) / 2
        Ki, Kj = kernel(X, X[[i]])[:, 0], kernel(X, X[[j]])[:, 0]
        computed += 2
        E[:] = E + di * Ki + dj * Kj + new_b - b
        a[i], a[j], b = new_i, new_j, new_b
        done += 1
        if trace is not None:
            trace.append((i, j, a.copy(), b))
        return True

    def examine(j):
        if not kkt_bad(y[j] * E[j], a[j], C, tol):
            return 0
        free = np.flatnonzero((a > 0) & (a < C))
        if len(free) > 1:
            i = free[np.argmax(np.abs(E[free] - E[j]))]
            if take(i, j):
                return 1
        for i in rng.permutation(free):
            if take(i, j):
                return 1
        for i in rng.permutation(m):
            if take(i, j):
                return 1
        return 0

    every, changed = True, 0
    while (changed > 0 or every) and done < steps:
        changed = 0
        pool = range(m) if every else np.flatnonzero((a > 0) & (a < C))
        for j in pool:
            changed += examine(j)
        if every:
            every = False
        elif changed == 0:
            every = True
    return a, b, done, computed


def dual_value(a, y, K):
    """D(alpha) = sum alpha - (alpha y)' K (alpha y) / 2."""
    v = a * y
    return a.sum() - v @ K @ v / 2


def exact_run(rule):
    """Run on the eight machines in fractions, with no cap."""
    P = [[int(v) for v in r] for r in X]
    Y = [int(v) for v in y]
    G = [[Fr(p[0] * q[0] + p[1] * q[1]) for q in P] for p in P]
    a, rows = [Fr(0)] * 8, []
    while True:
        F = [sum(a[j] * Y[j] * G[i][j] for j in range(8)) - Y[i]
             for i in range(8)]
        up = [i for i in range(8) if Y[i] > 0 or a[i] > 0]
        low = [i for i in range(8) if Y[i] < 0 or a[i] > 0]
        i = min(up, key=lambda t: F[t])
        gap = max(F[t] for t in low) - F[i]
        if gap <= 0:
            return a, F, rows

        def gain(t):
            e = G[i][i] + G[t][t] - 2 * G[i][t]
            return (F[t] - F[i]) ** 2 / e if F[t] > F[i] else Fr(-1)
        j = max(low, key=gain if rule == "second" else lambda t: F[t])
        eta = G[i][i] + G[j][j] - 2 * G[i][j]
        t = (F[j] - F[i]) / eta
        free = t
        t = min(t, a[i]) if Y[i] < 0 else t
        t = min(t, a[j]) if Y[j] > 0 else t
        a = list(a)
        a[i] += Y[i] * t
        a[j] -= Y[j] * t
        rows.append((i + 1, j + 1, gap, eta, free, t, list(a)))


def by_hand():
    for rule in ("first", "second"):
        print(rule, "order")
        a, F, rows = exact_run(rule)
        for i, j, gap, eta, free, t, new in rows:
            print(f"i {i} j {j} gap {str(gap):>5} eta {str(eta):>2}"
                  f" free {str(free):>5} t {str(t):>5}")
        print("alpha", [str(v) for v in a])
        print("offsets", sorted({str(-F[i]) for i in (3, 4, 6)}))


def platt_hand():
    tr = []
    a, b, k, _ = platt(X, y, tol=1e-6, trace=tr)
    print("Platt loops:", k, "steps, alpha", a[[3, 4, 6]], "b", round(b, 4))
    for i, j, new, bb in tr[:4]:
        print(f"pair {i + 1} {j + 1}  b {bb:7.3f}")


def clouds(m, seed=SEED, n=2, shift=1.0):
    """Two overlapping Gaussian clouds with balanced labels."""
    rng = np.random.default_rng(seed)
    t = np.where(np.arange(m) % 2 == 0, -1.0, 1.0)
    A = rng.normal(size=(m, n))
    A[:, 0] += shift * t
    return A, t


def against_sklearn():
    for name, A, t, C, g in (
            ("nine C=1/2", X9, y9, 0.5, None),
            ("eight C=1/20", X, y, 0.05, None),
            ("clouds linear", *clouds(400), 1.0, None),
            ("clouds gauss", *clouds(400), 1.0, 0.5)):
        k = linear if g is None else gaussian(g)
        for rule in ("first", "second"):
            a, b, n, _ = smo(A, t, C, k, rule, tol=1e-6)
            s = SVC(C=C, kernel="linear" if g is None else "rbf",
                    gamma=g or "scale", tol=1e-6, shrinking=False)
            s.fit(A, t)
            v = np.zeros(len(t))
            v[s.support_] = s.dual_coef_[0]
            K = k(A, A)
            dD = abs(dual_value(a, t, K) - dual_value(np.abs(v), t, K))
            print(f"{name:13s} {rule:6s} {n:6d} {s.n_iter_[0]:6d}"
                  f"  dD {dD:.0e}  db {abs(b - s.intercept_[0]):.0e}")


def rules():
    A, t = clouds(1000)
    for C in (1.0, 100.0):
        K = gaussian(0.5)
        for rule in ("first", "second"):
            a, b, n, c = smo(A, t, C, K, rule, steps=60000, cache=300)
            print(f"C {C:5.0f}  {rule:6s}  steps {n:6d}  columns {c:6d}"
                  f"  sv {(a > 1e-8).sum()}")
        a, b, n, c = platt(A, t, C, K, steps=60000)
        print(f"C {C:5.0f}  platt   steps {n:6d}  columns {c:6d}"
              f"  sv {(a > 1e-8).sum()}")


def small_checks():
    a, b, n, _ = smo(np.array([[-1.0], [0.0], [1.0]]),
                     np.array([-1.0, 1.0, -1.0]),
                     kernel=gaussian(np.log(2)), tol=1e-10)
    print("three points:", a * 17, "b * 17:", round(b * 17, 6), "steps", n)
    tw = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]])
    a, b, n, _ = smo(tw, np.array([1.0, 1.0, -1.0]), C=1.0, tol=1e-9)
    print("twin points:", a, "b", round(b, 3), "steps", n)
    t = np.array([-1.0, 1.0])
    a, b, n, _ = smo(np.array([[1.0, 1.0], [1.0, 1.0]]), t, C=1.0, tol=1e-9)
    print("clash:", a, "steps", n)


def cost(m, rule="second", C=1.0, cache=200):
    A, t = clouds(m)
    a, b, n, c = smo(A, t, C, gaussian(0.5), rule, cache=cache)
    return n, c, int((a > 1e-8).sum())


def scale_table():
    rows = [(m, *cost(m)) for m in (250, 500, 1000, 2000, 4000)]
    for m, n, c, sv in rows:
        print(f"m {m:5d}  steps {n:5d}  cols {c:5d}  sv {sv:4d}"
              f"  read {c * m:.1e} of {m * m:.1e}")
    r = np.array(rows, dtype=float)
    p = np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0]
    q = np.polyfit(np.log(r[:, 0]), np.log(r[:, 0] * r[:, 2]), 1)[0]
    print(f"steps grow like m^{p:.2f}, entries read like m^{q:.2f}")


def slsqp(m):
    A, t = clouds(m)
    return solve_soft(1.0, A, t)


def timing(path=Path(__file__).parent / "ch12_timing.json"):
    """Wall-clock seconds, best of three. Not part of the printed run."""
    def best(f):
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            f()
            out.append(time.perf_counter() - t0)
        return min(out)
    res = {"slsqp": [], "smo": [], "sklearn": []}
    for m in (100, 200, 400, 800):
        A, t = clouds(m)
        res["slsqp"].append((m, best(lambda: slsqp(m))))
    for m in (250, 500, 1000, 2000, 4000, 8000):
        A, t = clouds(m)
        res["smo"].append((m, best(lambda: smo(A, t, 1.0, gaussian(0.5)))))
        res["sklearn"].append((m, best(lambda: SVC(
            C=1.0, gamma=0.5, cache_size=200).fit(A, t))))
    path.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, [(m, round(s, 3)) for m, s in v])
        r = np.array(v)
        print(" exponent", np.polyfit(np.log(r[:, 0]), np.log(r[:, 1]), 1)[0])


if __name__ == "__main__":
    if sys.argv[1:] == ["timing"]:
        timing()
        sys.exit()
    for size in (1000, 50000, 10 ** 6):
        print(f"m {size:8d}  Q needs {8 * size ** 2 / 1e9:g} GB")
    by_hand()
    platt_hand()
    small_checks()
    against_sklearn()
    rules()
    scale_table()
