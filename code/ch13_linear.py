"""Chapter 13: Linear SVMs at scale, Pegasos and dual coordinate descent."""
import json
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.svm import SVC, LinearSVC

from ch07_soft import X9, solve_soft, y9
from ch09_kernel import rings
from ch12_smo import smo
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
CUT = -0.876  # median hidden score, so the classes balance


def with_offset(A, s=1.0):
    """Append the constant feature s, so that b = s * w[-1]."""
    if sparse.issparse(A):
        col = sparse.csr_matrix(np.full((A.shape[0], 1), s))
        return sparse.hstack([A, col], format="csr")
    return np.column_stack([A, np.full(len(A), s)])


def rows_of(A):
    """Each row as (indices, values): the only form the solvers read."""
    if sparse.issparse(A):
        A = A.tocsr()
        p = A.indptr
        return [(A.indices[p[i]:p[i + 1]], A.data[p[i]:p[i + 1]])
                for i in range(A.shape[0])]
    idx = np.arange(A.shape[1])
    return [(idx, A[i]) for i in range(len(A))]


def primal(w, A, t, C):
    """J(w) = |w|^2 / 2 + C sum of hinges, with no offset."""
    return w @ w / 2 + C * np.maximum(0, 1 - t * (A @ w)).sum()


def dual_value(a, w):
    """D(alpha) for the dual with no balance, given w = sum a y x."""
    return a.sum() - w @ w / 2


def pegasos(rows, t, n, lam, T, seed=SEED, project=False, marks=()):
    """Stochastic subgradient steps, with w stored as c * v."""
    m = len(t)
    picks = np.random.default_rng(seed).integers(m, size=T)
    v, c, vv = np.zeros(n), 1.0, 0.0
    saved = []
    for k in range(1, T + 1):
        i = picks[k - 1]
        idx, val = rows[i]
        vi = v[idx]
        active = c * t[i] * (vi @ val) < 1
        c *= 1 - 1 / k
        if c == 0.0:
            c, vv = 1.0, 0.0
            v[:] = 0.0
            vi = v[idx]
        if active:
            u = t[i] * val / (lam * k * c)
            vv += 2 * (vi @ u) + u @ u
            v[idx] += u
        if project:
            size = c * np.sqrt(max(vv, 0.0))
            if size * np.sqrt(lam) > 1:
                c /= size * np.sqrt(lam)
        if k in marks:
            saved.append((k, c * v))
    return (c * v, saved) if marks else c * v


def dcd(rows, t, n, C, tol=1e-3, epochs=1000, seed=SEED, shuffle=True,
        trace=None):
    """Dual coordinate descent: one alpha at a time, w kept current."""
    m = len(t)
    rng = np.random.default_rng(seed)
    a, w = [0.0] * m, np.zeros(n)
    q = [val @ val for _, val in rows]
    steps = 0
    for epoch in range(1, epochs + 1):
        low, high = np.inf, -np.inf
        for i in rng.permutation(m) if shuffle else range(m):
            if q[i] == 0.0:
                continue
            idx, val = rows[i]
            g = t[i] * (w[idx] @ val) - 1
            pg = g
            if a[i] == 0.0:
                pg = min(g, 0.0)
            elif a[i] == C:
                pg = max(g, 0.0)
            low, high = min(low, pg), max(high, pg)
            if pg != 0.0:
                new = min(max(a[i] - g / q[i], 0.0), C)
                w[idx] += (new - a[i]) * t[i] * val
                a[i] = new
                steps += 1
        if trace is not None:
            trace.append((epoch, np.array(a), w.copy()))
        if high - low <= tol:
            break
    return np.array(a), w, epoch, steps


def sparse_data(m, n=20000, per=30, noise=0.05, seed=SEED):
    """Word counts by a Zipf law; labels from a hidden linear rule."""
    rng = np.random.default_rng(seed)
    p = 1 / np.arange(1, n + 1)
    p /= p.sum()
    cols = rng.choice(n, size=m * per, p=p)
    rows = np.repeat(np.arange(m), per)
    A = sparse.csr_matrix((np.ones(m * per), (rows, cols)), shape=(m, n))
    A.sum_duplicates()
    size = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    A = sparse.diags(1 / size) @ A
    hidden = np.random.default_rng(SEED).normal(size=n)
    t = np.where(A @ hidden > CUT, 1.0, -1.0)
    flip = rng.random(m) < noise
    t[flip] = -t[flip]
    return A.tocsr(), t


def fourier(A, D, gamma, seed=SEED):
    """Random Fourier features for exp(-gamma |x - z|^2)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(A.shape[1], D))
    u = rng.uniform(0, 2 * np.pi, D)
    return lambda B: np.sqrt(2 / D) * np.cos(B @ W + u)


def nystroem(A, k, gamma, seed=SEED):
    """Features from k landmarks, with inner products K_k W^-1 K_k'."""
    rng = np.random.default_rng(seed)
    L = A[rng.choice(len(A), k, replace=False)]

    def K(P, Q):
        return np.exp(-gamma * ((P[:, None] - Q[None]) ** 2).sum(axis=2))
    vals, V = np.linalg.eigh(K(L, L))
    keep = vals > 1e-10
    T = V[:, keep] / np.sqrt(vals[keep])
    return lambda B: K(B, L) @ T


def pegasos_hand(A, t, lam, picks):
    """Exact steps with Fractions, on the plain update of the text."""
    w = [Fr(0)] * A.shape[1]
    out = []
    for k, i in enumerate(picks, 1):
        x = [Fr(v) for v in A[i]]
        score = Fr(t[i]) * sum(a * b for a, b in zip(w, x))
        keep = 1 - Fr(1, k)
        w = [keep * a for a in w]
        if score < 1:
            w = [a + Fr(t[i]) * b / (lam * k) for a, b in zip(w, x)]
        out.append((k, i + 1, score, list(w)))
    return out


def dcd_hand(A, t, C, order):
    """Exact coordinate steps with Fractions."""
    w = [Fr(0)] * A.shape[1]
    a = [Fr(0)] * len(t)
    out = []
    for i in order:
        x = [Fr(v) for v in A[i]]
        q = sum(v * v for v in x)
        g = Fr(t[i]) * sum(u * v for u, v in zip(w, x)) - 1
        new = min(max(a[i] - g / q, Fr(0)), C)
        w = [u + (new - a[i]) * Fr(t[i]) * v for u, v in zip(w, x)]
        a[i] = new
        out.append((i + 1, g, q, new, list(w)))
    return out


def dec(v):
    return "(" + ", ".join(f"{float(x):.3f}" for x in v) + ")"


def hand():
    print("-- hand runs: eight machines, s = 1, C = 1/2, lam = 1/4")
    Z = with_offset(X)
    picks = np.random.default_rng(SEED).integers(8, size=6)
    print("picks", picks + 1)
    for k, i, sc, w in pegasos_hand(Z, y, Fr(1, 4), picks):
        print(f"{k} machine {i} score {sc} w {[str(u) for u in w]}")
    run = dcd_hand(Z, y, Fr(1, 2), range(8))
    for i, g, q, new, w in run:
        print(f"i {i} g {float(g):.3f} q {q} a {float(new):.3f} w {dec(w)}")
    a = np.array([float(r[3]) for r in run])
    w = np.array([float(u) for u in run[-1][4]])
    print(f"after one pass J {primal(w, Z, y, 0.5):.3f}"
          f"  D {dual_value(a, w):.3f}")


def against_exact():
    print("-- matches on small data")
    for name, A, t in (("eight", X, y), ("nine", X9, y9)):
        Z = with_offset(A)
        a, w, ep, st = dcd(rows_of(Z), t, 3, 0.5, tol=1e-9, epochs=20000)
        ref = LinearSVC(loss="hinge", C=0.5, fit_intercept=False,
                        tol=1e-9, max_iter=2_000_000,
                        random_state=SEED).fit(Z, t)
        J = primal(w, Z, t, 0.5)
        print(f"{name}: w {w[:2]} b {w[2]:.3f} epochs {ep}"
              f" gap {J - dual_value(a, w):.0e}")
        print(f"  LinearSVC w {ref.coef_[0][:2]} b {ref.coef_[0][2]:.3f}"
              f" dJ {primal(ref.coef_[0], Z, t, 0.5) - J:.0e}")
    Z = with_offset(X9)
    r = rows_of(Z)
    a, ws, ep, st = dcd(r, y9, 3, 0.5, tol=1e-10, epochs=20000)
    J = primal(ws, Z, y9, 0.5)
    print("Pegasos on the nine, lam = 2/9")
    for T in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        w = pegasos(r, y9, 3, 2 / 9, T)
        w2 = pegasos(r, y9, 3, 2 / 9, T, project=True)
        print(f"T {T:7d}  w {w}  dJ {primal(w, Z, y9, 0.5) - J:.5f}"
              f"  ball {primal(w2, Z, y9, 0.5) - J:.5f}")


def scaling_offset():
    print("-- the offset feature s on the nine, C = 1/2")
    exact = solve_soft(0.5, X9, y9).x
    w = (exact * y9) @ X9
    print(f"Chapter 7: w {w}  b -2.900")
    for s in (1, 3, 10, 30, 100, 300):
        Z = with_offset(X9, s)
        a, w, ep, st = dcd(rows_of(Z), y9, 3, 0.5, tol=1e-6,
                           epochs=200000)
        print(f"s {s:3d}  w {w[:2]}  b {s * w[2]:.3f}  "
              f"y'a {a @ y9:.5f}  epochs {ep}")


def pitfalls():
    print("-- pitfalls")
    Z = with_offset(X9)
    r = rows_of(Z)
    for C in (0.5, 100.0, 10000.0):
        lam = 1 / (9 * C)
        a, ws, ep, st = dcd(r, y9, 3, C, tol=1e-7, epochs=200000)
        J = primal(ws, Z, y9, C)
        for T in (10 ** 4, 10 ** 6):
            w = pegasos(r, y9, 3, lam, T)
            w2 = pegasos(r, y9, 3, lam, T, project=True)
            print(f"C {C:5g} T {T:7d} J* {J:8.1f}  plain "
                  f"{primal(w, Z, y9, C):10.1f}  ball "
                  f"{primal(w2, Z, y9, C):10.1f}")
    A, t = sparse_data(4000)
    Z = with_offset(A)
    r = rows_of(Z)
    n = Z.shape[1]
    for sh in (True, False):
        a, w, ep, st = dcd(r, t, n, 1.0, tol=1e-2, shuffle=sh, epochs=2000)
        print(f"shuffle {sh!s:5}  epochs {ep}  steps {st}")


def sparse_run():
    print("-- sparse data: n = 20000, about 26 words, C = 1")
    Ate, tte = sparse_data(5000, seed=5)
    Zt = with_offset(Ate)
    for m in (2000, 8000, 32000):
        A, t = sparse_data(m)
        Z = with_offset(A)
        r = rows_of(Z)
        n = Z.shape[1]
        a, w, ep, st = dcd(r, t, n, 1.0, tol=1e-2)
        p = pegasos(r, t, n, 1 / m, 10 * m)
        J = primal(w, Z, t, 1.0)
        ref = LinearSVC(loss="hinge", C=1.0, fit_intercept=False,
                        tol=1e-2, max_iter=100000,
                        random_state=SEED).fit(Z, t)
        print(f"m {m:5d} dcd {np.mean((Zt @ w > 0) == (tte > 0)):.3f}"
              f" J {J:9.1f} gap {J - dual_value(a, w):.2f}"
              f" ep {ep} sv {(a > 0).sum()}")
        print(f"        peg {np.mean((Zt @ p > 0) == (tte > 0)):.3f}"
              f" J {primal(p, Z, t, 1.0):9.1f}"
              f"  LinearSVC {ref.score(Zt, tte):.3f}"
              f" J {primal(ref.coef_[0], Z, t, 1.0):9.1f}")
    A, t = sparse_data(8000)
    g = SVC(C=1.0, gamma=1.0).fit(A, t)
    print(f"Gaussian m 8000: {g.score(Ate, tte):.3f}"
          f" sv {g.n_support_.sum()}")


def bridge():
    print("-- rings: a linear solver on approximate kernel features")
    A, t = rings(4000, SEED, noise=0.2)
    Ate, tte = rings(4000, 7, noise=0.2)
    g = SVC(C=1.0, gamma=2.0).fit(A, t)
    plain = LinearSVC(C=1.0, max_iter=20000).fit(A, t)
    print(f"plain linear {plain.score(Ate, tte):.3f}")
    print(f"SVC Gaussian {g.score(Ate, tte):.3f}  sv {g.n_support_.sum()}")
    x, z = A[0], A[1]
    exact = np.exp(-2 * ((x - z) ** 2).sum())
    for D in (10, 100, 1000, 10000):
        phi = fourier(A, D, 2.0)
        Za = with_offset(phi(A))
        a, w, ep, st = dcd(rows_of(Za), t, D + 1, 1.0, tol=1e-2, epochs=200)
        acc = np.mean((with_offset(phi(Ate)) @ w > 0) == (tte > 0))
        print(f"Fourier D {D:5d} K {phi(A[[0]])[0] @ phi(A[[1]])[0]:.3f}"
              f" (exact {exact:.3f})  test {acc:.3f}")
    for k in (10, 30, 100, 300):
        phi = nystroem(A, k, 2.0)
        P = phi(A)
        Za = with_offset(P)
        a, w, ep, st = dcd(rows_of(Za), t, P.shape[1] + 1, 1.0, tol=1e-2,
                           epochs=200)
        acc = np.mean((with_offset(phi(Ate)) @ w > 0) == (tte > 0))
        print(f"Nystroem k {k:4d}  test {acc:.3f}")


class Rows(sparse.csr_matrix):
    """A sparse matrix that answers len(), as Columns expects."""
    def __len__(self):
        return self.shape[0]


def timing(path=Path(__file__).parent / "ch13_timing.json"):
    """Wall-clock seconds, best of three. Not part of the printed run."""
    def best(f, n=3):
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = f()
            out.append(time.perf_counter() - t0)
        return min(out), r
    res = {"smo": [], "svc": [], "dcd": [], "pegasos": [], "liblinear": []}
    Ate, tte = sparse_data(5000, seed=5)
    Zt = with_offset(Ate)
    for m in (1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000):
        A, t = sparse_data(m)
        Z = with_offset(A)
        r = rows_of(Z)
        n = Z.shape[1]
        if m <= 8000:
            R = Rows(A)
            s, (a, b, k, _) = best(lambda: smo(
                R, t, 1.0, lambda P, Q: (P @ Q.T).toarray(), tol=1e-2,
                cache=300), 1)
            res["smo"].append((m, s, int(k)))
        if m <= 32000:
            s, g = best(lambda: SVC(C=1.0, kernel="linear", tol=1e-2,
                                    cache_size=500).fit(A, t), 1)
            res["svc"].append((m, s, g.score(Ate, tte)))
        s, o = best(lambda: dcd(r, t, n, 1.0, tol=1e-2))
        res["dcd"].append((m, s, np.mean((Zt @ o[1] > 0) == (tte > 0))))
        s, w = best(lambda: pegasos(r, t, n, 1 / m, 10 * m))
        res["pegasos"].append((m, s, np.mean((Zt @ w > 0) == (tte > 0))))
        s, g = best(lambda: LinearSVC(
            loss="hinge", C=1.0, fit_intercept=False, tol=1e-2,
            max_iter=100000).fit(Z, t))
        res["liblinear"].append((m, s, g.score(Zt, tte)))
        print(m, {k: v[-1] for k, v in res.items() if v}, flush=True)
    path.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        q = np.array(v)
        print(k, "exponent", np.polyfit(np.log(q[:, 0]), np.log(q[:, 1]), 1)[0])


if __name__ == "__main__":
    if sys.argv[1:] == ["timing"]:
        timing()
        sys.exit()
    hand()
    against_exact()
    scaling_offset()
    pitfalls()
    sparse_run()
    bridge()
