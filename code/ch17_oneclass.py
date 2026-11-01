"""Chapter 17: One-class SVM and novelty detection."""
import warnings
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
TOL = 1e-7

# The four healthy machines and the four that need service.
HEALTHY = X[y == -1].astype(float)
SERVICE = X[y == 1].astype(float)
# Three points on a line, as in Chapter 11, K(x, z) = 2^-((x - z)^2).
LINE = np.array([[-1.0], [0.0], [1.0]])
LG = np.log(2.0)
GAMMA = 1.0


def rbf(A, B, gamma):
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d)


def lin(A, B):
    return A @ B.T


def solve_oc(K, nu):
    """Dual of the one-class SVM by a general solver."""
    m = len(K)
    cap = 1 / (nu * m)
    a0 = np.full(m, 1 / m)
    r = minimize(lambda a: 0.5 * a @ K @ a, a0, jac=lambda a: K @ a,
                 bounds=[(0, cap)] * m, method="SLSQP",
                 constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                              "jac": lambda a: np.ones(m)},
                 options={"ftol": 1e-14, "maxiter": 2000})
    return r.x


def solve_ball(K, nu):
    """Dual of the smallest ball by a general solver."""
    m = len(K)
    cap = 1 / (nu * m)
    a0 = np.full(m, 1 / m)
    d = np.diag(K)
    r = minimize(lambda a: a @ K @ a - d @ a, a0,
                 jac=lambda a: 2 * K @ a - d,
                 bounds=[(0, cap)] * m, method="SLSQP",
                 constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                              "jac": lambda a: np.ones(m)},
                 options={"ftol": 1e-14, "maxiter": 2000})
    return r.x


def rho_range(K, a, nu):
    """Interval of rho allowed by the KKT conditions."""
    cap = 1 / (nu * len(K))
    g = K @ a
    free = (a > 1e-6) & (a < cap - 1e-6)
    if free.any():
        return g[free].min(), g[free].max()
    lo = g[a >= cap - 1e-6].max() if (a >= cap - 1e-6).any() else -np.inf
    hi = g[a <= 1e-6].min() if (a <= 1e-6).any() else np.inf
    return lo, hi


def oc_smo(K, nu, tol=1e-6, cap_steps=200000):
    """Pair steps on the one-class dual, with maximal violation."""
    m = len(K)
    cap = 1 / (nu * m)
    a = np.zeros(m)
    whole = int(np.floor(nu * m))
    a[:whole] = cap
    if whole < m:
        a[whole] = 1 - whole * cap
    g = K @ a
    steps = 0
    while steps < cap_steps:
        up = np.where(a < cap - 1e-12)[0]
        dn = np.where(a > 1e-12)[0]
        i = up[np.argmin(g[up])]
        j = dn[np.argmax(g[dn])]
        gap = g[j] - g[i]
        if gap <= tol:
            break
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        d = min(gap / max(eta, 1e-12), cap - a[i], a[j])
        a[i] += d
        a[j] -= d
        g += d * (K[i] - K[j])
        steps += 1
    return a, (g[i] + g[j]) / 2, steps


def exact_line():
    """Check the line example in fractions, K entries 1, 1/2, 1/16."""
    K = [[Fr(1), Fr(1, 2), Fr(1, 16)],
         [Fr(1, 2), Fr(1), Fr(1, 2)],
         [Fr(1, 16), Fr(1, 2), Fr(1)]]
    cases = [(Fr(1, 3), [Fr(8, 17), Fr(1, 17), Fr(8, 17)], Fr(9, 17)),
             (Fr(3, 4), [Fr(4, 9), Fr(1, 9), Fr(4, 9)], Fr(5, 9))]
    out = []
    for nu, a, rho in cases:
        cap = 1 / (nu * 3)
        assert sum(a) == 1 and all(0 <= ai <= cap for ai in a)
        g = [sum(k * ai for k, ai in zip(row, a)) for row in K]
        for ai, gi in zip(a, g):
            if ai == 0:
                assert gi >= rho
            elif ai == cap:
                assert gi <= rho
            else:
                assert gi == rho
        out.append([gi - rho for gi in g])
    return out


def exact_ball():
    """Smallest circle around the healthy machines, in fractions."""
    pts = [[Fr(int(v)) for v in p] for p in HEALTHY]
    a = [Fr(0), Fr(1, 2), Fr(0), Fr(1, 2)]
    c = [sum(ai * p[k] for ai, p in zip(a, pts)) for k in range(2)]
    r2 = max((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for p in pts)
    d = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for p in pts]
    for ai, di in zip(a, d):
        assert di <= r2
        assert ai == 0 or di == r2
    s = [(float(p[0] - c[0]) ** 2 + float(p[1] - c[1]) ** 2)
         for p in SERVICE]
    return c, r2, d, s


def healthy(m, rng):
    """Readings of healthy machines: a curved band set by the load."""
    s = rng.uniform(0, 1, m)
    a = 1 + 3 * s + rng.normal(0, 0.15, m)
    b = 1 + 2 * s ** 2 + rng.normal(0, 0.15, m)
    return np.column_stack([a, b])


def faults(m, kind, rng):
    """Kind A runs hot for its load; kind B shakes and stays cool."""
    c = {"A": (2.3, 2.7), "B": (4.2, 1.6)}[kind]
    return np.array(c) + rng.normal(0, 0.3, (m, 2))


def fleet():
    """Scaled train and test sets of the fleet."""
    rng = np.random.default_rng(SEED)
    tr = healthy(300, rng)
    te = healthy(300, rng)
    fa = faults(30, "A", rng)
    fb = faults(30, "B", rng)
    sc = StandardScaler().fit(tr)
    return tr, te, fa, fb, sc


def fit(Xs, nu, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return OneClassSVM(nu=nu, gamma=gamma, tol=1e-6).fit(Xs)


def share_rows(nus=(0.02, 0.05, 0.1, 0.2, 0.5, 0.8)):
    tr, te, fa, fb, sc = fleet()
    Z, Zt = sc.transform(tr), sc.transform(te)
    Zf = sc.transform(np.vstack([fa, fb]))
    rows = []
    for nu in nus:
        o = fit(Z, nu, GAMMA)
        s = o.decision_function(Z)
        out = np.mean(s < -1e-6)
        sv = len(o.support_) / len(Z)
        fal = np.mean(o.predict(Zt) == -1)
        det = np.mean(o.predict(Zf) == -1)
        rows.append((nu, out, sv, fal, det))
    return rows


def area(o, sc, lo, hi, rng, n=20000):
    """Share of a box that the model accepts."""
    U = rng.uniform(lo, hi, (n, 2))
    return np.mean(o.predict(sc.transform(U)) == 1)


def gamma_rows(nu=0.05, gammas=(0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300)):
    """Choose gamma from healthy data alone, then look at the faults."""
    tr, te, fa, fb, sc = fleet()
    rng = np.random.default_rng(SEED + 1)
    idx = rng.permutation(len(tr))
    fold = np.array_split(idx, 5)
    lo, hi = tr.min(axis=0) - 1, tr.max(axis=0) + 1
    Z = sc.transform(tr)
    Zt = sc.transform(te)
    Zf = sc.transform(np.vstack([fa, fb]))
    lab = np.r_[np.zeros(len(te)), np.ones(len(Zf))]
    rows = []
    for g in gammas:
        miss = []
        for k in range(5):
            held = fold[k]
            keep = np.setdiff1d(idx, held)
            o = fit(Z[keep], nu, g)
            miss.append(np.mean(o.predict(Z[held]) == -1))
        o = fit(Z, nu, g)
        acc = area(o, sc, lo, hi, np.random.default_rng(SEED + 2))
        s = -o.decision_function(np.vstack([Zt, Zf]))
        rows.append((g, np.mean(miss), acc, len(o.support_) / len(Z),
                     roc_auc_score(lab, s),
                     np.mean(o.predict(Zf) == -1),
                     np.mean(o.predict(Zt) == -1)))
    return rows


def pick_gamma(rows, nu=0.05, slack=1.5):
    """Tightest model whose held-out alarms stay near the promise."""
    good = [r for r in rows if r[1] <= slack * nu]
    return min(good, key=lambda r: r[2])[0]


def novelty():
    """A classifier and a one-class model meet an unseen fault."""
    tr, te, fa, fb, sc = fleet()
    Z = sc.transform(tr)
    rng = np.random.default_rng(SEED + 3)
    Za = sc.transform(faults(60, "A", rng))
    clf = SVC(kernel="rbf", gamma=1.0, C=1.0, class_weight="balanced")
    clf.fit(np.vstack([Z, Za]), np.r_[-np.ones(len(Z)), np.ones(len(Za))])
    o = fit(Z, 0.05, 1.0)
    res = {}
    for name, A in (("healthy", te), ("seen A", fa), ("new B", fb)):
        S = sc.transform(A)
        res[name] = (np.mean(clf.predict(S) == 1),
                     np.mean(o.predict(S) == -1))
    return clf, o, sc, res


def spoiled():
    """Faults hidden in the training set."""
    rows = []
    for share in (0.0, 0.05, 0.1):
        rng = np.random.default_rng(SEED + 4)
        n = int(round(300 * share / (1 - share)))
        tr = np.vstack([healthy(300, rng), faults(n, "A", rng)])
        te = healthy(300, rng)
        fb = faults(60, "A", rng)
        sc = StandardScaler().fit(tr)
        for nu in (0.02, 0.15):
            o = fit(sc.transform(tr), nu, 1.0)
            rows.append((share, nu,
                         np.mean(o.predict(sc.transform(te)) == -1),
                         np.mean(o.predict(sc.transform(fb)) == -1)))
    return rows


def rare():
    """Precision at a low rate of faults."""
    tr, te, fa, fb, sc = fleet()
    o = fit(sc.transform(tr), 0.05, GAMMA)
    fal = np.mean(o.predict(sc.transform(te)) == -1)
    det = np.mean(o.predict(sc.transform(np.vstack([fa, fb]))) == -1)
    rows = []
    for p in (0.5, 0.1, 0.01, 0.001):
        rows.append((p, p * det / (p * det + (1 - p) * fal)))
    return fal, det, rows


def far_point():
    """Chapter 11 asked what to do far from the data."""
    tr, te, fa, fb, sc = fleet()
    Z = sc.transform(tr)
    y2 = np.r_[-np.ones(len(Z)), np.ones(60)]
    Za = sc.transform(faults(60, "A", np.random.default_rng(SEED + 3)))
    clf = SVC(kernel="rbf", gamma=1.0, C=1.0, class_weight="balanced")
    clf.fit(np.vstack([Z, Za]), y2)
    o = fit(Z, 0.05, GAMMA)
    p = sc.transform(np.array([[50.0, 50.0]]))
    return (clf.decision_function(p)[0], clf.intercept_[0],
            o.decision_function(p)[0], -o.offset_[0],
            len(Z) * 0.05)


def main():
    print("-- exact")
    (s1, s2) = exact_line()
    print("nu 1/3  f at points", *s1)
    print("nu 3/4  f at points", *s2)
    c, r2, d, sv = exact_ball()
    print("centre", *c, " R^2", r2)
    print("d^2 healthy", *d)
    print("d^2 service", *sv)

    print("-- line")
    KL = rbf(LINE, LINE, LG)
    for nu in (1 / 3, 0.75, 0.9):
        a = solve_oc(KL, nu)
        print(f"nu {nu:.3f} alpha {a} rho {rho_range(KL, a, nu)[0]:.4f}")
    o = OneClassSVM(nu=0.75, gamma=LG, tol=1e-9).fit(LINE)
    print("sklearn coef", o.dual_coef_[0], "offset", o.offset_)
    print("sklearn f", o.decision_function(LINE))

    print("-- linear")
    K4 = lin(HEALTHY, HEALTHY)
    for nu in (0.01, 0.5, 1.0):
        a = solve_oc(K4, nu)
        print(f"nu {nu:.2f} alpha {a} w {a @ HEALTHY}"
              f" rho {rho_range(K4, a, nu)[0]:.2f}")
    a = solve_oc(K4, 0.01)
    r = rho_range(K4, a, 0.01)[0]
    print("service scores", SERVICE @ (a @ HEALTHY) - r)
    C4 = HEALTHY - HEALTHY.mean(axis=0)
    a = solve_oc(lin(C4, C4), 0.5)
    print("centred alpha", a, "w", np.abs(a @ C4))

    print("-- ball")
    for name, K in (("gauss", rbf(HEALTHY, HEALTHY, 0.25)),
                    ("linear", K4)):
        print(name, "oc  ", solve_oc(K, 0.5))
        print(name, "ball", solve_ball(K, 0.5))
    print("small gamma", solve_oc(rbf(HEALTHY, HEALTHY, 1e-3), 0.01))

    print("-- healthy four")
    o = OneClassSVM(nu=0.25, gamma=0.25, tol=1e-9).fit(HEALTHY)
    print("sv", o.support_ + 1, "healthy",
          o.decision_function(HEALTHY) / (0.25 * 4))
    print("service", o.decision_function(SERVICE) / (0.25 * 4))

    print("-- smo")
    tr, te, fa, fb, sc = fleet()
    Z = sc.transform(tr)
    K = rbf(Z, Z, GAMMA)
    a, rho, st = oc_smo(K, 0.1)
    o = fit(Z, 0.1, GAMMA)
    print(f"steps {st}  rho {rho:.5f}  sklearn {o.offset_[0] / 30:.5f}")
    gap = np.abs(K @ a - rho - o.decision_function(Z) / 30).max()
    print(f"largest score gap {gap:.1e}")

    print("-- shares")
    print(" nu   outside  sv   alarm  caught")
    for r in share_rows():
        print(f"{r[0]:.2f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}")

    print("-- gamma")
    print("gamma  held  area   sv    auc  caught  alarm")
    gr = gamma_rows()
    for r in gr:
        print(f"{r[0]:5g}  {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f}"
              f" {r[5]:.3f}  {r[6]:.3f}")
    print("chosen gamma", pick_gamma(gr))

    print("-- novelty")
    print("          svc   one-class")
    for k, v in novelty()[3].items():
        print(f"{k:8s} {v[0]:.3f}  {v[1]:.3f}")

    print("-- far")
    f = far_point()
    print(f"svc score {f[0]:.4f}  b {f[1]:.4f}")
    print(f"one-class {f[2]:.4f}  -rho {f[3]:.4f}")

    print("-- spoiled")
    print("share  nu   alarm  caught")
    for r in spoiled():
        print(f"{r[0]:.2f} {r[1]:.2f}  {r[2]:.3f}  {r[3]:.3f}")

    print("-- rare")
    fal, det, rows = rare()
    print(f"alarm {fal:.3f}  caught {det:.3f}")
    for p, q in rows:
        print(f"fault rate {p:.3f}  precision {q:.3f}")


if __name__ == "__main__":
    main()
