"""Chapter 18: Why a wide margin generalises."""
from math import ceil, comb, e, exp, log, sqrt

import numpy as np
from sklearn.svm import SVC

from ch09_kernel import cover, labellings
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
DELTA = 0.05


def hoeffding_gap(m, delta=DELTA):
    """Gap that one fixed rule exceeds with probability at most delta."""
    return sqrt(log(1 / delta) / (2 * m))


def finite_gap(m, size, delta=DELTA):
    """The same gap for all of a finite set of rules at once."""
    return sqrt((log(size) + log(1 / delta)) / (2 * m))


def sauer(m, d):
    """Sum of binomials that bounds the growth function."""
    return sum(comb(m, i) for i in range(min(d, m) + 1))


def vc_gap(m, d, delta=DELTA):
    """Gap of the VC bound, with Sauer's sum for the growth function."""
    return sqrt(2 * log(sauer(m, d)) / m) + hoeffding_gap(m, delta)


def luckiest(k, m, draws, rng, p=0.3):
    """Mean largest gap among k independent rules with true error p."""
    wrong = rng.binomial(m, p, size=(draws, k))
    return (p - wrong / m).max(axis=1).mean()


def rbf(A, B, gamma):
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d)


def ramp(t):
    """One above the 0-1 loss and below the hinge."""
    return np.clip(1 - t, 0, 1)


def margin_gap(lam, r2, m, delta=DELTA):
    """Complexity and confidence, norm found from the data."""
    k = max(0, ceil(log(max(lam, 1), 2)))
    return (2 * 2 ** k * sqrt(r2 / m)
            + sqrt(((k + 1) * log(2) + log(1 / delta)) / (2 * m)))


def fit_line(A, lab, C=1e6):
    s = SVC(kernel="linear", C=C, tol=1e-6).fit(A, lab)
    return s.coef_[0], s.intercept_[0], s


def report_line(A, lab, name, delta=DELTA):
    """Every bound for a hard-margin line on the points A."""
    w, b, s = fit_line(A, lab)
    m = len(lab)
    r2 = (A ** 2).sum(axis=1).max() + 1     # lifted points
    lam = sqrt(w @ w + b * b)
    rho = 1 / sqrt(w @ w)
    print(f"{name}: rho {rho:.3f}  R2 {r2:.2f}  lam {lam:.2f}"
          f"  R2 lam2 {r2 * lam ** 2:.0f}")
    print(f"  margin term {margin_gap(lam, r2, m, delta):.2f}"
          f"  plain {2 * lam * sqrt(r2 / m):.2f}")


def loo_errors(Z, lab, C, gamma=None):
    """Exact leave-one-out errors. A tie counts as an error."""
    wrong = 0
    for k in range(len(lab)):
        keep = np.arange(len(lab)) != k
        kind = "linear" if gamma is None else "rbf"
        s = SVC(kernel=kind, C=C, gamma=gamma or "scale", tol=1e-6)
        s.fit(Z[keep], lab[keep])
        t = lab[k] * s.decision_function(Z[[k]])[0]
        wrong += t <= 1e-6
    return wrong


def dual_no_offset(K, lab, C, passes=4000, tol=1e-8):
    """Coordinate ascent on the dual with no balance condition."""
    Q = np.outer(lab, lab) * K
    a = np.zeros(len(lab))
    for _ in range(passes):
        most = 0.0
        for i in range(len(lab)):
            new = min(C, max(0.0, a[i] + (1 - Q[i] @ a) / Q[i, i]))
            most = max(most, abs(new - a[i]))
            a[i] = new
        if most < tol:
            break
    return a


def padded(m, n, radius, rng, flip=0.05):
    """A signal on x1, plus noise of fixed length in the other n - 1."""
    lab = rng.choice([-1.0, 1.0], size=m)
    Z = np.zeros((m, n))
    Z[:, 0] = lab * (0.5 + 0.5 * rng.random(m))
    N = rng.standard_normal((m, n - 1))
    Z[:, 1:] = radius * N / np.linalg.norm(N, axis=1, keepdims=True)
    bad = rng.random(m) < flip
    return Z, np.where(bad, -lab, lab)


def dimension_run(ns=(2, 10, 100, 1000, 10000), m=200, draws=10):
    """Noise of length 1 against length sqrt(n - 1), with n rising."""
    rows = []
    rng = np.random.default_rng(SEED)
    for n in ns:
        for mode, radius in (("fixed", 1.0), ("grow", sqrt(n - 1))):
            got = []
            for _ in range(draws):
                A, a = padded(m, n, radius, rng)
                T, t = padded(2000, n, radius, rng)
                s = SVC(kernel="linear", C=1.0).fit(A, a)
                w, b = s.coef_[0], s.intercept_[0]
                r2 = (A ** 2).sum(axis=1).max() + 1
                lam = sqrt(w @ w + b * b)
                fit = ramp(a * (A @ w + b)).mean()
                got.append((r2, lam, np.mean(np.sign(T @ w + b) != t),
                            fit + margin_gap(lam, r2, m)))
            rows.append((n, mode, *np.mean(got, axis=0)))
    return rows


def noisy_rings(m, rng, noise=0.45):
    """Two rings that overlap, in the unit disc after rescaling."""
    lab = rng.choice([-1.0, 1.0], size=m)
    r = np.where(lab > 0, 2.0, 1.0) + noise * rng.standard_normal(m)
    t = rng.uniform(0, 2 * np.pi, size=m)
    return np.column_stack([r * np.cos(t), r * np.sin(t)]), lab


CS = (0.01, 0.1, 1, 10, 100, 1000, 10000)
GS = (0.01, 0.1, 1, 10, 100, 1000)
GAMMA = 1.0


def one_fit(A, a, T, t, gamma, C):
    """Train, test, support share, ramp, norm, and bound for one fit."""
    s = SVC(kernel="rbf", gamma=gamma, C=C).fit(A, a)
    sv = s.support_vectors_
    q = (s.dual_coef_ @ rbf(sv, sv, gamma) @ s.dual_coef_.T).item()
    b = s.intercept_[0]
    lam = sqrt(q + b * b)
    f = a * s.decision_function(A)
    fit = ramp(f).mean()
    return (np.mean(f <= 0), np.mean(s.predict(T) != t),
            len(sv) / len(a), fit, lam,
            fit + margin_gap(lam, 2.0, len(a)))


def sweep(pairs, m=400, draws=20, test=4000):
    """Means over draws of one_fit for each (gamma, C)."""
    rng = np.random.default_rng(SEED)
    rows = np.zeros((len(pairs), 6))
    for _ in range(draws):
        A, a = noisy_rings(m, rng)
        T, t = noisy_rings(test, rng)
        for j, (g, C) in enumerate(pairs):
            rows[j] += one_fit(A, a, T, t, g, C)
    return rows / draws


def norm_growth(ms=(25, 50, 100, 200, 400), gamma=20.0, draws=5):
    """1/rho^2 of a Gaussian fit with large C, for true and random labels."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for m in ms:
        got = []
        for _ in range(draws):
            A, a = noisy_rings(m, rng, noise=0.08)
            one = []
            for lab in (a, rng.permutation(a)):
                s = SVC(kernel="rbf", gamma=gamma, C=1e3).fit(A, lab)
                sv = s.support_vectors_
                one.append((s.dual_coef_ @ rbf(sv, sv, gamma)
                            @ s.dual_coef_.T).item())
            got.append(one)
        rows.append((m, *np.mean(got, axis=0)))
    return rows


def size_run(ms=(100, 200, 400, 800, 1600, 3200), C=1.0, draws=10):
    """Bound and true gap against the sample size."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for m in ms:
        got = []
        for _ in range(draws):
            A, a = noisy_rings(m, rng)
            T, t = noisy_rings(4000, rng)
            got.append(one_fit(A, a, T, t, GAMMA, C))
        rows.append((m, *np.mean(got, axis=0)))
    return rows


if __name__ == "__main__":
    # One fixed rule, and many.
    for m in (100, 200, 1000):
        print(f"m {m:5d}  fixed {hoeffding_gap(m):.3f}"
              f"  115 rules {finite_gap(m, 115):.3f}")
    rng = np.random.default_rng(SEED)
    for k in (1, 10, 100, 1000, 10000):
        print(f"rules {k:6d}  largest gap {luckiest(k, 100, 2000, rng):.3f}"
              f"  bound {finite_gap(100, k):.3f}")
    print()

    # Counting labellings.
    print("lines on the eight:", labellings(X, 1), " Cover:", cover(8, 2),
          " Sauer:", sauer(8, 3), " e-form:",
          round((e * 8 / 3) ** 3, 1))
    for m in (3, 4, 5, 8, 20, 100):
        print(f"m {m:3d}  2^m {2 ** m:>31d}  lines {cover(m, 2):6d}"
              f"  Sauer {sauer(m, 3):6d}")
    for m, d in ((8, 3), (1000, 3), (1000, 101), (1000, 1001)):
        print(f"VC gap m {m:5d} d {d:5d}: {vc_gap(m, d):.3f}")
    print()

    # Shattering in the plane.
    from ch09_kernel import separable
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    hit = 0
    for code in range(8):
        lab = np.where([(code >> i) & 1 for i in range(3)], 1.0, -1.0)
        hit += abs(lab.sum()) == 3 or separable(tri, lab)
    print("three points:", hit, "of 8")
    sq = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    print("square, XOR labels:", separable(sq, np.array([1., 1., -1., -1.])))
    print()

    # The eight machines.
    report_line(X, y, "raw")
    Z = X - X.mean(axis=0)
    print("centre", X.mean(axis=0))
    report_line(Z, y, "centred")
    print("support vectors 3 of 8, loo errors", loo_errors(X, y, 1e6))
    w, b, s = fit_line(Z, y)
    K = Z @ Z.T + 1
    a = dual_no_offset(K, y, 1e6)
    print("no offset: alpha", a)
    print("  sum", round(a.sum(), 3))
    print("  errors allowed by a K_ii >= 1:", int(np.sum(a * np.diag(K) >= 1)))
    print("  R^2 sum alpha / m:",
          round(np.diag(K).max() * a.sum() / 8, 2))
    print()

    # Gaussian space: nine machines' widths come from Chapter 11.
    for g in (0.1, 1, 10):
        K = rbf(X, X, g)
        s = SVC(kernel="precomputed", C=1e6).fit(K, y)
        dc = s.dual_coef_[0]
        q = dc @ K[np.ix_(s.support_, s.support_)] @ dc
        lam = sqrt(q + s.intercept_[0] ** 2)
        print(f"gamma {g:4}: rho {1 / sqrt(q):.3f}  1/rho^2 {q:7.2f}"
              f"  sv {len(s.support_)}  lam {lam:.2f}"
              f"  term {margin_gap(lam, 2.0, 8):.2f}")
    print()

    # True labels against random labels.
    print("     m  true  random")
    for m, t1, t2 in norm_growth():
        print(f"{m:6d} {t1:5.1f} {t2:7.1f}")
    print()

    # More features, same radius.
    print("     n  mode      R2    lam   test  bound     VC")
    for n, mode, r2, lam, er, gp in dimension_run():
        print(f"{n:6d}  {mode:5s} {r2:7.1f} {lam:6.2f} {er:6.3f}"
              f" {gp:6.2f} {vc_gap(200, n + 1):6.2f}")
    print()

    # Gaussian kernel on rings.
    head = " train   test     sv   ramp    lam  bound"
    print("       C" + head)
    for C, r in zip(CS, sweep([(GAMMA, C) for C in CS])):
        print(f"{C:8g} {r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f}"
              f" {r[4]:6.2f} {r[5]:6.2f}")
    print()
    print("   gamma" + head)
    for g, r in zip(GS, sweep([(g, 1.0) for g in GS])):
        print(f"{g:8g} {r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f}"
              f" {r[4]:6.2f} {r[5]:6.2f}")
    print()

    print("     m  train   test     sv    lam  bound")
    for m, tr, ts, sv, fit, lam, gp in size_run():
        print(f"{m:6d} {tr:6.3f} {ts:6.3f} {sv:6.3f} {lam:6.2f} {gp:6.2f}")
    print()

    # Leave-one-out against the support vector count, one draw.
    rng = np.random.default_rng(SEED + 2)
    A, a = noisy_rings(200, rng)
    T, t = noisy_rings(4000, rng)
    print("     C    sv    loo   test")
    for C in (0.1, 1, 10, 100):
        s = SVC(kernel="rbf", gamma=GAMMA, C=C).fit(A, a)
        test = np.mean(np.sign(s.decision_function(T)) != t)
        print(f"{C:6g} {len(s.support_) / 200:5.3f}"
              f" {loo_errors(A, a, C, GAMMA) / 200:6.3f} {test:6.3f}")

    # The kernel bound without offset, same draw.
    K = rbf(A, A, GAMMA) + 1
    print("     C  J-H bound    sv")
    for C in (0.1, 1, 10):
        al = dual_no_offset(K, a, C, passes=400, tol=1e-6)
        f = (al * a) @ K
        xi = np.maximum(0, 1 - a * f)
        print(f"{C:6g} {np.mean(al * 2 + xi >= 1):10.3f}"
              f" {np.mean(al > 1e-9):5.3f}")
