"""Chapter 16: Support vector regression."""
import warnings
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR

from machines import X

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
TOL = 1e-6

# Four points of the worked example.
HX = np.array([0.0, 1, 2, 4])
HY = np.array([0.0, 1, 3, 2])
HE = 0.5
HC = 1.0


def tube_loss(r, eps):
    """Epsilon-insensitive loss of a residual."""
    return np.maximum(0, np.abs(r) - eps)


def huber(r, c):
    a = np.abs(r)
    return np.where(a <= c, r ** 2 / 2, c * a - c ** 2 / 2)


def exact_check():
    """Check every KKT condition of the worked example in fractions."""
    x = [Fr(0), Fr(1), Fr(2), Fr(4)]
    t = [Fr(0), Fr(1), Fr(3), Fr(2)]
    e, c = Fr(1, 2), Fr(1)
    w, b = Fr(1, 2), Fr(1, 2)
    up = [Fr(0), Fr(0), Fr(1), Fr(0)]
    dn = [Fr(5, 8), Fr(0), Fr(0), Fr(3, 8)]
    beta = [u - d for u, d in zip(up, dn)]
    r = [ti - (w * xi + b) for xi, ti in zip(x, t)]
    xi = [max(Fr(0), ri - e) for ri in r]
    xs = [max(Fr(0), -ri - e) for ri in r]
    assert sum(beta) == 0
    assert sum(bi * xi_ for bi, xi_ in zip(beta, x)) == w
    for i in range(4):
        assert 0 <= up[i] <= c and 0 <= dn[i] <= c
        assert up[i] * (e + xi[i] - r[i]) == 0
        assert dn[i] * (e + xs[i] + r[i]) == 0
        assert (c - up[i]) * xi[i] == 0
        assert (c - dn[i]) * xs[i] == 0
    primal = w ** 2 / 2 + c * (sum(xi) + sum(xs))
    dual = (sum(bi * ti for bi, ti in zip(beta, t))
            - e * sum(abs(bi) for bi in beta) - w ** 2 / 2)
    print("residuals", [str(ri) for ri in r])
    print("beta", [str(bi) for bi in beta])
    print("primal", primal, "dual", dual)


def solve_dual(K, t, C, eps):
    """Dual in alpha, alpha* by a general solver; returns the two."""
    m = len(t)
    Q = np.block([[K, -K], [-K, K]])
    p = np.r_[eps - t, eps + t]
    s = np.r_[np.ones(m), -np.ones(m)]

    def fun(z):
        return 0.5 * z @ Q @ z + p @ z

    res = minimize(fun, np.zeros(2 * m), jac=lambda z: Q @ z + p,
                   bounds=[(0, C)] * (2 * m), method="SLSQP",
                   constraints={"type": "eq", "fun": lambda z: s @ z,
                                "jac": lambda z: s},
                   options={"maxiter": 500, "ftol": 1e-12})
    z = np.where(res.x < 1e-9, 0, res.x)
    return z[:m], z[m:]


def offset_range(K, t, up, dn, C, eps):
    """Interval of b allowed by the KKT conditions."""
    g = t - K @ (up - dn)
    lo = np.r_[g[up < C - TOL] - eps, g[dn > TOL] + eps]
    hi = np.r_[g[up > TOL] - eps, g[dn < C - TOL] + eps]
    return lo.max(), hi.min()


def kinds(r, eps):
    """Name each point by its place relative to the tube."""
    out = []
    for ri in r:
        if abs(ri) < eps - TOL:
            out.append("inside")
        elif abs(ri) <= eps + TOL:
            out.append("edge")
        else:
            out.append("outside")
    return out


def svr_smo(K, t, C, eps, tol=1e-3, limit=200000):
    """Maximal-violation SMO on the 2m variables (alpha, alpha*)."""
    m = len(t)
    s = np.r_[np.ones(m), -np.ones(m)]
    p = np.r_[eps - t, eps + t]
    z = np.zeros(2 * m)
    grad = p.copy()     # gradient of the minimised objective
    steps = 0
    idx = np.r_[np.arange(m), np.arange(m)]

    def column(i):
        return s[i] * s * K[idx[i], idx]

    while steps < limit:
        v = -s * grad
        up = ((s > 0) & (z < C)) | ((s < 0) & (z > 0))
        low = ((s > 0) & (z > 0)) | ((s < 0) & (z < C))
        i = np.where(up)[0][np.argmax(v[up])]
        j = np.where(low)[0][np.argmin(v[low])]
        gap = v[i] - v[j]
        if gap < tol:
            break
        eta = max(K[idx[i], idx[i]] + K[idx[j], idx[j]]
                  - 2 * K[idx[i], idx[j]], 1e-12)
        d = gap / eta   # move s_i z_i up and s_j z_j down by d
        room_i = C - z[i] if s[i] > 0 else z[i]
        room_j = z[j] if s[j] > 0 else C - z[j]
        d = min(d, room_i, room_j)
        qi, qj = s[i] * column(i), s[j] * column(j)
        z[i] += s[i] * d
        z[j] -= s[j] * d
        z = np.clip(z, 0, C)
        grad += s[i] * d * qi * s[i] - s[j] * d * qj * s[j]
        steps += 1
    b = 0.5 * (v[i] + v[j])
    return z[:m], z[m:], b, steps


def show_hand():
    print("loss", tube_loss(np.array([-2, -.5, .2, 1.5]), HE))
    exact_check()
    K = np.outer(HX, HX)
    up, dn = solve_dual(K, HY, HC, HE)
    print("slsqp alpha ", up)
    print("slsqp alpha*", dn)
    print("products", up * dn)
    w = (up - dn) @ HX
    print("w", round(w, 6), "range", np.round(
        offset_range(K, HY, up, dn, HC, HE), 6))
    m = SVR(kernel="linear", C=HC, epsilon=HE, tol=1e-8)
    m.fit(HX[:, None], HY)
    print("SVR w", m.coef_.ravel(), "b", m.intercept_,
          "sv", m.support_ + 1, "coef", m.dual_coef_.ravel())
    print("our smo", np.round(svr_smo(K, HY, HC, HE, 1e-8)[2], 4))
    r = HY - m.predict(HX[:, None])
    print("kinds", kinds(r, HE))


def show_path():
    print(" C      w      b  range          sv  slack")
    K = np.outer(HX, HX)
    for C in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1, 10):
        up, dn = solve_dual(K, HY, C, HE)
        w = (up - dn) @ HX
        lo, hi = offset_range(K, HY, up, dn, C, HE)
        bb = (lo + hi) / 2
        sl = tube_loss(HY - w * HX - bb, HE).sum()
        n = int(((up + dn) > TOL).sum())
        print(f"{C:5.2f} {w:6.3f} {bb:6.3f}  [{lo:.3f}, {hi:.3f}]"
              f"  {n}  {sl:.3f}")
    m = SVR(kernel="linear", C=0.1, epsilon=HE, tol=1e-8)
    m.fit(HX[:, None], HY)
    print("SVR at C = 0.1: w", m.coef_.ravel(), "b", m.intercept_)


def show_eps():
    print("eps     w      b  sv  outside")
    for e in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2):
        m = SVR(kernel="linear", C=100, epsilon=e, tol=1e-8)
        m.fit(HX[:, None], HY)
        r = HY - m.predict(HX[:, None])
        out = int((np.abs(r) > e + 1e-6).sum())
        print(f"{e:4.2f} {m.coef_[0, 0]:6.3f} {m.intercept_[0]:6.3f}"
              f"  {len(m.support_)}  {out}")


def show_tie():
    """Equal pull on both sides: a stretch of offsets."""
    K = np.outer(HX, HX) * 0
    up, dn = solve_dual(K, HY, HC, 1.5)
    print("flat: range", offset_range(K, HY, up, dn, HC, 1.5))
    up, dn = solve_dual(K, HY, HC, 1.75)
    print("wide: range", offset_range(K, HY, up, dn, HC, 1.75))


def show_far():
    """Worked example: far points, K = I."""
    x = np.array([[0.0], [10], [20]])
    t = np.array([0.0, 1, 4])
    m = SVR(kernel="rbf", gamma=1, C=2, epsilon=0.5, tol=1e-8).fit(x, t)
    print("far b", m.intercept_, "beta", m.dual_coef_.ravel())
    print("fit", m.predict(x), "far away", m.predict([[100.0]]))


def show_machines():
    """Temperature rise from vibration on the eight machines."""
    a, t = X[:, :1].astype(float), X[:, 1].astype(float)
    m = SVR(kernel="linear", C=1, epsilon=1, tol=1e-8).fit(a, t)
    r = t - m.predict(a)
    print("eight: w", m.coef_.ravel(), "b", m.intercept_,
          "sv", m.support_ + 1)
    print("coef", m.dual_coef_.ravel())
    print("residuals", r)


def line_data(m=30, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, m)
    return x, 1 + 0.5 * x + rng.normal(0, 0.3, m)


def fit_three(x, t):
    """Slopes and offsets: least squares, ridge, SVR on one feature."""
    a = x[:, None]
    ls = LinearRegression().fit(a, t)
    ri = Ridge(alpha=1.0).fit(a, t)
    sv = SVR(kernel="linear", C=1, epsilon=0.3, tol=1e-6).fit(a, t)
    return {"ls": (ls.coef_[0], ls.intercept_),
            "ridge": (ri.coef_[0], ri.intercept_),
            "svr": (sv.coef_[0, 0], sv.intercept_[0])}


def show_outlier():
    x, t = line_data()
    print("rise    ls       ridge    svr     (slope)")
    for rise in (0, 5, 20, 50, 500):
        xx, tt = np.r_[x, 8], np.r_[t, 5 + rise]
        f = fit_three(xx, tt)
        print(f"{rise:4d}  {f['ls'][0]:7.3f}  {f['ridge'][0]:7.3f}"
              f"  {f['svr'][0]:7.3f}")
    print("far x   ls       ridge    svr     (slope)")
    for far in (20, 60, 200):
        xx, tt = np.r_[x, far], np.r_[t, 0]
        f = fit_three(xx, tt)
        print(f"{far:4d}  {f['ls'][0]:7.3f}  {f['ridge'][0]:7.3f}"
              f"  {f['svr'][0]:7.3f}")


def curve(m=60, sigma=0.2, seed=SEED, spoil=0.0):
    """Noisy sine; a share `spoil` of the labels gets a large shift."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, m)
    t = np.sin(2 * np.pi * x) + rng.normal(0, sigma, m)
    if spoil:
        bad = rng.random(m) < spoil
        t[bad] += rng.choice([-1.0, 1.0], bad.sum()) * rng.uniform(2, 5, bad.sum())
    return x[:, None], t


def truth(x):
    return np.sin(2 * np.pi * x.ravel())


GRID = np.linspace(0, 1, 401)[:, None]
GAMMA = 10.0


def rmse(model):
    return np.sqrt(np.mean((model.predict(GRID) - truth(GRID)) ** 2))


def eps_sweep(sigma=0.2, C=10.0):
    a, t = curve(sigma=sigma)
    rows = []
    for e in (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2):
        m = SVR(C=C, gamma=GAMMA, epsilon=e, tol=1e-5).fit(a, t)
        rows.append((e, len(m.support_), rmse(m)))
    return rows


def show_sweep():
    print(" eps   sv  rmse")
    for e, n, s in eps_sweep():
        print(f"{e:4.2f}  {n:3d}  {s:.3f}")
    m, sig = 60, 0.2
    print("rule", round(3 * sig * np.sqrt(np.log(m) / m), 3))
    a, t = curve()
    print("    C   sv  rmse")
    for C in (0.01, 0.1, 1, 10, 100, 1000, 10000):
        f = SVR(C=C, gamma=GAMMA, epsilon=0.15, tol=1e-5).fit(a, t)
        print(f"{C:6.2f} {len(f.support_):3d}  {rmse(f):.3f}")


def show_rival():
    """SVR against kernel ridge, clean and spoiled labels, 40 draws."""
    print("spoil   ridge  svr    ridge sv  svr sv")
    for spoil in (0.0, 0.1, 0.2):
        a0, b0, n0 = [], [], []
        for k in range(40):
            a, t = curve(seed=SEED + k, spoil=spoil)
            kr = KernelRidge(kernel="rbf", gamma=GAMMA, alpha=0.1)
            kr.fit(a, t - t.mean())
            sv = SVR(C=10, gamma=GAMMA, epsilon=0.15, tol=1e-5).fit(a, t)
            a0.append(np.sqrt(np.mean(
                (kr.predict(GRID) + t.mean() - truth(GRID)) ** 2)))
            b0.append(rmse(sv))
            n0.append(len(sv.support_))
        print(f"{spoil:4.1f}  {np.mean(a0):.3f}  {np.mean(b0):.3f}"
              f"  {60:7d}  {np.mean(n0):6.1f}")


def show_scale():
    a, t = curve()
    t = 1000 * t
    plain = SVR(gamma=GAMMA).fit(a, t)
    print("default R2", round(plain.score(a, t), 3),
          "sv", len(plain.support_))
    scaled = TransformedTargetRegressor(
        SVR(gamma=GAMMA), transformer=StandardScaler()).fit(a, t)
    print("scaled y R2", round(scaled.score(a, t), 3),
          "sv", len(scaled.regressor_.support_))
    same = SVR(gamma=GAMMA, C=1000, epsilon=100).fit(a, t)
    print("C, eps x 1000 R2", round(same.score(a, t), 3),
          "sv", len(same.support_))


def show_nu():
    a, t = curve()
    print(" nu   eps   outside  sv")
    for nu in (0.1, 0.25, 0.5, 0.75):
        m = NuSVR(nu=nu, C=10, gamma=GAMMA, tol=1e-6).fit(a, t)
        r = t - m.predict(a)
        free = np.abs(m.dual_coef_.ravel()) < 10 - 1e-6
        e = np.median(np.abs(r[m.support_[free]]))
        out = np.mean(np.abs(r) > e + 1e-4)
        print(f"{nu:4.2f} {e:.3f}  {out:.3f}  {len(m.support_) / 60:.3f}")


def show_smo():
    a, t = curve()
    d = (a - a.T) ** 2
    K = np.exp(-GAMMA * d)
    up, dn, b, steps = svr_smo(K, t, 10.0, 0.15)
    m = SVR(C=10, gamma=GAMMA, epsilon=0.15, tol=1e-3).fit(a, t)
    beta = up - dn
    f = K @ beta + b
    print("our steps", steps, "SVR steps", m.n_iter_)
    print("largest product", (up * dn).max(),
          "sv", int((np.abs(beta) > 1e-9).sum()), len(m.support_))
    print("largest gap to SVR", np.abs(f - m.predict(a)).max())


if __name__ == "__main__":
    warnings.simplefilter("ignore", ConvergenceWarning)
    show_hand()
    show_path()
    show_eps()
    show_tie()
    show_far()
    show_machines()
    show_outlier()
    show_sweep()
    show_rival()
    show_scale()
    show_nu()
    show_smo()
