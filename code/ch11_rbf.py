"""Chapter 11: polynomial and Gaussian kernels, gamma, C, and hills."""
from fractions import Fraction as F
from math import comb, exp, factorial, log, sqrt

import numpy as np
from sklearn.svm import SVC

from ch07_soft import X9, y9
from ch09_kernel import rings
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def poly(A, B, c=1.0, d=2):
    """K(x, z) = (x'z + c)^d for all pairs of rows."""
    return (A @ B.T + c) ** d


def sq_dist(A, B):
    """Squared distances between all pairs of rows."""
    return ((A[:, None] - B[None]) ** 2).sum(axis=2)


def gauss(A, B, gamma=1.0):
    """K(x, z) = exp(-gamma |x - z|^2)."""
    return np.exp(-gamma * sq_dist(A, B))


def laplace(A, B, gamma=1.0):
    """K(x, z) = exp(-gamma |x - z|_1)."""
    return np.exp(-gamma * np.abs(A[:, None] - B[None]).sum(axis=2))


def term_weights(c, d):
    """Coefficient of (x'z)^k in (x'z + c)^d, for k = 0, ..., d."""
    return [comb(d, k) * c ** (d - k) for k in range(d + 1)]


def features(n, d, c):
    """Length of the polynomial map."""
    return comb(n + d, d) if c > 0 else comb(n + d - 1, d)


def phi_k(x, k, gamma=0.5):
    """Entry k of the Gaussian map of a scalar x."""
    return np.exp(-gamma * x ** 2) * np.sqrt((2 * gamma) ** k
                                             / factorial(k)) * x ** k


def series(x, z, gamma, terms):
    """Partial sum of phi_k(x) phi_k(z) for k < terms."""
    return sum(phi_k(x, k, gamma) * phi_k(z, k, gamma)
               for k in range(terms))


def hand():
    """Three points on a line, gamma = ln 2, in exact fractions."""
    k = {0: F(1), 1: F(1, 2), 2: F(1, 16)}
    a = F(2) / (1 + k[2])  # f(0) - f(1) = a (1 + k2) = 2
    b = 1 - a
    return a, b


def f_hand(x, a=32 / 17, b=-15 / 17):
    """Score of the hand example at x (array or scalar)."""
    h = lambda c: 2.0 ** (-(x - c) ** 2)
    return 2 * a * h(0) - a * (h(-1) + h(1)) + b


def fit_rbf(A, t, gamma, C=1e6):
    """SVC with a Gaussian kernel, tight tolerance."""
    return SVC(gamma=gamma, C=C, tol=1e-6, max_iter=2000000).fit(A, t)


def weights(s):
    """Multipliers of a fitted SVC, in order of the data."""
    a = np.zeros(s.shape_fit_[0])
    a[s.support_] = np.abs(s.dual_coef_[0])
    return a


def identity_solution(t):
    """Multipliers and offset when K = I and C is large."""
    nu = t.sum() / len(t)
    return 1 - t * nu, nu


def linear_constant(a, t, A):
    """The constant that turns the Gaussian offset into the linear b."""
    return -0.5 * np.sum(a * t * (A ** 2).sum(axis=1))


def noisy(m=120, seed=SEED):
    """The rings with heavy noise: a split that no method can make clean."""
    return rings(m, seed, noise=0.35)


GAMMAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
CS = [1e-1, 1, 10, 100, 1000]


def surface(A, t, Ate, tte):
    """Test accuracy over the grid GAMMAS by CS."""
    return np.array([[SVC(gamma=g, C=C).fit(A, t).score(Ate, tte)
                      for C in CS] for g in GAMMAS])


if __name__ == "__main__":
    print("-- term weights of (x'z + c)^3")
    for c in (0, 0.5, 1, 2, 10):
        w = term_weights(c, 3)
        print(f"c {c:>4}  ", " ".join(f"{v:7.2f}" for v in w),
              f"  length {features(2, 3, c)}")
    print("-- same inner product, c = 2")
    u, v = np.array([1.0, 2.0]), np.array([3.0, -1.0])
    print("direct", (u @ v + 2) ** 3, " sum",
          sum(term_weights(2, 3)[k] * (u @ v) ** k for k in range(4)))

    print("-- degree against scale")
    for name, Z in (("raw", X), ("tens", X / 10)):
        for d in (2, 4, 8, 12, 16):
            K = poly(Z, Z, 1, d)
            e = np.linalg.eigvalsh(K)
            rank = (e > 1e-14 * e[-1]).sum()
            print(f"{name:>4} d {d:2d}  max K {K.max():8.1e}  "
                  f"K22 {K[1, 1]:8.1e}  seen rank {rank}")
    print("-- hard margin by degree, units of ten")
    Z = X / 10
    for d in (1, 2, 3, 4, 8):
        s = SVC(kernel="poly", degree=d, gamma=1, coef0=1, C=1e6,
                tol=1e-6, max_iter=2000000).fit(Z, y)
        print(f"d {d}  sv {sorted(s.support_ + 1)}  "
              f"train {s.score(Z, y):.3f}")
    print("-- raw units, degree 12: what the solver sees")
    K = poly(X, X, 1, 12)
    s = SVC(kernel="precomputed", C=1e6, tol=1e-3).fit(K, y)
    print("sv", sorted(s.support_ + 1), " iterations", s.n_iter_[0],
          " least margin", round((y * s.decision_function(K)).min(), 3))
    print("largest alpha", f"{weights(s).max():.1e}", " least diagonal",
          f"{np.diag(K).min():.1e}")

    print("-- shifted rings, degree 4, by c")
    A, t = rings()
    Z = (A + [1.0, 0.5]) / 3
    for c in (0, 1, 100):
        s = SVC(kernel="poly", degree=4, gamma=1, coef0=c, C=100, tol=1e-4,
                max_iter=3000000).fit(Z, t)
        print(f"c {c:3d}  train {s.score(Z, t):.3f}  sv {len(s.support_)}")

    print("-- the series, x = 1, z = 2, gamma = 1/2")
    print("exact", round(exp(-0.5), 4))
    for t in (1, 2, 3, 5, 8, 12):
        print(f"terms {t:2d}  {series(1.0, 2.0, 0.5, t):.4f}")
    print("-- hills on three points, gamma = ln 2")
    a, b = hand()
    print("a", a, " b", b, " width", round(2 / sqrt(4 * a), 3))
    print("f(0.5)", round(float(f_hand(0.5)), 4), " f(1.5)",
          round(float(f_hand(1.5)), 4), " f(3)", round(float(f_hand(3.0)), 4))
    x3 = np.array([[-1.0], [0.0], [1.0]])
    t3 = np.array([-1.0, 1.0, -1.0])
    s = fit_rbf(x3, t3, log(2))
    print("SVC alpha", weights(s), " b", s.intercept_.round(4)[0])
    print("exact     ", np.array([32 / 17, 64 / 17, 32 / 17]),
          " b", round(-15 / 17, 4))

    print("-- eight machines, gamma against C")
    for g in (0.01, 0.1, 0.3, 1, 10):
        row = []
        for C in (1, 1000):
            s = SVC(gamma=g, C=C, tol=1e-6, max_iter=2000000).fit(X, y)
            row.append(f"C={C}: sv {len(s.support_)}, "
                       f"{int(round(s.score(X, y) * 8))}/8")
        K = gauss(X, X, g)
        print(f"gamma {g:5}  " + "  ".join(row)
              + f"  eig {np.linalg.eigvalsh(K)[0]:.4f}")
    print("-- gamma small, C = 2.5 / (2 gamma)")
    for g in (0.1, 0.01, 0.001, 0.0001):
        s = fit_rbf(X, y, g, 2.5 / (2 * g))
        beta = 2 * g * weights(s)
        print(f"gamma {g:6}  beta {beta[[3, 4, 6]]}  "
              f"b {s.intercept_[0]:.3f}  "
              f"sum {s.intercept_[0] + linear_constant(beta, y, X):.3f}")
    print("-- gamma large on the eight machines, C = 1000")
    s = fit_rbf(X, y, 10, 1000)
    print("alpha", weights(s), " b", round(abs(s.intercept_[0]), 4))
    print("-- K = I with three + and one -")
    print(*identity_solution(np.array([1.0, 1, 1, -1])))
    print("-- nine machines, hard margin in the Gaussian space")
    for g in (0.1, 1, 10):
        s = fit_rbf(X9, y9, g, 1e8)
        print(f"gamma {g:4}  sv {len(s.support_)}  "
              f"train {int(round(s.score(X9, y9) * 9))}/9  "
              f"width {2 / sqrt(weights(s).sum()):.3f}")

    print("-- far from the data, f at (50, 50)")
    far = np.array([[50.0, 50.0]])
    s = fit_rbf(X, y, 0.1, 1000)
    print("gaussian", round(s.decision_function(far)[0], 4),
          " b", round(s.intercept_[0], 4))
    s = SVC(kernel="poly", degree=3, gamma=1, coef0=1, C=1000).fit(X / 10, y)
    print("cubic", round(s.decision_function(far / 10)[0], 1))

    print("-- rings, noise 0.08, m = 60")
    A, t = rings()
    Ate, tte = rings(2000, 7)
    for g in (0.001, 0.1, 1, 10, 1000):
        s = SVC(gamma=g, C=1000).fit(A, t)
        print(f"gamma {g:6}  sv {len(s.support_):2d}  "
              f"train {s.score(A, t):.3f}  test {s.score(Ate, tte):.3f}")
    print("-- rings, noise 0.35, m = 120")
    A, t = noisy()
    Ate, tte = rings(4000, 7, noise=0.35)
    for g, C in ((0.001, 1), (0.01, 1000), (0.1, 10), (1, 1), (1, 1000),
                 (100, 1), (1000, 1000)):
        s = SVC(gamma=g, C=C).fit(A, t)
        print(f"gamma {g:6}  C {C:4}  sv {len(s.support_):3d}  "
              f"train {s.score(A, t):.3f}  test {s.score(Ate, tte):.3f}")
    print("-- laplace against gaussian, same rings")
    for g in (0.1, 1, 10):
        kl = SVC(kernel="precomputed", C=10).fit(laplace(A, A, g), t)
        kg = SVC(gamma=g, C=10).fit(A, t)
        print(f"gamma {g:4}  laplace {kl.score(laplace(Ate, A, g), tte):.3f}"
              f"  gaussian {kg.score(Ate, tte):.3f}")

    print("-- one feature in other units")
    stretch = np.array([1.0, 100.0])
    A2, Ate2 = A * stretch, Ate * stretch
    mu, sd = A2.mean(axis=0), A2.std(axis=0)
    r = [SVC(gamma=1, C=10).fit(A2, t).score(Ate2, tte),
         SVC(gamma="scale", C=10).fit(A2, t).score(Ate2, tte),
         SVC(gamma=1, C=10).fit((A2 - mu) / sd, t).score((Ate2 - mu) / sd,
                                                         tte)]
    print("gamma 1 {:.3f}  scale {:.3f}  standardised {:.3f}".format(*r))
    print("gamma scale on the machines", round(1 / (2 * X.var()), 4))
