"""Chapter 8: the hinge loss, its relatives, and subgradient descent.

The data are those of Chapter 7. Every result of the descent is
compared with the dual solution of ch07_soft.py.
"""
from fractions import Fraction as F

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC, LinearSVC

from ch07_soft import X9, soft_line, solve_soft, y9
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def hinge(t):
    return np.maximum(0, 1 - t)


def logistic(t):
    return np.logaddexp(0, -t)


def squared(t):
    return (1 - t) ** 2


def objective(w, b, C, X=X9, y=y9):
    """J(w, b): the soft margin with the slacks removed."""
    return w @ w / 2 + C * hinge(y * (X @ w + b)).sum()


def subgradient(w, b, C, X=X9, y=y9):
    """One subgradient of J: beta_i = 1 inside, 0 on the edge."""
    beta = (y * (X @ w + b) < 1).astype(float)
    return w - C * (beta * y) @ X, -C * beta @ y, beta


def descend(C, X=X9, y=y9, steps=200_000, eta=0.2, marks=()):
    """Subgradient descent with eta / sqrt(k + 1), best point kept."""
    w, b = np.zeros(X.shape[1]), 0.0
    best = objective(w, b, C, X, y), w, b
    avg, total, trace = np.zeros(len(y)), 0.0, []
    for k in range(steps):
        gw, gb, beta = subgradient(w, b, C, X, y)
        step = eta / np.sqrt(k + 1)
        avg += step * beta
        total += step
        w, b = w - step * gw, b - step * gb
        J = objective(w, b, C, X, y)
        if J < best[0]:
            best = J, w, b
        if k + 1 in marks:
            trace.append((k + 1, J, *best))
    return best, C * avg / total, trace


def fit_logistic(C, X=X9, y=y9):
    """Minimise |w|^2 / 2 + C sum log(1 + exp(-t)), in centred data."""
    c = X.mean(axis=0)
    Z = X - c

    def f(v):
        w, b = v[:-1], v[-1]
        t = y * (Z @ w + b)
        beta = 1 / (1 + np.exp(t))
        return (w @ w / 2 + C * logistic(t).sum(),
                np.r_[w - C * (beta * y) @ Z, -C * beta @ y])
    v = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="BFGS",
                 options={"gtol": 1e-9, "maxiter": 5000}).x
    return v[:-1], v[-1] - v[:-1] @ c


def fit_squared(C, X=X9, y=y9):
    """Minimise |w|^2 / 2 + C sum (1 - t)^2: a linear system."""
    A = np.c_[X, np.ones(len(y))]
    P = np.eye(A.shape[1]) / (2 * C)
    P[-1, -1] = 0
    v = np.linalg.solve(A.T @ A + P, A.T @ y)
    return v[:-1], v[-1]


def wrong(w, b, X=X9, y=y9):
    return int(np.sum(y * (X @ w + b) < 0))


def leave_one_out(C, X, y):
    """Errors when each point is predicted from all the others."""
    e = 0
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        m = SVC(kernel="linear", C=C, tol=1e-8).fit(X[k], y[k])
        e += m.predict(X[[i]])[0] != y[i]
    return int(e)


def clouds(n, rng):
    """Two overlapping Gaussian classes in the plane."""
    yc = rng.choice([-1.0, 1.0], n)
    return rng.normal(size=(n, 2)) + 3 + yc[:, None], yc


def sweep(Cs, rounds=100, m=30, seed=SEED):
    """Mean training and test error over many small samples."""
    rng = np.random.default_rng(seed)
    train, test = np.zeros(len(Cs)), np.zeros(len(Cs))
    for _ in range(rounds):
        Xa, ya = clouds(m, rng)
        Xt, yt = clouds(1000, rng)
        for j, C in enumerate(Cs):
            f = SVC(kernel="linear", C=C).fit(Xa, ya)
            train[j] += (1 - f.score(Xa, ya)) / rounds
            test[j] += (1 - f.score(Xt, yt)) / rounds
    return train, test


if __name__ == "__main__":
    # Step 1: three losses and their slopes.
    t = np.array([-1, 0, 0.5, 1, 2, 5.0])
    print("t       ", t)
    print("hinge   ", hinge(t))
    print("logistic", logistic(t))
    print("squared ", squared(t))
    print("-d log  ", 1 / (1 + np.exp(t)))
    print("-d sq   ", 2 * (1 - t))

    # Step 2: J gives the objective of Chapter 7 without slacks.
    for w, b in ((np.array([1.0, 2.0]), -11.0),
                 (np.array([0.4, 0.5]), -2.9)):
        print("J", round(objective(w, b, 0.5), 6),
              " hinge sum", round(hinge(y9 * (X9 @ w + b)).sum(), 6),
              " wrong", wrong(w, b))

    # Step 3: lambda = 1 / (m C). Two copies, half the penalty.
    a1 = soft_line(0.5, X9, y9)
    a2 = soft_line(0.25, np.vstack([X9, X9]), np.r_[y9, y9])
    print("one copy", a1[0], round(a1[1], 3),
          " two copies", a2[0], round(a2[1], 3))

    # Step 4: the multipliers as a subgradient that vanishes.
    C = 0.5
    w, b, xi, a = soft_line(C, X9, y9)
    beta = a / C
    print("beta", beta)
    print("zero subgradient", np.abs(w - C * (beta * y9) @ X9).max()
          < 1e-6, abs(C * beta @ y9) < 1e-6)

    # Step 5: four steps by hand, eta = 1/20, exact fractions.
    Xf = [[F(int(v)) for v in row] + [F(1)] for row in X9]
    yf = [F(int(v)) for v in y9]
    th = [F(0)] * 3
    for k in range(5):
        s = [yi * sum(u * v for u, v in zip(row, th))
             for row, yi in zip(Xf, yf)]
        act = [i + 1 for i, si in enumerate(s) if si < 1]
        J = (th[0] ** 2 + th[1] ** 2) / 2 + sum(
            (1 - si) / 2 for si in s if si < 1)
        print(f"k {k}  w {float(th[0]):.4f} {float(th[1]):.4f}  "
              f"b {float(th[2]):.4f}  J {float(J):.4f}  act",
              *act)
        g = [(th[0] if j == 0 else th[1] if j == 1 else 0) - sum(
            yf[i - 1] * Xf[i - 1][j] for i in act) / 2
            for j in range(3)]
        th = [u - v / 20 for u, v in zip(th, g)]

    # Step 6: a subgradient need not point downhill.
    w0, b0 = np.array([0.4, 0.5]), -2.9
    gw, gb, _ = subgradient(w0, b0, C)
    print("at the minimiser: g", gw.round(9), round(gb, 9))
    print("J before", round(objective(w0, b0, C), 4), " after",
          round(objective(w0 - 0.1 * gw, b0 - 0.1 * gb, C), 4))

    # Step 7: the descent against the dual solution of Chapter 7.
    marks = (10, 100, 1000, 10_000, 100_000, 200_000)
    bst, avg, trace = descend(C, marks=marks)
    for k, Jk, Jb, wb, bb in trace:
        print(f"k {k:6d}  J {Jk:.5f}  best {Jb:.5f}  "
              f"w {wb[0]:.4f} {wb[1]:.4f}  b {bb:.4f}")
    print(f"dual  J {objective(w, b, C):.5f}  "
          f"w {w[0]:.4f} {w[1]:.4f}  b {b:.4f}")

    # Step 8: the running average of beta estimates the multipliers.
    print("avg C beta", avg)
    print("alpha     ", a)

    # Step 9: two more checks on Chapter 7.
    (Jb, wb, bb), _, _ = descend(1.0, X, y)
    print(f"eight, C = 1: w {wb[0]:.4f} {wb[1]:.4f}  b {bb:.4f}  "
          f"J {Jb:.5f}")
    (Jb, wb, bb), _, _ = descend(0.05, X, y)
    print(f"tie, C = 1/20: w {wb[0]:.4f} {wb[1]:.4f}  b {bb:.4f}")

    # Step 10: centring helps the offset (Chapter 14).
    c = X9.mean(axis=0)
    for name, Xd in (("raw", X9), ("centred", X9 - c)):
        (Jb, wb, bb), _, _ = descend(C, Xd, y9, steps=2000)
        print(f"{name:7s} 2000 steps: best J {Jb:.4f}  "
              f"w {wb[0]:.4f} {wb[1]:.4f}")

    # Step 11: three losses on the nine machines.
    C = 0.5
    fits = (("hinge", *soft_line(C, X9, y9)[:2]),
            ("logistic", *fit_logistic(C)), ("squared", *fit_squared(C)))
    for name, w, b in fits:
        s = y9 * (X9 @ w + b)
        print(f"{name:8s} w {w[0]:.3f} {w[1]:.3f}  b {b:6.3f}  "
              f"angle {np.degrees(np.arctan2(w[1], w[0])):.1f}  "
              f"wrong {wrong(w, b)}")
    print("pulls", 1 / (1 + np.exp(
        y9 * (X9 @ fits[1][1] + fits[1][2]))))

    # Step 12: a far machine on the correct side, C = 1.
    Xo, yo = np.vstack([X, [16.0, 6.0]]), np.append(y, 1.0)
    for name, fit in (("hinge", lambda A, u: soft_line(1, A, u)[:2]),
                      ("logistic", lambda A, u: fit_logistic(1, A, u)),
                      ("squared", lambda A, u: fit_squared(1, A, u))):
        w1, b1 = fit(X, y)
        w2, b2 = fit(Xo, yo)
        print(f"{name:8s} w {w1[0]:.3f} {w1[1]:.3f} -> "
              f"{w2[0]:.3f} {w2[1]:.3f}  b {b1:.3f} -> {b2:.3f}")

    # Step 13: logistic on separable data, as C grows.
    for C in (1, 100, 1e4, 1e5):
        w, b = fit_logistic(C, X, y)
        print(f"C {C:8.0f}  norm {np.linalg.norm(w):6.2f}  "
              f"angle {np.degrees(np.arctan2(w[1], w[0])):.2f}")
    print("hard margin angle", round(np.degrees(np.arctan2(2, 1)), 2))

    # Step 14: LinearSVC penalises the offset unless told otherwise.
    for s in (1, 100):
        m = LinearSVC(loss="hinge", C=0.5, intercept_scaling=s, tol=1e-8,
                      max_iter=2_000_000).fit(X9, y9)
        print(f"LinearSVC scaling {s:3d}: w {m.coef_[0]}  "
              f"b {m.intercept_[0]:.3f}")

    # Step 15: is a training error worth it?
    print("C          ", *(f"{C:5}" for C in (0.1, 0.2, 0.5, 1, 2.5, 10)))
    print("leave-one-out",
          *(f"{leave_one_out(C, X, y):5d}"
            for C in (0.1, 0.2, 0.5, 1, 2.5, 10)))
    Cs = (0.01, 0.1, 1, 10, 100, 1000)
    train, test = sweep(Cs)
    print("C    ", *(f"{C:6}" for C in Cs))
    print("train", *(f"{100 * v:6.1f}" for v in train))
    print("test ", *(f"{100 * v:6.1f}" for v in test))
