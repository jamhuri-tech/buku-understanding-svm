"""Chapter 3: the perceptron on the eight machines."""
import collections
import itertools

import numpy as np
from scipy.optimize import minimize

from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def perceptron(X, y, order=None, w=None, b=0.0, max_epochs=5000):
    """Cycle through the data until one clean pass.

    Return w, b, the mistake count of each point, the epochs used,
    and the list of updates. Epochs is None if no pass was clean.
    """
    order = range(len(y)) if order is None else order
    w = np.zeros(X.shape[1]) if w is None else np.array(w, dtype=float)
    a, log = np.zeros(len(y), dtype=int), []
    for epoch in range(1, max_epochs + 1):
        clean = True
        for i in order:
            if y[i] * (w @ X[i] + b) <= 0:       # a tie is a mistake
                w, b = w + y[i] * X[i], b + y[i]
                a[i] += 1
                log.append((epoch, i + 1, w.copy(), b))
                clean = False
        if clean:
            return w, b, a, epoch, log
    return w, b, a, None, log


def width(w, b, X=X, y=y):
    """Street width of a separating line, as found."""
    return 2 * (y * (X @ w + b)).min() / np.linalg.norm(w)


def centred_width(w, X=X, y=y):
    """Width of the same direction with the line moved to the middle."""
    s = X @ w
    return (s[y > 0].min() - s[y < 0].max()) / np.linalg.norm(w)


def lift(X):
    """Append a 1 to every point."""
    return np.column_stack([X, np.ones(len(X))])


def bound(X, y, v):
    """R, gamma and Novikoff's limit for lifted weights v."""
    Z = lift(X)
    R2 = (Z ** 2).sum(axis=1).max()
    g = (y * (Z @ v)).min() / np.linalg.norm(v)
    return R2, g, R2 / g ** 2


def best_lifted(X, y):
    """Lifted weights with the largest lifted margin."""
    Z = lift(X) * y[:, None]
    r = minimize(lambda v: v @ v, np.ones(Z.shape[1]), jac=lambda v: 2 * v,
                 constraints={"type": "ineq", "fun": lambda v: Z @ v - 1,
                              "jac": lambda v: Z},
                 method="SLSQP", tol=1e-12, options={"maxiter": 1000})
    return r.x


def survey(X, y):
    """Run every order of the data from zero."""
    rows = []
    for p in itertools.permutations(range(len(y))):
        w, b, a, ep, _ = perceptron(X, y, p)
        rows.append((*w, b, a.sum(), ep, width(w, b, X, y),
                     centred_width(w, X, y)))
    return np.array(rows)


if __name__ == "__main__":
    Xc = X - X.mean(axis=0)

    # Step 1: the order 1, 2, ..., 8 on the raw data.
    w, b, a, ep, log = perceptron(X, y)
    for e, i, v, c in log[:4]:
        print(f"pass {e} machine {i}  w {v}  b {c:g}")
    print("raw   : w", w, "b", b, "mistakes", a.sum(), "passes", ep)
    print("width :", round(width(w, b), 3))
    print("counts:", a)
    print("sum a y x =", (a * y) @ X, " sum a y =", a @ y)

    # Step 2: the same order on centred data.
    wc, bc, ac, ep, log = perceptron(Xc, y)
    for e, i, v, c in log:
        print(f"pass {e} machine {i}  w {v}  b {c:g}")
    print("centred: mistakes", ac.sum(), "passes", ep,
          "width", round(width(wc, bc, Xc), 3))

    # Step 3: Novikoff's limit for the known line and the best one.
    for name, D, v in (("raw", X, np.array([1, 2, -11.0])),
                       ("centred", Xc, np.array([1, 2, -0.5]))):
        for tag, u in (("svm ", v), ("best", best_lifted(D, y))):
            R2, g, lim = bound(D, y, u)
            print(f"{name:8s}{tag} R2 {R2:5.1f}  gamma {g:.4f}"
                  f"  limit {lim:7.1f}")

    # Step 4: all 40320 orders, from zero.
    for name, D in (("raw", X), ("centred", Xc)):
        T = survey(D, y)
        print(f"{name}: mistakes {T[:, 3].min():.0f} to {T[:, 3].max():.0f}"
              f", mean {T[:, 3].mean():.1f}; passes up to {T[:, 4].max():.0f}")
        print(f"  distinct lines {len(np.unique(T[:, :3], axis=0))}"
              f", width {T[:, 5].min():.3f} to {T[:, 5].max():.3f}"
              f", mean {T[:, 5].mean():.3f}")
        print(f"  reach 0.894: {(T[:, 5] > 0.8944).sum()}"
              f", after centring the line: {(T[:, 6] > 0.8944).sum()}"
              f", mean {T[:, 6].mean():.3f}")

    # Step 5: random starting points, in the fixed order.
    rng = np.random.default_rng(SEED)
    res = []
    for _ in range(2000):
        s = rng.normal(0, 3, 3)
        v, c, _, _, _ = perceptron(Xc, y, w=s[:2], b=s[2])
        res.append(width(v, c, Xc))
    res = np.array(res)
    print(f"random starts: width {res.min():.3f} to {res.max():.3f}"
          f", mean {res.mean():.3f}, reach 0.894: {(res > 0.8944).sum()}")

    # Step 6: no separating line, no stop.
    Xr, yr = np.vstack([X, [2, 2]]), np.append(y, 1.0)
    w, b, a, ep, log = perceptron(Xr, yr, max_epochs=2000)
    seen = collections.Counter((*v, c) for _, _, v, c in log)
    big = max(np.hypot(*v) for _, _, v, _ in log)
    print("rogue :", "stopped" if ep else "no clean pass in 2000")
    print("  mistakes", a.sum(), " largest |w|", round(big, 2),
          " weights seen", len(seen))
