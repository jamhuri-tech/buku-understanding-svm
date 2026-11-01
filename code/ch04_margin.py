"""Chapter 4: the hard-margin problem as a quadratic program."""
import numpy as np
from scipy.optimize import linprog, minimize

from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
TOL = 1e-6                 # a slack below this counts as zero


def solve_qp(P, q, A, c, z0):
    """Minimise z'Pz/2 + q'z subject to A z >= c."""
    return minimize(
        lambda z: z @ P @ z / 2 + q @ z, z0,
        jac=lambda z: P @ z + q, method="SLSQP",
        constraints={"type": "ineq", "fun": lambda z: A @ z - c,
                     "jac": lambda z: A},
        tol=1e-12, options={"maxiter": 500})


def problem(X, y):
    """The matrices of the hard-margin problem."""
    m, n = X.shape
    P = np.diag(np.r_[np.ones(n), 0.0])     # no cost on b
    A = y[:, None] * np.column_stack([X, np.ones(m)])
    return P, np.zeros(n + 1), A, np.ones(m)


def hard_margin(X, y, z0=None):
    """Return w, b and the solver's report."""
    P, q, A, c = problem(X, y)
    z0 = np.zeros(X.shape[1] + 1) if z0 is None else z0
    r = solve_qp(P, q, A, c, z0)
    return r.x[:-1], r.x[-1], r


def slacks(w, b, X=X, y=y):
    """How far each constraint is from being active."""
    return y * (X @ w + b) - 1


def width(w):
    return 2 / np.linalg.norm(w)


def feasible(X, y):
    """Does any pair meet every constraint? A linear program decides."""
    A = -y[:, None] * np.column_stack([X, np.ones(len(y))])
    r = linprog(np.zeros(A.shape[1]), A_ub=A, b_ub=-np.ones(len(y)),
                bounds=(None, None))
    return r.status == 0


def best_gap(X, y, n=720001):
    """Widest street by a sweep of directions (2-D, Equation 2.10)."""
    t = np.linspace(-np.pi, np.pi, n)
    s = X @ np.array([np.cos(t), np.sin(t)])
    return (s[y > 0].min(axis=0) - s[y < 0].max(axis=0)).max()


def generate(X, y, first):
    """Constraint generation: solve on a few points, add the worst."""
    keep, rounds = list(first), []
    while True:
        w, b, _ = hard_margin(X[keep], y[keep])
        s = slacks(w, b, X, y)
        rounds.append((list(keep), w, b, s))
        if s.min() > -TOL:
            return rounds
        keep.append(int(s.argmin()))


if __name__ == "__main__":
    # Step 1: the matrices.
    P, q, A, c = problem(X, y)
    print("P diagonal:", np.diag(P))
    print("A rows 1, 5:", A[0], A[4])
    print("smallest eigenvalue of P:", np.linalg.eigvalsh(P).min())

    # Step 2: the solver on the eight machines.
    w, b, r = hard_margin(X, y)
    s = slacks(w, b)
    print("w", w, " b", round(b, 3), " width", round(width(w), 3))
    print("steps", r.nit, " success", r.success)
    print("slack", s)
    print("active machines:", np.where(s < TOL)[0] + 1)

    # Step 3: the bridge gives the same weights.
    p, q0 = np.array([4.4, 3.8]), X[3]
    d = p - q0
    print("2(p - q)/delta^2 =", 2 * d / (d @ d))

    # Step 4: the answer ignores the order and the start.
    rng = np.random.default_rng(SEED)
    res, steps = [], []
    for _ in range(2000):
        o = rng.permutation(8)
        v, c0, r = hard_margin(X[o], y[o], rng.normal(0, 5, 3))
        res.append([*v, c0])
        steps.append(r.nit)
    res = np.array(res)
    print("2000 orders and starts: lines",
          len(np.unique(res.round(5), axis=0)),
          " largest error", round(np.abs(res - [1, 2, -11]).max(), 7),
          " steps", min(steps), "to", max(steps))

    # Step 5: delete one machine at a time.
    for i in range(8):
        k = np.arange(8) != i
        v, c0, _ = hard_margin(X[k], y[k])
        act = np.where(k & (slacks(v, c0) < TOL))[0] + 1
        print(f"without {i + 1}: w {v} b {c0:7.3f}"
              f"  width {width(v):.3f}  active {act}")

    # Step 6: constraint generation from machines 4 and 5.
    for keep, v, c0, s in generate(X, y, [3, 4]):
        print("points", [i + 1 for i in keep], " w", v,
              " b", round(c0, 3), " worst", round(s.min(), 3),
              "at", s.argmin() + 1)

    # Step 7: a machine in the wrong place.
    Xr, yr = np.vstack([X, [2, 2]]), np.append(y, 1.0)
    v, c0, r = hard_margin(Xr, yr)
    print("rogue: success", r.success, " status", r.status)
    print("  message:", r.message)
    print("  worst slack of the returned point:",
          round(slacks(v, c0, Xr, yr).min(), 3))
    lam = np.array([.5, .25, .25, 0, 0, 0, 0, 0, 1])
    print("  sum lam A =", lam @ problem(Xr, yr)[2],
          " sum lam =", lam.sum())

    # Step 8: one ninth machine, walking towards the healthy hull.
    for t in (0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.05):
        z = np.array([5, 5.0]) - 2 * t
        Xn, yn = np.vstack([X, z]), np.append(y, 1.0)
        if feasible(Xn, yn):
            v, c0, r = hard_margin(Xn, yn)
            print(f"t {t:4.2f}  feasible  sweep {best_gap(Xn, yn):.3f}"
                  f"  solver {'ok' if r.success else 'gave up'}")
        else:
            print(f"t {t:4.2f}  no feasible pair")

    # Step 9: a cost on b moves the line of machines 4 and 5.
    D, e = X[[3, 4]], y[[3, 4]]
    P2, q2, A2, c2 = problem(D, e)
    z = solve_qp(np.eye(3), q2, A2, c2, np.zeros(3)).x
    v, c0, _ = hard_margin(D, e)
    print("no cost on b: w", v, " b", round(c0, 3),
          " width", round(width(v), 3))
    print("cost on b   : w", z[:2], " b", round(z[2], 3),
          " width", round(width(z[:2]), 3))
    z = solve_qp(np.eye(3), np.zeros(3), A, np.ones(8), np.zeros(3)).x
    print("eight, cost on b: w", z[:2], " b", round(z[2], 3))
