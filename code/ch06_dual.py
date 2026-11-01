"""Chapter 6: the dual problem, its solution, and support vectors."""
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog, minimize

from ch04_margin import TOL, hard_margin
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101


def dual_matrix(X=X, y=y):
    """Q has the entries y_i y_j x_i'x_j."""
    Z = y[:, None] * X
    return Z @ Z.T


def dual_value(a, Q):
    """The dual objective, sum(a) - a'Qa/2."""
    return a.sum() - a @ Q @ a / 2


def solve_dual(Q, y=y, a0=None):
    """Maximise the dual: a >= 0, a'y = 0. Needs only Q and y."""
    a0 = np.zeros(len(y)) if a0 is None else a0
    return minimize(
        lambda a: a @ Q @ a / 2 - a.sum(), a0,
        jac=lambda a: Q @ a - 1, method="SLSQP",
        bounds=[(0, None)] * len(y),
        constraints={"type": "eq", "fun": lambda a: a @ y,
                     "jac": lambda a: y},
        tol=1e-12, options={"maxiter": 500})


def recover(a, X=X, y=y, tol=TOL):
    """The line from the multipliers: w, b, and the support set."""
    S = np.where(a > tol)[0]
    w = (a[S] * y[S]) @ X[S]
    b = np.mean(y[S] - X[S] @ w)           # average over S
    return w, b, S


def hull_weights(a, y=y):
    """Scale of the bridge and the weights of its two ends."""
    A = a[y > 0].sum()
    return A, a / A


def alpha_range(w, b, X=X, y=y, tol=TOL):
    """Least and greatest alpha_i over all dual solutions."""
    edge = np.abs(y * (X @ w + b) - 1) < tol
    m = len(y)
    Aeq = np.vstack([(y[:, None] * X).T, y])
    beq = np.r_[w, 0.0]
    ends = np.zeros((m, 2))
    box = [(0, None) if e else (0, 0) for e in edge]
    for i in range(m):
        c = np.zeros(m)
        c[i] = 1.0
        ends[i, 0] = linprog(c, A_eq=Aeq, b_eq=beq, bounds=box).fun
        ends[i, 1] = -linprog(-c, A_eq=Aeq, b_eq=beq, bounds=box).fun
    return ends


def kind(lo, hi, on_edge, tol=TOL):
    """Name the role of one point."""
    if lo > tol:
        return "essential"
    if hi > tol:
        return "optional"
    return "idle" if on_edge else "off edge"


def exact_active(S):
    """Solve Q a + b y = 1, a'y = 0 on S in fractions."""
    xs = [[F(int(v)) for v in X[i]] for i in S]
    ys = [F(int(y[i])) for i in S]
    k = len(S)
    M = [[ys[i] * ys[j] * (xs[i][0] * xs[j][0] + xs[i][1] * xs[j][1])
          for j in range(k)] + [ys[i], F(1)] for i in range(k)]
    M.append(ys + [F(0), F(0)])
    for c in range(k + 1):                  # Gauss-Jordan
        p = next(r for r in range(c, k + 1) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(k + 1):
            if r != c:
                M[r] = [u - M[r][c] * v for u, v in zip(M[r], M[c])]
    return [row[-1] for row in M]


if __name__ == "__main__":
    # Step 1: the matrix of the dual.
    Q = dual_matrix()
    print("rank of Q:", np.linalg.matrix_rank(Q), " of", len(y))
    print("least eigenvalue:", round(np.linalg.eigvalsh(Q).min(), 9) + 0)

    # Step 2: the exact solution on the active set {4, 5, 7}.
    sol = exact_active([3, 4, 6])
    print("alpha 4, 5, 7 and b:", [str(v) for v in sol])

    # Step 3: the solver sees only Q and y.
    r = solve_dual(Q)
    print("alpha", r.x, " dual value", round(-r.fun, 6))
    w, b, S = recover(r.x)
    print("w", w, " b", round(b, 6), " support", S + 1)
    print("sum alpha", round(r.x.sum(), 6), " |w|^2", round(w @ w, 6))

    # Step 4: alpha as bridge weights.
    A, lam = hull_weights(r.x)
    p, q = lam[y > 0] @ X[y > 0], lam[y < 0] @ X[y < 0]
    print("A", round(A, 6), " p", p, " q", q,
          " bridge", round(np.linalg.norm(p - q), 6))

    # Step 5: other combinations that give the same w.
    a3 = np.array([16, 2, 7, 2, 15, 0, 1, 0.0])
    print("perceptron counts: balance", a3 @ y)
    Aeq = np.vstack([(y[:, None] * X).T, y])
    low = linprog(np.ones(8), A_eq=Aeq, b_eq=np.r_[w, 0.0],
                  bounds=[(0, None)] * 8).x
    print("smallest sum", low)
    print("sum", round(low.sum(), 3),
          " dual value", round(dual_value(low, Q), 3))

    # Step 6: rogue machine z = (2, 2): the dual has no maximum.
    Xr, yr = np.vstack([X, [2, 2]]), np.append(y, 1.0)
    Qr = dual_matrix(Xr, yr)
    ray = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 1])
    print("ray: balance", ray @ yr, " Qray", np.abs(Qr @ ray).max(),
          " sum", ray.sum())
    for t in (1, 10, 100):
        print(f"  t = {t:3d}  dual value {dual_value(t * ray, Qr):.1f}")
    r9 = solve_dual(Qr, yr)
    print(f"solver: status {r9.status}  sum alpha {r9.x.sum():.1e}")

    # Step 7: a ninth machine on an edge.
    Xn, yn = np.vstack([X, [2, 5]]), np.append(y, 1.0)
    v, c0 = np.array([1.0, 2.0]), -11.0
    ends = alpha_range(v, c0, Xn, yn)
    edge = np.abs(yn * (Xn @ v + c0) - 1) < TOL
    for i in np.where(edge)[0]:
        print(f"machine {i + 1}: alpha in [{ends[i, 0]:.3f}, "
              f"{ends[i, 1]:.3f}]  {kind(*ends[i], edge[i])}")
    Qn = dual_matrix(Xn, yn)
    for a0 in (np.zeros(9), np.ones(9), np.arange(9.0)):
        print("start", a0[:2], " alpha 4, 5, 7, 9:",
              solve_dual(Qn, yn, a0).x[[3, 4, 6, 8]])

    # Step 8: an idle point on an edge.
    Xi = np.array([[4, 3], [4, 4], [6, 4.0]])
    yi = np.array([-1, 1, 1.0])
    v, c0, _ = hard_margin(Xi, yi)
    print("idle case: w", v.round(6), " b", round(c0, 6))
    ends = alpha_range(v, c0, Xi, yi)
    for i in range(3):
        print(f"point {i + 1}: alpha in [{ends[i, 0]:.3f}, "
              f"{ends[i, 1] + 0.0:.3f}]  {kind(*ends[i], True)}")

    # Step 9: removing one machine from the eight or the nine.
    for name, Xd, yd in (("eight", X, y), ("nine", Xn, yn)):
        for i in (3, 4, 6, 8):
            if i < len(yd):
                keep = np.arange(len(yd)) != i
                v, c0, _ = hard_margin(Xd[keep], yd[keep])
                print(f"{name}, without {i + 1}: w", (v + 0).round(3),
                      " width", round(2 / np.linalg.norm(v), 3))

    # Step 10: only inner products matter.
    rng = np.random.default_rng(SEED)
    R = np.linalg.qr(rng.normal(size=(1000, 1000)))[0][:, :2]
    Xb = X @ R.T                           # 8 points in 1000 dimensions
    Gdiff = np.abs(Xb @ Xb.T - X @ X.T).max()
    print("features", Xb.shape[1], " largest change in G",
          round(Gdiff, 9) + 0)
    ab = solve_dual(dual_matrix(Xb, y)).x
    print("alpha", ab)
    wb, bb, _ = recover(ab, Xb, y)
    z = np.array([5.0, 4.0])
    score = (ab * y) @ (X @ z) + bb
    print("variables: primal", Xb.shape[1] + 1, " dual", len(y))
    print("score of a new machine", round(score, 6),
          " from w", round((z @ R.T) @ wb + bb, 6))
