"""Chapter 2: the geometry of a separating line on the eight machines."""
import numpy as np
from scipy.optimize import linprog
from sklearn.svm import SVC

from machines import X, y

np.set_printoptions(precision=3, suppress=True)


def foot(x, w, b):
    """Closest point of the line w'x + b = 0 to x."""
    return x - (x @ w + b) / (w @ w) * w


def canonical(d):
    """Centre and scale direction d; return w, b, width."""
    s = X @ d
    hi, lo = s[y > 0].min(), s[y < 0].max()
    assert hi > lo
    k = 2 / (hi - lo)
    return k * d, -k * (hi + lo) / 2, (hi - lo) / np.linalg.norm(d)


def margins(w, b):
    """Smallest functional and geometric margins over the data."""
    m = (y * (X @ w + b)).min()
    return m, m / np.linalg.norm(w)


def separable(X, y):
    """Is there any w, b with y_i (w'x_i + b) >= 1 for all i?"""
    A = -y[:, None] * np.column_stack([X, np.ones(len(y))])
    r = linprog(np.zeros(A.shape[1]), A_ub=A,
                b_ub=-np.ones(len(y)), bounds=(None, None))
    return r.status == 0


def point_segment(p, a, c):
    """Closest point to p on the segment from a to c."""
    d = c - a
    t = np.clip((p - a) @ d / (d @ d), 0, 1)
    return a + t * d


def hull_distance(P, Q):
    """Shortest corner-to-edge distance between two point sets (2-D)."""
    best = np.inf
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            for i in range(len(B)):
                for j in range(i):
                    best = min(best, np.linalg.norm(
                        p - point_segment(p, B[i], B[j])))
    return best


w, b = np.array([1.0, 2.0]), -11.0

# Step 1: scores as heights, and the feet of three machines.
print("scores  :", X @ w + b)
for i in (3, 4, 6):
    print("foot of", i + 1, ":", foot(X[i], w, b))

# Step 2: the three knobs.
print("offset  :", round(-b / np.linalg.norm(w), 3))
for k in (1, 3, 0.5):
    print("k =", k, " margins", np.round(margins(k * w, k * b), 3))

# Step 3: direction, centre, scale.
for d in ([1, 1], [1, 3], [3, 1], [1, 2]):
    v, c, width = canonical(np.array(d, dtype=float))
    print("dir", d, "w", v, "b", round(c, 3), "width", round(width, 3))

# Step 4: three tests of separability.
Xr = np.vstack([X, [2, 2]])
yr = np.append(y, 1.0)
Xx = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
yx = np.array([-1, -1, 1, 1], dtype=float)
for name, A, c in (("machines", X, y), ("rogue", Xr, yr),
                   ("xor", Xx, yx)):
    print(f"{name:8s} separable: {separable(A, c)}")

# Step 5: the XOR labels become separable with one new feature.
Xl = np.column_stack([Xx, Xx[:, 0] * Xx[:, 1]])
print("lifted   separable:", separable(Xl, yx))

# Step 6: widest street against the distance between the hulls.
delta = hull_distance(X[y > 0], X[y < 0])
clf = SVC(kernel="linear", C=1e6).fit(X, y)
print("hulls   :", round(delta, 3))
print("street  :", round(2 / np.linalg.norm(clf.coef_), 3))

# Step 7: new units, for all machines and for machines 4 and 8.
pair = [3, 7]
for name, A, c in (("all ", X, y), ("pair", X[pair], y[pair])):
    for s in ([1, 1], [1, 10], [10, 1]):
        clf = SVC(kernel="linear", C=1e8).fit(A * s, c)
        v = clf.coef_.ravel() * s      # back in the original units
        print(name, s, "x1 +", round(v[1] / v[0], 3), "x2 =",
              round(-clf.intercept_[0] / v[0], 3),
              " width", round(2 / np.linalg.norm(clf.coef_), 3))
