"""Chapter 1: one hard-margin SVM by hand, confirmed with scikit-learn."""
import numpy as np
from sklearn.svm import SVC

from machines import X, y

np.set_printoptions(precision=3, suppress=True)


def street_width(w, b):
    """Width of the street of a separating line, centred or not."""
    s = X @ w + b
    assert np.all(y * s > 0)
    return (s[y > 0].min() - s[y < 0].max()) / np.linalg.norm(w)


# Step 1: four separating lines and their streets.
lines = {"A": ([1, 1], -7.5), "B": ([1, 3], -14),
         "C": ([2, 1], -11.5), "D": ([1, 2], -11)}
for name, (w, b) in lines.items():
    print("line", name, "width", round(street_width(np.array(w), b), 3))

# Step 2: the shortest bridge between the two hulls.
d = X[6] - X[4]
t = (X[3] - X[4]) @ d / (d @ d)
p = X[4] + t * d
bridge = p - X[3]
print("t =", t.round(3), " p =", p, " bridge =", bridge)
print("bridge length:", np.linalg.norm(bridge).round(3))
print("midpoint:", (p + X[3]) / 2)

# Step 3: the three active constraints give w and b.
sv = [3, 4, 6]
A = np.column_stack([X[sv], np.ones(3)])
w1, w2, b = np.linalg.solve(A, y[sv])
w = np.array([w1, w2])
print("w =", w, " b =", b.round(3))

# Step 4: stationarity and balance give the multipliers.
Z = (X[sv] * y[sv, None]).T
M = np.vstack([Z, y[sv]])
a = np.linalg.solve(M, [w[0], w[1], 0])
print("alpha =", a)

# Step 5: every constraint.
print("margins y*f:", y * (X @ w + b))
print("width:", (2 / np.linalg.norm(w)).round(3))

# Step 6: two new machines, in two ways.
new = np.array([[2, 2], [5, 3.2]])
print("f(new), primal:", new @ w + b)
print("f(new), dual  :", new @ X[sv].T @ (a * y[sv]) + b)
print("distance:", ((new @ w + b) / np.linalg.norm(w)).round(3))

# Step 7: scikit-learn, with a huge C for a hard margin.
clf = SVC(kernel="linear", C=1e6).fit(X, y)
print("support :", clf.support_ + 1)
print("alpha*y :", clf.dual_coef_.ravel().round(3))
print("w       :", clf.coef_.ravel().round(3))
print("b       :", clf.intercept_.round(3))
print("predict :", clf.predict(new))
