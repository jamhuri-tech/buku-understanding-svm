"""Search for the eight-machine data set: small integer readings whose
hard-margin SVM has three support vectors and simple exact values."""
from fractions import Fraction as F
from itertools import product

import numpy as np

SEED = 20261101
rng = np.random.default_rng(SEED)


def solve(sv, X, y):
    """Exact KKT system for support vectors sv (alpha_i, b)."""
    k = len(sv)
    A = [[F(int(y[i] * y[j] * (X[i] @ X[j]))) for j in sv] + [F(int(y[i]))] for i in sv]
    A.append([F(int(y[i])) for i in sv] + [F(0)])
    r = [F(1)] * k + [F(0)]
    n = k + 1
    for c in range(n):
        p = next((q for q in range(c, n) if A[q][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        r[c], r[p] = r[p], r[c]
        for q in range(n):
            if q != c and A[q][c] != 0:
                f = A[q][c] / A[c][c]
                A[q] = [u - f * v for u, v in zip(A[q], A[c])]
                r[q] -= f * r[c]
    return [r[i] / A[i][i] for i in range(n)]


found = []
for _ in range(1500):
    neg = rng.integers(1, 6, (4, 2))
    pos = rng.integers(3, 9, (4, 2))
    X = np.vstack([neg, pos])
    if len({tuple(p) for p in X}) < 8:
        continue
    y = np.array([-1] * 4 + [1] * 4)
    for a, b, c in product(range(4), range(4), range(4, 8)):
        for sv in ([a, b, c] if a < b else None, [a, 4 + b, c] if 4 + b < c else None):
            if sv is None:
                continue
            s = solve(sv, X, y)
            if s is None or any(v <= 0 for v in s[:3]):
                continue
            w = [sum(s[k] * int(y[i]) * int(X[i][d]) for k, i in enumerate(sv)) for d in (0, 1)]
            bb = s[3]
            mar = [int(y[i]) * (w[0] * int(X[i][0]) + w[1] * int(X[i][1]) + bb) for i in range(8)]
            oth = [mar[i] for i in range(8) if i not in sv]
            if min(oth) <= 1 or w[0] <= 0 or w[1] <= 0 or w[0] == w[1]:
                continue
            ds = [v.denominator for v in s[:3] + w + [bb]]
            if max(ds) <= 4 and sum(w) <= 3 and min(oth) >= F(3, 2):
                found.append((max(ds), X.tolist(), [i + 1 for i in sv], s, w))
found.sort(key=lambda t: (t[0], sum(v.numerator for v in t[4])))
for f in found[:6]:
    print(f[0], f[1], f[2], [str(v) for v in f[3]], [str(v) for v in f[4]])
print(len(found))
