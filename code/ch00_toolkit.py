"""Chapter 0: the toolkit on the eight machines."""
import numpy as np

from machines import X, y

np.set_printoptions(precision=3, suppress=True)

d = np.array([1.0, 2.0])
print("lengths :", np.linalg.norm(X, axis=1).round(3))
print("x4'x5   :", X[3] @ X[4])
cos = X[3] @ X[6] / (np.linalg.norm(X[3]) * np.linalg.norm(X[6]))
print("cos(x4,x7):", round(cos, 3))
print("score   :", X @ d - 11)
print("distance:", ((X @ d - 11) / np.linalg.norm(d)).round(3))

G = X @ X.T
print("G[:4,:4] =\n", G[:4, :4].astype(int))
lam = np.linalg.eigvalsh(G)[::-1]
print("two largest eigenvalues:", lam[:2].round(3))
print("rank of G:", np.linalg.matrix_rank(G))
v = y
print("y'Gy =", v @ G @ v, " |X'y|^2 =", (X.T @ v) @ (X.T @ v))
print("X'y =", X.T @ v)
