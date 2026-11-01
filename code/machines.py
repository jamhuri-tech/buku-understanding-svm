"""The eight machines used throughout the book.

Columns: vibration (mm/s), temperature rise (tens of degrees C).
y = +1 needs service, y = -1 healthy.
"""
import numpy as np

X = np.array([[3, 2], [1, 1], [1, 3], [4, 3],
              [4, 4], [4, 5], [6, 3], [5, 7]], dtype=float)
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)

# Hard-margin solution (Fractions in code/search_mini.py).
w = np.array([1.0, 2.0])
b = -11.0
alpha = np.array([0, 0, 0, 2.5, 2.0, 0, 0.5, 0])

if __name__ == "__main__":
    assert np.allclose((alpha * y) @ X, w)
    assert abs(alpha @ y) < 1e-12
    margins = y * (X @ w + b)
    assert margins.min() == 1 and sum(margins == 1) == 3
    print("margins", margins, "width", 2 / np.linalg.norm(w))
