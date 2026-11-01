"""Chapter 15: Several classes and probabilities."""
import warnings
from fractions import Fraction as Fr
from itertools import combinations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
HARD = 1e6
GAMMA = 0.02
COST = 1.0
WIDE = 0.2      # the over-fitted machine of the Platt experiment
WIDE_C = 10.0

# Eleven machines: 1 healthy, 2 worn, 3 loose.
X3 = np.vstack([X, [[8, 1], [9, 1], [10, 2]]]).astype(float)
c3 = np.r_[(y > 0) + 1, [3, 3, 3]]


def pairs(k):
    return list(combinations(range(1, k + 1), 2))


def ovo_scores(x):
    """Canonical lines for each pair; positive means the larger class."""
    a, b = x[..., 0], x[..., 1]
    return {(1, 2): a + 2 * b - 11,
            (1, 3): (2 * a - b - 10) / 5,
            (2, 3): (a - b - 5) / 2}


def ovr_scores(x):
    """Canonical one-versus-rest scores; positive claims the class."""
    a, b = x[..., 0], x[..., 1]
    return np.stack([(26 - 3 * a - 4 * b) / 2, a + 6 * b - 23,
                     (a - b - 5) / 2], axis=-1)


def votes(x):
    """Votes for classes 1, 2, 3 at the points x."""
    v = np.zeros(np.shape(x)[:-1] + (3,), dtype=int)
    for (i, j), f in ovo_scores(np.asarray(x, float)).items():
        v[..., i - 1] += f < 0
        v[..., j - 1] += f >= 0
    return v


def margin_sum(x):
    """Scores for each class: sum of signed pair scores."""
    s = np.zeros(np.shape(x)[:-1] + (3,))
    for (i, j), f in ovo_scores(np.asarray(x, float)).items():
        s[..., i - 1] -= f
        s[..., j - 1] += f
    return s


def show_hand():
    """Fit each pair and each rest problem and compare with the hand."""
    print("eleven machines, pairs")
    for i, j in pairs(3):
        keep = (c3 == i) | (c3 == j)
        t = np.where(c3[keep] == j, 1, -1)
        s = SVC(kernel="linear", C=HARD).fit(X3[keep], t)
        print(f"{i} v {j}: w {s.coef_[0]}  b {s.intercept_[0]:.3f}  "
              f"sv {X3[keep][s.support_].astype(int).tolist()}")
    print("eleven machines, rest")
    for k in (1, 2, 3):
        t = np.where(c3 == k, 1, -1)
        s = SVC(kernel="linear", C=HARD).fit(X3, t)
        print(f"{k} v rest: w {s.coef_[0]}  b {s.intercept_[0]:.3f}  "
              f"n {np.linalg.norm(s.coef_[0]):.3f}")


def show_votes():
    print("point        f12    f13    f23  votes   sums")
    for p in ((5, 2), (7, 5), (9, 3), (6, 1.5)):
        q = np.array(p, float)
        f = ovo_scores(q)
        print(f"{str(p):11s}{f[1, 2]:6.2f} {f[1, 3]:6.2f} {f[2, 3]:6.2f}"
              f"  {votes(q)}  {margin_sum(q)}")
    s = SVC(kernel="linear", C=HARD).fit(X3, c3)
    q = np.array([[6, 1.5]])
    print("SVC", s.predict(q), "break_ties",
          SVC(kernel="linear", C=HARD, break_ties=True).fit(
              X3, c3).predict(q))
    swap = 4 - c3
    print("classes renamed: SVC", SVC(kernel="linear", C=HARD).fit(
        X3, swap).predict(q), "break_ties", SVC(
        kernel="linear", C=HARD, break_ties=True).fit(X3, swap).predict(q))


def show_rest():
    print("point        f1     f2     f3   claims  distances")
    N = np.array([2.5, np.sqrt(37), np.sqrt(2) / 2])
    for p in ((5, 2), (6, 2.5), (6, 0), (3, 7)):
        f = ovr_scores(np.array(p, float))
        print(f"{str(p):11s}{f[0]:6.2f} {f[1]:6.2f} {f[2]:6.2f}"
              f"  {(f > 0).astype(int)}  {f / N}")


def areas(lo=0.0, hi=12.0, n=1200):
    """Share of a square with no clear answer, for the two schemes."""
    t = (np.arange(n) + 0.5) / n * (hi - lo) + lo
    G = np.stack(np.meshgrid(t, t), axis=-1)
    tri = (votes(G) == 1).all(axis=-1)
    f = ovr_scores(G)
    claims = (f > 0).sum(axis=-1)
    return tri.mean(), (claims == 0).mean(), (claims > 1).mean()


def digits(keep=10):
    d = load_digits()
    use = d.target < keep
    return d.data[use], d.target[use]


def steps_ovo(A, t):
    s = make_pipeline(StandardScaler(), SVC(gamma=GAMMA, C=COST))
    s.fit(A, t)
    m = s[-1]
    return int(np.sum(m.n_iter_)), len(m.n_iter_), int(len(m.support_))


def steps_ovr(A, t):
    s = make_pipeline(StandardScaler(),
                      OneVsRestClassifier(SVC(gamma=GAMMA, C=COST)))
    s.fit(A, t)
    it = [int(np.sum(e.n_iter_)) for e in s[-1].estimators_]
    sv = sorted(set(i for e in s[-1].estimators_ for i in e.support_))
    return sum(it), len(it), len(sv)


def weighted_steps(k):
    """Steps times points per problem, ten thousands of kernel rows."""
    A, t = digits(k)
    m = len(t)
    s = make_pipeline(StandardScaler(), SVC(gamma=GAMMA, C=COST)).fit(A, t)
    pw = 0
    start = np.r_[0, np.cumsum([np.sum(t == c) for c in range(k)])]
    for n, (i, j) in zip(s[-1].n_iter_, combinations(range(k), 2)):
        pw += n * (start[i + 1] - start[i] + start[j + 1] - start[j])
    r = make_pipeline(StandardScaler(),
                      OneVsRestClassifier(SVC(gamma=GAMMA, C=COST))).fit(A, t)
    rw = sum(int(np.sum(e.n_iter_)) * m for e in r[-1].estimators_)
    return m, pw, rw


def show_cost():
    print(" k     m  ovo probs steps   entries  "
          "ovr probs steps   entries")
    for k in (3, 5, 7, 10):
        m, pw, rw = weighted_steps(k)
        A, t = digits(k)
        so, no, _ = steps_ovo(A, t)
        sr, nr, _ = steps_ovr(A, t)
        print(f"{k:2d} {m:5d} {no:9d} {so:5d} {pw:9.2e} "
              f"{nr:9d} {sr:5d} {rw:9.2e}")
    p = 1 + np.log((sr / nr) / (so / no)) / np.log(m / (2 * m / k))
    print(f"steps grow as size^{p - 1:.2f}, predicted ratio "
          f"{k ** p / ((k - 1) * 2 ** (p - 1)):.2f}, seen {rw / pw:.2f}")


def models():
    lin = dict(max_iter=20000, random_state=SEED)
    return {
        "ovo": make_pipeline(StandardScaler(), SVC(gamma=GAMMA, C=COST)),
        "ovr": make_pipeline(StandardScaler(), OneVsRestClassifier(
            SVC(gamma=GAMMA, C=COST))),
        "linear ovr": make_pipeline(StandardScaler(),
                                    LinearSVC(C=0.01, dual=True, **lin)),
        "crammer-singer": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.01, multi_class="crammer_singer", dual=True,
                      **lin)),
        "code 2k": make_pipeline(StandardScaler(), OutputCodeClassifier(
            SVC(gamma=GAMMA, C=COST), code_size=2.0, random_state=SEED)),
    }


def show_accuracy():
    A, t = digits()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, p in models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            s = cross_val_score(p, A, t, cv=cv)
        print(f"{name:15s}{s.mean():.3f}")


def crammer_eleven():
    """Weight vectors of Crammer and Singer on the eleven machines."""
    Z = np.c_[X3 - X3.mean(axis=0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        s = LinearSVC(C=10.0, multi_class="crammer_singer", dual=True,
                      fit_intercept=False, max_iter=200000, tol=1e-6,
                      random_state=SEED).fit(Z, c3)
    return s, X3.mean(axis=0)


def show_crammer():
    s, c = crammer_eleven()
    W = s.coef_
    print("w1", W[0], "w2", W[1], "w3", W[2])
    print("sum", W.sum(axis=0))
    print("right", int(np.sum(s.predict(X3 - c) == c3)), "of", len(c3))


def code_distance(M):
    """Smallest Hamming distance between rows of a code matrix."""
    return min(int(np.sum(a != b)) for a, b in combinations(M, 2))


CODE = np.array([[+1, +1, +1, +1, +1, +1, +1],
                 [-1, -1, -1, -1, +1, +1, +1],
                 [-1, -1, +1, +1, -1, -1, +1],
                 [-1, +1, -1, +1, -1, +1, -1]])


def decode(bits, M=CODE):
    d = np.sum(M != bits, axis=1)
    return d, int(np.argmin(d)) + 1


def show_code():
    print("distance", code_distance(CODE))
    bits = CODE[2].copy()
    bits[0] = -bits[0]
    d, k = decode(bits)
    print("received", bits, "distances", d, "class", k)


def platt_fit(f, t, smooth=True, steps=100, tol=1e-8):
    """Fit P(y = +1 | f) = 1 / (1 + exp(A f + B)) by Newton's method."""
    f = np.asarray(f, float)
    pos = t > 0
    n1, n0 = pos.sum(), (~pos).sum()
    hi, lo = ((n1 + 1) / (n1 + 2), 1 / (n0 + 2)) if smooth else (1.0, 0.0)
    g = np.where(pos, hi, lo)
    A, B = 0.0, float(np.log((n0 + 1) / (n1 + 1)))

    def nll(A, B):
        u = A * f + B
        return np.sum(g * np.logaddexp(0, u)
                      + (1 - g) * np.logaddexp(0, -u))

    v = nll(A, B)
    for k in range(steps):
        p = np.exp(-np.logaddexp(0, A * f + B))
        r = g - p
        G = np.array([f @ r, r.sum()])
        if np.max(np.abs(G)) < tol:
            break
        w = p * (1 - p) + 1e-12
        H = np.array([[w @ (f * f), w @ f], [w @ f, w.sum()]])
        d = -np.linalg.solve(H, G)
        s = 1.0
        while s > 1e-10:
            new = nll(A + s * d[0], B + s * d[1])
            if new <= v + 1e-4 * s * (G @ d):
                break
            s /= 2
        A, B, v = A + s * d[0], B + s * d[1], new
    return A, B, k


def sigmoid(f, A, B):
    return np.exp(-np.logaddexp(0, A * f + B))


def brier(p, t):
    return float(np.mean((p - (t > 0)) ** 2))


def bins(p, t, n=10):
    """Mean forecast, observed share, count per equal-width bin."""
    g = np.minimum((p * n).astype(int), n - 1)
    return [(k, p[g == k].mean(), np.mean(t[g == k] > 0), np.sum(g == k))
            for k in range(n) if np.any(g == k)]


def murphy(p, t, n=10):
    """Reliability, resolution, and uncertainty over bins."""
    m, base = len(t), np.mean(t > 0)
    rel = sum(c * (q - r) ** 2 for _, q, r, c in bins(p, t, n)) / m
    res = sum(c * (r - base) ** 2 for _, q, r, c in bins(p, t, n)) / m
    return rel, res, base * (1 - base)


def ece(p, t, n=10):
    return sum(c * abs(q - r) for _, q, r, c in bins(p, t, n)) / len(t)


def even_odd():
    """Digits, +1 for an odd digit, split in half."""
    d = load_digits()
    t = 2 * (d.target % 2) - 1
    return train_test_split(d.data, t, test_size=0.5, stratify=t,
                            random_state=SEED)


def machine(gamma, c=COST):
    return make_pipeline(StandardScaler(), SVC(gamma=gamma, C=c))


def fold_scores(gamma, A, t, c=COST):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(machine(gamma, c), A, t, cv=cv,
                             method="decision_function")


def study(gamma=WIDE, c=WIDE_C):
    """Every quantity of the Platt experiment for one machine."""
    A, Ate, t, tte = even_odd()
    s = machine(gamma, c).fit(A, t)
    ftr, fte = s.decision_function(A), s.decision_function(Ate)
    fcv = fold_scores(gamma, A, t, c)
    return dict(t=t, tte=tte, ftr=ftr, fte=fte, fcv=fcv,
                b=float(s[-1].intercept_[0]),
                train=platt_fit(ftr, t), held=platt_fit(fcv, t))


def naive_forecast(f):
    """Clip f to [-1, 1] and map to [0, 1]: a score, not a probability."""
    return (np.clip(f, -1, 1) + 1) / 2


def show_platt():
    S = study()
    t, tte, ftr, fte, fcv = (S[k] for k in ("t", "tte", "ftr", "fte", "fcv"))
    b = S["b"]
    print(f"offset b {b:.3f}  median test score {np.median(fte):.3f}  "
          f"cut -B/A {-S['held'][1] / S['held'][0]:.3f}")
    print(f"train m {len(t)}  test m {len(tte)}  "
          f"share odd {np.mean(tte > 0):.3f}")
    print(f"accuracy train {np.mean(np.sign(ftr) == t):.3f}  "
          f"test {np.mean(np.sign(fte) == tte):.3f}")
    print(f"share |f| > 1 train {np.mean(np.abs(ftr) > 1):.3f}  "
          f"folds {np.mean(np.abs(fcv) > 1):.3f}  "
          f"test {np.mean(np.abs(fte) > 1):.3f}")
    for name, q in (("training", S["train"]), ("folds", S["held"])):
        print(f"{name:9s}A {q[0]:7.3f}  B {q[1]:6.3f}  "
              f"Newton steps {q[2]}")
    for name, q in (("training", S["train"]), ("folds", S["held"])):
        p = sigmoid(fte, *q[:2])
        print(f"{name:9s}accuracy of p > 1/2 {np.mean((p > .5) == (tte > 0)):.3f}")
    print("forecast          Brier  reliab  resol  uncert    ece")
    rows = (("clipped score", naive_forecast(fte)),
            ("sigmoid, train", sigmoid(fte, *S["train"][:2])),
            ("sigmoid, folds", sigmoid(fte, *S["held"][:2])),
            ("base rate", np.full(len(tte), np.mean(t > 0))))
    for name, p in rows:
        r = murphy(p, tte)
        print(f"{name:17s}{brier(p, tte):.4f}  {r[0]:.4f} {r[1]:.4f} "
              f"{r[2]:.4f}  {ece(p, tte):.3f}")


def show_gamma():
    print("gamma    acc     A(train)  A(folds)  Brier(tr) Brier(fo)")
    for g in (0.002, 0.02, 0.1, 0.2):
        S = study(g)
        acc = np.mean(np.sign(S["fte"]) == S["tte"])
        ptr = sigmoid(S["fte"], *S["train"][:2])
        pcv = sigmoid(S["fte"], *S["held"][:2])
        print(f"{g:5.3f}  {acc:.3f}  {S['train'][0]:9.3f} "
              f"{S['held'][0]:9.3f}  {brier(ptr, S['tte']):.4f}    "
              f"{brier(pcv, S['tte']):.4f}")


def show_svc():
    """SVC with probability=True, and the disagreement with predict."""
    A, Ate, t, tte = even_odd()
    p = make_pipeline(StandardScaler(),
                      SVC(gamma=WIDE, C=WIDE_C, probability=True,
                          random_state=SEED)).fit(A, t)
    pr = p.predict_proba(Ate)[:, 1]
    print(f"SVC probA {p[-1].probA_[0]:.3f}  probB {p[-1].probB_[0]:.3f}  "
          f"Brier {brier(pr, tte):.4f}")
    d = np.sum((pr > .5) != (p.predict(Ate) > 0))
    print(f"sign and p > 1/2 differ on {d} of {len(tte)}")


def couple(r):
    """Pairwise coupling, second method of Wu, Lin, and Weng."""
    k = len(r)
    Q = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                Q[i, i] += r[j, i] ** 2
                Q[i, j] = -r[j, i] * r[i, j]
    S = np.block([[Q, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
    return np.linalg.solve(S, np.r_[np.zeros(k), 1])[:k]


def show_couple():
    p = np.array([0.5, 0.3, 0.2])
    r = p[:, None] / (p[:, None] + p[None, :])
    np.fill_diagonal(r, 0)
    print("consistent", couple(r))
    c = np.array([[0, .6, .4], [.4, 0, .6], [.6, .4, 0]])
    print("cycle", couple(c))
    c = np.array([[0, .8, .3], [.2, 0, .6], [.7, .4, 0]])
    print("mixed", couple(c))


def show_exact():
    """Exact fit of Worked Example: q right at f = +1 and at f = -1."""
    q = Fr(3, 4)
    print("A", -np.log(float(q) / (1 - float(q))))
    f = np.array([1., 1, 1, 1, -1, -1, -1, -1])
    t = np.array([1, 1, 1, -1, -1, -1, -1, 1])
    print("Newton", platt_fit(f, t, smooth=False)[:2])
    print("smoothed", platt_fit(f, t)[:2])


if __name__ == "__main__":
    show_hand()
    show_votes()
    show_rest()
    print("areas", np.round(areas(), 4))
    show_cost()
    show_accuracy()
    show_crammer()
    show_code()
    show_exact()
    show_platt()
    show_gamma()
    show_svc()
    show_couple()
