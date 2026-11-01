"""Chapter 19: Case study on the Wine data (UCI, through scikit-learn).

`python3 ch19_case.py closed` stops before the test set is opened.
"""
import sys
import warnings
from collections import Counter

import numpy as np
from scipy.stats import beta
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.set_printoptions(precision=3, suppress=True, linewidth=64)
SEED = 20261101
TEST = 54
FOLDS = 5
REPEATS = 4
CS = 10.0 ** np.arange(-2, 4)
GS = 10.0 ** np.arange(-4, 2)
PICK = 0.01      # C of the linear machine, from linear_search
FLOOR = 0.9      # least probability for an automatic label
SHORT = ["alcohol", "malic acid", "ash", "ash alkalinity", "magnesium",
         "total phenols", "flavanoids", "nonflav. phenols",
         "proanthocyanins", "colour", "hue", "OD280/OD315", "proline"]


def load():
    """Lock the test set first: everything else uses train only."""
    d = load_wine()
    tr, te = train_test_split(np.arange(len(d.target)), test_size=TEST,
                              stratify=d.target, random_state=SEED)
    return d.data, d.target, tr, te


def folds(seed=SEED):
    return StratifiedKFold(FOLDS, shuffle=True, random_state=seed)


def one_se(res):
    """Smallest settings within one standard error of the best cell."""
    m = res["mean_test_score"]
    s = res["std_test_score"] / np.sqrt(FOLDS)
    top = int(np.argmax(m))
    keys = sorted(k for k in res if k.startswith("param_"))
    ok = [i for i in range(len(m)) if m[i] >= m[top] - s[top]]
    return min(ok, key=lambda i: tuple(float(res[k][i]) for k in keys))


def gaussian():
    return make_pipeline(StandardScaler(), SVC())


def searches():
    """The candidates: each one is a whole selection procedure."""
    def search(model, grid):
        return lambda seed: GridSearchCV(model, grid, cv=folds(seed),
                                         refit=one_se)
    C = {"svc__C": CS}
    G = {"svc__C": CS, "svc__gamma": GS}
    L = {"linearsvc__C": CS}
    lin = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    rest = make_pipeline(StandardScaler(), LinearSVC(
        dual=True, max_iter=20000, random_state=SEED))
    cs = make_pipeline(StandardScaler(), LinearSVC(
        multi_class="crammer_singer", max_iter=20000, random_state=SEED))
    ovr = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC()))
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000))
    return {
        "Gaussian ovo": search(gaussian(), G),
        "Gaussian ovr": search(ovr, {"onevsrestclassifier__estimator__C": CS,
                                     "onevsrestclassifier__estimator__gamma":
                                     GS}),
        "linear ovo": search(lin, C),
        "linear ovr": search(rest, L),
        "Crammer-Singer": search(cs, L),
        "logistic": search(logit, {"logisticregression__C": CS}),
        "forest": lambda seed: RandomForestClassifier(300, random_state=SEED),
    }


def nested(make, A, c):
    """Repeated outer folds; the inner search sees training folds only."""
    acc, picks = [], Counter()
    for r in range(REPEATS):
        ok = 0
        for a, b in folds(SEED + 1 + r).split(A, c):
            m = make(SEED + 101 + r).fit(A[a], c[a])
            ok += np.sum(m.predict(A[b]) == c[b])
            if hasattr(m, "best_params_"):
                picks[tuple(float(v) for k, v in
                            sorted(m.best_params_.items()))] += 1
        acc.append(ok / len(c))
    return np.array(acc), picks


def grid_search(A, c):
    g = GridSearchCV(gaussian(), {"svc__C": CS, "svc__gamma": GS},
                     cv=folds(), refit=one_se).fit(A, c)
    return g, g.cv_results_["mean_test_score"].reshape(len(CS), len(GS))


def linear_search(A, c, cs=CS):
    g = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="linear")),
                     {"svc__C": cs}, cv=folds(), refit=one_se).fit(A, c)
    return g.best_params_["svc__C"], g.cv_results_["mean_test_score"]


def chosen(probability=False):
    """The model fixed by the protocol of Section 19.5."""
    return make_pipeline(StandardScaler(), SVC(
        kernel="linear", C=PICK, probability=probability,
        random_state=SEED))


def linear_fit(A, c):
    p = chosen().fit(A, c)
    return p, p[-1].coef_, p[-1].intercept_


def stability(A, c, draws=300):
    """Share of bootstrap fits that agree with the sign of each weight."""
    _, w, _ = linear_fit(A, c)
    rng = np.random.default_rng(SEED)
    same = np.zeros(w.shape)
    for _ in range(draws):
        i = rng.integers(0, len(c), len(c))
        if len(np.unique(c[i])) < 3:
            continue
        same += np.sign(linear_fit(A[i], c[i])[1]) == np.sign(w)
    return same / draws


def probabilities(A, c):
    return cross_val_predict(chosen(True), A, c, cv=folds(),
                             method="predict_proba")


def brier(p, c):
    return np.mean(np.sum((p - np.eye(3)[c]) ** 2, axis=1))


def plane(A):
    """Two principal components of the training part, for display."""
    s = StandardScaler().fit(A)
    return s, PCA(2).fit(s.transform(A))


def clopper(k, m, d=0.05):
    """Exact two-sided interval for the error rate."""
    lo = 0.0 if k == 0 else beta.ppf(d / 2, k, m - k + 1)
    hi = 1.0 if k == m else beta.ppf(1 - d / 2, k + 1, m - k)
    return lo, hi


def understand(X, y, A, c):
    print("wines", len(y), "features", X.shape[1], "classes",
          np.bincount(y))
    print("train", np.bincount(c), "constant accuracy",
          round(np.bincount(c).max() / len(c), 3))
    print("smallest sd", round(A.std(0).min(), 3), " proline sd",
          round(A.std(0)[12], 1))
    print("proline share of total variance",
          round(A.var(0)[12] / A.var(0).sum(), 4))
    r = np.corrcoef(A.T)
    print("corr flavanoids, total phenols", round(r[6, 5], 3))
    print("corr flavanoids, OD280/OD315", round(r[6, 11], 3))
    print("largest other |corr|", round(np.sort(np.abs(
        r[np.triu_indices(13, 1)]))[-3], 3))


def hand(A):
    """Two wines: where the squared distance comes from."""
    a, b = A[0], A[1]
    z = StandardScaler().fit(A).transform(A[:2])
    d, e = (a - b) ** 2, (z[0] - z[1]) ** 2
    g = 1 / (13 * A.var())
    print("wine a", a[[0, 6, 12]], " wine b", b[[0, 6, 12]])
    print("raw d2", round(d.sum(), 1), " proline", round(d[12], 1),
          " share", round(d[12] / d.sum(), 5))
    print("std d2", round(e.sum(), 3), " proline", round(e[12], 3),
          " share", round(e[12] / e.sum(), 3))
    print("gamma scale", f"{g:.3e}", " K raw", round(np.exp(-g * d.sum()), 3),
          " K std", round(np.exp(-e.sum() / 13), 3))


def first_runs(X, y, A, c):
    show = lambda m: round(cross_val_score(m, A, c, cv=folds()).mean(), 3)
    print("constant        ", show(DummyClassifier(
        strategy="most_frequent")))
    print("raw Gaussian    ", show(SVC()))
    print("raw linear      ", show(SVC(kernel="linear")))
    print("scaled Gaussian ", show(gaussian()))
    print("scaled linear   ", show(make_pipeline(
        StandardScaler(), SVC(kernel="linear"))))
    print("sorted, no shuffle", cross_val_score(gaussian(), X, y,
                                                cv=KFold(3)))


def compare(A, c):
    res = {}
    for name, make in searches().items():
        acc, picks = nested(make, A, c)
        res[name] = acc
        print(f"{name:15s} {acc.mean():.3f}  low {acc.min():.3f}"
              f"  high {acc.max():.3f}  {len(picks)} settings")
        if name == "Gaussian ovo":
            for k, v in picks.most_common(3):
                print("   C, gamma", k, "chosen", v, "of",
                      REPEATS * FOLDS)
    return res


def read_model(A, c):
    g = make_pipeline(StandardScaler(), SVC(C=10.0, gamma=0.1)).fit(A, c)
    print("Gaussian C 10, gamma 0.1: support vectors",
          g[-1].n_support_.sum())
    p, w, b = linear_fit(A, c)
    s = p[-1]
    print("linear: support vectors", s.n_support_, "total",
          s.n_support_.sum(), "share",
          round(s.n_support_.sum() / len(c), 3))
    d = cross_val_score(chosen(), A, c, cv=LeaveOneOut())
    print("leave-one-out errors", int(len(c) - d.sum()), "of", len(c))
    print("at the cap", np.sum(np.isclose(np.abs(s.dual_coef_),
                                          PICK).any(axis=0)))
    for i, name in enumerate(("0 v 1", "0 v 2", "1 v 2")):
        top = np.argsort(-np.abs(w[i]))[:3]
        print(name, "width", round(2 / np.linalg.norm(w[i]), 3), "b",
              round(b[i], 3), [SHORT[j] for j in top])
    print("w 0 v 2")
    print(w[1])
    z = p[0].transform(A[:1])[0]
    print("z wine a")
    print(z)
    print("b", round(b[1], 3), " f", round(w[1] @ z + b[1], 3),
          " class of wine a", c[0])
    q = stability(A, c)
    print("least sign share, top three of each pair", np.round(
        [q[i, np.argsort(-np.abs(w[i]))[:3]].min() for i in range(3)], 3))
    print("flavanoids, phenols sign kept, 0 v 2", q[1, [6, 5]])
    keep = np.delete(np.arange(13), 6)
    print("w phenols, 0 v 2: all", round(w[1, 5], 3), " no flavanoids",
          round(linear_fit(A[:, keep], c)[1][1, 5], 3))
    print("accuracy without flavanoids", round(cross_val_score(
        chosen(), A[:, keep], c, cv=folds()).mean(), 3))


def calibrate(A, c):
    p = probabilities(A, c)
    top, pick = p.max(1), p.argmax(1)
    keep = top >= FLOOR
    print("out-of-fold Brier", round(brier(p, c), 4), " base rate",
          round(brier(np.tile(np.bincount(c) / len(c), (len(c), 1)), c), 4))
    print("largest p: median", round(np.median(top), 3), " least",
          round(top.min(), 3))
    print(f"p >= {FLOOR}: kept {keep.sum()} of {len(c)}, right",
          np.sum(pick[keep] == c[keep]), " sent back", np.sum(~keep))
    print("wrong among sent back", np.sum(pick[~keep] != c[~keep]))


def final(A, c, B, e):
    h = chosen().fit(A, c).predict(B)
    k = int(np.sum(h != e))
    print("test wines", len(e), " errors", k, " accuracy",
          round(1 - k / len(e), 4))
    print("macro F1", round(f1_score(e, h, average="macro"), 4))
    print(confusion_matrix(e, h))
    print("exact 95% for error", np.round(clopper(k, len(e)), 4))
    print("Hoeffding upper limit, delta 0.05",
          round(k / len(e) + np.sqrt(np.log(20) / (2 * len(e))), 4))
    q = chosen(True).fit(A, c)
    p = q.predict_proba(B)
    keep = p.max(1) >= FLOOR
    print("predict and largest p differ on", np.sum(p.argmax(1) != h))
    print("test Brier", round(brier(p, e), 4))
    print(f"p >= {FLOOR}: kept {keep.sum()} of {len(e)}, right",
          np.sum(p.argmax(1)[keep] == e[keep]), " sent back",
          np.sum(~keep))
    t = chosen().fit(A, c)
    d = t[-1].coef_ @ t[0].transform(B).T + t[-1].intercept_[:, None]
    for i in np.flatnonzero(h != e):
        print("error: true", e[i], "said", h[i], "p", p[i])
        print("pair scores 0v1 0v2 1v2", d[:, i])
    v = np.array([(d[0] > 0) * 1 + (d[1] > 0) * 1, (d[0] <= 0) * 1
                  + (d[2] > 0) * 1, (d[1] <= 0) * 1 + (d[2] <= 0) * 1])
    print("votes from the pair signs agree with predict:",
          np.all(v.argmax(0) == h))


if __name__ == "__main__":
    X, y, tr, te = load()
    A, c = X[tr], y[tr]
    print("-- data");  understand(X, y, A, c)
    print("-- two wines");  hand(A)
    print("-- first runs");  first_runs(X, y, A, c)
    g, M = grid_search(A, c)
    print("-- grid: rows C, columns gamma");  print(M)
    print("best", round(M.max(), 3), "one-SE pick", g.best_params_)
    print("linear one-SE C", *linear_search(A, c))
    wide = 10.0 ** np.arange(-5, -1)
    print("smaller C: pick", *linear_search(A, c, wide))
    print("-- nested");  compare(A, c)
    print("-- reading");  read_model(A, c)
    print("-- probabilities");  calibrate(A, c)
    if sys.argv[1:] != ["closed"]:
        print("-- test set, opened once")
        final(A, c, X[te], y[te])
        print("final model on all wines: support vectors",
              chosen().fit(X, y)[-1].n_support_.sum())
