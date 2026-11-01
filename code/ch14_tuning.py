"""Chapter 14: Scaling, tuning, and validation."""
import time
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC

from ch07_soft import X9, y9
from ch11_rbf import noisy
from ch13_linear import dcd, rows_of, sparse_data, with_offset
from machines import X, y

np.set_printoptions(precision=3, suppress=True)
SEED = 20261101
CS = 10.0 ** np.arange(-2, 5)
GAMMAS = 10.0 ** np.arange(-5, 2)
HARD = 1e6


def cancer():
    """The Wisconsin data of scikit-learn; +1 is malignant."""
    d = load_breast_cancer()
    return d.data, 2.0 * (d.target == 0) - 1, d.feature_names


def folds(k=5, seed=SEED):
    return StratifiedKFold(k, shuffle=True, random_state=seed)


def rbf_pipe(**kw):
    return make_pipeline(StandardScaler(), SVC(**kw))


def grid_search(A, t, cv, scoring="accuracy"):
    """Search C and gamma with the scaler inside every fold."""
    grid = {"svc__C": CS, "svc__gamma": GAMMAS}
    return GridSearchCV(rbf_pipe(), grid, cv=cv, scoring=scoring).fit(A, t)


def surface_of(gs):
    """Mean score, shape (gamma, C)."""
    return gs.cv_results_["mean_test_score"].reshape(len(CS), -1).T


def imbalanced(m, share=0.05, seed=SEED):
    """Two overlapping clouds; a share of the points is positive."""
    rng = np.random.default_rng(seed)
    k = int(round(m * share))
    A = rng.standard_normal((m, 2))
    A[:k] += [2.0, 1.5]
    return A, np.r_[np.ones(k), -np.ones(m - k)]


def loo_line(i, A=X, t=y):
    """Hard-margin line without point i, and the score of point i."""
    keep = np.arange(len(t)) != i
    s = SVC(C=HARD, kernel="linear", tol=1e-8).fit(A[keep], t[keep])
    w, b = s.coef_[0], s.intercept_[0]
    return w, b, t[i] * (A[i] @ w + b)


def standardise_by_hand():
    print("mean", X.mean(0), "sd", X.std(0))
    Z = (X - X.mean(0)) / X.std(0)
    D = X * [1, 10]
    Y = (D - D.mean(0)) / D.std(0)
    print("same after a change of unit:", np.allclose(Z, Y))
    p, q, r = X[3], X[4], X[6]
    for name, u in (("raw", np.array([1.0, 1.0])),
                    ("deg", np.array([1.0, 10.0])),
                    ("std", 1 / X.std(0))):
        d45 = np.sum((u * (p - q)) ** 2)
        d47 = np.sum((u * (p - r)) ** 2)
        print(f"{name}  d45 {d45:6.2f}  d47 {d47:5.2f}  "
              f"K45 {np.exp(-d45 / 2):.2e}  K47 {np.exp(-d47 / 2):.3f}")


def lines_by_hand():
    """The hard margin in three sets of units, read back in raw units."""
    for name, u in (("raw", np.ones(2)), ("deg", np.array([1.0, 10.0])),
                    ("std", 1 / X.std(0))):
        c = X.mean(0) * u if name == "std" else np.zeros(2)
        s = SVC(C=HARD, kernel="linear", tol=1e-9).fit(X * u - c, y)
        w = s.coef_[0]
        back = w * u
        k = 1 / back[0]
        print(f"{name}: sv {s.support_ + 1}  width "
              f"{2 / np.linalg.norm(w):.3f}  line x1 + {back[1] * k:.3f} x2"
              f" = {(c @ w - s.intercept_[0]) * k:.3f}")


def units_cancer():
    A, t, names = cancer()
    v = A.var(0)
    top = np.argsort(v)[::-1][:2]
    print(f"sd from {A.std(0).min():.4f} to {A.std(0).max():.1f}")
    print("share of variance:", [f"{names[i]} {v[i] / v.sum():.3f}"
                                 for i in top])
    print("rule: predict the larger class", round((t < 0).mean(), 3))
    cv = folds()
    for name, m in (
            ("raw gamma=1      ", SVC(gamma=1.0)),
            ("raw gamma=scale  ", SVC()),
            ("std gamma=scale  ", rbf_pipe())):
        s = cross_val_score(m, A, t, cv=cv)
        print(f"{name} {s.mean():.3f}")


def scalers():
    A, t, _ = cancer()
    Xtr, Xte, ttr, tte = train_test_split(
        A, t, test_size=0.25, stratify=t, random_state=SEED)
    for s in (StandardScaler(), MinMaxScaler(), RobustScaler(),
              MaxAbsScaler()):
        q = cross_val_score(make_pipeline(s, SVC()), Xtr, ttr, cv=folds())
        print(f"{type(s).__name__:15s} {q.mean():.3f}")


def scale_rule():
    """Standardised data: the typical squared distance is 2n."""
    A, t, _ = cancer()
    Z = StandardScaler().fit_transform(A)
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, len(Z), (2, 20000))
    d = np.sum((Z[i] - Z[j]) ** 2, axis=1)
    k = i != j
    print(f"mean squared distance {d[k].mean():.1f}, 2n = {2 * Z.shape[1]}")
    print(f"gamma scale {1 / (Z.shape[1] * Z.var()):.4f}")


def leak_scaler():
    """Fit the scaler on all the data, or on each training fold."""
    A, t, _ = cancer()
    gaps = []
    for r in range(20):
        cv = folds(seed=SEED + r)
        good = cross_val_score(rbf_pipe(), A, t, cv=cv).mean()
        Z = StandardScaler().fit_transform(A)
        bad = cross_val_score(SVC(), Z, t, cv=cv).mean()
        gaps.append((bad, good))
    g = np.array(gaps)
    print(f"cancer: leak {g[:, 0].mean():.4f}  clean {g[:, 1].mean():.4f}"
          f"  gap {np.mean(g[:, 0] - g[:, 1]):+.4f}")
    # A heavy tail in one feature and few training points.
    rng = np.random.default_rng(SEED)
    bad, good = [], []
    for r in range(300):
        m = 40
        tt = np.r_[np.ones(m // 2), -np.ones(m // 2)]
        B = rng.standard_normal((m, 2))
        B[:, 0] += tt
        B[:, 1] = rng.standard_cauchy(m)
        te = np.arange(m) % 4 == 0
        tr = ~te
        p = StandardScaler().fit(B[tr])
        good.append(SVC(gamma=1.0).fit(p.transform(B[tr]), tt[tr])
                    .score(p.transform(B[te]), tt[te]))
        Z = StandardScaler().fit_transform(B)
        bad.append(SVC(gamma=1.0).fit(Z[tr], tt[tr]).score(Z[te], tt[te]))
    print(f"tail: leak {np.mean(bad):.3f}  clean {np.mean(good):.3f}"
          f"  gap {np.mean(bad) - np.mean(good):+.3f}")


def leak_selection():
    """Choose features on all the data, then validate: pure noise."""
    rng = np.random.default_rng(SEED)
    A = rng.standard_normal((50, 2000))
    t = np.r_[np.ones(25), -np.ones(25)]
    rng.shuffle(t)
    pick = SelectKBest(f_classif, k=20)
    Z = pick.fit_transform(A, t)
    m = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    print(f"noise, picked before folds {cross_val_score(m, Z, t, cv=folds()).mean():.3f}")
    m = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20),
                      SVC(kernel="linear"))
    print(f"noise, picked inside folds {cross_val_score(m, A, t, cv=folds()).mean():.3f}")


def loo_hand():
    errors = 0
    for i in range(8):
        w, b, s = loo_line(i)
        errors += s <= 1e-6
        print(f"without {i + 1}: w {w}  b {b:7.3f}  yf {s:6.2f}")
    print("LOO errors", errors, "of 8")
    s = cross_val_score(SVC(C=HARD, kernel="linear"), X, y, cv=LeaveOneOut())
    print("cross_val_score", s)


def loo_errors(C, A=X9, t=y9):
    m = len(t)
    s = np.zeros(m)
    for i in range(m):
        k = np.arange(m) != i
        g = SVC(C=C, kernel="linear", tol=1e-8).fit(A[k], t[k])
        s[i] = t[i] * g.decision_function(A[[i]])[0]
    return int(np.sum(s <= 1e-6))


def loo_by_C():
    """Nine machines: errors are constant on stretches of C."""
    for C in (0.01, 0.1, 0.5, 1, 10, 100):
        g = SVC(C=C, kernel="linear", tol=1e-8).fit(X9, y9)
        print(f"C {C:6.2f}  sv {g.n_support_.sum()}  "
              f"LOO errors {loo_errors(C)}")
    grid = 10.0 ** np.linspace(-2, 2, 81)
    e = [loo_errors(C) for C in grid]
    runs = 1 + sum(p != q for p, q in zip(e, e[1:]))
    print(f"81 values of C, {runs} stretches of equal errors")


def fold_noise():
    A, t, _ = cancer()
    Xtr, Xte, ttr, tte = train_test_split(
        A, t, test_size=0.25, stratify=t, random_state=SEED)
    s = cross_val_score(rbf_pipe(), Xtr, ttr, cv=folds())
    print("folds", s, "mean", round(s.mean(), 3), "sd", round(s.std(ddof=1), 3))
    r = [cross_val_score(rbf_pipe(), Xtr, ttr, cv=folds(seed=SEED + i)).mean()
         for i in range(50)]
    print(f"50 shuffles: mean {np.mean(r):.3f} sd {np.std(r, ddof=1):.4f}"
          f" range {min(r):.3f} {max(r):.3f}")


def grid_run():
    A, t, _ = cancer()
    Xtr, Xte, ttr, tte = train_test_split(
        A, t, test_size=0.25, stratify=t, random_state=SEED)
    print("train", len(ttr), "test", len(tte))
    gs = grid_search(Xtr, ttr, folds())
    b = gs.best_params_
    print(f"best C {b['svc__C']:g} gamma {b['svc__gamma']:g} "
          f"cv {gs.best_score_:.3f}")
    print(f"test {gs.score(Xte, tte):.3f}")
    i = gs.best_index_
    sd = gs.cv_results_["std_test_score"][i] * np.sqrt(5 / 4)
    se = sd / np.sqrt(5)
    S = surface_of(gs)
    print(f"cells within one standard error ({se:.3f}): "
          f"{int(np.sum(S >= gs.best_score_ - se))} of {S.size}")
    return gs


def cost_of_C():
    """Steps of the solver over a small grid: the large C pays."""
    A, t = noisy(1000)
    Z = StandardScaler().fit_transform(A)
    C = 10.0 ** np.arange(-1, 4)
    S = np.array([[SVC(C=c, gamma=g).fit(Z, t).n_iter_[0] for c in C]
                  for g in (0.1, 1.0, 10.0)])
    print("steps by C", S.sum(0))
    print(f"share of the largest C {S[:, -1].sum() / S.sum():.2f}")


def warm():
    """Coordinate descent over a path in C, cold and warm."""
    A, t = sparse_data(2000)
    Z = with_offset(A)
    r = rows_of(Z)
    n = Z.shape[1]
    for step in (0.5, 0.1):
        cold = warm_ = 0
        a = None
        for C in 10.0 ** np.arange(-2.0, 0.01, step):
            cold += dcd(r, t, n, C, tol=1e-2)[2]
            _, _, ep, a = warm_dcd(r, t, n, C, a)
            warm_ += ep
        print(f"ratio {10 ** step:.2f}: cold {cold} passes, "
              f"warm {warm_} passes")


def warm_dcd(rows, t, n, C, start, tol=1e-2, epochs=1000):
    """dcd of Chapter 13 from a given alpha, clipped into the new box."""
    m = len(t)
    a = np.zeros(m) if start is None else np.minimum(start, C)
    w = np.zeros(n)
    Q = np.zeros(m)
    for i, (idx, val) in enumerate(rows):
        w[idx] += a[i] * t[i] * val
        Q[i] = val @ val
    rng = np.random.default_rng(SEED)
    for e in range(1, epochs + 1):
        hi, lo = -np.inf, np.inf
        for i in rng.permutation(m):
            idx, val = rows[i]
            G = t[i] * (w[idx] @ val) - 1
            P = G
            if a[i] <= 0:
                P = min(G, 0)
            elif a[i] >= C:
                P = max(G, 0)
            hi, lo = max(hi, P), min(lo, P)
            if P != 0:
                new = min(max(a[i] - G / Q[i], 0), C)
                w[idx] += (new - a[i]) * t[i] * val
                a[i] = new
        if hi - lo <= tol:
            break
    return w, None, e, a


def linear_tuning():
    """C for a linear model: one axis only."""
    A, t, _ = cancer()
    Xtr, Xte, ttr, tte = train_test_split(
        A, t, test_size=0.25, stratify=t, random_state=SEED)
    p = make_pipeline(StandardScaler(),
                      LinearSVC(loss="hinge", max_iter=200000))
    gs = GridSearchCV(p, {"linearsvc__C": 10.0 ** np.arange(-4, 3)},
                      cv=folds()).fit(Xtr, ttr)
    print("linear: C", gs.best_params_["linearsvc__C"],
          f"cv {gs.best_score_:.3f}  test {gs.score(Xte, tte):.3f}")


def kernels_too():
    """The kernel is one more entry of the grid."""
    A, t, _ = cancer()
    Xtr, Xte, ttr, tte = train_test_split(
        A, t, test_size=0.25, stratify=t, random_state=SEED)
    grid = [
        {"svc__kernel": ["linear"], "svc__C": CS[:5]},
        {"svc__kernel": ["poly"], "svc__degree": [2, 3], "svc__coef0": [1.0],
         "svc__C": CS[:5], "svc__gamma": GAMMAS[1:6]},
        {"svc__kernel": ["rbf"], "svc__C": CS, "svc__gamma": GAMMAS}]
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                      cv=folds()).fit(Xtr, ttr)
    q = gs.best_params_
    print(len(gs.cv_results_["params"]), "candidates; best",
          q["svc__kernel"], f"cv {gs.best_score_:.3f}")
    for k in ("linear", "poly", "rbf"):
        c = gs.cv_results_["param_svc__kernel"] == k
        print(f"  {k:6s} {gs.cv_results_['mean_test_score'][c].max():.3f}")


def nested_noise(reps=40):
    """Random labels: every true accuracy is one half."""
    rng = np.random.default_rng(SEED)
    grid = {"svc__C": 10.0 ** np.arange(-1, 4),
            "svc__gamma": 10.0 ** np.arange(-4, 1)}
    best, outer = [], []
    for r in range(reps):
        A = rng.standard_normal((60, 10))
        t = np.r_[np.ones(30), -np.ones(30)]
        rng.shuffle(t)
        gs = GridSearchCV(rbf_pipe(), grid, cv=folds(5, SEED + r))
        best.append(gs.fit(A, t).best_score_)
        outer.append(cross_val_score(gs, A, t, cv=folds(5, SEED + 1000 + r))
                     .mean())
    best, outer = np.array(best), np.array(outer)
    print(f"noise, {reps} data sets: best cell {best.mean():.3f}  "
          f"nested {outer.mean():.3f}")
    return best, outer


def nested_cancer():
    A, t, _ = cancer()
    t0 = time.time()
    gs = GridSearchCV(rbf_pipe(), {"svc__C": CS, "svc__gamma": GAMMAS},
                      cv=folds(5, SEED + 1))
    s = cross_val_score(gs, A, t, cv=folds())
    print("outer folds", s)
    print(f"nested {s.mean():.3f}; plain best cell "
          f"{gs.fit(A, t).best_score_:.3f}")
    print("fits", 5 * (5 * len(CS) * len(GAMMAS) + 1))


def cases(tte, s, cut=0.0):
    """Counts and scores for one threshold on the score."""
    p = s > cut
    tp, fp = np.sum(p & (tte > 0)), np.sum(p & (tte < 0))
    fn, tn = np.sum(~p & (tte > 0)), np.sum(~p & (tte < 0))
    return tp, fp, fn, tn


def metrics_hand():
    """A cruder score: the temperature feature alone."""
    f = X[:, 1] - 3.5
    print("scores", f)
    for cut in (0.0, -1.0):
        tp, fp, fn, tn = cases(y, f, cut)
        print(f"cut {cut:4.1f}  tp {tp} fp {fp} fn {fn} tn {tn}  "
              f"prec {tp / (tp + fp):.3f}  rec {tp / (tp + fn):.3f}  "
              f"F1 {2 * tp / (2 * tp + fp + fn):.3f}")
    print(f"AUC {roc_auc_score(y, f):.4f}")


def report(name, tte, s):
    p = np.where(s > 0, 1.0, -1.0)
    print(f"{name:10s} acc {np.mean(p == tte):.3f}  "
          f"prec {precision_score(tte, p, zero_division=0):.3f}  "
          f"rec {recall_score(tte, p):.3f}  "
          f"F1 {f1_score(tte, p, zero_division=0):.3f}  "
          f"AUC {roc_auc_score(tte, s):.3f}")


def lin_pipe(**kw):
    return rbf_pipe(kernel="linear", C=1.0, **kw)


def imbalance_run():
    A, t = imbalanced(2000)
    Ate, tte = imbalanced(20000, seed=SEED + 1)
    print("train positives", int((t > 0).sum()), "of", len(t))
    report("always -1", tte, -np.ones(len(tte)))
    plain = lin_pipe().fit(A, t)
    s = plain.decision_function(Ate)
    report("plain", tte, s)
    w = lin_pipe(class_weight="balanced").fit(A, t)
    report("balanced", tte, w.decision_function(Ate))
    # The threshold is chosen on held-out scores of the training set.
    h = cross_val_predict(lin_pipe(), A, t, cv=folds(),
                          method="decision_function")
    cuts = np.sort(h)
    best = max(cuts, key=lambda c: f1_score(t, np.where(h > c, 1.0, -1.0)))
    print(f"cut from folds {best:.3f}")
    report("moved cut", tte, s - best)
    print("balanced weights", w[-1].class_weight_)
    print("cos of the two directions", round(float(
        plain[-1].coef_[0] @ w[-1].coef_[0]
        / np.linalg.norm(plain[-1].coef_) / np.linalg.norm(w[-1].coef_)), 4))


def tune_imbalance():
    A, t = imbalanced(2000)
    Ate, tte = imbalanced(20000, seed=SEED + 1)
    grid = {"svc__C": 10.0 ** np.arange(-2, 3),
            "svc__gamma": 10.0 ** np.arange(-2, 2),
            "svc__class_weight": [None, "balanced"]}
    for sc in ("accuracy", "f1", "roc_auc"):
        gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                          cv=folds(), scoring=sc).fit(A, t)
        q = gs.best_params_
        s = gs.decision_function(Ate)
        p = np.where(s > 0, 1.0, -1.0)
        print(f"{sc:9s} C {q['svc__C']:g} gamma {q['svc__gamma']:g} "
              f"{q['svc__class_weight']}  F1 "
              f"{f1_score(tte, p, zero_division=0):.3f} AUC "
              f"{roc_auc_score(tte, s):.3f}")


def weights_hand():
    """Six machines at small C: balance ties the two sums together."""
    u, v = X[:6], y[:6]
    for w in (None, "balanced"):
        g = SVC(C=0.01, kernel="linear", tol=1e-8, class_weight=w).fit(u, v)
        a = np.zeros(6)
        a[g.support_] = np.abs(g.dual_coef_[0])
        print(f"weight {w}: alpha {np.round(a, 4)}")
        print(f"  w {g.coef_[0].round(4)}  b {g.intercept_[0]:.4f}"
              f"  predicted +1: {int(np.sum(g.predict(u) > 0))}")


if __name__ == "__main__":
    warnings.simplefilter("ignore", ConvergenceWarning)
    standardise_by_hand()
    lines_by_hand()
    units_cancer()
    scale_rule()
    scalers()
    leak_scaler()
    leak_selection()
    loo_hand()
    loo_by_C()
    fold_noise()
    grid_run()
    cost_of_C()
    warm()
    linear_tuning()
    kernels_too()
    nested_noise()
    nested_cancer()
    metrics_hand()
    imbalance_run()
    tune_imbalance()
    weights_hand()
