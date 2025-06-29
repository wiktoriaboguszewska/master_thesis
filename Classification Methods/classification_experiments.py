import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from creating_shift_train_test import sample_fixed_class_prior

from BBSC import bbsc
from MLLS import mlls
from TC import tc

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1) Loading all six datasets

def load_all():
    # (A) Cancer
    can = load_breast_cancer()
    X_cancer, y_cancer = can.data, can.target

    # (B) Credit (OpenML 31)
    cr = fetch_openml(data_id=31, as_frame=True)
    X_credit = pd.get_dummies(cr.data)
    y_credit = np.where(cr.target == 'good', 1, 0)

    # (C) Blood (UCI 176)
    bd = fetch_ucirepo(id=176)
    X_blood = bd.data.features
    y_blood = bd.data.targets.values.ravel()

    # (D) Bank (OpenML 1558)
    bk = fetch_openml(data_id=1558, as_frame=True)
    X_bank = pd.get_dummies(bk.data)
    y_bank = np.where(bk.target == '2', 1, 0)

    # (E) Phoneme (OpenML 1489)
    ph = fetch_openml(data_id=1489, as_frame=False)
    X_phoneme = ph.data
    y_phoneme = np.where(ph.target == '2', 1, 0)

    # (F) Banknote (OpenML 1462)
    bn = fetch_openml(data_id=1462, as_frame=False)
    X_banknote, y_banknote = bn.data, np.where(bn.target=='2',1,0)

    return {
        "cancer":   (X_cancer,   y_cancer),
        "credit":   (X_credit,   y_credit),
        "blood":    (X_blood,    y_blood),
        "bank":     (X_bank,     y_bank),
        "phoneme":  (X_phoneme,  y_phoneme),
        "banknote": (X_banknote,y_banknote),
    }

# 2) Defining runners for each method & classifier

def get_classifier(clf_name):
    if clf_name == 'logreg':
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='lbfgs', max_iter=5000)
        )
    elif clf_name == 'rf':
        base = RandomForestClassifier(n_estimators=100, random_state=0)
    else:
        raise ValueError("Unknown clf_name")
    return CalibratedClassifierCV(base, method='isotonic', cv=5)


def run_naive(X_tr, y_tr, X_te, clf_name):
    clf = get_classifier(clf_name)
    clf.fit(X_tr, y_tr)
    return (clf.predict_proba(X_te)[:,1] >= 0.5).astype(int)


def run_bbsc(X_tr, y_tr, X_te, clf_name):
    q_x = bbsc(X_tr, y_tr, X_te, clf_name=clf_name)
    return (q_x[:,1] >= 0.5).astype(int)


def run_mlls(X_tr, y_tr, X_te, clf_name):
    q_x, _ = mlls(X_tr, y_tr, X_te, clf_name=clf_name, epochs=200, q_init=0.5, tol=1e-4)
    return (q_x[:,1] >= 0.5).astype(int)


def run_tc(X_tr, y_tr, X_te, clf_name):
    q_x, _, _ = tc(X_tr, y_tr, X_te, clf_name=clf_name)
    return (q_x[:,1] >= 0.5).astype(int)

if __name__ == "__main__":
    np.random.seed(0)

    all_data = load_all()
    p1_train, p1_test = 0.5, 0.1
    n_trials = 20
    methods = [
        ("Naive", run_naive),
        ("BBSC", run_bbsc),
        ("EM", run_mlls),
        ("Threshold", run_tc)
    ]
    classifiers = ['logreg', 'rf']

    for name, (X, y) in all_data.items():
        rows = []
        for clf_name in classifiers:
            for method_label, runner in methods:
                accs = []
                for i in range(n_trials):
                    X0_tr, X0_te, y0_tr, y0_te = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=i
                    )
                    X_tr, y_tr, X_te, y_te = sample_fixed_class_prior(
                        X0_tr, y0_tr, X0_te, y0_te,
                        p1_train=p1_train,
                        p1_test=p1_test,
                        random_state=i
                    )
                    y_pred = runner(X_tr, y_tr, X_te, clf_name)
                    accs.append(np.mean(y_pred == y_te))
                avg_acc = np.mean(accs)
                rows.append((f"{method_label}_{clf_name}", avg_acc))

        df_avg = pd.DataFrame(rows, columns=["Method", "AvgAccuracy"])
        out = f"{name}_results.txt"
        df_avg.to_csv(out, sep="	", index=False)
        print(f"Saved average accuracies for {name} to {out}")
