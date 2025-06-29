import sqlite3
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from creating_shift_train_test import sample_fixed_class_prior
from MLLS import mlls
from TC import tc

# Loading all datasets
def load_all():
    can = load_breast_cancer()
    X_cancer, y_cancer = can.data, can.target

    cr = fetch_openml(data_id=31, as_frame=True)
    X_credit = pd.get_dummies(cr.data)
    y_credit = np.where(cr.target == 'good', 1, 0)

    bd = fetch_ucirepo(id=176)
    X_blood = bd.data.features
    y_blood = bd.data.targets.values.ravel()

    bk = fetch_openml(data_id=1558, as_frame=True)
    X_bank = pd.get_dummies(bk.data)
    y_bank = np.where(bk.target == '2', 1, 0)

    ph = fetch_openml(data_id=1489, as_frame=False)
    X_phoneme = ph.data
    y_phoneme = np.where(ph.target == '2', 1, 0)

    bn = fetch_openml(data_id=1462, as_frame=False)
    X_banknote, y_banknote = bn.data, np.where(bn.target == '2', 1, 0)

    return {
        'cancer': (X_cancer, y_cancer),
        'credit': (X_credit, y_credit),
        'blood': (X_blood, y_blood),
        'bank': (X_bank, y_bank),
        'phoneme': (X_phoneme, y_phoneme),
        'banknote': (X_banknote, y_banknote)
    }

# Configuration
DB = "estimators.db"
DATASETS = load_all()
CLASSIFIERS = ["logreg", "rf"]
METHODS = ["EM", "TC"]
P1_TRAIN = 0.1
P1_TESTS = [0.2, 0.5, 0.9]
SHIFT_SUFFIX = {0.2: "01", 0.5: "04", 0.9: "08"}
N_TRIALS = 50

# Connecting to database
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Runner functions
def estimate_mlls(X_tr, y_tr, X_te, clf_name):
    _, q1 = mlls(X_tr, y_tr, X_te, clf_name=clf_name, epochs=200, q_init=0.5, tol=1e-4)
    return q1

def estimate_tc(X_tr, y_tr, X_te, clf_name):
    _, q1, _ = tc(X_tr, y_tr, X_te, clf_name=clf_name)
    return q1

estimators = {
    "EM": estimate_mlls,
    "TC": estimate_tc
}

# Main loop
for dataset_name, (X, y) in DATASETS.items():
    for method in METHODS:
        for clf_name in CLASSIFIERS:
            estimator = estimators[method]
            for p1_test in P1_TESTS:
                s_val = SHIFT_SUFFIX[p1_test]
                table_name = f"{dataset_name}_{method}_{clf_name}_{s_val}"

                # Creating table
                cur.execute(f"DROP TABLE IF EXISTS '{table_name}'")
                cur.execute(f"""
                    CREATE TABLE '{table_name}' (
                        trial INTEGER,
                        pi_est REAL
                    )
                """)

                # Running trials
                for trial in range(N_TRIALS):
                    X0_tr, X0_te, y0_tr, y0_te = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=trial
                    )
                    X_tr, y_tr, X_te, y_te = sample_fixed_class_prior(
                        X0_tr, y0_tr, X0_te, y0_te,
                        p1_train=P1_TRAIN,
                        p1_test=p1_test,
                        random_state=trial
                    )
                    pi_est = estimator(X_tr, y_tr, X_te, clf_name)
                    cur.execute(f"INSERT INTO '{table_name}' (trial, pi_est) VALUES (?, ?)", (trial, pi_est))
                conn.commit()

conn.close()
