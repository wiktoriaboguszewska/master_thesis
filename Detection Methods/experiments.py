import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sqlite3
import os
from sklearn.model_selection import train_test_split
from creating_shift_train_test import sample_fixed_class_prior
from BBSD_KS import BBSD_KS
from BBSD_PT import BBSD_PT
from sklearn import datasets
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_dataset(dataset_name):
    if dataset_name == "cancer":
        data = datasets.load_breast_cancer()
        return data.data, data.target
    elif dataset_name == "credit":
        data = fetch_openml(data_id=31, as_frame=True)
        X = pd.get_dummies(data.data)
        y = np.where(data.target == 'good', 1, 0)
        return X.values, y
    elif dataset_name == "blood":
        data = fetch_ucirepo(id=176)
        X = data.data.features
        y = data.data.targets.values.ravel()
        return X.values, y
    elif dataset_name == "bank":
        data = fetch_openml(data_id=1558, as_frame=True)
        X = pd.get_dummies(data.data)
        y = np.where(data.target == '2', 1, 0)
        return X.values, y
    elif dataset_name == "phoneme":
        data = fetch_openml(data_id=1489, as_frame=False)
        X = data.data
        y = np.where(data.target == '2', 1, 0)
        return X, y
    elif dataset_name == "banknote":
        data = fetch_openml(data_id=1462, as_frame=False)
        X = data.data
        y = np.where(data.target == '2', 1, 0)
        return X, y
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

def table_exists(conn, table_name):
    """Check if a table already exists in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    result = cursor.fetchone()
    return result is not None

def compare_classifiers_on_shift(dataset_name="cancer", method="BBSD_KS", alpha=0.05, n_trials=50, db_file="pvalues.db"):
    X, y = load_dataset(dataset_name)
    classifiers = ["logreg", "rf", "svc", "gb", "dnn"]
    pi_train = 0.1
    s_values = np.linspace(0, 0.8, 9)

    # Creating or connecting to database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for pi_test in pi_train + s_values:
        s_value_str = str(round(pi_test - pi_train, 1)).replace('.', '')  # e.g., 0.1 -> '01'

        for clf in classifiers:
            table_name = f"{dataset_name.lower()}_{method.lower()}_{clf.lower()}_{s_value_str}"

            if table_exists(conn, table_name):
                print(f"Skipping existing table: {table_name}")
                continue  # Skip already done

            print(f"Calculating for table: {table_name}")

            p_values = []

            for trial in range(n_trials):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=trial)
                X_train_s, y_train_s, X_test_s, y_test_s = sample_fixed_class_prior(
                    X_train, y_train, X_test, y_test, p1_train=pi_train, p1_test=pi_test, random_state=trial
                )

                if method == "BBSD_KS":
                    pval = BBSD_KS(X_train_s, y_train_s, X_test_s, clf_name=clf, alpha=alpha)
                elif method == "BBSD_PT":
                    pval = BBSD_PT(X_train_s, y_train_s, X_test_s, clf_name=clf, alpha=alpha)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                p_values.append((pval,))

            # Creating table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    p_value REAL
                )
            """)
            # Inserting all p-values
            cursor.executemany(f"INSERT INTO {table_name} (p_value) VALUES (?)", p_values)
            conn.commit()
            print(f"Saved {len(p_values)} p-values to table {table_name}")

    conn.close()

# === Running for all combinations ===
datasets_list = ["cancer", "credit", "blood", "bank", "phoneme", "banknote"]
methods = ["BBSD_KS", "BBSD_PT"]

db_filename = "pvalues.db"
if not os.path.exists(db_filename):
    # Creating new empty file if missing
    open(db_filename, 'a').close()

for dataset in datasets_list:
    for method in methods:
        compare_classifiers_on_shift(dataset_name=dataset, method=method, alpha=0.05, n_trials=100, db_file=db_filename)
