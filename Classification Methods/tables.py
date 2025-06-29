import sqlite3
import pandas as pd
import numpy as np

# --- configuration ------------------------------------------------------
DB = "accuracies.db"

DATASETS   = ["cancer", "credit", "blood", "bank", "phoneme", "banknote"]
CLASSIFIERS = ["logreg", "rf"]
METHODS     = ["EM", "BBSC", "Threshold", "Naive"]

# the four scenarios to tabulate
SCENARIOS = [
    (0.1, 0.9),
    (0.1, 0.5),
]

# --- helper to fetch the mean accuracy from a table ----------------------
def fetch_mean(cur, dataset, method, clf, p1_train, p1_test):
    tbl = f"{dataset}_{method}_{clf}_t{p1_train}_s{p1_test}"
    cur.execute(f"SELECT AVG(accuracy) FROM '{tbl}'")
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else np.nan

# --- main ---------------------------------------------------------------
def main():
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()

    for p1_train, p1_test in SCENARIOS:
        rows = []
        for dataset in DATASETS:
            for clf in CLASSIFIERS:
                # fetching mean accuracy for each method
                vals = {
                    method: fetch_mean(cur, dataset, method, clf, p1_train, p1_test)
                    for method in METHODS
                }
                rows.append({
                    "Dataset":      dataset,
                    "Classifier":   clf,
                    "EM":           vals["EM"],
                    "BBSC":         vals["BBSC"],
                    "Threshold_calibration": vals["Threshold"],
                    "Naive":        vals["Naive"],
                })

        df = pd.DataFrame(rows, columns=[
            "Dataset",
            "Classifier",
            "EM",
            "BBSC",
            "Threshold_calibration",
            "Naive"
        ])

        # output filename like accuracy_03_07.csv
        fn = f"accuracy_{int(p1_train*10):02d}_{int(p1_test*10):02d}.csv"
        df.to_csv(fn, index=False)
        print(f"Saved {fn}")

    conn.close()

if __name__ == "__main__":
    main()
