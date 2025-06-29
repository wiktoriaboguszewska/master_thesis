import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# --- configuration ------------------------------------------------------
DB = "accuracies.db"
DATASETS = ["cancer", "credit", "blood", "bank", "phoneme", "banknote"]
CLASSIFIERS = ["logreg", "rf"]
METHODS = ["Naive", "BBSC", "EM", "Threshold"]

# Mapping each internal method‐name to the desired legend label:
DISPLAY_NAME = {
    "Naive":     "Naive",
    "BBSC":      "BBSC",
    "EM":        "MLLS",
    "Threshold": "TC"
}

# fixed train prior
P1_TRAIN = 0.1
# test priors and computed shifts
P1_TESTS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
SHIFTS = [p - P1_TRAIN for p in P1_TESTS]

# --- helper to fetch mean accuracy from DB --------------------------------
def fetch_mean_accuracy(cur, dataset, method, clf, p1_train, p1_test):
    table = f"{dataset}_{method}_{clf}_t{p1_train}_s{p1_test}"
    cur.execute(f"SELECT AVG(accuracy) FROM '{table}'")
    result = cur.fetchone()
    return result[0] if result is not None else np.nan

# --- main plotting loop -------------------------------------------------
def main():
    conn = sqlite3.connect(DB)
    cur  = conn.cursor()

    for dataset in DATASETS:
        for clf in CLASSIFIERS:
            plt.figure(figsize=(9, 5))
            for method in METHODS:
                means = []
                for p1_test in P1_TESTS:
                    mean_acc = fetch_mean_accuracy(
                        cur, dataset, method, clf,
                        P1_TRAIN, p1_test
                    )
                    means.append(mean_acc)
                # choose line width: default 1, EM=2, BBSC=3
                lw = 1
                if method == "EM":
                    lw = 2
                elif method == "BBSC":
                    lw = 3
                plt.plot(
                    SHIFTS, means, marker='o', label=DISPLAY_NAME[method], linewidth=lw
                )

            plt.xlabel("Shift magnitude  $s=\pi'-\pi$", fontsize=18)
            plt.ylabel("Mean Accuracy", fontsize=18)
            if dataset == "bank":
                plt.ylim(0.4, 1.0)
            elif dataset == "banknote":
                plt.ylim(0.8, 1.0)
            elif dataset == "cancer":
                plt.ylim(0.8, 1.0)
            else:
                plt.ylim(0.0, 1.0)
            plt.legend(title="Method", fontsize=16, title_fontsize=17)
            plt.grid(True)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            outname = f"{dataset}_{clf}.png"
            plt.savefig(outname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot {outname}")

    conn.close()

if __name__ == "__main__":
    main()
