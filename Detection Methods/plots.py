import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DB_FILE   = "pvalues.db"
ALPHA     = 0.05
DATASETS  = ["cancer", "credit", "blood", "bank",
             "phoneme", "banknote"]
CLASSIFIERS = ["logreg", "rf", "svc", "gb", "dnn"]
METHODS   = ["bbsd_ks", "bbsd_pt"]
SHIFT_VAL = np.linspace(0.0, 0.8, 9)
def shift_suffix(s):
    """Naming used in experiments.py"""
    return str(round(s, 1)).replace('.', '')    # e.g. 0.3 → "03"

# ---------------------------------------------------------------------
def rejection_rate(cur, table, alpha=ALPHA):
    """
    mean( 1{p < α} ) for a given table; returns np.nan if table missing
    or all entries are NULL.
    """
    try:
        cur.execute(f'SELECT p_value FROM "{table}"')
        raw = [r[0] for r in cur.fetchall() if r[0] is not None]
        return np.nan if not raw else np.mean(np.array(raw) < alpha)
    except sqlite3.OperationalError:
        # table does not exist
        return np.nan

# ---------------------------------------------------------------------
def plot_combo(cur, ds, method, outdir="."):
    rates = {clf: [] for clf in CLASSIFIERS}

    for s in SHIFT_VAL:
        suf = shift_suffix(s)
        for clf in CLASSIFIERS:
            tbl = f"{ds}_{method}_{clf}_{suf}"
            rates[clf].append(rejection_rate(cur, tbl))

    # ---------- draw ---------------------------------------------------
    plt.figure(figsize=(9, 5))
    for clf, y in rates.items():
        plt.plot(SHIFT_VAL, y, marker="o", label=clf)

    plt.axhline(ALPHA, color="red", ls="--", lw=1.5,
                label=f"α = {ALPHA}")
    plt.xlabel(r"Shift magnitude  $s=\pi'-\pi$", fontsize=18)
    plt.ylabel(r"Rejection rate of $H_0$", fontsize=18)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title="Classifier", fontsize=16, title_fontsize=17)
    plt.tight_layout()

    fname = f"{method.upper()}_{ds}.png"
    plt.savefig(Path(outdir)/fname)
    plt.close()
    print("saved", fname)

# ---------------------------------------------------------------------
def main():
    db = Path(DB_FILE)
    if not db.exists():
        raise FileNotFoundError(db)

    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        for ds in DATASETS:
            for meth in METHODS:
                plot_combo(cur, ds, meth)

if __name__ == "__main__":
    main()
