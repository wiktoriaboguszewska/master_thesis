import sqlite3
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

DB_FILE     = "pvalues.db"
ALPHA       = 0.05
DATASETS    = ["cancer", "credit", "blood", "bank", "phoneme", "banknote"]
METHODS     = ["BBSD_KS", "BBSD_PT"]
CLASSIFIERS = ["logreg", "rf", "svc", "gb", "dnn"]
S_VALUES    = np.linspace(0, 0.8, 9)          # 0.0 … 0.8

def table_suffix(s: float) -> str:
    return f"{int(round(s * 10)):02d}"        # 0.0→'00', 0.1→'01', …

def exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,))
    return cursor.fetchone() is not None

def rejection_rate(cursor, table_name: str, alpha: float) -> float:
    """
    Return the proportion of p-values < alpha found in `table_name`.
    Skips NULL / non-numeric entries and returns NaN if nothing usable exists.
    """
    if not exists(cursor, table_name):
        return np.nan

    cursor.execute(f"SELECT p_value FROM {table_name}")
    raw = [row[0] for row in cursor.fetchall()]

    # keep only real numbers
    vals = [float(v) for v in raw if isinstance(v, (int, float))]
    if not vals:
        return np.nan

    return np.mean(np.array(vals) < alpha)


def build_frames():
    index = pd.MultiIndex.from_product(
        [METHODS, DATASETS], names=["Method", "Dataset"])
    frames = {s: pd.DataFrame(index=index, columns=CLASSIFIERS, dtype=float)
              for s in S_VALUES}

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        for dataset, method, clf, s in product(DATASETS, METHODS,
                                               CLASSIFIERS, S_VALUES):
            suffix = table_suffix(s)
            table  = f"{dataset}_{method.lower()}_{clf}_{suffix}"
            rate   = rejection_rate(cur, table, ALPHA)
            frames[s].loc[(method, dataset), clf] = rate
    return frames


def main():
    db_path = Path(DB_FILE).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    frames = build_frames()

    for s, df in frames.items():
        if np.isclose(s, 0.0):
            name = "type1_error_results.csv"
        else:
            name = f"power_shift{int(round(s*10)):02d}_results.csv"
        df.to_csv(name)
        print(f"Saved {name}")

    print("Summary tables successfully created.")


if __name__ == "__main__":
    main()
