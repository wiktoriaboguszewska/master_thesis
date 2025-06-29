import sqlite3
import pandas as pd

# Configuration
db_path = "estimators.db"
datasets = ['cancer', 'credit', 'blood', 'bank', 'phoneme', 'banknote']
classifiers = ['logreg', 'rf']
methods = ['EM', 'TC']
p1_train = 0.1
p1_tests = [0.2, 0.5, 0.9]
shift_suffix = {0.2: '01', 0.5: '04', 0.9: '08'}

# Connecting to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Iterating through each scenario
for p1_test in p1_tests:
    s_val = shift_suffix[p1_test]
    records = []

    for dataset in datasets:
        for clf in classifiers:
            row = {
                'Dataset': dataset,
                'Classifier': clf
            }
            for method in methods:
                table = f"{dataset}_{method}_{clf}_{s_val}"
                try:
                    cursor.execute(f"SELECT AVG(pi_est) FROM '{table}'")
                    mean = cursor.fetchone()[0]
                except Exception as e:
                    mean = None
                row[method] = round(mean, 4) if mean is not None else 'NA'
            records.append(row)

    # Saving to CSV
    df = pd.DataFrame(records)
    csv_filename = f"estimators_{s_val}.csv"
    df.to_csv(csv_filename, index=False)

conn.close()
