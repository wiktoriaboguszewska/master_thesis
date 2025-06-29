import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Creating output directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Configuration
db_path = "estimators.db"
datasets = ['cancer', 'credit', 'blood', 'bank', 'phoneme', 'banknote']
classifiers = ['logreg', 'rf']
methods = ['EM', 'TC']
p1_train = 0.1
p1_tests = [0.2, 0.5, 0.9]
shift_suffix = {0.2: '01', 0.5: '04', 0.9: '08'}

# Connecting to DB
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for dataset in datasets:
    for clf in classifiers:
        for p1_test in p1_tests:
            s_val = shift_suffix[p1_test]
            true_pi = p1_test
            data = []

            for method in methods:
                table = f"{dataset}_{method}_{clf}_{s_val}"
                try:
                    df = pd.read_sql_query(f"SELECT pi_est FROM '{table}'", conn)
                    df['Method'] = method
                    data.append(df)
                except Exception as e:
                    print(f"Could not read from table {table}: {e}")
                    continue

            if data:
                all_data = pd.concat(data)
                plt.figure(figsize=(9, 5))
                ax = plt.gca()
                all_data.boxplot(column='pi_est', by='Method', ax=ax, widths=0.5)
                plt.axhline(y=true_pi, color='red', linestyle='--', linewidth=2, label=f"True π′ = {true_pi}")
                plt.title("")
                plt.suptitle("")
                plt.ylabel("Estimated π′", fontsize=18)
                plt.xlabel("Method", fontsize=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.ylim(0, 1)
                plt.legend(fontsize=15)
                plt.grid(True)

                filename = f"plots/{dataset}_{clf}_{s_val}.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved {filename}")

conn.close()
