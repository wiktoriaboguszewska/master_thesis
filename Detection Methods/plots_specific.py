import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
import pandas as pd
import textwrap, json, math

DB_FILE = "pvalues.db"
assert os.path.exists(DB_FILE), "pvalues.db not found in working directory"

ALPHA = 0.05
SHIFT_VAL = np.linspace(0.0, 0.8, 9)

def shift_suffix(s):
    return str(round(s,1)).replace('.','')

def rejection_rate(cur, table):
    try:
        cur.execute(f'SELECT p_value FROM "{table}"')
        raw = [r[0] for r in cur.fetchall() if r[0] is not None]
        return np.nan if not raw else np.mean(np.array(raw) < ALPHA)
    except sqlite3.OperationalError:
        return np.nan

def collect_rates(cur, dataset, method, clf):
    rates=[]
    for s in SHIFT_VAL:
        suf=shift_suffix(s)
        tbl=f"{dataset}_{method}_{clf}_{suf}"
        rates.append(rejection_rate(cur,tbl))
    return rates

wanted = {
    "cancer": [("bbsd_ks","rf"),("bbsd_ks","gb"),("bbsd_pt","rf"),("bbsd_pt","gb")],
    "bank": [("bbsd_ks","gb"),("bbsd_ks","svc"),("bbsd_pt","gb"),("bbsd_pt","svc")]
}

figs=[]
with sqlite3.connect(DB_FILE) as conn:
    cur=conn.cursor()
    for ds, combos in wanted.items():
        plt.figure(figsize=(9,5))
        for meth,clf in combos:
            y=collect_rates(cur,ds,meth,clf)
            # Use circle markers for KS, square markers for PT
            marker_style = 'o' if meth.lower() == 'bbsd_ks' else 's'
            label=f"{meth.upper()}_{clf}"
            plt.plot(SHIFT_VAL,y,marker=marker_style,label=label)

        # Horizontal line at significance level
        plt.axhline(ALPHA, ls="--", lw=1.5, color='red', label=f"α = {ALPHA}")

        plt.xlabel(r"Shift magnitude  $s=\pi'-\pi$",  fontsize=18)
        plt.ylabel(r"Rejection rate of $H_0$",  fontsize=18)
        #plt.title(f"{ds.capitalize()} dataset", fontsize=20)
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        fname=f"{ds}_method_comparison.png"
        plt.savefig(fname)
        figs.append(fname)
        plt.close()

print("Generated files:", figs)

