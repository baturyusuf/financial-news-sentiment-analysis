import os
import re
import sys
import subprocess
from itertools import product

import pandas as pd


def parse_metric(pattern: str, text: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def main():
    # Sweep aralığı (hızlı/etkili)
    K_LIST = [0.6, 0.8, 0.9, 1.0, 1.1]
    MIN_LIST = [0.10, 0.15, 0.20]

    results = []

    for k_vol, min_thr in product(K_LIST, MIN_LIST):
        env = os.environ.copy()
        env["K_VOL"] = str(k_vol)
        env["MIN_THR"] = str(min_thr)
        env["SAVE_MODELS"] = "0"  # sweep sırasında modelleri overwrite etmesin

        print(f"\n=== RUN: K_VOL={k_vol} MIN_THR={min_thr} ===")

        p = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            env=env
        )

        out = (p.stdout or "") + "\n" + (p.stderr or "")

        kept = parse_metric(r"Adaptive labeling:\s*kept\s*(\d+)", out)
        pr_auc = parse_metric(r"\[Validation\]\s*PR-AUC.*:\s*([0-9.]+)", out)
        roc_auc = parse_metric(r"\[Validation\]\s*ROC-AUC.*:\s*([0-9.]+)", out)
        bacc = parse_metric(r"BalancedAcc=([0-9.]+)", out)
        mcc = parse_metric(r"MCC=([0-9.]+)", out)
        best_thr = parse_metric(r"Best threshold:\s*([0-9.]+)", out)

        results.append({
            "K_VOL": k_vol,
            "MIN_THR": min_thr,
            "kept_rows": kept,
            "PR_AUC": pr_auc,
            "ROC_AUC": roc_auc,
            "BalancedAcc": bacc,
            "MCC": mcc,
            "best_thr": best_thr,
            "returncode": p.returncode,
        })

        # Eğer bir koşu patladıysa logu ekrana bas
        if p.returncode != 0:
            print("Run failed. Output:")
            print(out)

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/label_sweep.csv", index=False)

    # En iyi 5 (öncelik MCC, sonra PR_AUC)
    df_ranked = df.sort_values(["MCC", "PR_AUC"], ascending=False)
    print("\n==== TOP 5 (by MCC, then PR_AUC) ====")
    print(df_ranked.head(5).to_string(index=False))

    print("\nSaved: results/label_sweep.csv")


if __name__ == "__main__":
    main()
