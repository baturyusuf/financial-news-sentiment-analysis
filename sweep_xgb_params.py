# sweep_xgb_params.py
import os
import re
import csv
import sys
import time
import random
import subprocess
from pathlib import Path


MEAN_RE = re.compile(
    r"\[CV\]\s*MEAN RESULTS\s*"
    r"PR_AUC=(?P<pr>[0-9.]+)\s*\|\s*"
    r"ROC_AUC=(?P<roc>[0-9.]+)\s*\|\s*"
    r"BalancedAcc=(?P<bacc>[0-9.]+)\s*\|\s*"
    r"MCC=(?P<mcc>[0-9.]+)"
)


def run_one(trial_id: int, env_overrides: dict) -> dict:
    env = os.environ.copy()
    env.update(env_overrides)

    # main.py çıktısı çok; capture edip parse ediyoruz
    p = subprocess.run(
        [sys.executable, "main.py"],
        env=env,
        capture_output=True,
        text=True
    )

    out = (p.stdout or "") + "\n" + (p.stderr or "")

    m = MEAN_RE.search(out)
    if not m:
        return {
            "trial": trial_id,
            "returncode": p.returncode,
            "parsed": 0,
            "PR_AUC": "",
            "ROC_AUC": "",
            "BalancedAcc": "",
            "MCC": "",
            **env_overrides
        }

    return {
        "trial": trial_id,
        "returncode": p.returncode,
        "parsed": 1,
        "PR_AUC": float(m.group("pr")),
        "ROC_AUC": float(m.group("roc")),
        "BalancedAcc": float(m.group("bacc")),
        "MCC": float(m.group("mcc")),
        **env_overrides
    }


def sample_params(rng: random.Random) -> dict:
    """
    Küçük veri için overfit azaltıcı aralıklar.
    """
    max_depth = rng.choice([2, 3, 4])
    min_child_weight = rng.choice([5.0, 10.0, 20.0, 30.0])
    subsample = rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])
    colsample = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9])
    reg_lambda = rng.choice([1.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    gamma = rng.choice([0.0, 0.5, 1.0, 2.0])
    learning_rate = rng.choice([0.01, 0.02, 0.03, 0.05, 0.07])
    early_stop = rng.choice([80, 120, 150, 200])

    # n_estimators yüksek, early stopping seçsin
    n_estimators = 6000

    return {
        "XGB_MAX_DEPTH": str(max_depth),
        "XGB_MIN_CHILD_WEIGHT": str(min_child_weight),
        "XGB_SUBSAMPLE": str(subsample),
        "XGB_COLSAMPLE_BYTREE": str(colsample),
        "XGB_REG_LAMBDA": str(reg_lambda),
        "XGB_GAMMA": str(gamma),
        "XGB_LEARNING_RATE": str(learning_rate),
        "XGB_N_ESTIMATORS": str(n_estimators),
        "XGB_EARLY_STOPPING": str(early_stop),
        "XGB_TREE_METHOD": "hist",
    }


def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    out_csv = Path("results/xgb_sweep.csv")

    # Sabit koşullar (senin en iyi label ayarın + CV)
    base_env = {
        "CV_MODE": "1",
        "CV_POINTS": os.getenv("CV_POINTS", "0.60,0.70,0.80,0.90"),
        "SAVE_MODELS": "0",

        "K_VOL": "1.1",
        "MIN_THR": "0.2",
    }

    # Deneme sayısı
    N = int(os.getenv("N_TRIALS", "20"))
    seed = int(os.getenv("SWEEP_SEED", "42"))
    rng = random.Random(seed)

    rows = []
    t0 = time.time()

    for i in range(1, N + 1):
        hp = sample_params(rng)
        env = {**base_env, **hp}
        print(f"\n=== TRIAL {i}/{N} ===")
        print(" ".join([f"{k}={v}" for k, v in hp.items()]))

        res = run_one(i, env)
        rows.append(res)

        if res.get("parsed") == 1:
            print(f"[OK] MCC={res['MCC']:.4f} PR_AUC={res['PR_AUC']:.4f} BAcc={res['BalancedAcc']:.4f}")
        else:
            print("[WARN] CV mean parse edilemedi. main.py output formatını kontrol et.")

    # CSV yaz
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Top 10 (MCC desc, PR_AUC desc)
    good = [r for r in rows if r.get("parsed") == 1]
    good.sort(key=lambda r: (r["MCC"], r["PR_AUC"]), reverse=True)

    print("\n==== TOP 10 (by MCC, then PR_AUC) ====")
    for r in good[:10]:
        print(
            f"MCC={r['MCC']:.4f} PR={r['PR_AUC']:.4f} BAcc={r['BalancedAcc']:.4f} "
            f"depth={r['XGB_MAX_DEPTH']} mcw={r['XGB_MIN_CHILD_WEIGHT']} "
            f"sub={r['XGB_SUBSAMPLE']} col={r['XGB_COLSAMPLE_BYTREE']} "
            f"lam={r['XGB_REG_LAMBDA']} lr={r['XGB_LEARNING_RATE']} gamma={r['XGB_GAMMA']}"
        )

    dt = time.time() - t0
    print(f"\nSaved: {out_csv.as_posix()} | elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
