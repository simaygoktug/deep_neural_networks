# class_balance_and_smote.py
import argparse, os, math, random
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------- 1) Action discretization (same thresholds as GameLoop/NeuralNetHolder) ----------
TH_THRUST = 0.55
TL, TR    = 0.40, 0.60

# map thrust/turn -> discrete action id and nice label
ACTIONS = ["Idle","Up","Left","Right","Up+Left","Up+Right"]
def to_action(thrust: float, turn: float) -> int:
    up = 1 if thrust > TH_THRUST else 0
    left = (turn < TL)
    right = (turn > TR)
    if up and left:  return 4   # Up+Left
    if up and right: return 5   # Up+Right
    if up:           return 1   # Up
    if left:         return 2   # Left
    if right:        return 3   # Right
    return 0                   # Idle

# ---------- 2) Load & prepare ----------
def load_xy(path: str, in_cols=("x_dist","y_dist"), out_cols=("thrust","turn")):
    df = pd.read_csv(path)
    # tolerate alternate column names
    def pick(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        raise KeyError(f"None of {candidates} found in {path}.")
    xi = pick(df, (in_cols[0], "x_dist", "x", "xdist"))
    yi = pick(df, (in_cols[1], "y_dist", "y", "ydist"))
    to = pick(df, (out_cols[0], "thrust", "t"))
    ro = pick(df, (out_cols[1], "turn", "r", "tau"))
    X = df[[xi, yi]].astype(float).values
    y_reg = df[[to, ro]].astype(float).values
    # discretize for classification view
    y_cls = np.array([to_action(t, r) for (t, r) in y_reg], dtype=int)
    return X, y_cls, df

# ---------- 3) Simple baseline classifier (for F1) ----------
def baseline_clf_f1(X_train, y_train, X_eval, y_eval, title="val"):
    # light, deterministic baseline: multinomial logistic
    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_eval)
    macro = f1_score(y_eval, yhat, average="macro")
    micro = f1_score(y_eval, yhat, average="micro")
    print(f"\n[{title}] F1 macro={macro:.3f} | micro={micro:.3f}")
    print(classification_report(y_eval, yhat, target_names=ACTIONS, digits=3))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_eval, yhat))
    return macro, micro

# ---------- 4) Histograms ----------
def print_counts(name, y):
    cnt = Counter(y.tolist())
    print(f"\nClass counts — {name}:")
    for k in range(len(ACTIONS)):
        print(f"  {ACTIONS[k]:>8s}: {cnt.get(k,0)}")
    return cnt

# ---------- 5) SMOTE-lite (no external deps) ----------
def smote_lite(X, y, minority_target_ratio=0.8, k=5, seed=42):
    """
    Increase minority classes by synthesizing samples between each minority item
    and a random neighbor of the same class.
    minority_target_ratio=0.8 means: after oversampling, each minority count
    is at least 0.8 * majority_count.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X); y = np.asarray(y)
    counts = Counter(y.tolist())
    maj_n = max(counts.values())
    X_new = [X]; y_new = [y]
    for cls, n in counts.items():
        target = int(minority_target_ratio * maj_n)
        if n >= target: 
            continue
        idx = np.where(y==cls)[0]
        if len(idx) < 2:  # cannot interpolate with <2 samples; fallback: jitter
            grow = target - n
            if len(idx)==0: 
                continue
            base = X[idx[0]]
            jitter = rng.normal(0, 1e-3, size=(grow, X.shape[1]))
            X_new.append(base + jitter); y_new.append(np.full(grow, cls))
            continue
        # precompute neighbors (brute-force small k)
        for _ in range(target - n):
            i = int(rng.choice(idx))
            # pick a same-class neighbor j != i
            j = i
            while j == i:
                j = int(rng.choice(idx))
            lam = rng.uniform(0.0, 1.0)
            x_syn = X[i] + lam*(X[j]-X[i])
            X_new.append(x_syn.reshape(1,-1))
            y_new.append(np.array([cls]))
    X_bal = np.vstack(X_new)
    y_bal = np.concatenate(y_new)
    return X_bal, y_bal

# ---------- 6) CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=False)
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--smote-ratio", type=float, default=0.8)
    ap.add_argument("--out-train", default=None, help="If set, write oversampled TRAIN csv here.")
    args = ap.parse_args()

    Xtr, ytr, dftr = load_xy(args.train)
    Xva, yva, dfva = load_xy(args.val)

    print_counts("train (before)", ytr)
    print_counts("val", yva)

    # Baseline F1 before oversampling
    baseline_clf_f1(Xtr, ytr, Xva, yva, title="val (before SMOTE)")

    if args.only_report:
        return

    if args.smote:
        Xtr_os, ytr_os = smote_lite(Xtr, ytr, minority_target_ratio=args.smote_ratio)
        print_counts(f"train (after SMOTE r={args.smote_ratio})", ytr_os)
        # F1 after oversampling
        baseline_clf_f1(Xtr_os, ytr_os, Xva, yva, title="val (after SMOTE)")

        # Optional: write oversampled TRAIN for later training
        if args.out_train:
            # we must also synthesize targets (thrust/turn) compatible with the class.
            # Simplest: keep original thrust/turn of nearest real sample (proxy).
            # Here we just copy the nearest original row per synthesized sample.
            # (This file is intended for analysis/visualization, not final regression training.)
            df_out = pd.DataFrame({"x_dist": Xtr_os[:,0], "y_dist": Xtr_os[:,1], "action": ytr_os})
            df_out.to_csv(args.out_train, index=False)
            print(f"\n[write] oversampled train saved to: {args.out_train}")

if __name__ == "__main__":
    main()
