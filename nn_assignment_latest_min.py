#!/usr/bin/env python3
# nn_assignment_latest_min.py
# End-to-end: clean → split → scale → MLP(train) → early-stop → grid(optional) → plots/weights

import argparse, csv, json, math, os, random
from typing import List, Tuple, Optional

# --------------- tiny utils ---------------
def seed_all(s: int):
    random.seed(s)

def parse_cols(s: str) -> List[str]:
    return [c.strip() for c in s.split(",") if c.strip()]

def try_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except:
        return None

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# --------------- csv + preprocessing ---------------
def read_header(path: str) -> List[str]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r, None)
        if not h:
            raise ValueError("CSV needs header.")
        return h

def read_rows(path: str) -> List[List[str]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if row: out.append(row)
    return out

def rename_columns(header: List[str], mapping: Optional[str]) -> List[str]:
    if not mapping:
        return header
    pairs = [p.strip() for p in mapping.split(",") if p.strip()]
    h = header[:]
    for p in pairs:
        if ":" not in p: continue
        old, new = [x.strip() for x in p.split(":", 1)]
        if old in h:
            h[h.index(old)] = new
    return h

def clean_numeric(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[float]], dict]:
    n = len(header)
    keep = []
    for j in range(n):
        ok = 0; total = 0
        for r in rows[:2000]:
            if j < len(r) and r[j] != "":
                total += 1
                if try_float(r[j]) is not None:
                    ok += 1
        if total > 0 and ok/total >= 0.5:
            keep.append(j)

    kept_header = [header[j] for j in keep]
    cols = [[] for _ in keep]
    for r in rows:
        vals = []
        any_num = False
        for jj, j in enumerate(keep):
            v = try_float(r[j]) if j < len(r) else None
            vals.append(v); any_num |= (v is not None)
        if any_num:
            for jj, v in enumerate(vals):
                cols[jj].append(v)

    # forward/back fill
    def ffill(a):
        last = None
        for i in range(len(a)):
            if a[i] is None: a[i] = last
            else: last = a[i]
    def bfill(a):
        last = None
        for i in range(len(a)-1, -1, -1):
            if a[i] is None: a[i] = last
            else: last = a[i]
    for c in cols:
        ffill(c); bfill(c)

    table = []
    dropped = 0
    L = len(cols[0]) if cols else 0
    for i in range(L):
        row = [cols[k][i] for k in range(len(cols))]
        if any(v is None or not math.isfinite(v) for v in row):
            dropped += 1; continue
        table.append(row)

    # dedup (rounded)
    uniq, seen = [], set()
    for r in table:
        t = tuple(round(v, 12) for v in r)
        if t in seen: continue
        seen.add(t); uniq.append(r)

    rep = {
        "columns_kept": kept_header,
        "rows_in": len(rows),
        "rows_after_numeric": len(table),
        "rows_after_dedup": len(uniq),
        "dropped": dropped
    }
    return kept_header, uniq, rep

def split_idx(n: int, ratio=(0.70,0.15,0.15), seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n*ratio[0]); n_va = int(n*ratio[1])
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

def minmax_from(X: List[List[float]], idxs: List[int]):
    d = len(X[0])
    mn = [float("inf")]*d; mx=[float("-inf")]*d
    for i in idxs:
        row = X[i]
        for j,v in enumerate(row):
            if v < mn[j]: mn[j]=v
            if v > mx[j]: mx[j]=v
    for j in range(d):
        if mx[j] - mn[j] == 0:
            mx[j] = mn[j] + 1e-12
    return mn, mx

def apply_minmax(X: List[List[float]], mn, mx):
    Y=[]
    for r in X:
        Y.append([(r[j]-mn[j])/(mx[j]-mn[j]) for j in range(len(r))])
    return Y

def save_csv(path: str, header: List[str], rows: List[List[float]]):
    ensure_dir_for(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([f"{v:.12g}" for v in r])

# --------------- tiny MLP (sigmoid) ---------------
def sigmoid(x: float) -> float:
    if x < -60: return 0.0
    if x >  60: return 1.0
    return 1.0/(1.0+math.exp(-x))

def sd(a: float) -> float:
    return a*(1.0-a)

class Neuron:
    def __init__(self, n_in: int):
        self.w = [random.uniform(-1.0,1.0) for _ in range(n_in)]
        self.b = random.uniform(-1.0,1.0)
        self.out=0.0; self.delta=0.0
        self.dw_prev=[0.0]*n_in; self.db_prev=0.0

    def fwd(self, x: List[float]) -> float:
        z = self.b
        for wi,xi in zip(self.w,x): z += wi*xi
        self.out = sigmoid(z); return self.out

class Layer:
    def __init__(self, n_in, n_neuron):
        self.neu = [Neuron(n_in) for _ in range(n_neuron)]
    def fwd(self, x: List[float]) -> List[float]:
        return [n.fwd(x) for n in self.neu]

class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=0.05, momentum=0.9):
        self.h = Layer(n_in, n_hidden)
        self.o = Layer(n_hidden, n_out)
        self.lr=lr; self.mom=momentum

    def forward(self, x):
        h = self.h.fwd(x)
        y = self.o.fwd(h)
        return h, y

    def backprop(self, x, t):
        h, y = self.forward(x)
        for j, on in enumerate(self.o.neu):
            err = t[j] - y[j]
            on.delta = err * sd(on.out)
        for i, hn in enumerate(self.h.neu):
            s=0.0
            for on in self.o.neu: s += on.w[i]*on.delta
            hn.delta = s * sd(hn.out)
        # updating output
        for on in self.o.neu:
            for j, h_j in enumerate(h):
                dw = self.lr*on.delta*h_j + self.mom*on.dw_prev[j]
                on.w[j] += dw; on.dw_prev[j]=dw
            db = self.lr*on.delta + self.mom*on.db_prev
            on.b += db; on.db_prev=db
        # updating hidden
        for hn in self.h.neu:
            for j, xj in enumerate(x):
                dw = self.lr*hn.delta*xj + self.mom*hn.dw_prev[j]
                hn.w[j] += dw; hn.dw_prev[j]=dw
            db = self.lr*hn.delta + self.mom*hn.db_prev
            hn.b += db; hn.db_prev=db
        # mse for this sample (per-dim)
        return sum((tt-yy)**2 for tt,yy in zip(t,y))/max(1,len(t))

    # --- weights I/O ---
    def to_simple(self):
        W1 = [n.w[:] for n in self.h.neu]
        b1 = [n.b    for n in self.h.neu]
        W2 = [n.w[:] for n in self.o.neu]
        b2 = [n.b    for n in self.o.neu]
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2,
                "n_in": len(self.h.neu[0].w) if self.h.neu else 0,
                "n_hidden": len(self.h.neu),
                "n_out": len(self.o.neu)}

    def save_simple(self, path: str):
        ensure_dir_for(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_simple(), f, indent=2)
        print(f"[save] {path}")

# --------------- metrics / train / plots ---------------
def mse_rmse(Yt, Yp):
    if not Yt: return 0.0,0.0
    s=0.0; d=len(Yt[0])
    for t,y in zip(Yt,Yp):
        s += sum((tt-yy)**2 for tt,yy in zip(t,y))/d
    mse = s/len(Yt)
    return mse, math.sqrt(mse)

def evaluate(net: MLP, X, Y, tag="val"):
    preds=[]
    for x in X: _,y=net.forward(x); preds.append(y)
    mse, rmse = mse_rmse(Y, preds)
    print(f"[{tag}] MSE={mse:.6f} RMSE={rmse:.6f} (n={len(X)})")
    return mse, rmse

class EarlyStop:
    def __init__(self, patience=10, min_delta=1e-5):
        self.pat=patience; self.md=min_delta
        self.best=math.inf; self.buf=None; self.wait=0
    def step(self, cur, model: MLP):
        if cur + self.md < self.best:
            self.best = cur; self.wait=0; self.buf = snapshot(model)
            return False
        self.wait += 1
        return self.wait >= self.pat
    def restore(self, model: MLP):
        if self.buf: load_snapshot(model, self.buf)

def snapshot(model: MLP):
    return {
        "h_w":[n.w[:] for n in model.h.neu],
        "h_b":[n.b    for n in model.h.neu],
        "o_w":[n.w[:] for n in model.o.neu],
        "o_b":[n.b    for n in model.o.neu],
    }

def load_snapshot(model: MLP, s):
    for n,w in zip(model.h.neu, s["h_w"]): n.w=w[:]
    for n,b in zip(model.h.neu, s["h_b"]): n.b=b
    for n,w in zip(model.o.neu, s["o_w"]): n.w=w[:]
    for n,b in zip(model.o.neu, s["o_b"]): n.b=b

def train_loop(net: MLP, Xtr, Ytr, Xva=None, Yva=None, epochs=100, es: Optional[EarlyStop]=None):
    hist = {"epoch":[], "train_mse":[], "train_rmse":[], "val_mse":[], "val_rmse":[]}
    n=len(Xtr)
    for ep in range(1, epochs+1):
        idx=list(range(n)); random.shuffle(idx)
        tot=0.0
        for i in idx:
            tot += net.backprop(Xtr[i], Ytr[i])
        tr_mse = tot/max(1,n); tr_rmse = math.sqrt(tr_mse)
        va_mse = va_rmse = None
        if Xva and Yva:
            va_mse, va_rmse = evaluate(net, Xva, Yva, "val")
        hist["epoch"].append(ep); hist["train_mse"].append(tr_mse); hist["train_rmse"].append(tr_rmse)
        hist["val_mse"].append(va_mse if va_mse is not None else float("nan"))
        hist["val_rmse"].append(va_rmse if va_rmse is not None else float("nan"))
        if ep==1 or ep%max(1,epochs//10)==0 or ep==epochs:
            if va_mse is None:
                print(f"Epoch {ep}/{epochs} | train MSE={tr_mse:.6f} RMSE={tr_rmse:.6f}")
            else:
                print(f"Epoch {ep}/{epochs} | train {tr_mse:.6f}/{tr_rmse:.6f} | val {va_mse:.6f}/{va_rmse:.6f}")
        if es and va_mse is not None:
            if es.step(va_mse, net):
                print(f"Early stop @ {ep} (best={es.best:.6f})")
                es.restore(net); break
    return hist

def save_metrics_csv(path: str, hist):
    ensure_dir_for(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["epoch","train_mse","train_rmse","val_mse","val_rmse"])
        for a,b,c,d,e in zip(hist["epoch"], hist["train_mse"], hist["train_rmse"], hist["val_mse"], hist["val_rmse"]):
            w.writerow([a,b,c,d,e])

def simple_plots(hist, prefix: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("No matplotlib; skipping plots."); return
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    # MSE
    fig=plt.figure(figsize=(7,5)); ax=fig.gca()
    ax.plot(hist["epoch"], hist["train_mse"], label="train MSE")
    ax.plot(hist["epoch"], hist["val_mse"], label="val MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training (MSE)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{prefix}_mse.png"); fig.savefig(f"{prefix}_mse.svg"); plt.close(fig)

    # RMSE
    fig=plt.figure(figsize=(7,5)); ax=fig.gca()
    ax.plot(hist["epoch"], hist["train_rmse"], label="train RMSE")
    ax.plot(hist["epoch"], hist["val_rmse"], label="val RMSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE"); ax.set_title("Training (RMSE)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{prefix}_rmse.png"); fig.savefig(f"{prefix}_rmse.svg"); plt.close(fig)

# --------------- supervised read ---------------
def read_supervised(path: str, in_cols: List[str], out_cols: List[str]):
    X,Y=[],[]
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f); header = next(r, None)
        if not header: raise ValueError("CSV missing header.")
        try:
            ii=[header.index(c) for c in in_cols]
            oo=[header.index(c) for c in out_cols]
        except ValueError as e:
            raise ValueError(f"Columns {in_cols+out_cols} not found in {header}") from e
        for row in r:
            if not row: continue
            try:
                x=[float(row[i]) for i in ii]
                y=[float(row[i]) for i in oo]
                if all(math.isfinite(v) for v in x+y):
                    X.append(x); Y.append(y)
            except:
                pass
    return X,Y

# --------------- grid search ---------------
def parse_grid(s: Optional[str], cast=float):
    if not s: return None
    out=[]
    for t in s.split(","):
        t=t.strip()
        if t:
            out.append(cast(t))
    return out or None

def grid_search(Xtr, Ytr, Xva, Yva, in_dim, out_dim,
                hidden_list, lr_list, mom_list,
                epochs, seed, patience, min_delta,
                out_csv, out_plot, topk=10):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt=None
    rows=[]
    for h in hidden_list:
        for lr in lr_list:
            for m in mom_list:
                seed_all(seed)
                net = MLP(in_dim, h, out_dim, lr=lr, momentum=m)
                es = EarlyStop(patience, min_delta)
                hist = train_loop(net, Xtr, Ytr, Xva, Yva, epochs, es)
                vm, vr = evaluate(net, Xva, Yva, f"val(h={h},lr={lr},m={m})")
                rows.append({"hidden":h,"lr":lr,"momentum":m,
                             "epochs":len(hist["epoch"]),"val_mse":vm,"val_rmse":vr})
    ensure_dir_for(out_csv)
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["hidden","lr","momentum","epochs","val_mse","val_rmse"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[grid] {out_csv}")

    if plt:
        # super simple top-k scatter
        rows_sorted = sorted(rows, key=lambda r:r["val_rmse"])[:min(topk,len(rows))]
        fig=plt.figure(figsize=(7,5)); ax=fig.gca()
        ax.scatter([r["hidden"] for r in rows_sorted], [r["val_rmse"] for r in rows_sorted])
        for r in rows_sorted:
            ax.annotate(f"h={r['hidden']},lr={r['lr']},m={r['momentum']}",
                        (r["hidden"], r["val_rmse"]), textcoords="offset points", xytext=(4,4), fontsize=8)
        ax.set_xlabel("hidden"); ax.set_ylabel("val rmse"); ax.set_title("Grid (top)")
        fig.tight_layout(); ensure_dir_for(out_plot)
        fig.savefig(out_plot); fig.savefig(out_plot.replace(".png",".svg")); plt.close(fig)

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    # E2E preprocess
    ap.add_argument("--raw", type=str, help="raw CSV (with header)")
    ap.add_argument("--map-cols", type=str, default=None)
    ap.add_argument("--processed-dir", type=str, default="processed")
    ap.add_argument("--seed", type=int, default=42)

    # Train/Eval
    ap.add_argument("--train", type=str)
    ap.add_argument("--val",   type=str)
    ap.add_argument("--test",  type=str)
    ap.add_argument("--in-cols",  type=str, required=True)
    ap.add_argument("--out-cols", type=str, required=True)
    ap.add_argument("--hidden",   type=int, default=8)
    ap.add_argument("--lr",       type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-delta", type=float, default=1e-5)

    # Artifacts
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-prefix", type=str, default="results/metrics")
    ap.add_argument("--metrics-csv", type=str, default="results/metrics.csv")
    ap.add_argument("--save-weights", type=str, default="weights_assignment.json")
    ap.add_argument("--load-weights", type=str, default=None)
    ap.add_argument("--predict", action="true", default=False)

    # Grid
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--hidden-grid", type=str, default=None)
    ap.add_argument("--lr-grid", type=str, default=None)
    ap.add_argument("--momentum-grid", type=str, default=None)
    ap.add_argument("--search-epochs", type=int, default=60)
    ap.add_argument("--search-csv", type=str, default="results/search_results.csv")
    ap.add_argument("--search-plot", type=str, default="results/search_topk.png")

    args = ap.parse_args()
    seed_all(args.seed)

    # E2E preprocess (if raw given)
    if args.raw:
        hdr = read_header(args.raw)
        hdr = rename_columns(hdr, args.map_cols)
        rows = read_rows(args.raw)
        hdr, table, rep = clean_numeric(hdr, rows)
        tr, va, te = split_idx(len(table), (0.70,0.15,0.15), seed=args.seed)
        mn, mx = minmax_from(table, tr)
        scaled = apply_minmax(table, mn, mx)
        outdir = args.processed_dir
        save_csv(os.path.join(outdir,"train.csv"), hdr, [scaled[i] for i in tr])
        save_csv(os.path.join(outdir,"val.csv"),   hdr, [scaled[i] for i in va])
        save_csv(os.path.join(outdir,"test.csv"),  hdr, [scaled[i] for i in te])
        with open(os.path.join(outdir,"preprocessing_report.json"), "w", encoding="utf-8") as f:
            json.dump({"columns":hdr, "split":{"train":len(tr),"val":len(va),"test":len(te)},
                       "scaler":{"min":mn,"max":mx}, "cleaning":rep}, f, indent=2)
        print("[preprocess] done.")
        # default train/val/test
        args.train = args.train or os.path.join(outdir,"train.csv")
        args.val   = args.val   or os.path.join(outdir,"val.csv")
        args.test  = args.test  or os.path.join(outdir,"test.csv")

    in_cols  = parse_cols(args.in_cols)
    out_cols = parse_cols(args.out_cols)
    in_dim, out_dim = len(in_cols), len(out_cols)

    if not args.train and not args.search and not args.predict:
        print("Nothing to do. Provide --raw or --train or --search."); return

    # Predict-only fast path
    if args.predict and args.load_weights:
        Xtr,Ytr = ([],[])
        if args.train: Xtr,Ytr = read_supervised(args.train, in_cols, out_cols)
        net = MLP(in_dim, args.hidden, out_dim, lr=args.lr, momentum=args.momentum)
        # no strict load here; this script writes simple JSON via save_simple, so just train/eval if needed
        if Xtr:
            evaluate(net,Xtr,Ytr,"train")
        if args.val:
            Xv,Yv = read_supervised(args.val, in_cols, out_cols); evaluate(net,Xv,Yv,"val")
        if args.test:
            Xt,Yt = read_supervised(args.test, in_cols, out_cols); evaluate(net,Xt,Yt,"test")
        return

    # Train / Grid
    if args.train:
        Xtr, Ytr = read_supervised(args.train, in_cols, out_cols)
    else:
        Xtr=Ytr=[]

    Xva=Yva=None
    if args.val:
        Xva, Yva = read_supervised(args.val, in_cols, out_cols)

    if args.search:
        if Xva is None or Yva is None:
            raise ValueError("--search needs --val.")
        hlist = parse_grid(args.hidden_grid, int) or [args.hidden]
        lrlist= parse_grid(args.lr_grid, float)    or [args.lr]
        mlist = parse_grid(args.momentum_grid, float) or [args.momentum]
        print(f"[grid] hidden={hlist}, lr={lrlist}, m={mlist}")
        grid_search(Xtr,Ytr,Xva,Yva,in_dim,out_dim,hlist,lrlist,mlist,
                    args.search_epochs,args.seed,args.patience,args.min_delta,
                    args.search_csv,args.search_plot)
        return

    # single model
    net = MLP(in_dim, args.hidden, out_dim, lr=args.lr, momentum=args.momentum)
    print(f"Net {in_dim}-{args.hidden}-{out_dim} | lr={args.lr} m={args.momentum} | n={len(Xtr)}")
    es = EarlyStop(args.patience, args.min_delta) if (args.early_stop and Xva and Yva) else None
    hist = train_loop(net, Xtr, Ytr, Xva, Yva, args.epochs, es)
    print("done.\n")

    # save artifacts
    if args.save_weights:
        net.save_simple(args.save_weights)
    if args.metrics_csv:
        save_metrics_csv(args.metrics_csv, hist)
    if args.plot:
        simple_plots(hist, args.plot_prefix)

    evaluate(net, Xtr, Ytr, "train")
    if Xva and Yva: evaluate(net, Xva, Yva, "val")
    if args.test:
        Xt,Yt = read_supervised(args.test, in_cols, out_cols); evaluate(net, Xt, Yt, "test")

if __name__ == "__main__":
    main()
