#!/usr/bin/env python3
# CE889 – Final MLP
import argparse, csv, json, random
from math import exp, sqrt
from typing import List, Tuple, Optional

def set_seed(seed: Optional[int]):
    random.seed(42 if seed is None else seed)

def sigmoid(x: float) -> float:
    if x < -60: return 0.0
    if x >  60: return 1.0
    return 1.0 / (1.0 + exp(-x))

def sigmoid_deriv(a: float) -> float:
    return a * (1.0 - a)

def read_csv(path: str, in_cols: List[str], out_cols: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    X, Y = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            raise ValueError(f"CSV '{path}' needs a header with {in_cols+out_cols}")
        try:
            in_idx  = [header.index(c) for c in in_cols]
            out_idx = [header.index(c) for c in out_cols]
        except ValueError as e:
            raise ValueError(f"CSV '{path}' must contain columns {in_cols+out_cols}. Got: {header}") from e
        for row in r:
            if not row: continue
            try:
                X.append([float(row[i]) for i in in_idx])
                Y.append([float(row[i]) for i in out_idx])
            except:
                continue
    return X, Y

# --------------- model pieces ---------------
class Neuron:
    def __init__(self, n_inputs: int):
        self.w = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.b = random.uniform(-1.0, 1.0)
        self.out = 0.0
        self.delta = 0.0
        self.dw_prev = [0.0] * n_inputs
        self.db_prev = 0.0

    def forward(self, x: List[float]) -> float:
        z = self.b
        for wi, xi in zip(self.w, x):
            z += wi * xi
        self.out = sigmoid(z)
        return self.out

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, x: List[float]) -> List[float]:
        return [n.forward(x) for n in self.neurons]

class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 0.05, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.h = Layer(n_in, n_hidden)
        self.o = Layer(n_hidden, n_out)

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        h_out = self.h.forward(x)
        y_out = self.o.forward(h_out)
        return h_out, y_out

    def backprop(self, x: List[float], t: List[float]) -> float:
        h, y = self.forward(x)

        # output deltas
        for j, n in enumerate(self.o.neurons):
            err = t[j] - y[j]
            n.delta = err * sigmoid_deriv(n.out)

        # hidden deltas
        for i, hn in enumerate(self.h.neurons):
            s = 0.0
            for on in self.o.neurons:
                s += on.w[i] * on.delta
            hn.delta = s * sigmoid_deriv(hn.out)

        # update output layer
        for on in self.o.neurons:
            for j, h_out in enumerate(h):
                dw = self.lr * on.delta * h_out + self.momentum * on.dw_prev[j]
                on.w[j] += dw
                on.dw_prev[j] = dw
            db = self.lr * on.delta + self.momentum * on.db_prev
            on.b += db
            on.db_prev = db

        # update hidden layer
        for hn in self.h.neurons:
            for j, xj in enumerate(x):
                dw = self.lr * hn.delta * xj + self.momentum * hn.dw_prev[j]
                hn.w[j] += dw
                hn.dw_prev[j] = dw
            db = self.lr * hn.delta + self.momentum * hn.db_prev
            hn.b += db
            hn.db_prev = db

        mse = sum((tt - yy) ** 2 for tt, yy in zip(t, y)) / max(1, len(t))
        return mse

    def save(self, path: str):
        obj = {
            "hidden": {"w": [n.w for n in self.h.neurons], "b": [n.b for n in self.h.neurons]},
            "output": {"w": [n.w for n in self.o.neurons], "b": [n.b for n in self.o.neurons]},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"Saved weights to: {path}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for n, w in zip(self.h.neurons, obj["hidden"]["w"]): n.w = [float(v) for v in w]
        for n, b in zip(self.h.neurons, obj["hidden"]["b"]): n.b = float(b)
        for n, w in zip(self.o.neurons, obj["output"]["w"]): n.w = [float(v) for v in w]
        for n, b in zip(self.o.neurons, obj["output"]["b"]): n.b = float(b)
        print(f"Loaded weights from: {path}")

# --------------- metrics ---------------
def mse_rmse(Y_true: List[List[float]], Y_pred: List[List[float]]) -> Tuple[float, float]:
    if not Y_true: return 0.0, 0.0
    dim = len(Y_true[0])
    s = 0.0
    for t, y in zip(Y_true, Y_pred):
        s += sum((tt - yy) ** 2 for tt, yy in zip(t, y)) / dim
    mse = s / len(Y_true)
    return mse, sqrt(mse)

def evaluate(net: MLP, X: List[List[float]], Y: List[List[float]], tag: str):
    preds = []
    for x in X:
        _, y = net.forward(x)
        preds.append(y)
    mse, rmse = mse_rmse(Y, preds)
    print(f"[{tag}] MSE={mse:.6f} | RMSE={rmse:.6f}  (n={len(X)})")
    return mse, rmse

def train(net: MLP, Xtr: List[List[float]], Ytr: List[List[float]],
          Xva: Optional[List[List[float]]] = None, Yva: Optional[List[List[float]]] = None,
          epochs: int = 100):
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        idx = list(range(n))
        random.shuffle(idx)
        total = 0.0
        for i in idx:
            total += net.backprop(Xtr[i], Ytr[i])
        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            tr_mse = total / max(1, n)
            tr_rmse = sqrt(tr_mse)
            msg = f"Epoch {ep:4d}/{epochs} | train MSE={tr_mse:.6f} RMSE={tr_rmse:.6f}"
            if Xva and Yva:
                va_mse, va_rmse = evaluate(net, Xva, Yva, "val")
                msg += f" | (val MSE={va_mse:.6f} RMSE={va_rmse:.6f})"
            print(msg)

def parse_cols(s: str) -> List[str]:
    return [c.strip() for c in s.split(",") if c.strip()]

def main():
    ap = argparse.ArgumentParser(description="Plain MLP with momentum")
    ap.add_argument("--train", type=str, help="train.csv")
    ap.add_argument("--val",   type=str, help="val.csv")
    ap.add_argument("--test",  type=str, help="test.csv")
    ap.add_argument("--in-cols",  type=str, required=True, help="e.g. x_dist,y_dist")
    ap.add_argument("--out-cols", type=str, required=True, help="e.g. thrust,turn")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-weights", type=str, default=None)
    ap.add_argument("--load-weights", type=str, default=None)
    ap.add_argument("--predict", action="store_true", help="only evaluate, no training")
    args = ap.parse_args()

    set_seed(args.seed)
    in_cols  = parse_cols(args.in_cols)
    out_cols = parse_cols(args.out_cols)
    n_in, n_out = len(in_cols), len(out_cols)

    net = MLP(n_in, args.hidden, n_out, lr=args.lr, momentum=args.momentum)

    if args.predict:
        if not args.load_weights:
            print("--predict needs --load-weights")
            return
        net.load(args.load_weights)
        if args.train:
            Xtr, Ytr = read_csv(args.train, in_cols, out_cols)
            evaluate(net, Xtr, Ytr, "train")
        if args.val:
            Xva, Yva = read_csv(args.val, in_cols, out_cols)
            evaluate(net, Xva, Yva, "val")
        if args.test:
            Xte, Yte = read_csv(args.test, in_cols, out_cols)
            evaluate(net, Xte, Yte, "test")
        return

    if not args.train:
        print("Training mode requires --train.")
        return

    Xtr, Ytr = read_csv(args.train, in_cols, out_cols)
    Xva = Yva = None
    if args.val:
        Xva, Yva = read_csv(args.val, in_cols, out_cols)

    print(f"Network: {n_in}-{args.hidden}-{n_out} | lr={args.lr} | momentum={args.momentum}")
    print(f"Training on {len(Xtr)} samples for {args.epochs} epochs...\n")
    train(net, Xtr, Ytr, Xva, Yva, args.epochs)
    print("\nTraining complete.")

    if args.save_weights:
        net.save(args.save_weights)

    evaluate(net, Xtr, Ytr, "train")
    if Xva and Yva:
        evaluate(net, Xva, Yva, "val")
    if args.test:
        Xte, Yte = read_csv(args.test, in_cols, out_cols)
        evaluate(net, Xte, Yte, "test")

if __name__ == "__main__":
    main()
