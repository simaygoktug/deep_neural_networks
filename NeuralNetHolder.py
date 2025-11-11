# NeuralNetHolder.py
# Inference-only MLP holder for in-game predictions (thrust, turn)

import json
import math
import os
from typing import List, Tuple, Any

# ====== NN -> Controller tuning (hybrid autopilot) ======
THRUST_ON       = 0.25   # NN thrust eşiği
TURN_LEFT_T     = 0.40   # NN turn deadzone alt sınır
TURN_RIGHT_T    = 0.60   # NN turn deadzone üst sınır

FALL_BOOST_VY   = 0.28   # hızlı düşüşte garanti thrust
Y_PD_KP         = 0.0015 # hedef dikey hız için P katsayısı
Y_PD_OFFSET     = 0.08   # taban dikey hız ofseti (m/s benzeri)

X_DEAD_PX       = 25.0   # x mesafe deadzone (px)
X_DIR_GAIN      = 0.002  # x uzaklık -> yönlendirme kıvrılma şiddeti

MAX_TILT_DEG    = 30     # yumuşak açı limiti (stabilizasyon)
PRINT_EVERY     = 15     # debug çıktısı her N kare
# ========================================================

# --------- Tiny utilities ---------
def sigmoid(x: float) -> float:
    # numerical stability
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def affine(x, W, b):
    # W: [out, in], b: [out]
    return [dot(w_row, x) + b_i for w_row, b_i in zip(W, b)]

def minmax_scale(x, mn, mx):
    out = []
    for xi, a, b in zip(x, mn, mx):
        if b - a == 0:
            out.append(0.0)
        else:
            out.append((xi - a) / (b - a))
    return out

# --------- Helpers for robust JSON parsing ---------
def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _is_num_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(_is_number(x) for x in v)

def _is_num_matrix(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(_is_num_list(r) for r in v)

def _to_float_vec(v):  # assumes _is_num_list
    return [float(x) for x in v]

def _to_float_mat(M):  # assumes _is_num_matrix
    return [[float(x) for x in row] for row in M]

def _rec_collect(container: Any, path: str = ""):
    """Recursively collect all (path, value) pairs from dict/list trees."""
    items = []
    if isinstance(container, dict):
        for k, v in container.items():
            items += _rec_collect(v, f"{path}.{k}" if path else str(k))
    elif isinstance(container, list):
        # do not expand numeric lists/matrices; return them as values
        if _is_num_list(container) or _is_num_matrix(container):
            items.append((path, container))
        else:
            for i, v in enumerate(container):
                items += _rec_collect(v, f"{path}[{i}]")
    else:
        # primitives are ignored
        pass
    return items

def _safe_float(x):
    """Coerce various messy inputs to float. Returns None if not coercible."""
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        if s in {"", "-", "nan", "NaN", "None", "null"}:
            return None
        s = s.replace(",", ".")  # tolerate locale comma
        try:
            v = float(s)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None
    return None

# --------- MLP (inference only) ---------
class MLPInference:
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        # weights
        self.W1 = [[0.0] * n_in for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[0.0] * n_hidden for _ in range(n_out)]
        self.b2 = [0.0] * n_out

    def load_from_json(self, path: str):
        """Schema-agnostic loader: tries known keys, otherwise infers from shapes."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        def pick(container: Any, *names, default=None):
            """Case-insensitive key finder; also looks inside optional 'meta' block."""
            if isinstance(container, dict):
                for name in names:
                    if name in container:
                        return container[name]
                    for k in container.keys():
                        if k.lower() == name.lower():
                            return container[k]
                meta = container.get("meta") or container.get("Meta") or container.get("META")
                if isinstance(meta, dict):
                    return pick(meta, *names, default=default)
            return default

        # 1) Known-name fast path
        W1 = pick(obj, "W1", "w1", "weights_input_hidden", "weightsIH")
        b1 = pick(obj, "b1", "B1", "bias_hidden", "biasH")
        W2 = pick(obj, "W2", "w2", "weights_hidden_output", "weightsHO")
        b2 = pick(obj, "b2", "B2", "bias_output", "biasO")

        if all(v is not None for v in (W1, b1, W2, b2)) and _is_num_matrix(W1) and _is_num_list(b1) and _is_num_matrix(W2) and _is_num_list(b2):
            W1 = _to_float_mat(W1); b1 = _to_float_vec(b1)
            W2 = _to_float_mat(W2); b2 = _to_float_vec(b2)
        else:
            # 2) Schema-agnostic path: search for two matrices + two vectors that fit MLP shapes
            matrices = []
            vectors = []
            for p, v in _rec_collect(obj):
                if _is_num_matrix(v):
                    matrices.append((p, _to_float_mat(v)))
                elif _is_num_list(v):
                    vectors.append((p, _to_float_vec(v)))

            best = None
            for pW1, MW1 in matrices:
                h = len(MW1)
                n = len(MW1[0]) if h > 0 else 0
                # candidate b1
                b1_cands = [(pb, vb) for pb, vb in vectors if len(vb) == h]
                if not b1_cands:
                    continue
                # candidate W2: rows = out, cols = h
                W2_cands = [(pW2, MW2) for pW2, MW2 in matrices if len(MW2) > 0 and len(MW2[0]) == h]
                if not W2_cands:
                    continue
                for pb, vb in b1_cands:
                    for pW2, MW2 in W2_cands:
                        out = len(MW2)
                        b2_cands = [(pb2, vb2) for pb2, vb2 in vectors if len(vb2) == out]
                        if not b2_cands:
                            continue
                        # choose first valid combo
                        pb2, vb2 = b2_cands[0]
                        best = (MW1, vb, MW2, vb2)
                        break
                    if best:
                        break
                if best:
                    break

            if not best:
                raise ValueError("Could not infer matrices/vectors from the JSON file.")
            W1, b1, W2, b2 = best

        # sanity checks
        n_hidden = len(W1)
        n_in = len(W1[0]) if n_hidden > 0 else 0
        n_out = len(W2)
        if any(len(row) != n_in for row in W1):
            raise ValueError("W1 row lengths are inconsistent.")
        if any(len(row) != n_hidden for row in W2):
            raise ValueError("W2 columns must equal hidden size.")
        if len(b1) != n_hidden or len(b2) != n_out:
            raise ValueError("b1/b2 lengths are inconsistent with W1/W2.")

        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    def forward(self, x: List[float]) -> List[float]:
        # hidden
        h_lin = affine(x, self.W1, self.b1)
        h = [sigmoid(a) for a in h_lin]
        # output (both sigmoid because targets in [0,1])
        y_lin = affine(h, self.W2, self.b2)
        y = [sigmoid(a) for a in y_lin]
        return y

# --------- Game-facing holder ---------
class NeuralNetHolder:
    """
    In-game interface.
    Input: raw features [x_dist, y_dist]
    Output: [thrust, turn] in [0,1]
    """
    def __init__(self,
                 weights_path: str = "weights_assignment.json",
                 preprocess_report: str = "preprocessing_report.json"):
        super().__init__()
        self.ready = False
        self.mn = [0.0, 0.0]  # default if no report
        self.mx = [1.0, 1.0]
        self.net = None

        # Load scaler (robust to schema variants)
        if os.path.exists(preprocess_report):
            try:
                with open(preprocess_report, "r", encoding="utf-8") as f:
                    rep = json.load(f)

                def pick(container: Any, *names, default=None):
                    if isinstance(container, dict):
                        for name in names:
                            if name in container:
                                return container[name]
                            for k in container.keys():
                                if k.lower() == name.lower():
                                    return container[k]
                    return default

                scaler = pick(rep, "scaler", "Scaler", default=None)
                if isinstance(scaler, dict):
                    mn = pick(scaler, "min", "mins", "x_min")
                    mx = pick(scaler, "max", "maxs", "x_max")
                else:
                    # flat schema fallback
                    mn = pick(rep, "min", "mins", "x_min")
                    mx = pick(rep, "max", "maxs", "x_max")

                if isinstance(mn, list) and isinstance(mx, list) and len(mn) >= 2 and len(mx) >= 2:
                    self.mn = [float(mn[0]), float(mn[1])]
                    self.mx = [float(mx[0]), float(mx[1])]
            except Exception:
                # keep defaults
                pass

        # Load weights (sizes inferred inside)
        if os.path.exists(weights_path):
            try:
                self.net = MLPInference(n_in=2, n_hidden=8, n_out=2)  # placeholder; overwritten in load
                self.net.load_from_json(weights_path)
                self.ready = True
                print(f"[NeuralNetHolder] Loaded weights: {weights_path} | hidden={self.net.n_hidden}")
                # Optional visibility:
                # print(f"[NeuralNetHolder] Scaler min={self.mn}, max={self.mx}")
                # print(f"[NeuralNetHolder] Net dims: in={self.net.n_in}, hidden={self.net.n_hidden}, out={self.net.n_out}")
            except Exception as e:
                print(f"[NeuralNetHolder] Failed to load weights: {e}")
        else:
            print("[NeuralNetHolder] weights_assignment.json not found. Running in fallback mode.")

    def predict(self, input_row: List[float]) -> Tuple[float, float]:
        """
        input_row: list veya dict kabul edilir.
          - list/tuple: [x_dist, y_dist]
          - dict: {'x_dist': ..., 'y_dist': ...}
        returns: (thrust, turn) in [0,1]
        """
        if not self.ready:
            return (0.0, 0.5)

        # --- 1) raw değerleri güvenli şekilde al ---
        if isinstance(input_row, dict):
            raw_x = input_row.get("x_dist")
            raw_y = input_row.get("y_dist")
        else:
            # liste/tuple benzeri
            try:
                raw_x = input_row[0]
                raw_y = input_row[1]
            except Exception:
                # beklenmeyen format → yumuşak fallback
                return (0.0, 0.5)

        x_val = _safe_float(raw_x)
        y_val = _safe_float(raw_y)
        if x_val is None:
            x_val = 0.0
        if y_val is None:
            y_val = 0.0

        # --- 2) scale like training ---
        mn = self.mn if isinstance(self.mn, list) and len(self.mn) >= 2 else [0.0, 0.0]
        mx = self.mx if isinstance(self.mx, list) and len(self.mx) >= 2 else [1.0, 1.0]
        x_scaled = minmax_scale([x_val, y_val], mn, mx)

        # --- 3) forward pass ---
        y = self.net.forward(x_scaled)  # [thrust, turn] in [0,1]

        # --- 4) clamp ---
        t = max(0.0, min(1.0, float(y[0])))
        r = max(0.0, min(1.0, float(y[1])))
        return (t, r)
