import os, re, inspect, warnings, time
import numpy as np
import pandas as pd
import moabb

from typing import List, Tuple, Dict, Any
from moabb.paradigms import MotorImagery
from moabb.datasets import Stieger2021

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# ---- Config geral ----
DATA_DIR     = r"D:\dados_stieger"
SUBJECTS     = list(range(1, 10))
SESSIONS_USE = [1, 2, 3, 4, 5, 6, 7]

INTERVAL    = [0.5, 2.5]
RESAMPLE_HZ = 128

MAX_SPLITS = 5
SEED       = 42

OUT_CSV = "grid_timecurves_riem_svmrbf_stieger2021.csv"

# ---- Grids (você pode editar à vontade) ----
WIN_LIST_SEC  = [0.25, 0.50, 0.75, 1.00]
STEP_LIST_SEC = [0.05]  # pode virar grid também, se quiser

BANDS = [
    {"name": "nofilt",   "fmin": None, "fmax": None},  # sem filtro (usa o "raw" reamostrado)
    {"name": "mu",       "fmin": 8.0,  "fmax": 13.0},
    {"name": "beta",     "fmin": 13.0, "fmax": 30.0},
    {"name": "mu_beta",  "fmin": 8.0,  "fmax": 30.0},
    {"name": "theta",    "fmin": 4.0,  "fmax": 8.0},
    {"name": "alpha",    "fmin": 8.0,  "fmax": 12.0},
    {"name": "lowgamma", "fmin": 30.0, "fmax": 45.0},
]

CHANNEL_SETS = [
    {"name": "motor_17", "chs": [
        "FC3","FC1","FCz","FC2","FC4",
        "C5","C3","C1","Cz","C2","C4","C6",
        "CP3","CP1","CPz","CP2","CP4"
    ]},
    {"name": "central_9", "chs": ["C5","C3","C1","Cz","C2","C4","C6","CP1","CP2"]},
    {"name": "csp_like_11", "chs": ["FC1","FC2","C3","C1","Cz","C2","C4","CP1","CPz","CP2","FCz"]},
]

# ============================================================
# Dataset local
# ============================================================
class Stieger2021Local(Stieger2021):
    def __init__(self, interval=[0, 3], sessions=None, fix_bads=True, data_dir=None):
        sig = inspect.signature(super().__init__)
        if "fix_bads" in sig.parameters:
            super().__init__(interval=interval, sessions=sessions, fix_bads=fix_bads)
        else:
            super().__init__(interval=interval, sessions=sessions)
        self.data_dir = data_dir

    def data_path(self, subject, **kwargs):
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return []
        files = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".mat"):
                continue
            m = re.match(r"S(\d+)_Session_(\d+)\.mat", fname)
            if m and int(m.group(1)) == subject:
                ses = int(m.group(2))
                if self.sessions is None or ses in self.sessions:
                    files.append(os.path.join(self.data_dir, fname))
        return sorted(files)

# ============================================================
# Transform
# ============================================================
class CAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)

# ============================================================
# Pipeline
# ============================================================
def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("car", CAR()),
        ("cov", Covariances(estimator="oas")),
        ("ts",  TangentSpace(metric="riemann")),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", gamma="scale", C=C, probability=True)),
    ])

# ============================================================
# Helpers
# ============================================================
def list_available_sessions(ds: Stieger2021Local, subject: int) -> List[int]:
    paths = ds.data_path(subject)
    out = set()
    for p in paths:
        out.add(int(os.path.basename(p).split("_")[-1].split(".")[0]))
    return sorted(out)

def safe_stratified_cv(y: np.ndarray, max_splits: int, seed: int):
    _, cnt = np.unique(y, return_counts=True)
    min_cnt = int(cnt.min())
    if min_cnt < 2:
        return None
    n_splits = min(max_splits, min_cnt)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def make_windows(n_times: int, sfreq: float, t0: float, win_sec: float, step_sec: float):
    win = int(round(win_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if win < 2:
        return []
    if step < 1:
        step = 1
    out = []
    for start in range(0, n_times - win + 1, step):
        stop = start + win
        center = (start + stop - 1) / 2.0
        out.append((start, stop, float(t0 + center / sfreq)))
    return out

def timecurve_score_mean_proba(
    X: np.ndarray,
    y: np.ndarray,
    win_sec: float,
    step_sec: float,
    interval_t0: float,
    sfreq: float,
    cv,
    pipe: Pipeline
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Retorna:
      - t_centers (n_windows,)
      - acc_per_window (n_windows,)  (accuracy média nos folds)
      - classes (lista de labels)
    """
    n_trials, _, n_times = X.shape
    windows = make_windows(n_times, sfreq, interval_t0, win_sec, step_sec)
    if not windows:
        return np.array([]), np.array([]), []

    t_centers = np.array([w[2] for w in windows], float)
    acc = np.zeros(len(windows), float)

    for w_id, (a, b, _) in enumerate(windows):
        fold_scores = []
        for tr_idx, te_idx in cv.split(X, y):
            Xtr = X[tr_idx, :, a:b]
            Xte = X[te_idx, :, a:b]
            ytr = y[tr_idx]
            yte = y[te_idx]

            pipe_fold = build_pipeline(C=pipe.named_steps["clf"].C)
            pipe_fold.fit(Xtr, ytr)
            yhat = pipe_fold.predict(Xte)
            fold_scores.append(float(np.mean(yhat == yte)))

        acc[w_id] = float(np.mean(fold_scores))

    classes = list(np.unique(y))
    return t_centers, acc, classes

# ============================================================
# Main (grid)
# ============================================================
if __name__ == "__main__":

    rows = []
    run_id = 0

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj} ===")

        ds_list = Stieger2021Local(interval=INTERVAL, sessions=SESSIONS_USE, data_dir=DATA_DIR)
        ds_list.subject_list = [subj]
        avail = list_available_sessions(ds_list, subj)
        use_sessions = sorted(set(avail) & set(SESSIONS_USE))
        print(f"[INFO] Sessions: {use_sessions}")

        for ses in use_sessions:
            print(f"\n--- Subject {subj} | Session {ses} ---")

            ds = Stieger2021Local(interval=INTERVAL, sessions=[ses], data_dir=DATA_DIR)
            ds.subject_list = [subj]

            for band in BANDS:
                for chset in CHANNEL_SETS:
                    fmin = band["fmin"]
                    fmax = band["fmax"]
                    chs  = chset["chs"]

                    paradigm = MotorImagery(
                        n_classes=4,
                        resample=RESAMPLE_HZ,
                        channels=chs,
                        fmin=(None if fmin is None else float(fmin)),
                        fmax=(None if fmax is None else float(fmax)),
                    )

                    try:
                        X, y, _ = paradigm.get_data(ds, subjects=[subj])
                        if X is None or X.shape[0] == 0:
                            print(f"[WARN] vazio band={band['name']} ch={chset['name']}")
                            continue

                        cv = safe_stratified_cv(y, MAX_SPLITS, SEED)
                        if cv is None:
                            print(f"[WARN] sem CV band={band['name']} ch={chset['name']}")
                            continue

                        for win_sec in WIN_LIST_SEC:
                            for step_sec in STEP_LIST_SEC:
                                t0 = time.perf_counter()
                                run_id += 1

                                pipe = build_pipeline(C=1.0)

                                t_centers, acc, _ = timecurve_score_mean_proba(
                                    X=X, y=y,
                                    win_sec=float(win_sec),
                                    step_sec=float(step_sec),
                                    interval_t0=float(INTERVAL[0]),
                                    sfreq=float(RESAMPLE_HZ),
                                    cv=cv,
                                    pipe=pipe,
                                )

                                if t_centers.size == 0:
                                    continue

                                elapsed = time.perf_counter() - t0

                                # salva por janela (só accuracy média por tempo)
                                for w_i in range(len(t_centers)):
                                    rows.append({
                                        "run_id": int(run_id),
                                        "subject": int(subj),
                                        "session": int(ses),
                                        "band": band["name"],
                                        "fmin": (np.nan if fmin is None else float(fmin)),
                                        "fmax": (np.nan if fmax is None else float(fmax)),
                                        "channel_set": chset["name"],
                                        "n_channels": int(len(chs)),
                                        "win_sec": float(win_sec),
                                        "step_sec": float(step_sec),
                                        "time_center_s": float(t_centers[w_i]),
                                        "acc_mean": float(acc[w_i]),
                                        "n_trials": int(X.shape[0]),
                                        "n_splits": int(cv.get_n_splits()),
                                        "elapsed_s": float(elapsed),
                                        "pipeline": "riem+svm_rbf_time_resolved",
                                    })

                                print(
                                    f"[OK] band={band['name']} ch={chset['name']} "
                                    f"win={win_sec:.2f}s step={step_sec:.2f}s "
                                    f"-> mean(acc)={acc.mean():.3f}"
                                )

                    except Exception as e:
                        print(f"[ERROR] subj={subj} ses={ses} band={band['name']} ch={chset['name']} -> {e}")
                        continue

    if not rows:
        raise RuntimeError("Nada foi gerado. Ajuste grids / DATA_DIR / canais / bandas.")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[FINAL] Salvo em: {OUT_CSV}")
    print(df.head())
