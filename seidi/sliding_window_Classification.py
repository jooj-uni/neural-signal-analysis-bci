import os, re, inspect, warnings
import numpy as np
import pandas as pd
import moabb

from typing import List, Tuple
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

# Config
DATA_DIR     = r"D:\dados_stieger"
SUBJECTS     = list(range(1, 10))
SESSIONS_USE = [1, 2, 3, 4, 5, 6, 7]

INTERVAL    = [0.5, 2.5]
RESAMPLE_HZ = 128
FMIN, FMAX  = 0.5, 45.0

TARGET_21 = [
    "FC3","FC1","FCz","FC2","FC4",
    "C5","C3","C1","Cz","C2","C4","C6",
    "CP3","CP1","CPz","CP2","CP4"
]

WIN_SEC  = 1.0
STEP_SEC = 0.1

MAX_SPLITS = 5
SEED       = 42

OUT_CSV = "timecurves_riem_svmrbf_stieger2021.csv"


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


class CAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("car", CAR()),
        ("cov", Covariances(estimator="oas")),
        ("ts",  TangentSpace(metric="riemann")),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", gamma="scale", C=1.0, probability=True)),
    ])


def list_available_sessions(ds: Stieger2021Local, subject: int) -> List[int]:
    paths = ds.data_path(subject)
    sess = set()
    for p in paths:
        ses = int(os.path.basename(p).split("_")[-1].split(".")[0])
        sess.add(ses)
    return sorted(sess)


def make_windows(n_times: int, sfreq: float, t0: float,
                 win_sec: float, step_sec: float) -> List[Tuple[int, int, float]]:
    win = int(round(win_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if win < 2:
        raise ValueError("WIN_SEC muito pequeno para o sfreq.")
    if step < 1:
        step = 1

    out = []
    for start in range(0, n_times - win + 1, step):
        stop = start + win
        center = (start + stop - 1) / 2.0
        out.append((start, stop, float(t0 + center / sfreq)))
    return out


def safe_stratified_cv(y: np.ndarray, max_splits: int, seed: int):
    _, cnt = np.unique(y, return_counts=True)
    min_cnt = int(cnt.min())
    if min_cnt < 2:
        return None
    n_splits = min(max_splits, min_cnt)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


if __name__ == "__main__":

    rows_out = []

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj} ===")

        ds_list = Stieger2021Local(interval=INTERVAL, sessions=SESSIONS_USE, data_dir=DATA_DIR)
        ds_list.subject_list = [subj]
        avail = list_available_sessions(ds_list, subj)
        use_sessions = sorted(set(avail) & set(SESSIONS_USE))
        print(f"[INFO] Available sessions: {avail} | Using: {use_sessions}")

        for ses in use_sessions:
            print(f"\n--- Subject {subj} | Session {ses} ---")

            ds = Stieger2021Local(interval=INTERVAL, sessions=[ses], data_dir=DATA_DIR)
            ds.subject_list = [subj]

            paradigm = MotorImagery(
                n_classes=4,
                resample=RESAMPLE_HZ,
                fmin=FMIN,
                fmax=FMAX,
                channels=TARGET_21
            )

            try:
                X, y, _ = paradigm.get_data(ds, subjects=[subj])
                if X is None or X.shape[0] == 0:
                    print("[WARN] Sem épocas. Pulando.")
                    continue

                n_trials, n_ch, n_times = X.shape
                classes_all = np.unique(y)
                print(f"[INFO] X shape={X.shape} | classes={classes_all}")

                cv = safe_stratified_cv(y, MAX_SPLITS, SEED)
                if cv is None:
                    print("[WARN] Classes insuficientes para CV estratificada. Pulando.")
                    continue

                windows = make_windows(
                    n_times=n_times,
                    sfreq=float(RESAMPLE_HZ),
                    t0=float(INTERVAL[0]),
                    win_sec=float(WIN_SEC),
                    step_sec=float(STEP_SEC),
                )
                print(f"[INFO] Windows: {len(windows)} (WIN={WIN_SEC}s STEP={STEP_SEC}s)")

                for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
                    y_tr = y[tr_idx]
                    _, cnt_tr = np.unique(y_tr, return_counts=True)
                    if cnt_tr.min() < 2:
                        print(f"[WARN] Fold {fold}: treino insuficiente em alguma classe. Skip fold.")
                        continue

                    for w_id, (a, b, t_center) in enumerate(windows):
                        Xtr_w = X[tr_idx, :, a:b]
                        Xte_w = X[te_idx, :, a:b]

                        pipe = build_pipeline()
                        try:
                            pipe.fit(Xtr_w, y_tr)
                            proba = pipe.predict_proba(Xte_w)
                            classes = pipe.classes_

                            for local_i, global_trial in enumerate(te_idx):
                                y_true = y[global_trial]  # string
                                for c_i, c in enumerate(classes):
                                    rows_out.append({
                                        "subject": subj,
                                        "session": ses,
                                        "fold": fold,
                                        "trial": int(global_trial),
                                        "time_center_s": float(t_center),
                                        "window_start_s": float(INTERVAL[0] + a / RESAMPLE_HZ),
                                        "window_stop_s":  float(INTERVAL[0] + b / RESAMPLE_HZ),
                                        "y_true": y_true,          # string
                                        "class": c,                # string
                                        "proba": float(proba[local_i, c_i]),
                                        "pipeline": "riem+svm_rbf_time_resolved",
                                        "n_trials_session": int(n_trials),
                                        "n_channels": int(n_ch),
                                        "win_sec": float(WIN_SEC),
                                        "step_sec": float(STEP_SEC),
                                    })

                        except Exception as e:
                            print(f"[ERROR] subj={subj} ses={ses} fold={fold} w={w_id} t={t_center:.3f}s -> {e}")
                            continue

            except Exception as e:
                print(f"[ERROR] Falha subj={subj} ses={ses}: {e}")
                continue

    if not rows_out:
        raise RuntimeError("Nada foi gerado. Verifique DATA_DIR / sessões / parâmetros.")

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[FINAL] Salvo em: {OUT_CSV}")
    print(df.head())
