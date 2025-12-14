# ============================================================
# IMPORTS
# ============================================================
import os, re, inspect, warnings, time
import numpy as np
import pandas as pd
import mne, moabb

from typing import List, Dict
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation
from moabb.datasets import Stieger2021
from mne.decoding import CSP

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal

warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# ============================================================
# CONFIGURAÇÃO
# ============================================================
DATA_DIR          = r"D:\dados_stieger"
SUBJECTS          = list(range(1, 5))
SESSIONS_USE      = [1, 3, 7]
INTERVAL          = [0.5, 2.5]
RESAMPLE_HZ       = 128
FMIN, FMAX        = 0.5, 50.0
OUT_CSV           = "results_stieger2021_full.csv"

PSD_FMIN, PSD_FMAX = 8.0, 30.0

TARGET_21 = [
    "FC3","FC1","FCz","FC2","FC4",
    "C5","C3","C1","Cz","C2","C4","C6",
    "CP3","CP1","CPz","CP2","CP4"
]

BASE_PIPELINE_NAMES = [
    "csp+lda","csp+lr",
    "riem+lda","riem+lr",
    "riem+svm_lin","riem+svm_rbf"
]

PSD_PIPELINE_NAMES = [
    "psd_riem+lda","psd_riem+lr",
    "psd_riem+svm_lin","psd_riem+svm_rbf"
]

# ============================================================
# DATASET LOCAL
# ============================================================
class Stieger2021Local(Stieger2021):
    def __init__(self, interval=[0,3], sessions=None, fix_bads=True, data_dir=None):
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
            if fname.endswith(".mat"):
                m = re.match(r"S(\d+)_Session_(\d+)\.mat", fname)
                if m and int(m.group(1)) == subject:
                    ses = int(m.group(2))
                    if self.sessions is None or ses in self.sessions:
                        files.append(os.path.join(self.data_dir, fname))
        return sorted(files)

# ============================================================
# TRANSFORMERS
# ============================================================
class CAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)


class PSDCovariances(BaseEstimator, TransformerMixin):
    """
    PSD → SPD (com projeção SPD explícita)
    """
    def __init__(self, sfreq, fmin, fmax, reg=1e-6):
        self.sfreq = sfreq
        self.fmin  = fmin
        self.fmax  = fmax
        self.reg   = reg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_trials, n_ch, _ = X.shape
        covmats = np.zeros((n_trials, n_ch, n_ch))

        for i in range(n_trials):
            freqs, psd = signal.welch(
                X[i], fs=self.sfreq, axis=-1,
                detrend="constant", scaling="density"
            )
            band = (freqs >= self.fmin) & (freqs <= self.fmax)
            P = np.log(psd[:, band] + 1e-20)

            C = np.cov(P, bias=False)
            C = 0.5 * (C + C.T)

            # projeção SPD explícita
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals[eigvals < 1e-6] = 1e-6
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            covmats[i] = C + self.reg * np.eye(n_ch)

        return covmats

# ============================================================
# PIPELINES
# ============================================================
def build_base_pipelines():
    car = CAR()
    csp = CSP(n_components=8, log=True, norm_trace=False)

    cov = Covariances("oas")
    ts  = TangentSpace("riemann")

    return {
        "csp+lda": make_pipeline(car, csp, LDA()),
        "csp+lr" : make_pipeline(car, csp, LogisticRegression(max_iter=500)),

        "riem+lda": Pipeline([
            ("car",car),("cov",cov),("ts",ts),("clf",LDA())
        ]),
        "riem+lr": Pipeline([
            ("car",car),("cov",cov),("ts",ts),
            ("clf",LogisticRegression(max_iter=500))
        ]),
        "riem+svm_lin": Pipeline([
            ("car",car),("cov",cov),("ts",ts),
            ("sc",StandardScaler()),
            ("clf",SVC(kernel="linear"))
        ]),
        "riem+svm_rbf": Pipeline([
            ("car",car),("cov",cov),("ts",ts),
            ("sc",StandardScaler()),
            ("clf",SVC(kernel="rbf", gamma="scale"))
        ]),
    }


def build_psd_riem_pipelines():
    car = CAR()
    psd = PSDCovariances(RESAMPLE_HZ, PSD_FMIN, PSD_FMAX)
    ts  = TangentSpace("riemann")

    return {
        "psd_riem+lda": Pipeline([
            ("car",car),("psd",psd),("ts",ts),("clf",LDA())
        ]),
        "psd_riem+lr": Pipeline([
            ("car",car),("psd",psd),("ts",ts),
            ("clf",LogisticRegression(max_iter=500))
        ]),
        "psd_riem+svm_lin": Pipeline([
            ("car",car),("psd",psd),("ts",ts),
            ("sc",StandardScaler()),
            ("clf",SVC(kernel="linear"))
        ]),
        "psd_riem+svm_rbf": Pipeline([
            ("car",car),("psd",psd),("ts",ts),
            ("sc",StandardScaler()),
            ("clf",SVC(kernel="rbf", gamma="scale"))
        ]),
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    all_results = []

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj} ===")

        ds_list = Stieger2021Local(INTERVAL, SESSIONS_USE, data_dir=DATA_DIR)
        ds_list.subject_list = [subj]
        avail = sorted({
            int(os.path.basename(p).split("_")[-1].split(".")[0])
            for p in ds_list.data_path(subj)
        })

        for s in set(avail) & set(SESSIONS_USE):
            print(f"\n--- Session {s} ---")

            ds = Stieger2021Local(INTERVAL, [s], data_dir=DATA_DIR)
            ds.subject_list = [subj]

            paradigm = MotorImagery(
                n_classes=4, resample=RESAMPLE_HZ,
                fmin=FMIN, fmax=FMAX, channels=TARGET_21
            )

            # ---------- RAW ----------
            eval_raw = WithinSessionEvaluation(
                paradigm=paradigm, datasets=[ds],
                overwrite=True, hdf5_path=None
            )
            try:
                res = eval_raw.process(build_base_pipelines())
                res["channels_used"] = ",".join(TARGET_21)
                all_results.append(res)
            except Exception as e:
                print(f"[RAW ERROR] {e}")

            # ---------- PSD ----------
            try:
                X, y, _ = paradigm.get_data(ds, subjects=[subj])
                uniq, cnt = np.unique(y, return_counts=True)
                if cnt.min() < 2:
                    raise ValueError("Classes insuficientes para CV")

                cv = StratifiedKFold(
                    n_splits=min(5, cnt.min()),
                    shuffle=True, random_state=42
                )

                rows = []
                for name, pipe in build_psd_riem_pipelines().items():
                    print(f"[PSD] {name}")
                    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=1)
                    rows.append({
                        "score":float(scores.mean()),
                        "time":np.nan,
                        "samples":len(y),
                        "subject":subj,
                        "session":s,
                        "fold":np.nan,
                        "n_sessions":1,
                        "dataset":"Stieger2021",
                        "pipeline":name,
                        "run":np.nan,
                        "channels_used":",".join(TARGET_21)
                    })

                all_results.append(pd.DataFrame(rows))

            except Exception as e:
                print(f"[PSD ERROR] {e}")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_CSV, index=False)
    print("\n[FINAL] Saved:", OUT_CSV)
    print(results.head())
