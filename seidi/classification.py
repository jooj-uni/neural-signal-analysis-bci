import os, re, inspect, warnings, time
import numpy as np
import pandas as pd
import moabb

from typing import List
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

# ---- Config ----
DATA_DIR          = r"D:\dados_stieger"
SUBJECTS          = list(range(1, 5))
SESSIONS_USE      = [1, 2, 3, 4, 5, 6, 7]
INTERVAL          = [0.5, 2.5]
RESAMPLE_HZ       = 128
FMIN, FMAX        = 0.5, 45.0
OUT_CSV           = "results_stieger2021_full.csv"

COH_FMIN, COH_FMAX = 8.0, 30.0
COH_NPERSEG        = None
COH_NOVERLAP       = None

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

COH_PIPELINE_NAMES = [
    "cohpeak+lda","cohpeak+lr",
    "cohpeak+svm_lin","cohpeak+svm_rbf"
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
# Transformers
# ============================================================
class CAR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)

class CoherencePeakFeatures(BaseEstimator, TransformerMixin):
    """
    Para cada trial:
      - coerência entre todos os pares
      - pico (máximo) na banda vira feature do par
    Retorna: (n_trials, n_pairs)
    """
    def __init__(self, sfreq, fmin=8.0, fmax=30.0, nperseg=None, noverlap=None):
        self.sfreq = float(sfreq)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.nperseg = nperseg
        self.noverlap = noverlap

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _pairs(n_ch):
        return [(i, j) for i in range(n_ch - 1) for j in range(i + 1, n_ch)]

    def transform(self, X):
        X = np.asarray(X)
        n_trials, n_ch, _ = X.shape
        pairs = self._pairs(n_ch)
        feats = np.zeros((n_trials, len(pairs)), dtype=np.float32)

        for t in range(n_trials):
            Xt = X[t]
            for k, (i, j) in enumerate(pairs):
                f, Cxy = signal.coherence(
                    Xt[i], Xt[j],
                    fs=self.sfreq,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    detrend="constant",
                )
                band = (f >= self.fmin) & (f <= self.fmax)
                feats[t, k] = float(np.nanmax(Cxy[band])) if np.any(band) else 0.0

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

# ============================================================
# Pipelines
# ============================================================
def build_base_pipelines():
    car = CAR()
    csp = CSP(n_components=8, log=True, norm_trace=False)

    cov = Covariances("oas")
    ts  = TangentSpace("riemann")

    return {
        "csp+lda": make_pipeline(car, csp, LDA()),
        "csp+lr" : make_pipeline(car, csp, LogisticRegression(max_iter=500)),

        "riem+lda": Pipeline([("car",car),("cov",cov),("ts",ts),("clf",LDA())]),
        "riem+lr":  Pipeline([("car",car),("cov",cov),("ts",ts),
                              ("clf",LogisticRegression(max_iter=500))]),
        "riem+svm_lin": Pipeline([("car",car),("cov",cov),("ts",ts),
                                  ("sc",StandardScaler()),
                                  ("clf",SVC(kernel="linear"))]),
        "riem+svm_rbf": Pipeline([("car",car),("cov",cov),("ts",ts),
                                  ("sc",StandardScaler()),
                                  ("clf",SVC(kernel="rbf", gamma="scale"))]),
    }

def build_coh_feature_extractor():
    return Pipeline([
        ("car", CAR()),
        ("coh", CoherencePeakFeatures(
            sfreq=RESAMPLE_HZ,
            fmin=COH_FMIN,
            fmax=COH_FMAX,
            nperseg=COH_NPERSEG,
            noverlap=COH_NOVERLAP
        ))
    ])

def build_coh_classifiers():
    return {
        "cohpeak+lda": LDA(),
        "cohpeak+lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500)
        ),
        "cohpeak+svm_lin": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear")
        ),
        "cohpeak+svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale")
        ),
    }

# ============================================================
# Main
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

        for s in sorted(set(avail) & set(SESSIONS_USE)):
            print(f"\n--- Session {s} ---")

            ds = Stieger2021Local(INTERVAL, [s], data_dir=DATA_DIR)
            ds.subject_list = [subj]

            paradigm = MotorImagery(
                n_classes=4,
                resample=RESAMPLE_HZ,
                fmin=FMIN,
                fmax=FMAX,
                channels=TARGET_21
            )

            # ---- RAW (MOABB) ----
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

            # ---- Coherence-peak (features 1x + CV rápido) ----
            try:
                X, y, _ = paradigm.get_data(ds, subjects=[subj])
                uniq, cnt = np.unique(y, return_counts=True)
                if cnt.min() < 2:
                    raise ValueError("Classes insuficientes para CV")

                cv = StratifiedKFold(
                    n_splits=min(5, int(cnt.min())),
                    shuffle=True, random_state=42
                )

                t0 = time.perf_counter()
                feat_pipe = build_coh_feature_extractor()
                X_feat = feat_pipe.fit_transform(X)  # <- calcula coerência 1x aqui
                feat_time = time.perf_counter() - t0
                print(f"[COH] features: shape={X_feat.shape} | t={feat_time:.2f}s")

                rows = []
                clfs = build_coh_classifiers()

                for name, clf in clfs.items():
                    print(f"[COH] {name}")
                    t1 = time.perf_counter()
                    scores = cross_val_score(clf, X_feat, y, cv=cv, n_jobs=1)
                    t_clf = time.perf_counter() - t1

                    rows.append({
                        "score": float(scores.mean()),
                        "time": float(feat_time + t_clf),   # tempo total aproximado (feat + clf)
                        "samples": int(len(y)),
                        "subject": int(subj),
                        "session": int(s),
                        "fold": np.nan,
                        "n_sessions": 1,
                        "dataset": "Stieger2021",
                        "pipeline": name,
                        "run": np.nan,
                        "channels_used": ",".join(TARGET_21),
                        "coh_fmin": float(COH_FMIN),
                        "coh_fmax": float(COH_FMAX),
                        "coh_nperseg": (np.nan if COH_NPERSEG is None else float(COH_NPERSEG)),
                        "coh_noverlap": (np.nan if COH_NOVERLAP is None else float(COH_NOVERLAP)),
                        "coh_n_features": int(X_feat.shape[1]),
                    })

                all_results.append(pd.DataFrame(rows))

            except Exception as e:
                print(f"[COH ERROR] {e}")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_CSV, index=False)
    print("\n[FINAL] Saved:", OUT_CSV)
    print(results.head())
