# classification.py
import os, re, inspect, warnings, time
import numpy as np
import pandas as pd
import moabb

# ============================================================
# 0) Cache dirs (evita erro do MNE/MOABB) - antes de MNE/MOABB internos
# ============================================================
MNE_DATA_DIR  = r"D:\dados_stieger"
os.makedirs(MNE_DATA_DIR, exist_ok=True)
os.environ["MNE_DATA"] = MNE_DATA_DIR

# ============================================================
# Imports dependentes
# ============================================================
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
# Config
# ============================================================
DATA_DIR          = r"D:\dados_stieger"
SUBJECTS          = list(range(1, 64))
SESSIONS_USE      = [1,2,3,4,5,6,7]
INTERVAL          = [0.5, 2.5]
RESAMPLE_HZ       = 128
FMIN, FMAX        = 0.5, 45.0
OUT_CSV           = "results_stieger2021_full.csv"

# Coerência: pico na banda
COH_FMIN, COH_FMAX = 8.0, 30.0

# >>> solução simples/robusta: fixe Welch <<<
# 2s de época em 128 Hz -> 256 amostras. Use nperseg=128 (1s), overlap=64.
COH_NPERSEG  = 128
COH_NOVERLAP = 32

TARGET_21 = [
    "FC3","FC1","FCz","FC2","FC4",
    "C5","C3","C1","Cz","C2","C4","C6",
    "CP3","CP1","CPz","CP2","CP4"
]

# ============================================================
# Dataset local
# ============================================================
class Stieger2021Local(Stieger2021):
    """
    Lê .mat locais no formato: S{subject}_Session_{session}.mat
    """
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
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        X = np.asarray(X)
        # X: (n_trials, n_ch, n_times)
        return X - X.mean(axis=1, keepdims=True)

class CoherencePeakFeatures(BaseEstimator, TransformerMixin):
    """
    Solução simples:
      - calcula coerência (Welch) para pares não-redundantes (i<j)
      - feature = max(coerência) na banda [fmin, fmax]
    Retorna: (n_trials, n_pairs)

    Observação: não calcula pares redundantes (não usa i==j e não repete j<i).
    """
    def __init__(self, sfreq, fmin=8.0, fmax=30.0, nperseg=128, noverlap=64):
        self.sfreq = float(sfreq)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)

    @staticmethod
    def _pairs(n_ch):
        return [(i, j) for i in range(n_ch - 1) for j in range(i + 1, n_ch)]

    def fit(self, X, y=None):
        # cache dos pares
        X = np.asarray(X)
        self._n_ch_ = X.shape[1]
        self._pairs_ = self._pairs(self._n_ch_)
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_trials, n_ch, n_times = X.shape
        if n_ch != getattr(self, "_n_ch_", n_ch):
            self._pairs_ = self._pairs(n_ch)

        # nperseg não pode ser maior que n_times
        nperseg = min(self.nperseg, n_times)
        noverlap = min(self.noverlap, max(0, nperseg - 1))

        pairs = self._pairs_
        feats = np.zeros((n_trials, len(pairs)), dtype=np.float32)

        for t in range(n_trials):
            Xt = X[t]
            for k, (i, j) in enumerate(pairs):
                f, Cxy = signal.coherence(
                    Xt[i], Xt[j],
                    fs=self.sfreq,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="constant",
                )
                band = (f >= self.fmin) & (f <= self.fmax)
                if np.any(band):
                    v = np.nanmax(Cxy[band])
                    feats[t, k] = float(v) if np.isfinite(v) else 0.0
                else:
                    feats[t, k] = 0.0

        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# Pipelines RAW (MOABB)
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

# ============================================================
# Classificadores para coerência (simples e robustos)
# ============================================================
def build_coh_classifiers():
    # LDA comum pode ser instável; shrinkage é a alternativa simples que funciona.
    return {
        "cohpeak+lda_shrink": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LDA(solver="lsqr", shrinkage="auto")),
        ]),
        "cohpeak+lr": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]),
        "cohpeak+svm_lin": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="linear")),
        ]),
        "cohpeak+svm_rbf": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf", gamma="scale")),
        ]),
    }

# ============================================================
# Helpers
# ============================================================
def _available_sessions(ds_list, subj):
    paths = ds_list.data_path(subj)
    out = set()
    for p in paths:
        base = os.path.basename(p)
        try:
            ses = int(base.split("_")[-1].split(".")[0])
            out.add(ses)
        except Exception:
            pass
    return sorted(out)

def _make_cv(y):
    uniq, cnt = np.unique(y, return_counts=True)
    if cnt.min() < 2:
        return None
    n_splits = min(5, int(cnt.min()))
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    all_results = []

    for subj in SUBJECTS:
        print(f"\n=== Subject {subj} ===")

        ds_list = Stieger2021Local(INTERVAL, SESSIONS_USE, data_dir=DATA_DIR)
        ds_list.subject_list = [subj]
        avail = _available_sessions(ds_list, subj)

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

            # -------------------------
            # 1) RAW (MOABB Evaluation)
            # -------------------------
            try:
                eval_raw = WithinSessionEvaluation(
                    paradigm=paradigm,
                    datasets=[ds],
                    overwrite=True,
                    hdf5_path=None
                )
                res = eval_raw.process(build_base_pipelines())
                res["channels_used"] = ",".join(TARGET_21)
                all_results.append(res)
            except Exception as e:
                print(f"[RAW ERROR] {repr(e)}")

            # -----------------------------------------
            # 2) Coerência -> vetor de features -> classif.
            # -----------------------------------------
            try:
                X, y, _ = paradigm.get_data(ds, subjects=[subj])
                cv = _make_cv(y)
                if cv is None:
                    raise ValueError("Classes insuficientes para CV.")

                # feature extraction 1x
                t0 = time.perf_counter()
                feat_extractor = Pipeline([
                    ("car", CAR()),
                    ("coh", CoherencePeakFeatures(
                        sfreq=RESAMPLE_HZ,
                        fmin=COH_FMIN,
                        fmax=COH_FMAX,
                        nperseg=COH_NPERSEG,
                        noverlap=COH_NOVERLAP
                    ))
                ])
                X_feat = feat_extractor.fit_transform(X)
                feat_time = time.perf_counter() - t0

                # remove features constantes (global, fora do CV) — solução simples
                var = X_feat.var(axis=0)
                keep = var > 0
                if keep.sum() == 0:
                    raise ValueError("Coherence features degeneradas: variância zero em todas as features.")
                X_feat = X_feat[:, keep]

                print(f"[COH] X_feat: {X_feat.shape} | feat_time={feat_time:.2f}s | kept={keep.sum()}/{len(keep)}")

                rows = []
                for name, clf in build_coh_classifiers().items():
                    t1 = time.perf_counter()
                    scores = cross_val_score(clf, X_feat, y, cv=cv, n_jobs=1, error_score=np.nan)
                    t_clf = time.perf_counter() - t1

                    rows.append({
                        "score": float(np.nanmean(scores)),
                        "score_std": float(np.nanstd(scores)),
                        "time": float(feat_time + t_clf),
                        "samples": int(len(y)),
                        "subject": int(subj),
                        "session": int(s),
                        "dataset": "Stieger2021",
                        "pipeline": name,
                        "channels_used": ",".join(TARGET_21),

                        "coh_fmin": float(COH_FMIN),
                        "coh_fmax": float(COH_FMAX),
                        "coh_nperseg": int(COH_NPERSEG),
                        "coh_noverlap": int(COH_NOVERLAP),
                        "coh_n_features": int(X_feat.shape[1]),
                        "cv_n_splits": int(cv.get_n_splits()),
                    })

                all_results.append(pd.DataFrame(rows))

            except Exception as e:
                print(f"[COH ERROR] {repr(e)}")

    if len(all_results) == 0:
        raise RuntimeError("Nenhum resultado foi gerado (all_results vazio).")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_CSV, index=False)
    print("\n[FINAL] Saved:", OUT_CSV)
    print(results.head())
