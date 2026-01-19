import os, re, inspect, warnings, moabb
import numpy                 as np
import pandas                as pd
from moabb.paradigms         import MotorImagery
from moabb.datasets          import Stieger2021
from typing                  import List, Tuple
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import SVC
from pyriemann.estimation    import Covariances
from pyriemann.tangentspace  import TangentSpace
warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# ================= CONFIG =================
DATA_DIR     = r"/home/seidi/Downloads/Stieger"
SUBJECTS     = list(range(1, 6))
SESSIONS_USE = [1, 3, 5, 7]
INTERVAL     = [0.5, 2.5]
RESAMPLE_HZ  = 128
FMIN, FMAX   = 0.5, 45.0
TARGET_21    = ["FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4"]

WIN_SEC      = 1.0
STEP_SEC     = 0.1
TRAIN_WIN    = (0.5, 2.5)

MAX_SPLITS   = 5
SEED         = 42

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_CSV      = os.path.join(SCRIPT_DIR, "timecurves_riem_svmrbf_stieger2021.csv")

# ================= DATASET =================
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

# ================= HELPERS =================
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
    sess = set()
    for p in ds.data_path(subject):
        sess.add(int(os.path.basename(p).split("_")[-1].split(".")[0]))
    return sorted(sess)

def make_windows(n_times: int, sfreq: float, t0: float, win_sec: float, step_sec: float) -> List[Tuple[int, int, float]]:
    win, step = int(win_sec * sfreq), max(1, int(step_sec * sfreq))
    if win < 2:
        raise ValueError("WIN_SEC muito pequeno.")
    out       = []
    for a in range(0, n_times - win + 1, step):
        b     = a + win
        t     = t0 + ((a + b - 1) / 2) / sfreq
        out.append((a, b, float(t)))
    return out

def safe_stratified_cv(y: np.ndarray, max_splits: int, seed: int):
    _, cnt = np.unique(y, return_counts=True)
    if cnt.min() < 2:
        return None
    return StratifiedKFold(n_splits=min(max_splits, int(cnt.min())), shuffle=True, random_state=seed)

def to_sample(t: float, sfreq: float, t0: float) -> int:
    return int(round((t - t0) * sfreq))

def clamp(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))

# ================= MAIN =================
if __name__ == "__main__":

    rows_out = []
    for subj in SUBJECTS:
        print(f"\n=== Subject {subj} ===")
        ds_list                  = Stieger2021Local(interval=INTERVAL,sessions=SESSIONS_USE,data_dir=DATA_DIR)
        ds_list.subject_list     = [subj]
        avail                    = list_available_sessions(ds_list, subj)
        use_sessions             = sorted(set(avail) & set(SESSIONS_USE))

        for ses in use_sessions:
            print(f"\n--- Subject {subj} | Session {ses} ---")
            ds                  = Stieger2021Local(interval=INTERVAL,sessions=[ses],data_dir=DATA_DIR)
            ds.subject_list     = [subj]
            paradigm            = MotorImagery(n_classes=4, resample=RESAMPLE_HZ,fmin=FMIN, fmax=FMAX, channels=TARGET_21)

            try:
                X, y, _         = paradigm.get_data(ds, subjects=[subj])
                if X is None or X.size == 0:
                    continue

                n_trials, n_ch, n_times = X.shape
                cv              = safe_stratified_cv(y, MAX_SPLITS, SEED)
                if cv is None:
                    continue

                windows         = make_windows(n_times, RESAMPLE_HZ, INTERVAL[0], WIN_SEC, STEP_SEC)
                tr_a            = clamp(to_sample(TRAIN_WIN[0], RESAMPLE_HZ, INTERVAL[0]), 0, n_times - 2)
                tr_b            = clamp(to_sample(TRAIN_WIN[1], RESAMPLE_HZ, INTERVAL[0]), tr_a + 2, n_times)

                for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
                    pipe = build_pipeline()
                    pipe.fit(X[tr_idx, :, tr_a:tr_b], y[tr_idx])

                    for a, b, t_center in windows:
                        Xte       = X[te_idx, :, a:b]
                        proba     = pipe.predict_proba(Xte)
                        classes   = pipe.classes_
                        y_pred    = classes[np.argmax(proba, axis=1)]
                        acc       = float(np.mean(y_pred == y[te_idx]))

                        rows_out.append({
                            "subject"           : subj,
                            "session"           : ses,
                            "fold"              : fold,
                            "time_center_s"     : t_center,
                            "window_start_s"    : INTERVAL[0] + a / RESAMPLE_HZ,
                            "window_stop_s"     : INTERVAL[0] + b / RESAMPLE_HZ,
                            "acc"               : acc,
                            "pipeline"          : "riem+svm_rbf_trainfixed_testsliding",
                            "n_trials_session"  : int(n_trials),
                            "n_trials_test"     : int(len(te_idx)),
                            "n_channels"        : int(n_ch),
                            "win_sec"           : WIN_SEC,
                            "step_sec"          : STEP_SEC,
                            "train_win_start_s" : TRAIN_WIN[0],
                            "train_win_stop_s"  : TRAIN_WIN[1],
                        })

            except Exception as e:
                print(f"[ERROR] subj={subj} ses={ses}: {e}")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n[FINAL] Salvo em: {OUT_CSV}")
    print(df.head())