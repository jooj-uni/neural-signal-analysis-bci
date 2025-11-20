import os, re, inspect, warnings, mne, moabb
import numpy  as np
import pandas as pd

from typing                        import List, Dict
from scipy.signal                  import correlate
from moabb.paradigms               import MotorImagery
from moabb.evaluations             import WithinSessionEvaluation
from moabb.datasets                import Stieger2021
from mne.decoding                  import CSP
from sklearn.pipeline              import Pipeline, make_pipeline
from sklearn.base                  import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model          import LogisticRegression
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import TangentSpace

warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# Configuração
DATA_DIR     = r"D:\dados_stieger"
SUBJECTS     = [
    1,2,3,4,5,6,7,8,9,10,
    11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
    36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62
]
SESSIONS_USE = [1, 3, 5, 7]
INTERVAL     = [0.5, 2.5]
RESAMPLE_HZ  = 128
FMIN, FMAX   = 0.5, 50.0
MIN_PRESENT  = 11
OUT_CSV      = "results_stieger2021.csv"
TARGET_21    = [
    "FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4",
    "C6","CP3","CP1","CPz","CP2","CP4"
]

# Dataset local
class Stieger2021Local(Stieger2021):
    """Lê arquivos .mat diretamente de DATA_DIR."""

    def __init__(self, interval=[0, 3], sessions=None,
                 fix_bads=True, data_dir=None):
        sig = inspect.signature(super().__init__)
        if "fix_bads" in sig.parameters:
            super().__init__(interval=interval,
                             sessions=sessions,
                             fix_bads=fix_bads)
        else:
            super().__init__(interval=interval, sessions=sessions)
        self.data_dir = data_dir

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return []

        files = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".mat"):
                continue
            m_ses = re.search(r"Session_(\d+)", fname)
            if m_ses is None:
                continue
            ses = int(m_ses.group(1))
            if self.sessions is None or ses in self.sessions:
                files.append(os.path.join(self.data_dir, fname))

        return sorted(
            files,
            key=lambda f: int(
                re.search(r"Session_(\d+)", os.path.basename(f)).group(1)
            )
        )

# Pré-processadores
class CAR(BaseEstimator, TransformerMixin):
    """Referência comum por época: X <- X - média nos canais."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X - X.mean(axis=1, keepdims=True)


class MaxCrossCorr(BaseEstimator, TransformerMixin):
    """Matriz por época com o MÁXIMO da correlação cruzada normalizada entre 
    canais. Garante SPD via ‘floor’ de autovalores."""
    def __init__(self, reg=1e-6, max_lag=None, eps_spd=1e-6):
        self.reg     = reg
        self.max_lag = max_lag
        self.eps_spd = eps_spd

    def fit(self, X, y=None):
        self.n_channels_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_epochs, n_ch, n_times = X.shape
        out = np.empty((n_epochs, n_ch, n_ch), float)

        for ep in range(n_epochs):
            M = np.zeros((n_ch, n_ch), float)
            for i in range(n_ch):
                xi = X[ep, i]
                xi = (xi - xi.mean()) / (xi.std() + 1e-12)
                for j in range(i, n_ch):
                    xj = X[ep, j]
                    xj = (xj - xj.mean()) / (xj.std() + 1e-12)

                    corr = correlate(xi, xj, mode="full")
                    if self.max_lag is not None:
                        mid = corr.size // 2
                        lo  = max(0, mid - self.max_lag)
                        hi  = min(corr.size, mid + self.max_lag + 1)
                        corr = corr[lo:hi]

                    val     = np.max(np.abs(corr)) / n_times
                    M[i, j] = val
                    M[j, i] = val

            np.fill_diagonal(M, 1.0)
            M += self.reg * np.eye(n_ch)

            A    = 0.5 * (M + M.T)
            w, V = np.linalg.eigh(A)
            w    = np.clip(w, self.eps_spd, None)
            out[ep] = (V * w) @ V.T
        return out


class LaplacianMNE(BaseEstimator, TransformerMixin):
    """Aplica um filtro laplaciano espacial linear: X' = L @ X."""
    def __init__(self, lap_matrix):
        self.lap_matrix = np.asarray(lap_matrix)

    def fit(self, X, y=None):
        if X.shape[1] != self.lap_matrix.shape[0]:
            raise ValueError(
                f"LaplacianMNE: {X.shape[1]} canais, "
                f"mas lap_matrix é {self.lap_matrix.shape}"
            )
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.einsum("ij,etj->eti", self.lap_matrix, X)

# Pipelines
def pipeline_names() -> List[str]:
    """Nomes das pipelines (usados também para linhas NaN)."""
    return [
        "csp+lda","csp+lr",
        "riem+lda","riem+lr",
        # "maxcorr+lda","maxcorr+lr",
        "lap_riem+lda","lap_riem+lr",
        # "lap_maxcorr+lda","lap_maxcorr+lr",
    ]


def build_pipelines(lap_matrix) -> Dict[str, Pipeline]:
    csp      = CSP(n_components=8, log=True, norm_trace=False)
    riem_cov = Covariances(estimator="oas")
    riem_ts  = TangentSpace(metric="riemann")
    maxcorr  = MaxCrossCorr(reg=1e-6, max_lag=None, eps_spd=1e-6)
    car      = CAR()
    lap      = LaplacianMNE(lap_matrix)

    return {
        # CAR
        "csp+lda"        : make_pipeline(car, csp, LDA()),
        "csp+lr"         : make_pipeline(car, csp, LogisticRegression(max_iter=500, solver="lbfgs")),
        "riem+lda"       : Pipeline([("car",car),("cov",riem_cov),("tgs",riem_ts),("clf",LDA())]),
        "riem+lr"        : Pipeline([("car",car),("cov",riem_cov),("tgs",riem_ts),("clf",LogisticRegression(max_iter=500, solver="lbfgs"))
        ]),
        # "maxcorr+lda"    : Pipeline([  ("car",car),("mcorr",maxcorr),("tgs",riem_ts),("clf",LDA())
        # ]),
        # "maxcorr+lr"     : Pipeline([
        #     ("car",car),("mcorr",maxcorr),("tgs",riem_ts),
        #     ("clf",LogisticRegression(max_iter=500, solver="lbfgs"))
        # ]),
        # Laplacian MNE
        "lap_riem+lda"   : Pipeline([("lap",lap),("cov",riem_cov),("tgs",riem_ts),("clf",LDA())]),
        "lap_riem+lr"    : Pipeline([("lap",lap),("cov",riem_cov),("tgs",riem_ts), ("clf",LogisticRegression(max_iter=500, solver="lbfgs"))]),
        # "lap_maxcorr+lda": Pipeline([("lap",lap),("mcorr",maxcorr),("tgs",riem_ts),("clf",LDA())]),
        # "lap_maxcorr+lr" : Pipeline([("lap",lap),("mcorr",maxcorr),("tgs",riem_ts), ("clf",LogisticRegression(max_iter=500, solver="lbfgs"))]),
    }

# Utilitários
def available_sessions(ds: Stieger2021Local, subject: int) -> List[int]:
    paths = ds.data_path(subject)
    return sorted({int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in paths})


def channels_for_session(ds_one: Stieger2021Local, subject: int, session: int,  target21: List[str]) -> List[str]:
    try:
        subj_dict = ds_one._get_single_subject_data(subject)
    except Exception as e:
        print(
            f"Sujeito {subject} | Sessão {session}: erro ao carregar dados "
            f"(_get_single_subject_data): {e}"
        )
        return []

    runs = subj_dict.get(str(session), {})
    if not runs:
        return []

    present = None
    for raw in runs.values():
        eeg = set(raw.copy().pick("eeg").info["ch_names"])
        present = eeg if present is None else (present & eeg)

    if not present:
        return []
    return [ch for ch in target21 if ch in present]


def compute_laplacian_matrix_mne(ds_one: Stieger2021Local, subject: int,
                                 session: int, chs: List[str]) -> np.ndarray:
    """Constrói matriz laplaciana usando vizinhança real dos canais (MNE)."""
    subj_dict = ds_one._get_single_subject_data(subject)
    runs      = subj_dict.get(str(session), {})
    if not runs:
        raise RuntimeError(f"Nenhum run para sujeito {subject}, sessão {session}")

    raw        = next(iter(runs.values())).copy()
    raw.pick(chs)
    picks      = mne.pick_channels(raw.info["ch_names"], include=chs)
    lap_matrix = mne.channels.make_laplacian_matrix(raw.info, picks=picks)
    return lap_matrix


def nan_rows(subject: int, session: int, chs: List[str]) -> pd.DataFrame:
    cols = [
        "score","time","samples","subject","session","fold",
        "n_sessions","dataset","pipeline","run","channels_used",
    ]
    rows = []
    for pname in pipeline_names():
        rows.append({
            "score"        : np.nan,
            "time"         : np.nan,
            "samples"      : np.nan,
            "subject"      : subject,
            "session"      : session,
            "fold"         : np.nan,
            "n_sessions"   : np.nan,
            "dataset"      : "Stieger2021",
            "pipeline"     : pname,
            "run"          : np.nan,
            "channels_used": ",".join(chs) if chs else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)

# Main
if __name__ == "__main__":
    all_results = []

    for subj in SUBJECTS:
        print(f"\n=== Sujeito {subj} ===")
        ds_listing              = Stieger2021Local(
            interval=INTERVAL,
            sessions=SESSIONS_USE,
            data_dir=DATA_DIR,
        )
        ds_listing.subject_list = [subj]
        avail                   = available_sessions(ds_listing, subj)

        for s in sorted(set(SESSIONS_USE) & set(avail)):
            ds_one              = Stieger2021Local(
                interval=INTERVAL,
                sessions=[s],
                data_dir=DATA_DIR,
            )
            ds_one.subject_list = [subj]

            try:
                chs = channels_for_session(ds_one, subj, s, TARGET_21)
            except MemoryError:
                print(
                    f"Sujeito {subj} | Sessão {s}: MemoryError ao montar canais. NaN."
                )
                all_results.append(nan_rows(subj, s, []))
                continue
            except Exception as e:
                print(
                    f"Sujeito {subj} | Sessão {s}: erro inesperado ao obter canais "
                    f"({e}). NaN."
                )
                all_results.append(nan_rows(subj, s, []))
                continue

            if len(chs) < MIN_PRESENT:
                print(
                    f"Sujeito {subj} | Sessão {s}: {len(chs)}/21 → insuficiente. NaN."
                )
                all_results.append(nan_rows(subj, s, chs))
                continue

            print(f"Sujeito {subj} | Sessão {s}: usando {len(chs)} canais: {chs}")

            try:
                lap_matrix = compute_laplacian_matrix_mne(ds_one, subj, s, chs)
            except Exception as e:
                print(
                    f"Sujeito {subj} | Sessão {s}: erro no laplacian ({e}). NaN."
                )
                all_results.append(nan_rows(subj, s, chs))
                continue

            pipelines = build_pipelines(lap_matrix)

            paradigm   = MotorImagery(
                n_classes=4,
                resample=RESAMPLE_HZ,
                fmin=FMIN,
                fmax=FMAX,
                channels=chs,
            )
            evaluation = WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[ds_one],
                overwrite=True,
                hdf5_path=None,
            )

            try:
                res                  = evaluation.process(pipelines)
                res["subject"]       = res["subject"].astype(int)
                res["session"]       = res["session"].astype(int)
                res["channels_used"] = ",".join(chs)
                all_results.append(res)

            except MemoryError:
                print(
                    f"Sujeito {subj} | Sessão {s}: MemoryError na avaliação. NaN."
                )
                all_results.append(nan_rows(subj, s, chs))

            except Exception as e:
                print(
                    f"Sujeito {subj} | Sessão {s}: erro na avaliação ({e}). NaN."
                )
                all_results.append(nan_rows(subj, s, chs))

    if not all_results:
        raise RuntimeError(
            "Nada foi gerado. Verifique DATA_DIR / nomes de arquivos / filtros."
        )

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSalvo em: {os.path.join(os.getcwd(), OUT_CSV)}")
    print(results.head())
