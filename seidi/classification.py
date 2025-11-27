import os, re, inspect, warnings, time
import numpy  as np
import pandas as pd
import mne, moabb

from typing                        import List, Dict
from moabb.paradigms               import MotorImagery
from moabb.evaluations             import WithinSessionEvaluation
from moabb.datasets                import Stieger2021
from mne.decoding                  import CSP
from sklearn.pipeline              import Pipeline, make_pipeline
from sklearn.base                  import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model          import LogisticRegression
from sklearn.model_selection       import StratifiedKFold, cross_val_score
from pyriemann.estimation          import Covariances
from pyriemann.tangentspace        import TangentSpace

warnings.filterwarnings("ignore")
moabb.set_log_level("info")

# ============================================================
# CONFIGURAÇÃO (PRINCIPAL)
# ============================================================
DATA_DIR          = r"D:\dados_stieger"
SUBJECTS          = list(range(1, 64))   # 1 a 63
SESSIONS_USE      = [1, 3, 5, 7]
INTERVAL          = [0.5, 2.5]
RESAMPLE_HZ       = 128
FMIN, FMAX        = 0.5, 50.0
MIN_PRESENT       = 11
OUT_CSV           = "results_stieger2021_full.csv"
CHECKPOINT_EVERY  = 8                   # checkpoint a cada 8 sujeitos

TARGET_21         = [
    "FC3","FC1","FCz","FC2","FC4",
    "C5","C3","C1","Cz","C2","C4","C6",
    "CP3","CP1","CPz","CP2","CP4"
]

BASE_PIPELINE_NAMES = ["csp+lda","csp+lr","riem+lda","riem+lr"]
CSD_PIPELINE_NAMES  = ["csd_csp+lda","csd_csp+lr","csd_riem+lda","csd_riem+lr"]

def pipeline_names() -> List[str]:
    return BASE_PIPELINE_NAMES + CSD_PIPELINE_NAMES


# ============================================================
# DATASETS LOCAIS
# ============================================================
class Stieger2021Local(Stieger2021):
    """Lê arquivos .mat diretamente de DATA_DIR, padrão S{subj}_Session_{ses}.mat."""
    def __init__(self, interval=[0,3], sessions=None, fix_bads=True, data_dir=None):
        sig = inspect.signature(super().__init__)
        if "fix_bads" in sig.parameters:
            super().__init__(interval=interval, sessions=sessions, fix_bads=fix_bads)
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
            # Ex: S11_Session_7.mat
            m = re.match(r"S(\d+)_Session_(\d+)\.mat", fname)
            if m is None:
                continue
            subj_id = int(m.group(1))
            ses     = int(m.group(2))
            if subj_id != subject:
                continue
            if self.sessions is None or ses in self.sessions:
                files.append(os.path.join(self.data_dir, fname))
        return sorted(
            files,
            key=lambda f: int(
                re.search(r"Session_(\d+)", os.path.basename(f)).group(1)
            )
        )


class Stieger2021LocalCSD(Stieger2021Local):
    """
    Variante que aplica CSD dentro de _get_single_subject_data,
    usando compute_current_source_density nos canais EEG.
    """
    def _get_single_subject_data(self, subject):
        subj_dict = super()._get_single_subject_data(subject)
        print(f"\n[Stieger2021LocalCSD] Aplicando CSD em subject {subject}")
        for ses_key, runs in subj_dict.items():
            print(f"  Sessão {ses_key}: runs -> {list(runs.keys())}")
            for run_key, raw in runs.items():
                raw = raw.copy()
                print(f"    Run {run_key} | canais antes: {raw.info['ch_names']}")
                eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, misc=False)
                if len(eeg_picks) == 0:
                    print("    [AVISO] Nenhum canal EEG, pulando CSD.")
                    runs[run_key] = raw
                    continue

                eeg_names = [raw.ch_names[i] for i in eeg_picks]
                print(f"    EEG picks ({len(eeg_picks)}): {eeg_names}")

                # Referência média + CSD
                raw_eeg = raw.copy().pick(eeg_picks)
                raw_eeg.set_eeg_reference("average", projection=False)
                print("    [INFO] Aplicando compute_current_source_density...")
                raw_eeg_csd = mne.preprocessing.compute_current_source_density(raw_eeg)

                data_csd = raw_eeg_csd.get_data()
                print("    [INFO] Forma CSD:", data_csd.shape,
                      "| NaN:", np.isnan(data_csd).any(),
                      "| Inf:", np.isinf(data_csd).any())

                # injeta dados CSD de volta no raw original
                raw._data[eeg_picks, :] = data_csd
                runs[run_key]           = raw

        return subj_dict


# ============================================================
# PRÉ-PROCESSADOR
# ============================================================
class CAR(BaseEstimator, TransformerMixin):
    """Referência comum por época: X <- X - média nos canais."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X)
        return X - X.mean(axis=1, keepdims=True)


# ============================================================
# PIPELINES
# ============================================================
def build_base_pipelines() -> Dict[str, Pipeline]:
    csp      = CSP(n_components=8, log=True, norm_trace=False)
    riem_cov = Covariances(estimator="oas")
    riem_ts  = TangentSpace(metric="riemann")
    car      = CAR()
    return {
        "csp+lda" : make_pipeline(car, csp, LDA()),
        "csp+lr"  : make_pipeline(
            car, csp,
            LogisticRegression(max_iter=500, solver="lbfgs")
        ),
        "riem+lda": Pipeline([
            ("car",car),("cov",riem_cov),("tgs",riem_ts),("clf",LDA())
        ]),
        "riem+lr" : Pipeline([
            ("car",car),("cov",riem_cov),("tgs",riem_ts),
            ("clf",LogisticRegression(max_iter=500, solver="lbfgs"))
        ]),
    }


def build_csd_pipelines() -> Dict[str, Pipeline]:
    """
    Mesmas arquiteturas, mas nomes diferentes, para rodar no dataset CSD.
    """
    csp      = CSP(n_components=8, log=True, norm_trace=False)
    riem_cov = Covariances(estimator="oas")
    riem_ts  = TangentSpace(metric="riemann")
    car      = CAR()
    return {
        "csd_csp+lda" : make_pipeline(car, csp, LDA()),
        "csd_csp+lr"  : make_pipeline(
            car, csp,
            LogisticRegression(max_iter=500, solver="lbfgs")
        ),
        "csd_riem+lda": Pipeline([
            ("car",car),("cov",riem_cov),("tgs",riem_ts),("clf",LDA())
        ]),
        "csd_riem+lr" : Pipeline([
            ("car",car),("cov",riem_cov),("tgs",riem_ts),
            ("clf",LogisticRegression(max_iter=500, solver="lbfgs"))
        ]),
    }


# ============================================================
# UTILITÁRIOS
# ============================================================
def available_sessions(ds: Stieger2021Local, subject: int) -> List[int]:
    paths = ds.data_path(subject)
    return sorted({int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in paths})


def channels_for_session(ds_one: Stieger2021Local, subject: int,
                         session: int, target21: List[str]) -> List[str]:
    try:
        subj_dict = ds_one._get_single_subject_data(subject)
    except Exception as e:
        print(f"Sujeito {subject} | Sessão {session}: erro _get_single_subject_data: {e}")
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


def nan_rows(subject: int, session: int, chs: List[str],
             pipelines: List[str] = None) -> pd.DataFrame:
    if pipelines is None:
        pipelines = pipeline_names()
    cols = [
        "score","time","samples","subject","session","fold",
        "n_sessions","dataset","pipeline","run","channels_used"
    ]
    rows = []
    for pname in pipelines:
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


def save_checkpoint(all_results: List[pd.DataFrame],
                    out_csv: str, last_subject: int) -> None:
    if not all_results:
        return
    results_partial = pd.concat(all_results, ignore_index=True)
    ck_name         = f"results_stieger2021_checkpoint_subj_{last_subject:02d}.csv"
    results_partial.to_csv(ck_name, index=False)
    print(f"[CHECKPOINT] Parcial salvo em: {os.path.join(os.getcwd(), ck_name)}")
    results_partial.to_csv(out_csv, index=False)
    print(f"[CHECKPOINT] OUT_CSV atualizado: {os.path.join(os.getcwd(), out_csv)}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = []

    for i, subj in enumerate(SUBJECTS, start=1):
        print(f"\n=== Sujeito {subj} (#{i}) ===")

        ds_listing              = Stieger2021Local(
            interval=INTERVAL, sessions=SESSIONS_USE, data_dir=DATA_DIR
        )
        ds_listing.subject_list = [subj]
        avail                   = available_sessions(ds_listing, subj)
        print(f"[INFO] Sessões disponíveis para sujeito {subj}: {avail}")

        for s in sorted(set(SESSIONS_USE) & set(avail)):
            print(f"\n--- Sujeito {subj} | Sessão {s} ---")

            ds_raw              = Stieger2021Local(
                interval=INTERVAL, sessions=[s], data_dir=DATA_DIR
            )
            ds_raw.subject_list = [subj]

            ds_csd              = Stieger2021LocalCSD(
                interval=INTERVAL, sessions=[s], data_dir=DATA_DIR
            )
            ds_csd.subject_list = [subj]

            # ----------------- canais -----------------
            try:
                chs = channels_for_session(ds_raw, subj, s, TARGET_21)
            except MemoryError:
                print(f"Sujeito {subj} | Sessão {s}: MemoryError canais → NaN todas pipelines.")
                all_results.append(nan_rows(subj, s, [], pipelines=None))
                continue
            except Exception as e:
                print(f"Sujeito {subj} | Sessão {s}: erro canais ({e}) → NaN todas pipelines.")
                all_results.append(nan_rows(subj, s, [], pipelines=None))
                continue

            if len(chs) < MIN_PRESENT:
                print(f"Sujeito {subj} | Sessão {s}: {len(chs)}/21 insuficiente → NaN todas pipelines.")
                all_results.append(nan_rows(subj, s, chs, pipelines=None))
                continue

            print(f"Sujeito {subj} | Sessão {s}: usando {len(chs)} canais: {chs}")

            # ----------------- paradigma -----------------
            paradigm = MotorImagery(
                n_classes=4, resample=RESAMPLE_HZ,
                fmin=FMIN, fmax=FMAX, channels=chs
            )

            eval_raw = WithinSessionEvaluation(
                paradigm=paradigm, datasets=[ds_raw],
                overwrite=True, hdf5_path=None
            )

            base_pipes = build_base_pipelines()
            csd_pipes  = build_csd_pipelines()

            # ----------------- RAW (MOABB) -----------------
            try:
                print("[RAW] Rodando pipelines base (MOABB)...")
                res_raw                  = eval_raw.process(base_pipes)
                res_raw["subject"]       = res_raw["subject"].astype(int)
                res_raw["session"]       = res_raw["session"].astype(int)
                res_raw["channels_used"] = ",".join(chs)

                found_raw   = set(res_raw["pipeline"].unique())
                missing_raw = [p for p in BASE_PIPELINE_NAMES if p not in found_raw]
                if missing_raw:
                    print(f"[RAW] Pipelines ausentes: {missing_raw}")
                    all_results.append(nan_rows(subj, s, chs, pipelines=missing_raw))

                all_results.append(res_raw)

            except MemoryError:
                print(f"Sujeito {subj} | Sessão {s}: MemoryError RAW → NaN pipelines RAW.")
                all_results.append(nan_rows(subj, s, chs, pipelines=BASE_PIPELINE_NAMES))

            except Exception as e:
                print(f"Sujeito {subj} | Sessão {s}: erro avaliação RAW ({e}) → NaN pipelines RAW.")
                all_results.append(nan_rows(subj, s, chs, pipelines=BASE_PIPELINE_NAMES))

            # ----------------- CSD (get_data + CV manual) -----------------
            try:
                print("[CSD] Extraindo dados com CSD (get_data)...")
                # ds_csd já tem sessions=[s], então não precisa filtrar por sessão
                X_csd, y_csd, meta_csd = paradigm.get_data(dataset=ds_csd,
                                                           subjects=[subj])

                if X_csd.shape[0] == 0:
                    print(f"[CSD] Nenhuma época para sujeito {subj}, sessão {s} → NaN pipelines CSD.")
                    all_results.append(nan_rows(subj, s, chs, pipelines=CSD_PIPELINE_NAMES))
                else:
                    print(f"[CSD] X shape = {X_csd.shape}, n_trials = {len(y_csd)}")
                    print(f"[CSD] Sessões em meta_csd:", meta_csd["session"].unique())

                    rows_csd = []
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                    for pname, pipe in csd_pipes.items():
                        print(f"[CSD] Rodando {pname} (CV 5-fold)...")
                        t0 = time.perf_counter()
                        scores = cross_val_score(pipe, X_csd, y_csd, cv=cv, n_jobs=1)
                        t1 = time.perf_counter()
                        mean_score = float(np.mean(scores))
                        elapsed    = float(t1 - t0)

                        rows_csd.append({
                            "score"        : mean_score,
                            "time"         : elapsed,
                            "samples"      : float(len(y_csd)),
                            "subject"      : subj,
                            "session"      : s,
                            "fold"         : np.nan,
                            "n_sessions"   : 1.0,
                            "dataset"      : "Stieger2021",
                            "pipeline"     : pname,
                            "run"          : np.nan,
                            "channels_used": ",".join(chs),
                        })

                    res_csd = pd.DataFrame(rows_csd)
                    all_results.append(res_csd)

            except MemoryError:
                print(f"Sujeito {subj} | Sessão {s}: MemoryError CSD → NaN pipelines CSD.")
                all_results.append(nan_rows(subj, s, chs, pipelines=CSD_PIPELINE_NAMES))

            except Exception as e:
                print(f"Sujeito {subj} | Sessão {s}: erro avaliação CSD ({e}) → NaN pipelines CSD.")
                all_results.append(nan_rows(subj, s, chs, pipelines=CSD_PIPELINE_NAMES))

        # checkpoint por bloco de sujeitos
        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_results, OUT_CSV, last_subject=subj)

    if not all_results:
        raise RuntimeError("Nada foi gerado. Verifique DATA_DIR / nomes de arquivos / filtros.")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"\n[FINAL] Salvo em: {os.path.join(os.getcwd(), OUT_CSV)}")
    print(results.head())
