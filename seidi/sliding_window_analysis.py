import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from scipy.stats         import t

# =============== CONFIG ===============
SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
OUT_DIR              = os.path.join(SCRIPT_DIR, "plots_time_acc_ci")
CSV_PATH             = os.path.join(SCRIPT_DIR, "timecurves_riem_svmrbf_stieger2021.csv")

SUBJECTS_PER_FIG     = 8
ALPHA                = 0.05
YLIM                 = (0.0, 1.0)

CHANCE               = 0.25
SMOOTH               = False
SMOOTH_WINDOW        = 5


# ============== HELPERS ===============
def smooth(y, win):
    if win <= 1: return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


# =============== MAIN ================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df                 = pd.read_csv(CSV_PATH)
    required           = {"pipeline","subject","session","fold","time_center_s","window_start_s","acc"}
    missing            = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV não tem colunas necessárias: {sorted(missing)}")

    df["subject"]      = df["subject"].astype(int)
    df["session"]      = df["session"].astype(int)
    df["fold"]         = df["fold"].astype(int)

    t0                 = df["window_start_s"].min()
    df["t_rel"]        = df["time_center_s"] - t0

    subjects           = sorted(df["subject"].unique())
    sessions           = sorted(df["session"].unique())
    pipelines          = sorted(df["pipeline"].unique())

    for pipe in pipelines:

        dpipe          = df[df["pipeline"] == pipe].copy()
        stats          = (dpipe.groupby(["subject","session","t_rel"], observed=True)["acc"]
                               .agg(["mean","std","count"]).reset_index())

        stats["sem"]   = stats["std"] / np.sqrt(stats["count"])
        stats["tcrit"] = stats["count"].apply(lambda n: t.ppf(1 - ALPHA/2, n-1) if n > 1 else np.nan)
        stats["ci_low"]= stats["mean"] - stats["tcrit"] * stats["sem"]
        stats["ci_hi"] = stats["mean"] + stats["tcrit"] * stats["sem"]

        pipe_tag       = "".join(c if c.isalnum() or c in "-_+" else "_" for c in pipe)

        for k in range(0, len(subjects), SUBJECTS_PER_FIG):
            block       = subjects[k:k + SUBJECTS_PER_FIG]
            fig, axes   = plt.subplots(len(block), len(sessions),
                                       figsize=(4.6 * len(sessions), 2.6 * len(block)),
                                       sharex=True, sharey=True, constrained_layout=True)

            if len(block) == 1:    axes = axes[np.newaxis, :]
            if len(sessions) == 1: axes = axes[:, np.newaxis]

            for i, subj in enumerate(block):
                for j, ses in enumerate(sessions):
                    ax  = axes[i, j]
                    sub = stats[(stats.subject == subj) & (stats.session == ses)]
                    if sub.empty:
                        ax.set_axis_off()
                        continue

                    x   = sub["t_rel"].values
                    m   = sub["mean"].values
                    lo  = sub["ci_low"].values
                    hi  = sub["ci_hi"].values
                    if SMOOTH: m = smooth(m, SMOOTH_WINDOW)

                    ax.plot(x, m, lw=2)
                    ax.fill_between(x, lo, hi, alpha=0.25)

                    ax.axvline(0,       color="k", ls="--", lw=1)
                    ax.axhline(CHANCE,  color="k", ls="--", lw=1, alpha=0.8)

                    ax.set_ylim(*YLIM)
                    ax.grid(alpha=0.3)

                    if i == 0:              ax.set_title(f"Session {ses}")
                    if j == 0:              ax.set_ylabel(f"Subj {subj}\nAccuracy")
                    if i == len(block) - 1: ax.set_xlabel("Time relative to onset (s)")

            fig.suptitle(f"{pipe}\nAccuracy over time (mean ± 95% CI across folds)", y=1.02)

            out = os.path.join(OUT_DIR, f"{pipe_tag}_time_acc_ci_subjects_{block[0]}_{block[-1]}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {out}")

    print(f"\n[FINAL] Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
