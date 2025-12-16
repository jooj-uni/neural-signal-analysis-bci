import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# =========================
# CONFIG
# =========================
CSV_PATH = "timecurves_riem_svmrbf_stieger2021.csv"
OUT_DIR  = "plots_time_profiles_ci"

SUBJECTS_PER_FIG = 8
ALPHA = 0.05          # IC 95%
YLIM = (0.0, 1.0)

SMOOTH = False        # se quiser suavizar a MÉDIA
SMOOTH_WINDOW = 5


# =========================
# HELPERS
# =========================
def smooth(y, win):
    if win <= 1:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Tipos
    df["subject"] = df["subject"].astype(int)
    df["session"] = df["session"].astype(int)
    df["trial"]   = df["trial"].astype(int)

    # Tempo relativo à tentativa
    t0 = df["window_start_s"].min()
    df["t_rel"] = df["time_center_s"] - t0

    subjects = sorted(df["subject"].unique())
    sessions = sorted(df["session"].unique())
    classes  = sorted(df["class"].unique())

    cmap = plt.get_cmap("tab10")
    class_colors = {c: cmap(i % 10) for i, c in enumerate(classes)}

    # =====================================================
    # 1) Colapsar folds → média por TRIAL
    # =====================================================
    df_trial = (
        df
        .groupby(
            ["subject", "session", "trial", "class", "t_rel"],
            observed=True
        )["proba"]
        .mean()
        .reset_index()
    )

    # =====================================================
    # 2) Estatísticas entre TRIALS
    # =====================================================
    stats = (
        df_trial
        .groupby(
            ["subject", "session", "class", "t_rel"],
            observed=True
        )["proba"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    # t crítico para IC
    stats["tcrit"] = stats["count"].apply(
        lambda n: t.ppf(1 - ALPHA / 2, n - 1) if n > 1 else np.nan
    )

    stats["ci_low"]  = stats["mean"] - stats["tcrit"] * stats["sem"]
    stats["ci_high"] = stats["mean"] + stats["tcrit"] * stats["sem"]

    # =====================================================
    # 3) Plot
    # =====================================================
    for k in range(0, len(subjects), SUBJECTS_PER_FIG):
        block = subjects[k:k + SUBJECTS_PER_FIG]

        fig, axes = plt.subplots(
            len(block), len(sessions),
            figsize=(4.6 * len(sessions), 2.9 * len(block)),
            sharex=True, sharey=True,
            constrained_layout=True
        )

        if len(block) == 1:
            axes = axes[np.newaxis, :]
        if len(sessions) == 1:
            axes = axes[:, np.newaxis]

        for i, subj in enumerate(block):
            for j, ses in enumerate(sessions):
                ax = axes[i, j]

                sub = stats[
                    (stats.subject == subj) &
                    (stats.session == ses)
                ]

                if sub.empty:
                    ax.set_axis_off()
                    continue

                for cls in classes:
                    sc = sub[sub["class"] == cls]
                    if sc.empty:
                        continue

                    t_rel = sc["t_rel"].values
                    mean  = sc["mean"].values
                    lo    = sc["ci_low"].values
                    hi    = sc["ci_high"].values

                    if SMOOTH:
                        mean = smooth(mean, SMOOTH_WINDOW)

                    color = class_colors[cls]

                    ax.plot(
                        t_rel, mean,
                        color=color,
                        lw=2,
                        label=cls
                    )
                    ax.fill_between(
                        t_rel, lo, hi,
                        color=color,
                        alpha=0.25
                    )

                ax.axvline(0, color="k", ls="--", lw=1)
                ax.set_ylim(*YLIM)
                ax.grid(alpha=0.3)

                if i == 0:
                    ax.set_title(f"Session {ses}")
                if j == 0:
                    ax.set_ylabel(f"Subj {subj}\nP(class)")
                if i == len(block) - 1:
                    ax.set_xlabel("Time relative to trial onset (s)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", title="Class")

        fig.suptitle(
            "Temporal profile of class probability\n"
            "Mean ± 95% CI across trials",
            y=1.02
        )

        out = os.path.join(
            OUT_DIR,
            f"time_profile_ci_subjects_{block[0]}_{block[-1]}.png"
        )
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Saved: {out}")

    print(f"\n[FINAL] Done. Figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
