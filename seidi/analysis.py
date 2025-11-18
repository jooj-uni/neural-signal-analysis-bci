import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler

CHANCE        = 0.25

# CARREGAR DADOS
CSV_NAME      = "results_stieger2021.csv"
df            = pd.read_csv(os.path.join(os.getcwd(), CSV_NAME))
required      = {"pipeline", "subject", "session", "score"}
missing       = required - set(df.columns)
if missing:
    raise ValueError(f"CSV precisa conter: {sorted(required)}; faltam: {sorted(missing)}")

df["subject"] = df["subject"].astype(int)
df["session"] = df["session"].astype(int)

pipes         = sorted(df["pipeline"].unique())
subjs         = sorted(df["subject"].unique())
sess          = sorted(df["session"].unique())

# AGREGAÇÕES
g_pipe        = (df.groupby("pipeline")["score"]
                   .agg(["mean", "std", "count"])
                   .reindex(pipes))
g_pipe.to_csv("summary_by_pipeline.csv", index=True)

piv_sp        = (df.pivot_table(values="score",
                                index="subject",
                                columns="pipeline",
                                aggfunc=np.nanmean)
                  .reindex(index=subjs, columns=pipes))
piv_ep        = (df.pivot_table(values="score",
                                index="session",
                                columns="pipeline",
                                aggfunc=np.nanmean)
                  .reindex(index=sess, columns=pipes))

# FUNÇÕES AUXILIARES
def annotate_heatmap(ax, data, fontsize=11):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fontsize)

def heatmap(ax, M, xticks, yticks, title, xlabel, ylabel, vmin=0.0, vmax=1.0):
    arr      = np.array(M, dtype=float)
    mask     = np.ma.masked_invalid(arr)
    cmap     = plt.cm.inferno.copy()
    im       = ax.imshow(mask, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cb       = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    cb.ax.tick_params(labelsize=9)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels(xticks, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    annotate_heatmap(ax, arr, fontsize=10)

# FIGURA PRINCIPAL
np.random.seed(0)
fig           = plt.figure(figsize=(20, 11))
gs            = fig.add_gridspec(2, 4,
                                 width_ratios=[1.2, 1, 1, 1.2],
                                 height_ratios=[1, 1],
                                 hspace=0.35, wspace=0.35)
ax_hm_sp      = fig.add_subplot(gs[:, 0])  # span 2 linhas (esquerda)
ax_bar        = fig.add_subplot(gs[0, 1])  # grade 2×2 (linha superior)
ax_hm_ep      = fig.add_subplot(gs[0, 2])  # grade 2×2 (linha superior)
ax_lines      = fig.add_subplot(gs[1, 1])  # grade 2×2 (linha inferior)
ax_box        = fig.add_subplot(gs[1, 2])  # grade 2×2 (linha inferior)
ax_hm_best    = fig.add_subplot(gs[:, 3])  # span 2 linhas (direita)

# Remover grids de todos os eixos
for ax in [ax_hm_sp, ax_bar, ax_hm_ep, ax_lines, ax_box, ax_hm_best]:
    ax.grid(False)

# ---- (1) Barras: média por pipeline (± EP) ----
means         = g_pipe["mean"].to_numpy()
nobs          = g_pipe["count"].astype(float).to_numpy()
sems          = np.where(nobs > 1, g_pipe["std"].to_numpy() / np.sqrt(nobs), np.nan)
x             = np.arange(len(pipes))
ax_bar.bar(x, means, yerr=sems, capsize=4)
ax_bar.set_xticks(x, pipes, rotation=25, ha="right")
ax_bar.set_ylabel("Accuracy (média)")
ax_bar.set_title("Média por pipeline (± EP)")
ax_bar.set_ylim(0.0, 1.0)
ax_bar.axhline(CHANCE, linestyle="--", linewidth=1, color="k", alpha=0.8)

# ---- (2) Heatmap: Sujeito × Pipeline ----
heatmap(ax_hm_sp, piv_sp.values, pipes, subjs,
        title="Sujeito × Pipeline (média)",
        xlabel="Pipeline", ylabel="Sujeito")

# ---- (3) Heatmap: Sessão × Pipeline ----
heatmap(ax_hm_ep, piv_ep.values, pipes, sess,
        title="Sessão × Pipeline (média)",
        xlabel="Pipeline", ylabel="Sessão")

# ---- (4) Linhas: evolução por sessão (média ± IC95%) ----
# Cores bem distintas
color_list    = list(plt.get_cmap("tab20").colors)  # 20 cores distintas
ax_lines.set_prop_cycle(cycler(color=color_list))

for idx, p in enumerate(pipes):
    y_mean    = piv_ep[p].to_numpy()
    grp       = df[df["pipeline"] == p].groupby("session")["score"]
    y_std     = grp.std().reindex(sess).to_numpy()
    y_n       = grp.count().reindex(sess).to_numpy()
    y_sem     = np.where(y_n > 1, y_std / np.sqrt(y_n), np.nan)
    ax_lines.errorbar(sess, y_mean, yerr=1.96 * y_sem,
                      marker='o', capsize=4, label=p)

ax_lines.set_xlabel("Sessão")
ax_lines.set_ylabel("Accuracy (média)")
ax_lines.set_title("Evolução por sessão (média ± IC95%)")
ax_lines.legend(ncols=2, fontsize=8)
ax_lines.set_ylim(0.0, 1.0)
ax_lines.axhline(CHANCE, linestyle="--", linewidth=1, color="k", alpha=0.8)

# ---- (5) Boxplot por pipeline ----
data_per_pipe = [df.loc[df["pipeline"] == p, "score"].dropna().to_numpy()
                 for p in pipes]
ax_box.boxplot(data_per_pipe, labels=pipes, showmeans=True)
ax_box.set_xticklabels(pipes, rotation=25, ha="right")
ax_box.set_ylabel("Accuracy")
ax_box.set_title("Distribuição por pipeline")
ax_box.set_ylim(0.0, 1.0)
ax_box.axhline(CHANCE, linestyle="--", linewidth=1, color="k", alpha=0.8)

# ---- (6) Heatmap: Sujeito × Sessão (melhor pipeline) ----
best_pipe     = g_pipe["mean"].idxmax()
df_best       = df[df["pipeline"] == best_pipe]
subj_order    = (df_best.groupby("subject")["score"]
                     .mean()
                     .sort_values(ascending=False)
                     .index
                     .tolist())
piv_best      = (df_best.pivot_table(values="score",
                                     index="subject",
                                     columns="session",
                                     aggfunc=np.nanmean)
                 .reindex(index=subj_order, columns=sess))

heatmap(ax_hm_best,
        piv_best.values,
        sess,
        subj_order,
        title=f"[BEST] {best_pipe}: Sujeito × Sessão",
        xlabel="Sessão", ylabel="Sujeito")

# ---- Título e salvamento da figura principal ----
fig.suptitle("Stieger2021 — Comparações: pipeline / sujeito / sessão (NaN-safe)",
             fontsize=16)
out_path      = os.path.join(os.getcwd(), "results_overview_with_best_pipeline.pdf")
plt.savefig(out_path, dpi=150)

# ---- (7) Tempo médio por pipeline (figura separada) ----
# Espera-se uma coluna "time" no DataFrame (tempo em segundos, por exemplo).
# Se o nome for outro (ex.: "runtime", "duration"), basta trocar abaixo.
if "time" in df.columns:
    g_time = (df.groupby("pipeline")["time"]
                .agg(["mean", "std", "count"])
                .reindex(pipes))

    fig_time, ax_time = plt.subplots(figsize=(8, 5))

    means_t   = g_time["mean"].to_numpy()
    nobs_t    = g_time["count"].astype(float).to_numpy()
    sems_t    = np.where(nobs_t > 1, g_time["std"].to_numpy() / np.sqrt(nobs_t), np.nan)
    x_t       = np.arange(len(pipes))

    ax_time.bar(x_t, means_t, yerr=sems_t, capsize=4)
    ax_time.set_xticks(x_t, pipes, rotation=25, ha="right")
    
    ax_time.set_ylabel("Tempo médio (s)")
    ax_time.set_title("Tempo médio por pipeline (± EP)")

    fig_time.tight_layout()
    out_time_path = os.path.join(os.getcwd(), "results_time_by_pipeline.pdf")
    fig_time.savefig(out_time_path, dpi=150)
else:
    print("Coluna 'time' não encontrada em df; pulando análise de tempo por pipeline.")

# ---- Mostrar todas as figuras ----
plt.show()
