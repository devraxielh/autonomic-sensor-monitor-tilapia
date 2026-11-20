"""02 — comparación de detectores de falla: baselines, SOTA tabular y detector propuesto."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from utils import (DOUBLE_COL, N_REPEATS, N_SPLITS, OKABE_ITO, PROPOSED, RESULTS_DIR, add_panel_label,
                   ci95, cluster_bootstrap, cycle_tensor, mark_best, rep_fold_seed, save_figure, save_table, stat_macro_auroc)
from autonomic_monitor import DETECTORS, FAULT_LABEL, FAULTS, Detector, make_dataset

d = np.load(RESULTS_DIR / "windows.npz")
W, blocks, scale = d["W"], d["blocks"], d["scale"]


def tpr_at_fpr(y, s, fpr_max=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


rows, oof = [], []
t0 = time.time()
for rep in range(N_REPEATS):
    for fold in range(N_SPLITS):
        tr_idx, te_idx = np.where(blocks <= fold)[0], np.where(blocks == fold + 1)[0]
        tr = make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale)
        te = make_dataset(W, te_idx, rep_fold_seed(rep, fold, 1), scale)
        pred = pd.DataFrame({"rep": rep, "fold": fold, "cycle": te["cycle"], "y": te["y"],
                             "ftype": te["ftype"], "fchan": te["fchan"], "sev": te["sev"]})
        for name in DETECTORS:
            t1 = time.time()
            det = Detector(name, rep_fold_seed(rep, fold, 2)).fit(tr["X"], tr["y"])
            fit_s = time.time() - t1
            s = det.score(te["X"])
            pred[name] = s
            normal = te["ftype"] == "normal"
            rec = {"model": name, "rep": rep, "fold": fold, "n_train_cycles": len(tr_idx), "fit_seconds": fit_s,
                   "auprc": average_precision_score(te["y"], s), "auroc_pooled": roc_auc_score(te["y"], s),
                   "tpr_at_5fpr": tpr_at_fpr(te["y"], s)}
            for f in FAULTS:
                m = normal | (te["ftype"] == f)
                rec[f"auroc_{f}"] = roc_auc_score(te["y"][m], s[m])
            rec["macro_auroc"] = float(np.mean([rec[f"auroc_{f}"] for f in FAULTS]))
            rows.append(rec)
        oof.append(pred)
    print(f"rep {rep + 1}/{N_REPEATS} listo ({time.time() - t0:.0f}s)", flush=True)

scores = pd.DataFrame(rows)
scores.to_csv(RESULTS_DIR / "fold_scores.csv", index=False)
pd.concat(oof, ignore_index=True).to_parquet(RESULTS_DIR / "oof_predictions.parquet", index=False)

# --- tabla 2: resultados principales ---
metrics = {"macro_auroc": "Macro-AUROC", "auprc": "AUPRC", "tpr_at_5fpr": "TPR@5\\%FPR"}
order = scores.groupby("model")["macro_auroc"].mean().sort_values(ascending=False).index.tolist()
# IC 95 % del macro-AUROC por bootstrap de conglomerados (ciclos reales dentro de cada bloque)
_oof = pd.concat(oof, ignore_index=True)
_long = _oof.melt(id_vars=["rep", "fold", "cycle", "ftype"], value_vars=DETECTORS, var_name="model", value_name="score")
_long["model"] = pd.Categorical(_long.model, categories=DETECTORS, ordered=True)
_obs, _boots = cluster_bootstrap(stat_macro_auroc, cycle_tensor(_long, "score", ["model"], ["normal"] + FAULTS))
_lo, _hi = ci95(_boots)
CI = {m: (float(_lo[i]), float(_hi[i])) for i, m in enumerate(DETECTORS)}
t2 = []
for m in order:
    g = scores[scores.model == m]
    r = {"Detector": m}
    for k in metrics:
        r.update({f"{k}_mean": g[k].mean(), f"{k}_sd": g[k].std(ddof=1)})
    r.update({"macro_auroc_ci_lo": CI[m][0], "macro_auroc_ci_hi": CI[m][1]})
    for f in FAULTS:
        r[f"auroc_{f}"] = g[f"auroc_{f}"].mean()
    r["fit_seconds"] = g["fit_seconds"].mean()
    t2.append(r)
t2 = pd.DataFrame(t2)
tex = pd.DataFrame({"Detector": [m + (" (loop detector)" if m == PROPOSED else "") for m in t2["Detector"]]})
for k, lab in metrics.items():
    tex[lab + " $\\uparrow$"] = mark_best(t2[f"{k}_mean"], t2[f"{k}_sd"], higher_better=True)
    if k == "macro_auroc":
        tex["95\\% CI"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t2[f"{k}_ci_lo"], t2[f"{k}_ci_hi"])]
save_table(t2, "table_02_main_results_cv", tex, column_format="lcccc")

# --- figura 2: (a) distribución por bloque x repetición, (b) AUROC por tipo de falla ---
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.9), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
pal = {m: (OKABE_ITO[5] if m == PROPOSED else OKABE_ITO[1]) for m in order}
sns.boxplot(data=scores, y="model", x="macro_auroc", order=order, hue="model", palette=pal, legend=False,
            width=0.6, fliersize=0, linewidth=0.8, ax=axes[0])
sns.stripplot(data=scores, y="model", x="macro_auroc", order=order, color="0.2", size=1.6, alpha=0.5, ax=axes[0])
axes[0].axvline(0.5, color="0.4", ls=":", lw=0.8)
axes[0].set_xlabel("Macro-AUROC (5 blocks $\\times$ 10 repeats)"); axes[0].set_ylabel("")
add_panel_label(axes[0], "a", x=-0.45)
heat = t2.set_index("Detector")[[f"auroc_{f}" for f in FAULTS]]
heat.columns = [FAULT_LABEL[f] for f in FAULTS]
sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", vmin=0.5, vmax=1.0, cbar_kws={"label": "AUROC"},
            annot_kws={"fontsize": 8}, linewidths=0.4, ax=axes[1])
axes[1].set_ylabel(""); axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=40, ha="right", fontsize=8)
axes[1].grid(False); add_panel_label(axes[1], "b", x=-0.5)
save_figure(fig, "fig_02_main_performance_comparison")
print(t2[["Detector", "macro_auroc_mean", "macro_auroc_sd", "auprc_mean", "tpr_at_5fpr_mean", "fit_seconds"]
         + [f"auroc_{f}" for f in FAULTS]].round(3).to_string())
