"""05 — explicabilidad (SHAP), diagnóstico, reparación por canal, calibración, caso real y costo."""
import json
import pickle
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix

from utils import (NA, ALPHA_FA, DOUBLE_COL, N_SPLITS, OKABE_ITO, PROJECT_DIR, PROPOSED, RESULTS_DIR, SEED,
                   add_panel_label, ci95, cluster_bootstrap, rep_fold_seed, save_figure, save_table, stat_mean)
from autonomic_monitor import (CH_LABEL, CHANNELS, DETECTORS, FAULT_LABEL, FAULTS, N_READ, AutonomicMonitor,
                               Detector, feature_family, make_dataset)

d = np.load(RESULTS_DIR / "windows.npz")
W, blocks, scale = d["W"], d["blocks"], d["scale"]

# ---------------- SHAP del detector propuesto (último bloque: mayor conjunto de entrenamiento) ----------------
fold = N_SPLITS - 1
tr = make_dataset(W, np.where(blocks <= fold)[0], rep_fold_seed(0, fold, 0), scale)
te = make_dataset(W, np.where(blocks == fold + 1)[0], rep_fold_seed(0, fold, 1), scale)
det = Detector(PROPOSED, rep_fold_seed(0, fold, 2)).fit(tr["X"], tr["y"])
F = det.features(te["X"]); names = det.fx_.feature_names_
sv = np.asarray(shap.TreeExplainer(det.model_).shap_values(F))
sv = sv[:, :, 1] if sv.ndim == 3 and sv.shape[2] == 2 else (sv[1] if sv.ndim == 3 else sv)
imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
imp["family"] = imp.feature.map(feature_family); imp["channel"] = imp.feature.str.split("__").str[0]
imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
imp.sort_values("mean_abs_shap", ascending=False).to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
fam = imp.groupby("family").share.sum().sort_values()
# importancia por tipo de falla: |SHAP| medio dentro de cada tipo, agregado por familia
by_type = pd.DataFrame({FAULT_LABEL[f]: pd.Series(np.abs(sv[te["ftype"] == f]).mean(0), index=names).groupby(imp.set_index("feature").family).sum()
                        for f in FAULTS})
by_type = (by_type / by_type.sum()).loc[fam.index[::-1]]
by_type.to_csv(RESULTS_DIR / "shap_family_by_fault.csv")

fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 1.05})
ax[0].barh(fam.index, fam.values, color=OKABE_ITO[4], edgecolor="k", linewidth=0.5)
ax[0].set_xlabel("Share of mean |SHAP|"); add_panel_label(ax[0], "a", x=-0.95)
sns.heatmap(by_type, annot=True, fmt=".2f", cmap="viridis", vmin=0, cbar_kws={"label": "Share of mean |SHAP|"},
            annot_kws={"fontsize": 8}, linewidths=0.4, ax=ax[1])
ax[1].set_ylabel(""); ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40, ha="right", fontsize=8)
ax[1].grid(False); add_panel_label(ax[1], "b", x=-0.85)
save_figure(fig, "fig_04_explainability_shap_or_latent")

# ---------------- diagnóstico: tipo y canal ----------------
dg = pd.read_parquet(RESULTS_DIR / "diagnosis_predictions.parquet")
cm = confusion_matrix(dg.true_type, dg.pred_type, labels=FAULTS, normalize="true")
per = dg.groupby(["rep", "fold"]).apply(lambda g: pd.Series({
    "type_acc": accuracy_score(g.true_type, g.pred_type), "chan_acc": accuracy_score(g.true_chan, g.pred_chan),
    **{f"chan_acc_{c}": accuracy_score(g.true_chan[g.true_chan == c], g.pred_chan[g.true_chan == c]) for c in range(4)},
    **{f"type_acc_{f}": accuracy_score(g.true_type[g.true_type == f], g.pred_type[g.true_type == f]) for f in FAULTS}}))
heal = pd.read_csv(RESULTS_DIR / "healing_scores.csv")
hm = heal.groupby(["rep", "fold", "channel"])[["unrepaired", "profile", "model", "type_aware"]].mean().reset_index()

rows = [{"Item": "Fault-type accuracy (6 classes)", "v": per.type_acc}, {"Item": "Channel localization accuracy (4 classes)", "v": per.chan_acc}]
rows += [{"Item": f"  Localization, {lab}", "v": per[f"chan_acc_{c}"]} for c, lab in enumerate([CH_LABEL[c] for c in CHANNELS] + ["all channels"])]
rows += [{"Item": f"  Type accuracy, {FAULT_LABEL[f]}", "v": per[f"type_acc_{f}"]} for f in FAULTS]
# IC 95 % por bootstrap de conglomerados: tensores (rep, bloque, ciclo, tipo de falla) de aciertos
_dg = dg.sort_values(["rep", "fold", "cycle"])
_shape = (1, dg.rep.nunique(), dg.fold.nunique(), -1, len(FAULTS))
_t = lambda col: _dg.assign(v=col.astype(float)).pivot_table(index=["rep", "fold", "cycle"], columns="true_type", values="v")[FAULTS].to_numpy().reshape(_shape)
T_type, T_chan, T_true = _t(_dg.true_type == _dg.pred_type), _t(_dg.true_chan == _dg.pred_chan), _t(_dg.true_chan)
tensors = [T_type, T_chan] + [np.where(T_true == c, T_chan, np.nan) for c in range(4)] \
    + [np.where(np.arange(len(FAULTS)) == j, T_type, np.nan) for j in range(len(FAULTS))]
cis = [tuple(float(x[0]) for x in ci95(cluster_bootstrap(stat_mean, T)[1])) for T in tensors]
t5 = pd.DataFrame([{"Item": r["Item"], "mean": r["v"].mean(), "sd": r["v"].std(ddof=1),
                    "ci_lo": ci[0], "ci_hi": ci[1]} for r, ci in zip(rows, cis)])
save_table(t5, "table_05_diagnosis", pd.DataFrame({"Diagnosis task": t5.Item,
           "Accuracy (mean $\\pm$ SD)": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t5["mean"], t5.sd)],
           "95\\% cluster-bootstrap CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t5.ci_lo, t5.ci_hi)]}), column_format="lcc")
t6 = []
for c, ch in enumerate(CHANNELS):
    g = hm[hm.channel == c]
    t6.append({"Channel": CH_LABEL[ch], **{m: g[m].mean() for m in ("unrepaired", "profile", "model", "type_aware")},
               **{m + "_sd": g[m].std(ddof=1) for m in ("unrepaired", "profile", "model", "type_aware")},
               "reduction_model_pct": 100 * (1 - g["model"].mean() / g["unrepaired"].mean())})
t6 = pd.DataFrame(t6)
save_table(t6, "table_06_healing_by_channel", pd.DataFrame({"Channel": t6.Channel,
           **{lab: [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t6[m], t6[m + "_sd"])] for m, lab in
              (("unrepaired", "Unrepaired"), ("profile", "Normal profile"), ("model", "Cross-channel model"), ("type_aware", "Type-aware plan"))},
           "Reduction (\\%)": t6.reduction_model_pct.map("{:.1f}".format)}), column_format="lccccc")

# ---------------- calibración del detector (predicciones fuera de muestra) ----------------
oof = pd.read_parquet(RESULTS_DIR / "oof_predictions.parquet")
brier = {m: brier_score_loss(oof.y, oof[m]) for m in ("RandomForest", "LogisticRegression", "XGBoost", "LightGBM", "CatBoost")}

fig = plt.figure(figsize=(DOUBLE_COL, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.75, wspace=0.45)
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
sns.heatmap(pd.DataFrame(cm, index=[FAULT_LABEL[f] for f in FAULTS], columns=[FAULT_LABEL[f] for f in FAULTS]),
            annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Fraction of true type"},
            annot_kws={"fontsize": 9}, linewidths=0.4, ax=ax[0])
ax[0].set_xlabel("Diagnosed fault type"); ax[0].set_ylabel("True fault type"); ax[0].grid(False)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=25, ha="right", fontsize=9); ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=9)
add_panel_label(ax[0], "a", x=-0.2)
xpos = np.arange(3); wd = 0.2
for j, (m, lab) in enumerate((("unrepaired", "Unrepaired"), ("profile", "Normal profile"), ("model", "Cross-channel"), ("type_aware", "Type-aware"))):
    ax[1].bar(xpos + (j - 1.5) * wd, t6[m], wd, yerr=t6[m + "_sd"], capsize=1.5, error_kw={"lw": 0.6},
              color=["0.6", OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[5]][j], edgecolor="k", linewidth=0.4, label=lab)
ax[1].set_xticks(xpos); ax[1].set_xticklabels(["DO", "pH", "Turbidity"]); ax[1].set_xlabel("Faulty channel")
ax[1].set_ylabel("Error on corrupted readings (SD)"); ax[1].set_ylim(0, 2.6)
ax[1].legend(fontsize=8, loc="upper center", ncol=2, columnspacing=0.8); add_panel_label(ax[1], "b", x=-0.33)
for m, col, ls in (("RandomForest", OKABE_ITO[5], "-"), ("CatBoost", OKABE_ITO[4], "--"), ("LogisticRegression", OKABE_ITO[2], ":")):
    pt, pp = calibration_curve(oof.y, oof[m], n_bins=8, strategy="quantile")
    ax[2].plot(pp, pt, marker="o", ms=3, lw=1.1, color=col, ls=ls,
               label=f"{m.replace('LogisticRegression', 'Logistic reg.').replace('RandomForest', 'Random forest')} ({brier[m]:.3f})")
ax[2].plot([0, 1], [0, 1], color="0.4", ls=":", lw=0.8); ax[2].set_xlabel("Predicted fault probability")
ax[2].set_ylabel("Observed fault frequency"); ax[2].set_ylim(0, 1.5); ax[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax[2].legend(fontsize=8, loc="upper left", title="Detector (Brier score)", title_fontsize=8)
add_panel_label(ax[2], "c", x=-0.33)
save_figure(fig, "fig_05_error_calibration_or_case_study")

# ---------------- caso real: relleno por interpolación presente en el archivo fuente ----------------
full = make_dataset(W, np.arange(len(W)), rep_fold_seed(0, 9, 0), scale)
mon = AutonomicMonitor(PROPOSED, SEED, ALPHA_FA).fit(full)
raw = pd.read_excel(PROJECT_DIR / "data" / "raw" / "Data_Model_IoTMLCQ_2024.xlsx")
raw["Datetime"] = pd.to_datetime(raw["Datetime"]).dt.round("h")
case = {}
for h0 in (8, 12, 16):                              # ventanas de 6 horas dentro de la rampa interpolada
    seg = raw[raw.Datetime.dt.hour.between(h0, h0 + N_READ - 1)]
    wins = np.stack([g[CHANNELS].to_numpy() for _, g in seg.groupby(seg.Datetime.dt.date) if len(g) == N_READ])
    a = mon.analyze(wins)
    alarm = a["score"] > mon.threshold_
    case[f"hours_{h0}_{h0 + N_READ - 1}"] = {"n_windows": int(len(wins)), "flagged_frac": float(alarm.mean()),
        "diagnosed_gateway_backfill_frac": float((a["ftype"][alarm] == "gateway_backfill").mean()) if alarm.any() else 0.0,
        "rule_flagged_frac": float((a["rule_score"] >= 1.0).mean())}
case["real_cycles_oof_false_alarm"] = float((mon.calib_scores_ > mon.threshold_).mean())
(RESULTS_DIR / "case_study_source_backfill.json").write_text(json.dumps(case, indent=2))

t11 = pd.DataFrame([{"Windows": f"Interpolated, hours {k.split('_')[1]}--{k.split('_')[2]}", "n": v["n_windows"],
                     "flagged": v["flagged_frac"], "as_gateway": v["diagnosed_gateway_backfill_frac"], "rules": v["rule_flagged_frac"]}
                    for k, v in case.items() if k.startswith("hours")]
                   + [{"Windows": "Real cycles (out-of-sample scores)", "n": int(len(mon.calib_scores_)),
                       "flagged": case["real_cycles_oof_false_alarm"], "as_gateway": np.nan,
                       "rules": float((mon.rules_.score(full["X"][full["y"] == 0]) >= 1.0).mean())}])
t11_csv = t11.copy(); t11_csv["as_gateway"] = [NA if pd.isna(v) else v for v in t11.as_gateway]
save_table(t11_csv, "table_11_case_study_source_file", pd.DataFrame({
    "Windows": t11.Windows, "$n$": t11.n, "Alarm rate, monitor": t11.flagged.map("{:.3f}".format),
    "Diagnosed as gateway backfill": t11.as_gateway.map(lambda v: "--" if pd.isna(v) else f"{v:.3f}"),
    "Alarm rate, QC rules": t11.rules.map("{:.3f}".format)}), column_format="lcccc")

# ---------------- costo de cómputo (un hilo, una ventana) ----------------
one = te["X"][:1]
cost = []
for name in [n for n in DETECTORS if n != "Dummy"]:
    dd = Detector(name, SEED).fit(tr["X"], tr["y"])
    mdl = dd.model_.steps[-1][1] if hasattr(dd.model_, "steps") else dd.model_
    if hasattr(mdl, "n_jobs"):
        mdl.set_params(n_jobs=1)
    dd.score(one)
    lat = []
    for _ in range(100):
        t0 = time.perf_counter(); dd.score(one); lat.append((time.perf_counter() - t0) * 1e3)
    cost.append({"Component": name + (" (loop detector)" if name == PROPOSED else ""), "size_kb": len(pickle.dumps(dd.model_)) / 1024,
                 "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.quantile(lat, 0.95))})
for m in (mon.detector_.model_, mon.localizer_, mon.typer_, *mon.healer_.models_):
    m.set_params(n_jobs=1)
lat = []
for _ in range(100):
    t0 = time.perf_counter(); a = mon.analyze(one)
    mon.healer_.heal(one[0], a["est"][0], int(a["channel"][0]), a["ftype"][0], "type_aware"); lat.append((time.perf_counter() - t0) * 1e3)
size = sum(len(pickle.dumps(m)) for m in (mon.detector_.model_, mon.localizer_, mon.typer_, mon.healer_.models_)) / 1024
cost.append({"Component": "Full MAPE-K loop (analyze + plan + execute)", "size_kb": size,
             "latency_ms_median": float(np.median(lat)), "latency_ms_p95": float(np.quantile(lat, 0.95))})
t7 = pd.DataFrame(cost)
save_table(t7, "table_07_computational_cost", pd.DataFrame({"Component": t7.Component, "Model size (kB)": t7.size_kb.map("{:.0f}".format),
           "Median latency (ms)": t7.latency_ms_median.map("{:.2f}".format), "95th pct. latency (ms)": t7.latency_ms_p95.map("{:.2f}".format)}),
           column_format="lccc")
from importlib.metadata import version
LIBS = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap", "matplotlib", "seaborn"]
(RESULTS_DIR / "environment.json").write_text(json.dumps({"platform": platform.platform(), "machine": platform.machine(),
    "processor": platform.processor(), "python": platform.python_version(),
    "libraries": {lib: version(lib) for lib in LIBS}}, indent=2))

print(fam.round(3).to_string()); print(by_type.round(2).to_string()); print(t5.round(3).to_string()); print(t6.round(3).to_string())
print({k: round(v, 4) for k, v in brier.items()}); print(json.dumps(case, indent=1)); print(t7.round(2).to_string())
print(imp.sort_values("mean_abs_shap", ascending=False).head(8)[["feature", "share"]].round(3).to_string())
