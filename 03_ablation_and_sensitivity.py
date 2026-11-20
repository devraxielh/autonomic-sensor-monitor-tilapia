"""03 — ablación del lazo MAPE-K (self-healing, self-optimization) y análisis de sensibilidad."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import (ALPHA_FA, DOUBLE_COL, N_REPEATS, N_SPLITS, OKABE_ITO, PREVALENCE, PROPOSED, RESULTS_DIR,
                   DOWN, NA, add_panel_label, ci95, cluster_bootstrap, cycle_tensor, mark_best, rep_fold_seed,
                   save_figure, save_table, stat_e2e)
from autonomic_monitor import (ALL_CH, FAULT_LABEL, FAULTS, AutonomicMonitor, Detector, make_dataset,
                               run_loop_config, window_errors)

d = np.load(RESULTS_DIR / "windows.npz")
W, blocks, scale = d["W"], d["blocks"], d["scale"]

# nombre -> (detector, umbral, reparación)
CONFIGS = {
    "C0 No monitoring": None,
    "C1 QC rules + profile": ("rules", "fixed", "profile"),
    "C2 ML detect + profile": ("ml", "fixed", "profile"),
    "C3 + cross-channel healing": ("ml", "fixed", "model"),
    "C4 + type-aware plan": ("ml", "fixed", "type_aware"),
    "C4y + Youden threshold": ("ml", "youden", "type_aware"),
    "C5 + conformal threshold": ("ml", "conformal", "type_aware"),
    "C6 + online threshold (full loop)": ("ml", "online", "type_aware"),
    "Oracle analysis": ("oracle", None, "type_aware"),
}
ALPHAS = [0.02, 0.05, 0.10, 0.20]


def run_config(cfg, mon, a, te, alpha=ALPHA_FA):
    return run_loop_config(cfg, mon, a, te, alpha)


def err(out, te, sel):
    """Error absoluto medio de las 18 lecturas del ciclo, en desviaciones estándar del canal."""
    return float(window_errors(out, te, scale)[sel].mean()) if sel.any() else np.nan


cycle_rows = []
loop_rows, diag_rows, alpha_rows, heal_rows = [], [], [], []
t0 = time.time()
for rep in range(N_REPEATS):
    for fold in range(N_SPLITS):
        tr_idx, te_idx = np.where(blocks <= fold)[0], np.where(blocks == fold + 1)[0]
        tr = make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale)
        te = make_dataset(W, te_idx, rep_fold_seed(rep, fold, 1), scale)
        mon = AutonomicMonitor(PROPOSED, rep_fold_seed(rep, fold, 2), ALPHA_FA).fit(tr)
        a = mon.analyze(te["X"])
        normal, faulty = te["y"] == 0, te["y"] == 1
        for name, cfg in CONFIGS.items():
            alarm, out = run_config(cfg, mon, a, te)
            e_n, e_f = err(out, te, normal), err(out, te, faulty)
            rec = {"config": name, "rep": rep, "fold": fold, "n_train_cycles": len(tr_idx),
                   "false_alarm_rate": alarm[normal].mean(), "detection_rate": alarm[faulty].mean(),
                   "err_normal": e_n, "err_faulty": e_f,
                   "err_e2e": (1 - PREVALENCE) * e_n + PREVALENCE * e_f,
                   "err_corrupted": float((np.abs(out - te["clean"]) / scale)[te["mask"]].mean())}
            for f in FAULTS:
                rec[f"err_{f}"] = err(out, te, te["ftype"] == f)
                rec[f"det_{f}"] = alarm[te["ftype"] == f].mean()
            loop_rows.append(rec)
            cycle_rows.append(pd.DataFrame({"config": name.split(" ")[0], "rep": rep, "fold": fold, "cycle": te["cycle"],
                                            "ftype": te["ftype"], "err": window_errors(out, te, scale)}))
        # diagnóstico (sobre todas las fallas del bloque de prueba)
        diag_rows.append(pd.DataFrame({"rep": rep, "fold": fold, "cycle": te["cycle"][faulty], "true_type": te["ftype"][faulty],
                                       "pred_type": a["ftype"][faulty], "true_chan": te["fchan"][faulty],
                                       "pred_chan": a["channel"][faulty]}))
        # reparación por canal y método, con diagnóstico perfecto (aísla la calidad del Execute)
        for i in np.where(faulty & (te["fchan"] != ALL_CH))[0]:
            k, m = int(te["fchan"][i]), te["mask"][i]
            base = {"rep": rep, "fold": fold, "ftype": te["ftype"][i], "channel": k,
                    "unrepaired": (np.abs(te["X"][i] - te["clean"][i]) / scale)[m].mean()}
            for mode in ("profile", "model", "type_aware"):
                h = mon.healer_.heal(te["X"][i], a["est"][i], k, te["ftype"][i], mode)
                base[mode] = (np.abs(h - te["clean"][i]) / scale)[m].mean()
            heal_rows.append(base)
        # self-optimization: FA lograda vs objetivo
        for alpha in ALPHAS:
            for mode in ("conformal", "online"):
                alarm, _ = run_config(("ml", mode, "type_aware"), mon, a, te, alpha)
                alpha_rows.append({"rep": rep, "fold": fold, "alpha": alpha, "mode": mode,
                                   "false_alarm_rate": alarm[normal].mean(), "detection_rate": alarm[faulty].mean()})
        alpha_rows.append({"rep": rep, "fold": fold, "alpha": np.nan, "mode": "fixed",
                           "false_alarm_rate": (a["score"] >= 0.5)[normal].mean(),
                           "detection_rate": (a["score"] >= 0.5)[faulty].mean()})
    print(f"lazo rep {rep + 1}/{N_REPEATS} ({time.time() - t0:.0f}s)", flush=True)

loop = pd.DataFrame(loop_rows); loop.to_csv(RESULTS_DIR / "loop_scores.csv", index=False)
cyc = pd.concat(cycle_rows, ignore_index=True)
cyc.to_parquet(RESULTS_DIR / "loop_cycle_errors.parquet", index=False)
# IC 95 % por bootstrap de conglomerados (ciclos reales dentro de cada bloque)
SHORT = [n.split(" ")[0] for n in CONFIGS]
cyc["config"] = pd.Categorical(cyc.config, categories=SHORT, ordered=True)
_obs_e, boots_e = cluster_bootstrap(stat_e2e, cycle_tensor(cyc, "err", ["config"], ["normal"] + FAULTS))
E2E_CI = dict(zip(CONFIGS, zip(*ci95(boots_e))))
RED_CI = dict(zip(CONFIGS, zip(*ci95(100 * (1 - boots_e / boots_e[:, [0]])))))
pd.concat(diag_rows, ignore_index=True).to_parquet(RESULTS_DIR / "diagnosis_predictions.parquet", index=False)
heal = pd.DataFrame(heal_rows); heal.to_csv(RESULTS_DIR / "healing_scores.csv", index=False)
alph = pd.DataFrame(alpha_rows); alph.to_csv(RESULTS_DIR / "alpha_sweep.csv", index=False)

# --- sensibilidad 1: severidad y ruido residual (detector propuesto, todos los bloques) ---
sens_rows = []
SEVS, JITS = [0.5, 1.0, 1.5, 2.0, 3.0], [0.0, 0.01, 0.05, 0.10, 0.20]
for rep in range(3):
    for fold in range(N_SPLITS):
        tr_idx, te_idx = np.where(blocks <= fold)[0], np.where(blocks == fold + 1)[0]
        det = Detector(PROPOSED, rep_fold_seed(rep, fold, 2)).fit(**{k: make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale)[k] for k in ("X", "y")})
        rules = Detector("QC-Rules").fit(**{k: make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale)[k] for k in ("X", "y")})
        for s in SEVS:
            te = make_dataset(W, te_idx, rep_fold_seed(rep, fold, 3), scale, severity=s)
            for f in ("stuck", "drift", "spike", "noise"):
                m = (te["ftype"] == "normal") | (te["ftype"] == f)
                sens_rows.append({"kind": "severity", "level": s, "fault": f, "rep": rep, "fold": fold,
                                  "auroc": roc_auc_score(te["y"][m], det.score(te["X"][m])),
                                  "auroc_rules": roc_auc_score(te["y"][m], rules.score(te["X"][m]))})
        for j in JITS:      # el detector se reentrena con el mismo ruido residual que verá en operación
            trj = make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale, jitter=j)
            detj = Detector(PROPOSED, rep_fold_seed(rep, fold, 2)).fit(trj["X"], trj["y"])
            te = make_dataset(W, te_idx, rep_fold_seed(rep, fold, 4), scale, jitter=j)
            for f in ("stuck", "gap_fill", "gateway_backfill"):
                m = (te["ftype"] == "normal") | (te["ftype"] == f)
                sens_rows.append({"kind": "jitter", "level": j, "fault": f, "rep": rep, "fold": fold,
                                  "auroc": roc_auc_score(te["y"][m], detj.score(te["X"][m])),
                                  "auroc_rules": roc_auc_score(te["y"][m], rules.score(te["X"][m]))})
        # sensibilidad 2: hiperparámetros del detector propuesto
        tr = make_dataset(W, tr_idx, rep_fold_seed(rep, fold, 0), scale)
        te = make_dataset(W, te_idx, rep_fold_seed(rep, fold, 1), scale)
        for n_est in (25, 50, 100, 300, 600):
            for depth in (3, 6, None):
                s_ = Detector(PROPOSED, rep_fold_seed(rep, fold, 2), n_estimators=n_est, max_depth=depth).fit(tr["X"], tr["y"]).score(te["X"])
                au = np.mean([roc_auc_score(te["y"][(te["ftype"] == "normal") | (te["ftype"] == f)],
                                            s_[(te["ftype"] == "normal") | (te["ftype"] == f)]) for f in FAULTS])
                sens_rows.append({"kind": "hyper", "level": n_est, "fault": str(depth), "rep": rep, "fold": fold,
                                  "auroc": au, "auroc_rules": np.nan})
    print(f"sensibilidad rep {rep + 1}/3 ({time.time() - t0:.0f}s)", flush=True)
sens = pd.DataFrame(sens_rows); sens.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)

# --- tabla 3: ablación del lazo ---
base = loop[loop.config == "C0 No monitoring"].set_index(["rep", "fold"])["err_e2e"]
t3 = []
for name in CONFIGS:
    g = loop[loop.config == name].set_index(["rep", "fold"])
    red = 100 * (1 - g["err_e2e"] / base)
    lo, hi = map(float, RED_CI[name])
    t3.append({"Configuration": name, "false_alarm_rate": g.false_alarm_rate.mean(), "false_alarm_sd": g.false_alarm_rate.std(ddof=1),
               "detection_rate": g.detection_rate.mean(), "detection_sd": g.detection_rate.std(ddof=1),
               "err_faulty": g.err_faulty.mean(), "err_normal": g.err_normal.mean(),
               "err_e2e": g.err_e2e.mean(), "err_e2e_sd": g.err_e2e.std(ddof=1),
               "reduction_pct": red.mean(), "reduction_ci_lo": lo, "reduction_ci_hi": hi})
t3 = pd.DataFrame(t3)
# prevalencia de equilibrio: fracción mínima de ciclos con falla a partir de la cual el lazo reduce el error
e0 = t3.loc[0, "err_faulty"]
t3["breakeven_prevalence"] = [np.nan if n.startswith("C0") else (en / (en + e0 - ef) if e0 > ef else np.inf)
                              for n, en, ef in zip(t3.Configuration, t3.err_normal, t3.err_faulty)]
elig = ~t3.Configuration.str.startswith(("Oracle", "C0")).to_numpy()      # el mejor/segundo se marcan entre C1..C6
tex = pd.DataFrame({
    "Configuration": t3.Configuration,
    f"False alarms {DOWN}": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t3.false_alarm_rate, t3.false_alarm_sd)],
    "Detection $\\uparrow$": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t3.detection_rate, t3.detection_sd)],
    f"Error, faulty {DOWN}": mark_best(t3.err_faulty, higher_better=False, eligible=elig),
    f"Error, normal {DOWN}": mark_best(t3.err_normal, higher_better=False, eligible=elig),
    f"End-to-end error {DOWN}": mark_best(t3.err_e2e, t3.err_e2e_sd, higher_better=False, eligible=elig),
    "$\\Delta$ vs. C0 (\\%) [95\\% CI] $\\uparrow$": [f"{a:.1f} [{b:.1f}, {c:.1f}]" if not n.startswith("C0") else "--"
                                                       for n, a, b, c in zip(t3.Configuration, t3.reduction_pct, t3.reduction_ci_lo, t3.reduction_ci_hi)],
    f"Break-even prevalence (\\%) {DOWN}": ["--" if np.isnan(v) else ("none" if np.isinf(v) else f"{100 * v:.1f}") for v in t3.breakeven_prevalence]})
t3_csv = t3.copy()
t3_csv["breakeven_prevalence"] = [NA if np.isnan(v) else ("never" if np.isinf(v) else v) for v in t3.breakeven_prevalence]
save_table(t3_csv, "table_03_ablation_study", tex, column_format="lccccccc")

# --- tabla 8: auto-optimización del umbral (falsas alarmas logradas vs objetivo) ---
t8 = []
for mode, lab in (("fixed", "Fixed threshold 0.5"), ("conformal", "Static conformal"), ("online", "Online adaptive")):
    for a_, g in alph[alph["mode"] == mode].groupby("alpha", dropna=False):
        t8.append({"Threshold": lab, "target_alpha": a_, "false_alarm_mean": g.false_alarm_rate.mean(),
                   "false_alarm_sd": g.false_alarm_rate.std(ddof=1), "detection_mean": g.detection_rate.mean(),
                   "detection_sd": g.detection_rate.std(ddof=1)})
t8 = pd.DataFrame(t8)
t8_csv = t8.copy(); t8_csv["target_alpha"] = [NA if np.isnan(v) else v for v in t8.target_alpha]
save_table(t8_csv, "table_08_threshold_self_optimisation", pd.DataFrame({
    "Threshold": t8.Threshold, "Target $\\alpha$": ["--" if np.isnan(v) else f"{v:.2f}" for v in t8.target_alpha],
    f"Achieved false-alarm rate {DOWN}": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t8.false_alarm_mean, t8.false_alarm_sd)],
    "Detection rate $\\uparrow$": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t8.detection_mean, t8.detection_sd)]}), column_format="lccc")

# --- tablas 9 y 10: sensibilidad a la severidad y al ruido residual (AUROC medio) ---
def sens_table(kind, name, levels, fmt):
    g = sens[sens.kind == kind]
    rows = []
    for f in g.fault.unique():
        for det, col in (("Random forest", "auroc"), ("QC rules", "auroc_rules")):
            m = g[g.fault == f].groupby("level")[col].mean()
            rows.append({"Fault": FAULT_LABEL[f], "Detector": det, **{fmt(l): m[l] for l in levels}})
    df = pd.DataFrame(rows)
    tex = df.copy()
    for c in tex.columns[2:]:
        tex[c] = df[c].map("{:.3f}".format)
    save_table(df, name, tex, column_format="ll" + "c" * len(levels))
    return df

t9 = sens_table("severity", "table_09_sensitivity_severity", SEVS, lambda l: f"s={l:g}")
t10 = sens_table("jitter", "table_10_sensitivity_residual_noise", JITS, lambda l: f"{100 * l:g}\\%")

# --- figura 3: (a) ablación, (b) auto-optimización, (c) severidad, (d) tamaño de entrenamiento ---
fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.6), gridspec_kw={"hspace": 0.5, "wspace": 0.42})
names = list(CONFIGS)
means = [loop[loop.config == n].err_e2e.mean() for n in names]
cis = [tuple(map(float, E2E_CI[n])) for n in names]
cols = ["0.6", OKABE_ITO[0]] + [OKABE_ITO[1]] * 3 + [OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[5], "0.25"]
short = [n.split(" ")[0] for n in names]
ax[0, 0].bar(short, means, color=cols, edgecolor="k", linewidth=0.5,
             yerr=[[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]], capsize=2, error_kw={"lw": 0.8})
ax[0, 0].set_ylabel("End-to-end error (SD units)"); ax[0, 0].set_xlabel("Loop configuration")
ax[0, 0].tick_params(axis="x", labelsize=8)
add_panel_label(ax[0, 0], "a", x=-0.2)

for mode, lab, col, mk in (("conformal", "Static conformal", OKABE_ITO[4], "o"), ("online", "Online adaptive", OKABE_ITO[5], "s")):
    g = alph[alph["mode"] == mode].groupby("alpha")["false_alarm_rate"]
    ax[0, 1].errorbar(ALPHAS, g.mean().values, yerr=g.std(ddof=1).values,
                      marker=mk, ms=4, lw=1.2, capsize=2, color=col, label=lab)
fx = alph[alph["mode"] == "fixed"]["false_alarm_rate"].mean()
ax[0, 1].axhline(fx, color=OKABE_ITO[1], ls="--", lw=1, label="Fixed threshold 0.5")
ax[0, 1].plot([0, 0.22], [0, 0.22], color="0.4", ls=":", lw=0.8, label="Target = achieved")
ax[0, 1].set_xlabel("Target false-alarm rate $\\alpha$"); ax[0, 1].set_ylabel("Achieved false-alarm rate")
ax[0, 1].set_ylim(-0.02, 0.33); ax[0, 1].legend(fontsize=8, loc="upper left", ncol=2, columnspacing=0.8); add_panel_label(ax[0, 1], "b", x=-0.2)

sv = sens[sens.kind == "severity"]
for i, f in enumerate(("stuck", "drift", "spike", "noise")):
    g = sv[sv.fault == f].groupby("level")
    ax[1, 0].plot(SEVS, g.auroc.mean().values, marker="os^D"[i], ms=4, lw=1.2, color=OKABE_ITO[[4, 5, 2, 6][i]], label=FAULT_LABEL[f])
    ax[1, 0].plot(SEVS, g.auroc_rules.mean().values, ls=":", lw=1, color=OKABE_ITO[[4, 5, 2, 6][i]])
ax[1, 0].set_xlabel("Fault severity level"); ax[1, 0].set_ylabel("AUROC")
ax[1, 0].legend(fontsize=8, loc="lower right"); add_panel_label(ax[1, 0], "c", x=-0.2)

fs = pd.read_csv(RESULTS_DIR / "fold_scores.csv")
for m, col, mk in ((PROPOSED, OKABE_ITO[5], "o"), ("QC-Rules", OKABE_ITO[0], "s"), ("OneClassSVM", OKABE_ITO[2], "^")):
    g = fs[fs.model == m].groupby("n_train_cycles")["macro_auroc"]
    ax[1, 1].errorbar(g.mean().index, g.mean().values, yerr=g.std(ddof=1).values, marker=mk, ms=4, lw=1.2, capsize=2, color=col, label=m)
ax[1, 1].set_xlabel("Training cycles (expanding window)"); ax[1, 1].set_ylabel("Macro-AUROC")
ax[1, 1].set_xticks([28, 56, 84, 112, 140]); ax[1, 1].legend(fontsize=8, loc="lower right"); add_panel_label(ax[1, 1], "d", x=-0.2)
save_figure(fig, "fig_03_ablation_and_sensitivity")

print(t3.round(3).to_string())
print(alph.groupby(["mode", "alpha"], dropna=False)[["false_alarm_rate", "detection_rate"]].mean().round(3).to_string())
print(sens[sens.kind != "hyper"].groupby(["kind", "fault", "level"])[["auroc", "auroc_rules"]].mean().round(3).to_string())
print(sens[sens.kind == "hyper"].pivot_table(index="level", columns="fault", values="auroc").round(4).to_string())
print(heal.groupby("channel")[["unrepaired", "profile", "model", "type_aware"]].mean().round(3).to_string())
print(heal.groupby("ftype")[["unrepaired", "profile", "model", "type_aware"]].mean().round(3).to_string())
