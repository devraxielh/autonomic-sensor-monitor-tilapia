"""06 — análisis de robustez pedidos por la revisión.

(a) Lazo con otros detectores: ¿las conclusiones dependen de haber elegido Random Forest?
(b) Ajuste anidado con igual presupuesto: ¿el ranking de detectores sobrevive al ajuste de hiperparámetros?
Ninguno cambia el protocolo principal (hiperparámetros fijos); se reportan como comprobaciones.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

from utils import (ALPHA_FA, DOWN, N_SPLITS, PREVALENCE, RESULTS_DIR, UP, mark_best, rep_fold_seed, save_table)
from autonomic_monitor import (FAULTS, SUPERVISED, AutonomicMonitor, Detector, make_dataset, run_loop_config,
                               window_errors)

d = np.load(RESULTS_DIR / "windows.npz")
W, blocks, scale = d["W"], d["blocks"], d["scale"]
t0 = time.time()


def macro_auroc(te, s):
    normal = te["ftype"] == "normal"
    return float(np.mean([roc_auc_score(te["y"][normal | (te["ftype"] == f)], s[normal | (te["ftype"] == f)]) for f in FAULTS]))


# ---------------- (a) el lazo con otros detectores ----------------
LOOP_DETECTORS, LOOP_REPS = ["RandomForest", "CatBoost", "LogisticRegression"], 5
LOOP_CFG = {"C0": None, "C4": ("ml", "fixed", "type_aware"), "C4y": ("ml", "youden", "type_aware"),
            "C5": ("ml", "conformal", "type_aware"), "C6": ("ml", "online", "type_aware")}
rows = []
for det in LOOP_DETECTORS:
    for rep in range(LOOP_REPS):
        for fold in range(N_SPLITS):
            tr = make_dataset(W, np.where(blocks <= fold)[0], rep_fold_seed(rep, fold, 0), scale)
            te = make_dataset(W, np.where(blocks == fold + 1)[0], rep_fold_seed(rep, fold, 1), scale)
            mon = AutonomicMonitor(det, rep_fold_seed(rep, fold, 2), ALPHA_FA).fit(tr)
            a = mon.analyze(te["X"])
            normal, faulty = te["y"] == 0, te["y"] == 1
            for name, cfg in LOOP_CFG.items():
                alarm, out = run_loop_config(cfg, mon, a, te, ALPHA_FA)
                e = window_errors(out, te, scale)
                rows.append({"detector": det, "config": name, "rep": rep, "fold": fold, "fa": alarm[normal].mean(),
                             "det": alarm[faulty].mean(),
                             "e2e": (1 - PREVALENCE) * e[normal].mean() + PREVALENCE * e[faulty].mean()})
    print(f"lazo con {det} listo ({time.time() - t0:.0f}s)", flush=True)
lr = pd.DataFrame(rows); lr.to_csv(RESULTS_DIR / "loop_detector_robustness.csv", index=False)
t12 = lr.groupby(["detector", "config"], sort=False).agg(fa=("fa", "mean"), det=("det", "mean"), e2e=("e2e", "mean"),
                                                         e2e_sd=("e2e", lambda v: v.std(ddof=1))).reset_index()
t12["reduction_pct"] = [100 * (1 - e / t12[(t12.detector == dt) & (t12.config == "C0")].e2e.iloc[0])
                        for dt, e in zip(t12.detector, t12.e2e)]
tex = t12.copy()
e2e_txt = []
for dt in LOOP_DETECTORS:
    g = t12[t12.detector == dt]
    e2e_txt += mark_best(g.e2e, g.e2e_sd, higher_better=False, eligible=(g.config != "C0").to_numpy())
save_table(t12, "table_12_loop_detector_robustness", pd.DataFrame({
    "Detector of the loop": t12.detector, "Configuration": t12.config,
    f"False alarms {DOWN}": t12.fa.map("{:.3f}".format), f"Detection {UP}": t12.det.map("{:.3f}".format),
    f"End-to-end error {DOWN}": e2e_txt, f"$\\Delta$ vs. C0 (\\%) {UP}": t12.reduction_pct.map("{:.1f}".format)}),
    column_format="llcccc")

# ---------------- (b) ajuste anidado con igual presupuesto ----------------
SPACES = {
    "LogisticRegression": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "MLP": {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 3e-3]},
    "RandomForest": {"max_depth": [None, 4, 8, 12], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.3, 0.6]},
    "ExtraTrees": {"max_depth": [None, 4, 8, 12], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.3, 0.6]},
    "XGBoost": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.1, 0.3], "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.5, 0.9]},
    "LightGBM": {"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.1, 0.3], "min_child_samples": [3, 5, 10], "colsample_bytree": [0.5, 0.9]},
    "CatBoost": {"depth": [3, 4, 6], "learning_rate": [0.03, 0.1, 0.3], "l2_leaf_reg": [1, 3, 10]},
}
N_ITER, N_INNER, TUNE_REPS = 8, 3, 3
rows = []
for rep in range(TUNE_REPS):
    for fold in range(N_SPLITS):
        tr = make_dataset(W, np.where(blocks <= fold)[0], rep_fold_seed(rep, fold, 0), scale)
        te = make_dataset(W, np.where(blocks == fold + 1)[0], rep_fold_seed(rep, fold, 1), scale)
        inner = np.array_split(np.unique(tr["cycle"]), N_INNER)          # grupos contiguos de ciclos de entrenamiento
        seed = rep_fold_seed(rep, fold, 2)
        for name in SUPERVISED:
            default = macro_auroc(te, Detector(name, seed).fit(tr["X"], tr["y"]).score(te["X"]))
            best, best_params = -np.inf, None
            for params in ParameterSampler(SPACES[name], n_iter=min(N_ITER, int(np.prod([len(v) for v in SPACES[name].values()]))),
                                           random_state=rep_fold_seed(rep, fold, 5)):
                sc = []
                for held in inner:
                    va = np.isin(tr["cycle"], held)
                    s = Detector(name, seed, **params).fit(tr["X"][~va], tr["y"][~va]).score(tr["X"][va])
                    sc.append(roc_auc_score(tr["y"][va], s))
                if np.mean(sc) > best:
                    best, best_params = float(np.mean(sc)), params
            tuned = macro_auroc(te, Detector(name, seed, **best_params).fit(tr["X"], tr["y"]).score(te["X"]))
            rows.append({"model": name, "rep": rep, "fold": fold, "default": default, "tuned": tuned, "best_params": str(best_params)})
    print(f"ajuste anidado rep {rep + 1}/{TUNE_REPS} ({time.time() - t0:.0f}s)", flush=True)
tn = pd.DataFrame(rows); tn.to_csv(RESULTS_DIR / "nested_tuning.csv", index=False)
t13 = tn.groupby("model").agg(default=("default", "mean"), default_sd=("default", lambda v: v.std(ddof=1)),
                              tuned=("tuned", "mean"), tuned_sd=("tuned", lambda v: v.std(ddof=1))).reset_index()
t13["gain"] = t13.tuned - t13.default
t13["rank_default"] = t13["default"].rank(ascending=False).astype(int)
t13["rank_tuned"] = t13["tuned"].rank(ascending=False).astype(int)
t13 = t13.sort_values("default", ascending=False).reset_index(drop=True)
save_table(t13, "table_13_tuning_robustness", pd.DataFrame({
    "Detector": t13.model, f"Default macro-AUROC {UP}": mark_best(t13["default"], t13.default_sd),
    f"Tuned macro-AUROC {UP}": mark_best(t13.tuned, t13.tuned_sd), "Gain from tuning": t13.gain.map("{:+.3f}".format),
    "Rank (default)": t13.rank_default, "Rank (tuned)": t13.rank_tuned}), column_format="lccccc")
pd.set_option("display.width", 250)
print(t12.round(3).to_string()); print(t13.round(4).to_string())
