"""04 — inferencia estadística.

Inferencia principal: bootstrap por conglomerados. Las 50 evaluaciones (5 bloques x 10 repeticiones
de la inyección) comparten los mismos 28 ciclos reales por bloque y no son independientes; por eso se
remuestrean los ciclos reales dentro de cada bloque (todas las versiones inyectadas y repeticiones de
un ciclo se mantienen juntas) y se reporta el IC 95 % de cada diferencia.
Descriptivo: Friedman y Wilcoxon-Holm sobre las 50 evaluaciones (p-values optimistas).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from utils import (N_BOOTSTRAP, NA, PRIMARY_METRIC, PROPOSED, RESULTS_DIR, cluster_bootstrap, cycle_tensor, holm,
                   rank_biserial_wilcoxon, save_table, stat_e2e, stat_macro_auroc)
from autonomic_monitor import DETECTORS, FAULTS

TYPES = ["normal"] + FAULTS
tensor = lambda df, value, keys: cycle_tensor(df, value, keys, TYPES)
macro_auroc, e2e_error = stat_macro_auroc, stat_e2e


# ---------------- detectores ----------------
fs = pd.read_csv(RESULTS_DIR / "fold_scores.csv")
models = [m for m in DETECTORS if m in fs.model.unique()]
piv = fs.pivot_table(index=["rep", "fold"], columns="model", values=PRIMARY_METRIC)[models]
piv_fold = fs.pivot_table(index="fold", columns="model", values=PRIMARY_METRIC)[models]
stat, p_fr = friedmanchisquare(*[piv[c].values for c in piv.columns])
ranks = pd.DataFrame(rankdata(-piv.values, axis=1), columns=piv.columns).mean()

oof = pd.read_parquet(RESULTS_DIR / "oof_predictions.parquet")
long = oof.melt(id_vars=["rep", "fold", "cycle", "ftype"], value_vars=models, var_name="model", value_name="score")
long["model"] = pd.Categorical(long.model, categories=models, ordered=True)
A = tensor(long, "score", ["model"])
obs, boots = cluster_bootstrap(macro_auroc, A)
assert np.allclose(obs, piv.mean().values, atol=1e-9), "el macro-AUROC reconstruido no coincide con fold_scores"
ip = models.index(PROPOSED)

rows = [{"family": "Detectors", "comparison": f"Friedman ({len(models)} detectors, n={len(piv)})", "observed_diff": NA,
         "ci_lo": NA, "ci_hi": NA, "ci_excludes_zero": NA, "statistic": stat, "p_value": p_fr, "p_holm": NA,
         "effect_size": NA, "folds_won": NA, "outcome": "Detectors differ (descriptive)"}]
comp = []
for j, other in enumerate(models):
    if other == PROPOSED:
        continue
    a, b = piv[PROPOSED].values, piv[other].values
    w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    d = boots[:, ip] - boots[:, j]
    lo, hi = np.quantile(d, [0.025, 0.975])
    comp.append({"family": "Detectors", "comparison": f"{PROPOSED} vs. {other}", "observed_diff": obs[ip] - obs[j],
                 "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0), "statistic": w.statistic,
                 "p_value": w.pvalue, "effect_size": rank_biserial_wilcoxon(a, b),
                 "folds_won": f"{int((piv_fold[PROPOSED] > piv_fold[other]).sum())}/{len(piv_fold)}"})
for c, pa in zip(comp, holm([c["p_value"] for c in comp])):
    c["p_holm"] = pa
    c["outcome"] = ("Loop detector higher" if c["observed_diff"] > 0 else "Loop detector lower") if c["ci_excludes_zero"] else "Not distinguishable"
rows += sorted(comp, key=lambda r: -r["observed_diff"])

# ---------------- configuraciones del lazo ----------------
cyc = pd.read_parquet(RESULTS_DIR / "loop_cycle_errors.parquet")
configs = list(dict.fromkeys(cyc.config))
cyc["config"] = pd.Categorical(cyc.config, categories=configs, ordered=True)
E = tensor(cyc, "err", ["config"])
obs_e, boots_e = cluster_bootstrap(e2e_error, E)
loop = pd.read_csv(RESULTS_DIR / "loop_scores.csv"); loop["config"] = loop.config.str.split(" ").str[0]
lp = loop.pivot_table(index=["rep", "fold"], columns="config", values="err_e2e")
lp_fold = loop.pivot_table(index="fold", columns="config", values="err_e2e")
assert np.allclose(obs_e, lp[configs].mean().values, atol=1e-9), "el error reconstruido no coincide con loop_scores"
pairs = [("C6", "C0"), ("C6", "C1"), ("C6", "C4"), ("C6", "C4y"), ("C6", "C5"), ("C5", "C4y"), ("C4y", "C4"),
         ("C4y", "C0"), ("C4", "C3"), ("C3", "C2"), ("C4", "C0")]
comp = []
for x, y in pairs:
    a, b = lp[x].values, lp[y].values
    w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    d = boots_e[:, configs.index(x)] - boots_e[:, configs.index(y)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    comp.append({"family": "Loop (end-to-end error)", "comparison": f"{x} vs. {y}",
                 "observed_diff": obs_e[configs.index(x)] - obs_e[configs.index(y)], "ci_lo": lo, "ci_hi": hi,
                 "ci_excludes_zero": bool(lo > 0 or hi < 0), "statistic": w.statistic, "p_value": w.pvalue,
                 "effect_size": rank_biserial_wilcoxon(b, a),
                 "folds_won": f"{int((lp_fold[x] < lp_fold[y]).sum())}/{len(lp_fold)}"})
for c, pa in zip(comp, holm([c["p_value"] for c in comp])):
    c["p_holm"] = pa
    c["outcome"] = ("First lower error" if c["observed_diff"] < 0 else "First higher error") if c["ci_excludes_zero"] else "Not distinguishable"
rows += comp

t4 = pd.DataFrame(rows)
num = lambda v, f: "--" if isinstance(v, str) else f.format(v)
tex = pd.DataFrame({
    "Family": t4.family, "Comparison": t4.comparison,
    "Difference [95\\% cluster-bootstrap CI]": [
        "--" if isinstance(d_, str) else f"{d_:+.4f} [{lo:+.4f}, {hi:+.4f}]" for d_, lo, hi in zip(t4.observed_diff, t4.ci_lo, t4.ci_hi)],
    "Statistic": t4.statistic.map("{:.1f}".format), "$p$": t4.p_value.map("{:.2e}".format),
    "$p_{\\mathrm{Holm}}$": [num(v, "{:.2e}") for v in t4.p_holm],
    "$r_{rb}$": [num(v, "{:.2f}") for v in t4.effect_size],
    "Blocks won": [("--" if v == NA else v) for v in t4.folds_won], "Outcome (from CI)": t4.outcome})
save_table(t4, "table_04_statistical_significance", tex, column_format="llcccccll")
ranks.sort_values().rename("mean_rank").to_csv(RESULTS_DIR / "detector_mean_ranks.csv")
(RESULTS_DIR / "inference_summary.json").write_text(json.dumps({
    "n_bootstrap": N_BOOTSTRAP, "resampling_unit": "real test cycle within block (28 per block, 5 blocks)",
    "friedman_chi2": stat, "friedman_p": p_fr, "min_two_sided_wilcoxon_p_with_5_blocks": 0.0625}, indent=2))
pd.set_option("display.width", 250)
print(t4[["comparison", "observed_diff", "ci_lo", "ci_hi", "ci_excludes_zero", "p_holm", "folds_won", "outcome"]].to_string())
print(ranks.sort_values().round(2).to_string())
