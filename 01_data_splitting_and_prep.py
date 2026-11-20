"""01 — carga, esquema de validación temporal, resumen de cohorte y figura de datos."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (DATA_PATH, DOUBLE_COL, N_REPEATS, N_SPLITS, OKABE_ITO, PROJECT_DIR, RESULTS_DIR, SEED,
                   add_panel_label, save_figure, save_table)
from autonomic_monitor import (CH_LABEL, CHANNELS, FAULT_LABEL, FAULTS, N_READ, WindowFeatures, flat, inject,
                               load_windows)

W, blocks = load_windows(DATA_PATH)
scale = W.reshape(-1, 3).std(axis=0)
n_blocks = int(blocks.max()) + 1
assert n_blocks == N_SPLITS + 1

# --- esquema de validación: ventana expansiva por bloques cronológicos (sin barajar) ---
splits = [{"fold": f, "train_blocks": list(range(f + 1)), "test_block": f + 1,
           "n_train_cycles": int((blocks <= f).sum()), "n_test_cycles": int((blocks == f + 1).sum())}
          for f in range(N_SPLITS)]
np.savez(RESULTS_DIR / "windows.npz", W=W, blocks=blocks, scale=scale)
(RESULTS_DIR / "cv_scheme.json").write_text(json.dumps(
    {"seed": SEED, "n_repeats": N_REPEATS, "splits": splits, "faults": FAULTS,
     "note": "Bloque de prueba siempre posterior a los de entrenamiento; el perfil normal, el escalado y "
             "todos los modelos se ajustan solo con los bloques de entrenamiento."}, indent=2))

# --- tabla 1: resumen de cohorte ---
rows = []
for c, ch in enumerate(CHANNELS):
    v = W[:, :, c]
    lag1 = np.corrcoef(v[:, :-1].ravel(), v[:, 1:].ravel())[0, 1]
    day = np.corrcoef(v.mean(1)[:-1], v.mean(1)[1:])[0, 1]
    rows.append({"Channel": CH_LABEL[ch], "Source column": ch, "n readings": v.size, "Mean": v.mean(),
                 "SD": v.std(), "Min": v.min(), "Max": v.max(),
                 "Within-cycle lag-1 r": lag1, "Between-cycle lag-1 r": day})
t1 = pd.DataFrame(rows)
t1_tex = t1.copy()
for col in ["Mean", "SD", "Min", "Max", "Within-cycle lag-1 r", "Between-cycle lag-1 r"]:
    t1_tex[col] = t1[col].map(lambda x: f"{x:.3f}")
t1_tex["Source column"] = t1_tex["Source column"].map(lambda s: r"\texttt{" + s.replace("_", r"\_") + "}")
save_table(t1, "table_01_cohort_summary", t1_tex)

audit = pd.read_csv(PROJECT_DIR / "data_provenance_audit.csv")
cohort = {"n_cycles": int(len(W)), "readings_per_cycle": N_READ, "n_channels": 3, "n_real_readings": int(W.size // 3),
          "n_blocks": n_blocks, "cycles_per_block": 28, "source_rows": 4383,
          "original_frac_channels": float(audit.set_index("column").loc[CHANNELS, "original_frac"].mean()),
          "ramp_frac_channels": float(audit.set_index("column").loc[CHANNELS, "linear_ramp_frac"].mean()),
          "natural_zero_diffs": int((np.abs(np.diff(W, axis=1)) < 1e-9).sum()),
          "scale_sd": dict(zip(CHANNELS, map(float, scale)))}
(RESULTS_DIR / "cohort.json").write_text(json.dumps(cohort, indent=2))

# --- figura 1: perfiles normales por canal (arriba) y ejemplos de cada falla (abajo) ---
fig = plt.figure(figsize=(DOUBLE_COL, 6.0))
gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1, 1], hspace=0.8, wspace=2.6)
SHORT = {"oxigeno_scaled": "DO", "ph": "pH", "turbidez": "Turbidity"}
x = np.arange(N_READ)
for c, ch in enumerate(CHANNELS):
    ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
    for i in range(0, len(W), 4):
        ax.plot(x, W[i, :, c], color="0.75", lw=0.4, alpha=0.6)
    q = np.quantile(W[:, :, c], [0.1, 0.5, 0.9], axis=0)
    ax.fill_between(x, q[0], q[2], color=OKABE_ITO[c + 4], alpha=0.25, lw=0)
    ax.plot(x, q[1], color=OKABE_ITO[c + 4], lw=1.8, marker="o", ms=3)
    ax.set_xlabel("Reading index"); ax.set_ylabel(f"{SHORT[ch]} channel (a.u.)", fontsize=9); ax.set_xticks(x)
    ax.tick_params(labelsize=8); add_panel_label(ax, "abc"[c], x=-0.38)
rng = np.random.default_rng(SEED)
demo = int(np.argmin(np.abs(W[:, :, 0].std(1) - np.median(W[:, :, 0].std(1)))))
for j, f in enumerate(FAULTS):
    ax = fig.add_subplot(gs[1 + j // 3, 2 * (j % 3):2 * (j % 3) + 2])
    wf, k, m, s = inject(W[demo], f, rng, scale, severity=2.0, channel=0)
    ax.plot(x, W[demo, :, 0], color="0.35", lw=1.2, marker="o", ms=3, label="Clean")
    ax.plot(x, wf[:, 0], color=OKABE_ITO[5], lw=1.2, ls="--", marker="s", ms=3, label="Faulty")
    ax.set_xticks(x); ax.tick_params(labelsize=8)
    ax.set_xlabel("Reading index" if j >= 3 else ""); ax.set_ylabel("DO channel (a.u.)" if j % 3 == 0 else "", fontsize=9)
    ax.text(0.98, 0.04, FAULT_LABEL[f], transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    add_panel_label(ax, "defghi"[j], x=-0.38)
    if j == 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8)
save_figure(fig, "fig_01_data_distribution_or_manifold")

# --- comprobación anti-fuga: el transformador no ve el bloque de prueba ---
tr = blocks <= 0
fx = WindowFeatures().fit(flat(W[tr]))
assert np.allclose(fx.profile_mean_, W[tr].mean(0))
print(t1.round(3).to_string()); print(json.dumps(cohort, indent=1)); print(pd.DataFrame(splits).to_string())
