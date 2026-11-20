"""00_config_and_utils.py — configuración global, estilo Q1 y utilidades compartidas.

Proyecto: monitor autonómico (MAPE-K) de fallas e integridad de sensores IoT de calidad
de agua en cultivo de tilapia, con propiedades self-healing y self-optimization.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 2000
N_SPLITS = 5            # bloques cronológicos de prueba (ventana expansiva, 28 ciclos c/u)
N_REPEATS = 10          # repeticiones de la inyección de fallas por bloque
PREVALENCE = 0.15       # fracción de ciclos con falla en el flujo operativo simulado
ALPHA_FA = 0.05         # presupuesto de falsas alarmas del umbral auto-optimizado
PRIMARY_METRIC = "macro_auroc"
PROPOSED = "RandomForest"   # detector del lazo propuesto, fijado a priori


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_global_seed()

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_PATH = PROJECT_DIR / "dataset_ml_ready.csv"
RESULTS_DIR = SCRIPTS_DIR / "results"
TABLES_DIR = SCRIPTS_DIR / "tables"
FIGURES_DIR = SCRIPTS_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_DIR / "paper" / "figures"   # copia para que paper/ sea autocontenida
PAPER_TABLES_DIR = PROJECT_DIR / "paper" / "tables"
for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Estilo gráfico Q1 (sin títulos)
# ----------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
SINGLE_COL = 3.5   # pulgadas
DOUBLE_COL = 7.0


def set_publication_style(family: str = "serif") -> None:
    sns.set_theme(context="paper", style="ticks", palette=OKABE_ITO)
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


set_publication_style()


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Etiqueta de subfigura (a), (b)… en la esquina superior izquierda."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _strip_titles(fig) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")


def save_figure(fig, name: str, formats=("pdf", "png"), dpi: int = 600) -> list[Path]:
    """Guarda en vectorial + PNG 600 DPI, eliminando cualquier título residual."""
    _strip_titles(fig)
    for ax in fig.get_axes():
        sns.despine(ax=ax)
    paths = []
    for fmt in formats:
        p = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    sync_figures_to_paper([p.name for p in paths])
    return paths


def sync_figures_to_paper(names=None) -> list[Path]:
    """Copia las figuras de scripts/figures/ a paper/figures/ (todas, o solo `names`).

    El artículo incluye las figuras desde paper/figures/, de modo que la carpeta paper/ se pueda
    compilar o subir a Overleaf sin depender de scripts/. Sin `names`, además borra de paper/figures/
    las figuras fig_* que ya no existan en scripts/figures/.
    """
    import shutil

    if not PAPER_FIGURES_DIR.parent.exists():
        return []
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    src = [FIGURES_DIR / n for n in names] if names else sorted(FIGURES_DIR.glob("fig_*.*"))
    for f in src:
        shutil.copy2(f, PAPER_FIGURES_DIR / f.name)
    if not names:
        keep = {f.name for f in src}
        for old in PAPER_FIGURES_DIR.glob("fig_*.*"):
            if old.name not in keep:
                old.unlink()
    return [PAPER_FIGURES_DIR / f.name for f in src]


# ----------------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------------
def _latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def save_table(df: pd.DataFrame, name: str, df_tex: pd.DataFrame | None = None,
               index: bool = False, column_format: str | None = None) -> None:
    """CSV con valores completos + .tex booktabs (df_tex permite una versión formateada)."""
    df.to_csv(TABLES_DIR / f"{name}.csv", index=index)
    tex_df = (df_tex if df_tex is not None else df).copy()
    tex_df.columns = [c if ("\\" in str(c) or "$" in str(c)) else _latex_escape(c) for c in tex_df.columns]
    for c in tex_df.select_dtypes(include="object").columns:
        tex_df[c] = tex_df[c].map(lambda v: v if "\\" in str(v) else _latex_escape(v))
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    cf = column_format or ("l" + "c" * (tex_df.shape[1] - 1))
    lines = [f"\\begin{{tabular}}{{{cf}}}", "\\toprule", " & ".join(map(str, tex_df.columns)) + " \\\\", "\\midrule"]
    lines += [" & ".join(fmt(v) for v in row) + " \\\\" for row in tex_df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex = "\n".join(lines)
    (TABLES_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    sync_tables_to_paper([f"{name}.csv", f"{name}.tex"])


def fmt_mean_sd(values, decimals: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{decimals}f} $\\pm$ {v.std(ddof=1):.{decimals}f}"


# ----------------------------------------------------------------------------
# Estadística
# ----------------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot: int = N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = np.array([stat(rng.choice(v, size=v.size, replace=True)) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rank_biserial_wilcoxon(x, y) -> float:
    """Tamaño del efecto r de rango-biserial para Wilcoxon pareado."""
    from scipy.stats import rankdata

    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


def holm(pvals) -> np.ndarray:
    """p-values ajustados por Holm-Bonferroni."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mean_ci(values, decimals: int = 3) -> str:
    """Media con IC 95 % bootstrap, formateada para LaTeX."""
    lo, hi = bootstrap_ci(values)
    return f"{np.mean(values):.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def rep_fold_seed(rep: int, fold: int, stream: int = 0) -> int:
    """Semilla determinista por (repetición, bloque, flujo)."""
    return SEED + 100_000 * stream + 1_000 * rep + fold


def mark_best(means, sds=None, higher_better: bool = True, decimals: int = 3, eligible=None) -> list[str]:
    """Formatea `media ± sd` con el mejor en negrita y el segundo subrayado (solo entre los elegibles)."""
    m = np.asarray(means, float)
    el = np.ones(len(m), bool) if eligible is None else np.asarray(eligible, bool)
    order = [i for i in (np.argsort(-m) if higher_better else np.argsort(m)) if el[i]]
    rnd = np.round(m, decimals)
    best = [i for i in order if rnd[i] == rnd[order[0]]]
    rest = [i for i in order if i not in best]
    second = [i for i in rest if rnd[i] == rnd[rest[0]]] if rest else []
    out = []
    for i, v in enumerate(m):
        txt = f"{v:.{decimals}f}"
        if i in best:
            txt = f"\\textbf{{{txt}}}"
        elif i in second:
            txt = f"\\underline{{{txt}}}"
        if sds is not None:
            txt += f" $\\pm$ {np.asarray(sds, float)[i]:.{decimals}f}"
        out.append(txt)
    return out


UP, DOWN = "$\\uparrow$", "$\\downarrow$"
NA = "not_applicable"


# ----------------------------------------------------------------------------
# Bootstrap por conglomerados (unidad: ciclo real de prueba dentro de cada bloque)
# ----------------------------------------------------------------------------
# Las 50 evaluaciones (5 bloques x 10 repeticiones de la inyección) comparten los mismos ciclos reales y
# no son independientes. Todo intervalo de confianza del proyecto se calcula remuestreando ciclos reales
# dentro de cada bloque, manteniendo juntas todas las versiones inyectadas y repeticiones de un ciclo.
def cycle_tensor(df: pd.DataFrame, value: str, keys: list, types: list) -> np.ndarray:
    """(…claves, rep, bloque, ciclo, tipo) con los ciclos ordenados dentro de cada bloque."""
    piv = df.pivot_table(index=keys + ["rep", "fold", "cycle"], columns="ftype", values=value, observed=True)[types]
    lead = [df[k].nunique() for k in keys]
    return piv.to_numpy().reshape(*lead, df.rep.nunique(), df.fold.nunique(), -1, len(types))


def stat_macro_auroc(A: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """A: (modelo, rep, bloque, ciclo, tipo[0=normal]) -> macro-AUROC por modelo."""
    per_fold = []
    for f in range(A.shape[2]):
        G = A[:, :, f, idx[f], :]
        sn, sf = G[..., 0], G[..., 1:]
        gt = (sf[:, :, :, None, :] > sn[:, :, None, :, None]) + 0.5 * (sf[:, :, :, None, :] == sn[:, :, None, :, None])
        per_fold.append(gt.mean(axis=(2, 3)).mean(axis=-1).mean(axis=-1))
    return np.mean(per_fold, axis=0)


def stat_e2e(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """E: (config, rep, bloque, ciclo, tipo[0=normal]) -> error de extremo a extremo por configuración."""
    per_fold = []
    for f in range(E.shape[2]):
        G = E[:, :, f, idx[f], :]
        per_fold.append(((1 - PREVALENCE) * G[..., 0].mean(-1) + PREVALENCE * G[..., 1:].mean((-1, -2))).mean(-1))
    return np.mean(per_fold, axis=0)


def stat_mean(T: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """T: (clave, rep, bloque, ciclo, tipo) -> media (ignorando NaN) por clave."""
    G = np.stack([T[:, :, f, idx[f], :] for f in range(T.shape[2])], axis=2)
    return np.nanmean(G.reshape(G.shape[0], -1), axis=1)


def cluster_bootstrap(stat, T: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    """Devuelve (valor observado, matriz de remuestreos). Mismos índices para toda llamada con igual forma."""
    rng = np.random.default_rng(seed)
    n_fold, n_cyc = T.shape[2], T.shape[3]
    obs = stat(T, np.tile(np.arange(n_cyc), (n_fold, 1)))
    boots = np.array([stat(T, rng.integers(0, n_cyc, size=(n_fold, n_cyc))) for _ in range(n_boot)])
    return obs, boots


def ci95(boots: np.ndarray):
    return np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


def sync_tables_to_paper(names=None) -> list[Path]:
    """Copia las tablas de scripts/tables/ a paper/tables/ (todas, o solo `names`).

    El artículo y el suplemento incluyen las tablas con \\input{tables/<archivo>}, de modo que paper/ no
    depende de scripts/. Sin `names`, además borra de paper/tables/ las table_* huérfanas.
    """
    import shutil

    if not PAPER_TABLES_DIR.parent.exists():
        return []
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    src = [TABLES_DIR / n for n in names] if names else sorted(TABLES_DIR.glob("table_*.*"))
    for f in src:
        shutil.copy2(f, PAPER_TABLES_DIR / f.name)
    if not names:
        keep = {f.name for f in src}
        for old in PAPER_TABLES_DIR.glob("table_*.*"):
            if old.name not in keep:
                old.unlink()
    return [PAPER_TABLES_DIR / f.name for f in src]
