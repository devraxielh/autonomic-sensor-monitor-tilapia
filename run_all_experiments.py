"""run_all_experiments.py — ejecuta 01 -> 05 desde cero y verifica los criterios de salida.

Uso:  python scripts/run_all_experiments.py            (ejecución completa)
      python scripts/run_all_experiments.py --check    (solo verificación de salidas)
"""
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_data_splitting_and_prep.py", "02_baseline_and_sota_models.py", "03_ablation_and_sensitivity.py",
         "04_statistical_tests.py", "05_explainability_and_error.py", "06_robustness_checks.py"]
TABLES = ["table_01_cohort_summary", "table_02_main_results_cv", "table_03_ablation_study",
          "table_04_statistical_significance", "table_05_diagnosis", "table_06_healing_by_channel",
          "table_07_computational_cost", "table_08_threshold_self_optimisation", "table_09_sensitivity_severity",
          "table_10_sensitivity_residual_noise", "table_11_case_study_source_file",
          "table_12_loop_detector_robustness", "table_13_tuning_robustness"]
FIGURES = ["fig_01_data_distribution_or_manifold", "fig_02_main_performance_comparison",
           "fig_03_ablation_and_sensitivity", "fig_04_explainability_shap_or_latent",
           "fig_05_error_calibration_or_case_study"]


def run() -> None:
    subprocess.run([sys.executable, str(HERE.parent / "prepare_dataset.py")], check=True)
    for d in ("results", "tables", "figures"):
        shutil.rmtree(HERE / d, ignore_errors=True)
    for s in STEPS:
        t0 = time.time()
        print(f"\n=== {s} ===", flush=True)
        subprocess.run([sys.executable, s], cwd=HERE, check=True)
        print(f"--- {s}: {time.time() - t0:.0f}s", flush=True)
    sync_paper_figures()


def sync_paper_figures() -> None:
    """Deja paper/figures/ idéntica a scripts/figures/ (copia todo y borra huérfanas)."""
    sys.path.insert(0, str(HERE))
    import utils

    copied = utils.sync_figures_to_paper()
    print(f"figuras copiadas a paper/figures/: {len(copied)}", flush=True)
    copied = utils.sync_tables_to_paper()
    print(f"tablas copiadas a paper/tables/: {len(copied)}", flush=True)


def check() -> bool:
    from PIL import Image
    ok = True

    def report(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("[OK]   " if cond else "[FAIL] ") + msg)

    for t in TABLES:
        report((HERE / "tables" / f"{t}.csv").exists() and (HERE / "tables" / f"{t}.tex").exists(), f"tabla {t} (.csv + .tex)")
    for f in FIGURES:
        png, pdf = HERE / "figures" / f"{f}.png", HERE / "figures" / f"{f}.pdf"
        dpi = round(Image.open(png).info.get("dpi", (0, 0))[0]) if png.exists() else 0
        report(png.exists() and pdf.exists() and dpi >= 600, f"figura {f} (.pdf + .png, {dpi} dpi)")
    paper_figs = HERE.parent / "paper" / "figures"
    if paper_figs.parent.exists():
        same = all((paper_figs / f.name).exists() and (paper_figs / f.name).read_bytes() == f.read_bytes()
                   for f in (HERE / "figures").glob("fig_*.*"))
        report(same, "paper/figures/ contiene una copia idéntica de cada figura de scripts/figures/")
        paper_tabs = HERE.parent / "paper" / "tables"
        same = all((paper_tabs / f.name).exists() and (paper_tabs / f.name).read_bytes() == f.read_bytes()
                   for f in (HERE / "tables").glob("table_*.*"))
        report(same, "paper/tables/ contiene una copia idéntica de cada tabla de scripts/tables/")
    bad = []
    for p in HERE.glob("*.py"):
        if p.name == "run_all_experiments.py":
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r"(set_title|plt\.title|suptitle)\(([^)]*)\)", line):
                if m.group(2).strip().split(",")[0] not in ('""', "''"):
                    bad.append(f"{p.name}:{n}")
    report(not bad, "NO TITLES: sin títulos internos en figuras" + (f" -> {bad}" if bad else ""))
    import pandas as pd
    fs = pd.read_csv(HERE / "results" / "fold_scores.csv")
    report(fs.groupby("model").size().min() >= 50, "cada detector evaluado en 5 bloques x 10 repeticiones")
    audit = subprocess.run([sys.executable, str(HERE / "review" / "audit_assets.py")], capture_output=True, text=True)
    report(audit.returncode == 0, "auditoría de activos (review/audit_assets.py): " + audit.stdout.strip().splitlines()[-1])
    return ok


if __name__ == "__main__":
    if "--check" not in sys.argv:
        run()
    sys.exit(0 if check() else 1)
