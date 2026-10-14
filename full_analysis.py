"""
generate_figures.py
--------------------
Genera las figuras propuestas para el estudio de robustez del CVAI frente a
variabilidad estocástica en el posicionamiento de landmarks.

Entradas esperadas (mismo esquema que produce main_bash.py):

  results/cvai_uncertainty_summary.csv
      model, level, offset_mm, nominal_cvai, mean, std, p5, p95, cv,
      n_valid, n_invalid, n_samples

  results/cvai_uncertainty_samples.csv
      model, level, sample, cvai

Salidas: PNG en figures/, uno por cada bloque de la propuesta:
  1. cv_heatmap_model_level.png
  2. boxplot_by_level.png
  3. cv_band_by_level.png
  4. bland_altman.png
  5. misclassification_probability.png   (requiere samples)
  6. gray_zone_diagram.png               (requiere samples)
  7. category_flow.png                   (requiere samples, "alluvial" simplificado)

El script es tolerante a datasets parciales: si faltan niveles/modelos en
cvai_uncertainty_samples.csv simplemente usa lo disponible y avisa por consola.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from figure_style import apply_paper_style
apply_paper_style()

# ── Umbrales clínicos (Tabla 1 del informe) ─────────────────────────────────
THRESHOLDS = [3.4, 6.25, 8.75, 11.0]
CATEGORY_LABELS = [
    "Normal / Mínima",
    "Leve",
    "Moderada",
    "Mod. Severa",
    "Severa",
]
CATEGORY_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]


def categorize(cvai_values):
    """Asigna cada valor de CVAI a una categoría clínica (Tabla 1)."""
    cvai_values = np.asarray(cvai_values, dtype=float)
    idx = np.searchsorted(THRESHOLDS, cvai_values, side="right")
    return idx  # 0..4, indexa CATEGORY_LABELS


def load_data(summary_path: Path, samples_path: Path | None):
    summary = pd.read_csv(summary_path)
    samples = None
    if samples_path is not None and samples_path.exists():
        samples = pd.read_csv(samples_path)
    return summary, samples


# ── 1. Heatmap CV(%) modelo x nivel ─────────────────────────────────────────
def plot_cv_heatmap(summary: pd.DataFrame, out_dir: Path):
    pivot = summary.pivot_table(index="model", columns="level", values="cv")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1] + 3),
                                     max(4, 0.35 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma_r")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Nivel craneal")
    ax.set_ylabel("Modelo (phantom)")
    ax.set_title("Coeficiente de variación del CVAI (%) por modelo y nivel")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CV (%)")

    fig.tight_layout()
    fig.savefig(out_dir / "cv_heatmap_model_level.png", dpi=200)
    plt.close(fig)


# ── 2. Boxplot de CVAI por nivel (todas las muestras disponibles) ──────────
def plot_boxplot_by_level(samples: pd.DataFrame, out_dir: Path):
    levels = sorted(samples["level"].unique())
    data = [samples.loc[samples["level"] == lv, "cvai"].values for lv in levels]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(levels) + 2), 5))
    ax.boxplot(data, tick_labels=levels, showfliers=False)

    for i, thr in enumerate(THRESHOLDS):
        ax.axhline(thr, color="grey", linestyle="--", linewidth=0.8)
        ax.text(len(levels) + 0.55, thr, f"{thr}%", va="center",
                fontsize=7, color="grey", clip_on=False)

    ax.set_xlabel("Nivel craneal")
    ax.set_ylabel("CVAI (%)")
    ax.set_title("Distribución del CVAI bajo perturbación de landmarks, por nivel")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_by_level.png", dpi=200)
    plt.close(fig)


# ── 3. Banda P5-P95 de CVAI vs nivel, para modelos seleccionados ───────────
def plot_cv_band_by_level(summary: pd.DataFrame, out_dir: Path, models=None, max_models=4):
    models = models or summary["model"].unique()[:max_models]

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = summary[summary["model"] == m].sort_values("level")
        if sub.empty:
            continue
        ax.plot(sub["level"], sub["mean"], marker="o", label=m)
        ax.fill_between(sub["level"], sub["p5"], sub["p95"], alpha=0.15)

    for thr in THRESHOLDS:
        ax.axhline(thr, color="grey", linestyle="--", linewidth=0.7)

    ax.set_xlabel("Nivel craneal")
    ax.set_ylabel("CVAI (%) [media y banda P5-P95]")
    ax.set_title("Estabilidad del CVAI por nivel craneal")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_band_by_level.png", dpi=200)
    plt.close(fig)


# ── 4. Bland-Altman: nominal vs media de las muestras ───────────────────────
def plot_bland_altman(summary: pd.DataFrame, out_dir: Path):
    nominal = summary["nominal_cvai"].values
    mean_sim = summary["mean"].values
    diff = mean_sim - nominal
    avg = (mean_sim + nominal) / 2

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(avg, diff, s=14, alpha=0.7)
    ax.axhline(bias, color="black", linewidth=1)
    ax.axhline(loa_upper, color="red", linestyle="--", linewidth=1)
    ax.axhline(loa_lower, color="red", linestyle="--", linewidth=1)

    ax.text(ax.get_xlim()[1], bias, f" sesgo={bias:.2f}", va="center", fontsize=7)
    ax.text(ax.get_xlim()[1], loa_upper, f" +1.96SD={loa_upper:.2f}", va="center", fontsize=7)
    ax.text(ax.get_xlim()[1], loa_lower, f" -1.96SD={loa_lower:.2f}", va="center", fontsize=7)

    ax.set_xlabel("Media (nominal, media simulada) (%)")
    ax.set_ylabel("Media simulada − nominal (%)")
    ax.set_title("Bland–Altman: CVAI nominal vs media bajo incertidumbre")
    fig.tight_layout()
    fig.savefig(out_dir / "bland_altman.png", dpi=200)
    plt.close(fig)


# ── 5. Probabilidad de reclasificación diagnóstica ──────────────────────────
def plot_misclassification_probability(summary: pd.DataFrame, samples: pd.DataFrame, out_dir: Path):
    rows = []
    for (model, level), grp in samples.groupby(["model", "level"]):
        nom_row = summary[(summary["model"] == model) & (summary["level"] == level)]
        if nom_row.empty:
            continue
        nominal = nom_row["nominal_cvai"].values[0]
        nominal_cat = categorize([nominal])[0]
        sample_cats = categorize(grp["cvai"].values)
        p_misclass = np.mean(sample_cats != nominal_cat)
        rows.append({"model": model, "level": level, "nominal_cvai": nominal,
                      "p_misclass": p_misclass, "n": len(grp)})

    if not rows:
        print("  [misclassification] No hay solapamiento model/level entre "
              "summary y samples: figura omitida.")
        return

    res = pd.DataFrame(rows).sort_values("nominal_cvai")

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(res["nominal_cvai"], res["p_misclass"] * 100,
                     c=res["level"], cmap="viridis", s=40)
    for thr in THRESHOLDS:
        ax.axvline(thr, color="grey", linestyle="--", linewidth=0.7)

    ax.set_xlabel("CVAI nominal (%)")
    ax.set_ylabel("P(reclasificación de categoría) (%)")
    ax.set_title("Probabilidad de cambio de categoría diagnóstica bajo incertidumbre")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Nivel craneal")
    fig.tight_layout()
    fig.savefig(out_dir / "misclassification_probability.png", dpi=200)
    plt.close(fig)
    return res


# ── 6. Diagrama de "zona gris" diagnóstica ──────────────────────────────────
def plot_gray_zone(samples: pd.DataFrame, out_dir: Path, grid_points=200, p_thresh=0.10):
    all_cvai = samples["cvai"].values
    if len(all_cvai) < 10:
        print("  [gray zone] Muestras insuficientes: figura omitida.")
        return

    lo, hi = np.percentile(all_cvai, [1, 99])
    grid = np.linspace(lo, hi, grid_points)

    # Estimación empírica: para cada punto x de la rejilla, probabilidad de que
    # una medición "verdadera" en x observe un valor perturbado en categoría
    # distinta, usando el error empírico global (nominal vs muestras) como
    # proxy de la distribución de error a esa escala. Con más datos por
    # modelo/nivel esto se sustituye por la distribución local real.
    residuals = samples.groupby(["model", "level"])["cvai"].transform(
        lambda s: s - s.mean()
    ).values
    errs = np.random.choice(residuals, size=(grid_points, 300), replace=True)
    perturbed = grid[:, None] + errs
    base_cat = categorize(grid)
    pert_cat = categorize(perturbed.ravel()).reshape(perturbed.shape)
    p_cross = np.mean(pert_cat != base_cat[:, None], axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, p_cross * 100, color="black", linewidth=1.5)
    ax.axhline(p_thresh * 100, color="red", linestyle="--", linewidth=1,
               label=f"Umbral de zona gris ({p_thresh*100:.0f}%)")

    gray = p_cross >= p_thresh
    ax.fill_between(grid, 0, p_cross * 100, where=gray, color="red", alpha=0.15)

    for thr in THRESHOLDS:
        ax.axvline(thr, color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("CVAI (%)")
    ax.set_ylabel("P(cruce de umbral) (%)")
    ax.set_title("Zonas grises diagnósticas del CVAI bajo incertidumbre de landmarks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gray_zone_diagram.png", dpi=200)
    plt.close(fig)


# ── 7. Flujo de categorías (nominal -> muestras), tipo alluvial simplificado ─
# ── 7b. Riesgo de reclasificación y CV agregados por nivel craneal ─────────
def plot_reclassification_by_level(summary: pd.DataFrame, samples: pd.DataFrame,
                                     out_dir: Path, reference_level: int = 3):
    """
    Extiende el análisis de un único nivel (level3_misclassification_report)
    a los 7 niveles craneales: para cada nivel, calcula el % de muestras
    perturbadas que cae en una categoría distinta a la nominal (agregado
    sobre los 74 modelos) y el CV medio del CVAI en ese nivel.

    reference_level : nivel resaltado como plano de medición estándar
        (se dibuja en un color distinto en el panel izquierdo).
    """
    levels = sorted(summary["level"].unique())
    pct_reclass, mean_cv, mean_offset = [], [], []

    for level in levels:
        summ_lv = summary[summary["level"] == level]
        samp_lv = samples[samples["level"] == level]

        n_reclass, n_total = 0, 0
        for model, grp in samp_lv.groupby("model"):
            nom_row = summ_lv[summ_lv["model"] == model]
            if nom_row.empty:
                continue
            nominal_cat = categorize(nom_row["nominal_cvai"].values)[0]
            sample_cats = categorize(grp["cvai"].values)
            n_reclass += int((sample_cats != nominal_cat).sum())
            n_total += len(sample_cats)

        pct_reclass.append(100 * n_reclass / n_total if n_total else np.nan)
        mean_cv.append(summ_lv["cv"].mean())
        mean_offset.append(summ_lv["offset_mm"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # -- Panel izquierdo: % de reclasificación por nivel --
    ax = axes[0]
    colors = ["#DD8452" if lvl == reference_level else "#4C72B0" for lvl in levels]
    bars = ax.bar(levels, pct_reclass, color=colors, alpha=0.85, width=0.6)
    for b, v in zip(bars, pct_reclass):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Nivel craneal")
    ax.set_ylabel("Muestras reclasificadas (%)")
    ax.set_title("Riesgo de reclasificación por nivel")
    ax.set_xticks(levels)
    handles = [mpatches.Patch(color="#DD8452", label=f"Nivel {reference_level} (referencia)"),
               mpatches.Patch(color="#4C72B0", label="Otros niveles")]
    ax.legend(handles=handles, fontsize=8)

    # -- Panel derecho: CV medio por nivel, con offset_mm como eje superior --
    ax2 = axes[1]
    ax2.plot(levels, mean_cv, marker="o", color="#55A868")
    ax2.set_xlabel("Nivel craneal")
    ax2.set_ylabel("CV medio del CVAI (%)")
    ax2.set_title("Variabilidad media por nivel")
    ax2.set_xticks(levels)

    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(levels)
    ax2_top.set_xticklabels([f"{o:.0f}" for o in mean_offset], fontsize=8)
    ax2_top.set_xlabel("Desplazamiento medio respecto al plano de referencia (mm)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "reclassification_by_level.png", dpi=200)
    plt.close(fig)


def plot_category_flow(summary: pd.DataFrame, samples: pd.DataFrame, out_dir: Path,
                        level: int | None = None):
    """
    Matriz de transición de categoría diagnóstica (nominal vs. obtenida en
    las muestras perturbadas).

    level : si se indica, la matriz se calcula únicamente sobre ese nivel
        craneal (p.ej. level=3, el plano de medición estándar). Si es None
        (por defecto), se agregan todos los niveles disponibles.
    """
    summary_lv = summary if level is None else summary[summary["level"] == level]
    samples_lv = samples if level is None else samples[samples["level"] == level]

    rows = []
    for (model, lvl), grp in samples_lv.groupby(["model", "level"]):
        nom_row = summary_lv[(summary_lv["model"] == model) & (summary_lv["level"] == lvl)]
        if nom_row.empty:
            continue
        nominal_cat = categorize(nom_row["nominal_cvai"].values)[0]
        sample_cats = categorize(grp["cvai"].values)
        for c in sample_cats:
            rows.append((nominal_cat, c))

    if not rows:
        print("  [category flow] No hay solapamiento model/level: figura omitida.")
        return

    flow = pd.DataFrame(rows, columns=["from_cat", "to_cat"])
    n_cats = len(CATEGORY_LABELS)
    flow_matrix = np.zeros((n_cats, n_cats))
    for f, t in zip(flow["from_cat"], flow["to_cat"]):
        flow_matrix[f, t] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(flow_matrix, cmap="Blues")
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(CATEGORY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(CATEGORY_LABELS, fontsize=7)
    ax.set_xlabel("Categoría obtenida (muestra perturbada)")
    ax.set_ylabel("Categoría nominal")
    title = "Matriz de transición de categoría diagnóstica"
    title += " (todos los niveles)" if level is None else f" (nivel {level})"
    ax.set_title(title)

    for i in range(n_cats):
        for j in range(n_cats):
            v = int(flow_matrix[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > flow_matrix.max() / 2 else "black")

    fig.colorbar(im, ax=ax, label="N muestras")
    fig.tight_layout()
    out_name = "category_flow.png" if level is None else f"category_flow_level{level}.png"
    fig.savefig(out_dir / out_name, dpi=200)
    plt.close(fig)


# ── 8. Informe de reclasificación diagnóstica para un nivel concreto ────────
def level_misclassification_report(summary: pd.DataFrame, samples: pd.DataFrame,
                                     out_dir: Path, level: int = 3):
    """
    Para un nivel craneal dado (por defecto 3, el correspondiente a la
    medición tradicional con calibre/cefalómetro), calcula por modelo:
      - CVAI nominal y categoría nominal
      - P(reclasificación) empírica a partir de las N muestras perturbadas
      - hacia qué categoría(s) migran las muestras que sí cambian
    y agrega estadísticos globales sobre el conjunto de modelos.

    Devuelve el DataFrame por modelo y escribe:
      - level{N}_misclassification_table.csv
      - level{N}_misclassification_report.png
    """
    summ_lv = summary[summary["level"] == level]
    samp_lv = samples[samples["level"] == level]

    rows = []
    for model, grp in samp_lv.groupby("model"):
        nom_row = summ_lv[summ_lv["model"] == model]
        if nom_row.empty:
            continue
        nominal = nom_row["nominal_cvai"].values[0]
        nominal_cat = int(categorize([nominal])[0])
        sample_cats = categorize(grp["cvai"].values)
        n = len(sample_cats)
        p_misclass = float(np.mean(sample_cats != nominal_cat))

        # distribución de destino entre las muestras que SÍ cambian de categoría
        changed = sample_cats[sample_cats != nominal_cat]
        if len(changed) > 0:
            vals, counts = np.unique(changed, return_counts=True)
            dest = {CATEGORY_LABELS[v]: int(c) for v, c in zip(vals, counts)}
        else:
            dest = {}

        dist_to_threshold = min(abs(nominal - t) for t in THRESHOLDS)

        rows.append({
            "model": model,
            "nominal_cvai": nominal,
            "nominal_category": CATEGORY_LABELS[nominal_cat],
            "dist_to_nearest_threshold": dist_to_threshold,
            "n_samples": n,
            "p_misclassification": p_misclass,
            "destination_categories": dest,
        })

    if not rows:
        print(f"  [level {level} report] No hay solapamiento model/level "
              f"entre summary y samples para el nivel {level}: se omite.")
        return None

    res = pd.DataFrame(rows).sort_values("nominal_cvai").reset_index(drop=True)

    # ── estadísticos globales ──
    n_models = len(res)
    pooled_p = float(np.average(res["p_misclassification"], weights=res["n_samples"]))
    mean_p = float(res["p_misclassification"].mean())
    frac_over_10 = float((res["p_misclassification"] > 0.10).mean())
    frac_over_25 = float((res["p_misclassification"] > 0.25).mean())

    by_cat = res.groupby("nominal_category")["p_misclassification"].agg(
        ["mean", "std", "count"]
    ).reindex(CATEGORY_LABELS).dropna(how="all")

    print(f"\n=== Nivel {level}: resumen sobre {n_models} modelos ===")
    print(f"P(reclasificación) media (no ponderada): {mean_p*100:.1f}%")
    print(f"P(reclasificación) ponderada por nº de muestras: {pooled_p*100:.1f}%")
    print(f"% de modelos con P(reclasificación) > 10%: {frac_over_10*100:.1f}%")
    print(f"% de modelos con P(reclasificación) > 25%: {frac_over_25*100:.1f}%")
    print("\nPor categoría nominal:")
    print(by_cat.to_string())

    res_out = res.drop(columns=["destination_categories"]).copy()
    res_out.to_csv(out_dir / f"level{level}_misclassification_table.csv", index=False)

    # ── figura: dos paneles ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(res["nominal_cvai"], res["p_misclassification"] * 100,
                     c=res["dist_to_nearest_threshold"], cmap="coolwarm_r", s=30)
    for thr in THRESHOLDS:
        ax.axvline(thr, color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel("CVAI nominal en nivel %d (%%)" % level)
    ax.set_ylabel("P(reclasificación diagnóstica) (%)")
    ax.set_title(f"Nivel {level}: riesgo de cambio diagnóstico ({n_models} modelos)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Distancia al umbral más cercano (%)")

    ax2 = axes[1]
    ax2.hist(res["p_misclassification"] * 100, bins=20, color="#4C72B0", edgecolor="white")
    ax2.axvline(mean_p * 100, color="black", linestyle="-", linewidth=1.2,
                label=f"media={mean_p*100:.1f}%")
    ax2.set_xlabel("P(reclasificación) por modelo (%)")
    ax2.set_ylabel("Nº de modelos")
    ax2.set_title("Distribución del riesgo de reclasificación entre modelos")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"level{level}_misclassification_report.png", dpi=200)
    plt.close(fig)

    return res


# ── 9. Direccionalidad de la reclasificación (sobre- vs infra-estimación) ──
def directionality_report(summary: pd.DataFrame, samples: pd.DataFrame,
                            out_dir: Path, level: int = 3):
    """
    Para cada modelo en `level`, de las muestras que cambian de categoría
    respecto a la nominal, calcula cuántas suben de severidad (el ruido de
    landmark hace parecer el cráneo MÁS asimétrico de lo que es) y cuántas
    bajan (lo hace parecer MENOS asimétrico). Esto es clínicamente relevante:
    sobreestimación → posible sobretratamiento; infraestimación → casos que
    pasan desapercibidos.

    Escribe level{N}_directionality_table.csv y level{N}_directionality.png.
    Devuelve el DataFrame por modelo.
    """
    summ_lv = summary[summary["level"] == level]
    samp_lv = samples[samples["level"] == level]

    rows = []
    for model, grp in samp_lv.groupby("model"):
        nom_row = summ_lv[summ_lv["model"] == model]
        if nom_row.empty:
            continue
        nominal = nom_row["nominal_cvai"].values[0]
        nominal_cat = int(categorize([nominal])[0])
        sample_cats = categorize(grp["cvai"].values)
        n = len(sample_cats)

        n_up = int(np.sum(sample_cats > nominal_cat))    # más severo que lo real
        n_down = int(np.sum(sample_cats < nominal_cat))  # menos severo que lo real
        n_same = n - n_up - n_down

        rows.append({
            "model": model,
            "nominal_cvai": nominal,
            "nominal_category": CATEGORY_LABELS[nominal_cat],
            "n_samples": n,
            "p_overestimation": n_up / n,     # sube de categoría
            "p_underestimation": n_down / n,  # baja de categoría
            "p_misclassification": (n_up + n_down) / n,
            "net_bias": (n_up - n_down) / n,  # >0: sesgo a sobreestimar; <0: a infraestimar
        })

    if not rows:
        print(f"  [directionality] Sin datos para nivel {level}.")
        return None

    res = pd.DataFrame(rows).sort_values("nominal_cvai").reset_index(drop=True)
    res.to_csv(out_dir / f"level{level}_directionality_table.csv", index=False)

    n_models = len(res)
    mean_over = res["p_overestimation"].mean()
    mean_under = res["p_underestimation"].mean()
    mean_bias = res["net_bias"].mean()
    # Wilcoxon-signed-rank-like check simple: proporcion de modelos con sesgo neto en cada sentido
    frac_over_biased = (res["net_bias"] > 0.02).mean()
    frac_under_biased = (res["net_bias"] < -0.02).mean()
    frac_balanced = 1 - frac_over_biased - frac_under_biased

    print(f"\n=== Nivel {level}: direccionalidad de la reclasificación ({n_models} modelos) ===")
    print(f"P(sobreestimación de categoría) media: {mean_over*100:.1f}%")
    print(f"P(infraestimación de categoría) media: {mean_under*100:.1f}%")
    print(f"Sesgo neto medio (over - under): {mean_bias*100:+.1f} puntos")
    print(f"% modelos con sesgo neto hacia sobreestimar (>2pp): {frac_over_biased*100:.1f}%")
    print(f"% modelos con sesgo neto hacia infraestimar (>2pp): {frac_under_biased*100:.1f}%")
    print(f"% modelos aprox. simétricos: {frac_balanced*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(n_models)
    ax.bar(x, res["p_overestimation"] * 100, color="#C44E52", label="Sobreestimación (sube categoría)")
    ax.bar(x, -res["p_underestimation"] * 100, color="#4C72B0", label="Infraestimación (baja categoría)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Modelos (ordenados por CVAI nominal)")
    ax.set_ylabel("P(cambio de categoría) (%)")
    ax.set_title(f"Direccionalidad de la reclasificación — nivel {level}")
    ax.legend(fontsize=8, loc="upper left")

    ax2 = axes[1]
    ax2.scatter(res["nominal_cvai"], res["net_bias"] * 100, s=35,
                c=res["net_bias"], cmap="coolwarm", edgecolor="k", linewidth=0.3)
    ax2.axhline(0, color="black", linewidth=0.8)
    for thr in THRESHOLDS:
        ax2.axvline(thr, color="grey", linestyle="--", linewidth=0.7)
    ax2.set_xlabel("CVAI nominal (%)")
    ax2.set_ylabel("Sesgo neto (sobre − infra) (puntos %)")
    ax2.set_title("¿Hacia dónde se sesga el error, según severidad nominal?")

    fig.tight_layout()
    fig.savefig(out_dir / f"level{level}_directionality.png", dpi=200)
    plt.close(fig)

    return res


# ── 10. CVAI obtenido en muestras vs CVAI nominal, por modelo ──────────────
def plot_samples_vs_nominal_boxplot(summary: pd.DataFrame, samples: pd.DataFrame,
                                      out_dir: Path, level: int = 3,
                                      landscape: bool = False):
    """
    Boxplot con un bigote por modelo: eje X = CVAI nominal del modelo,
    eje Y = distribución de los CVAI obtenidos en las 500 muestras
    perturbadas. Los bigotes se colocan en su posición real de CVAI nominal
    (no en posiciones categóricas equiespaciadas), así que su dispersión a lo
    largo del eje X y su anchura relativa son comparables entre sí.

    landscape : si True, genera la figura con proporciones pensadas para
        ocupar una página completa apaisada (más ancha que alta).
    """
    summ_lv = summary[summary["level"] == level]
    samp_lv = samples[samples["level"] == level]

    data, positions, models = [], [], []
    for model, grp in samp_lv.groupby("model"):
        nom_row = summ_lv[summ_lv["model"] == model]
        if nom_row.empty:
            continue
        nominal = nom_row["nominal_cvai"].values[0]
        data.append(grp["cvai"].values)
        positions.append(nominal)
        models.append(model)

    if not data:
        print(f"  [samples vs nominal] Sin datos para nivel {level}.")
        return

    order = np.argsort(positions)
    data = [data[i] for i in order]
    positions = [positions[i] for i in order]
    models = [models[i] for i in order]

    # anchura de caja proporcional al espaciado típico entre modelos vecinos,
    # para que no se solapen mucho pero sigan siendo visibles
    span = max(positions) - min(positions)
    width = max(span / len(positions) * 0.6, 0.05)

    if landscape:
        # A4 apaisado ~11.7 x 8.3in; dejamos margen para márgenes/caption
        figsize = (11, 5.5)
    else:
        figsize = (max(10, 0.18 * len(positions) + 4), 6)

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(data, positions=positions, widths=width, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.5, linewidth=0.6),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))

    # boxplot() fija por defecto un tick con el valor exacto en cada caja;
    # con 70+ modelos eso satura el eje, así que lo sustituimos por una
    # rejilla numérica limpia.
    lims = [min(positions) - 1, max(positions) + 1]
    ticks = np.linspace(np.floor(lims[0]), np.ceil(lims[1]), 8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    ax.set_xlim(lims)
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1,
            label="Identidad (obtenido = nominal)")
    ax.plot([], [], color="#C44E52", linestyle="--", linewidth=1.1,
            label="Umbrales diagnósticos")

    for thr in THRESHOLDS:
        if lims[0] <= thr <= lims[1]:
            ax.axvline(thr, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.axhline(thr, color="#C44E52", linestyle="--", linewidth=1.1, alpha=0.8, zorder=0)
        ax.text(lims[1], thr, f" {thr}%", color="#C44E52", fontsize=8,
                va="center", ha="left", clip_on=False)

    ax.set_xlabel(f"CVAI nominal, nivel {level} (%)")
    ax.set_ylabel(f"CVAI obtenido en las muestras perturbadas, nivel {level} (%)")
    ax.set_title(f"Dispersión del CVAI muestreado vs CVAI nominal — "
                 f"{len(positions)} modelos, nivel {level}")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / f"level{level}_samples_vs_nominal_boxplot.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                         default=Path("results/cvai_uncertainty_summary.csv"))
    parser.add_argument("--samples", type=Path,
                         default=Path("results/cvai_uncertainty_samples.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--report-level", type=int, default=3,
                         help="Nivel craneal para el informe de reclasificación "
                              "por modelo (por defecto 3, medición tradicional).")
    parser.add_argument("--landscape", action="store_true",
                         help="Genera el boxplot samples-vs-nominal con proporciones "
                              "de página completa apaisada.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.summary.exists():
        sys.exit(f"No se encuentra {args.summary}")

    summary, samples = load_data(args.summary, args.samples)

    print(f"Summary: {len(summary)} filas, {summary['model'].nunique()} modelos, "
          f"niveles {sorted(summary['level'].unique())}")

    print("→ cv_heatmap_model_level.png")
    plot_cv_heatmap(summary, args.out)

    print("→ bland_altman.png")
    plot_bland_altman(summary, args.out)

    print("→ cv_band_by_level.png")
    plot_cv_band_by_level(summary, args.out)

    if samples is not None and not samples.empty:
        print(f"Samples: {len(samples)} filas")
        print("→ boxplot_by_level.png")
        plot_boxplot_by_level(samples, args.out)

        print("→ misclassification_probability.png")
        plot_misclassification_probability(summary, samples, args.out)

        print("→ gray_zone_diagram.png")
        plot_gray_zone(samples, args.out)

        print(f"→ category_flow_level{args.report_level}.png")
        plot_category_flow(summary, samples, args.out, level=args.report_level)

        print("→ reclassification_by_level.png")
        plot_reclassification_by_level(summary, samples, args.out,
                                         reference_level=args.report_level)

        print(f"→ level{args.report_level}_misclassification_report.png "
              f"+ level{args.report_level}_misclassification_table.csv")
        level_misclassification_report(summary, samples, args.out, level=args.report_level)

        print(f"→ level{args.report_level}_directionality.png "
              f"+ level{args.report_level}_directionality_table.csv")
        directionality_report(summary, samples, args.out, level=args.report_level)

        print(f"→ level{args.report_level}_samples_vs_nominal_boxplot.png")
        plot_samples_vs_nominal_boxplot(summary, samples, args.out, level=args.report_level,
                                          landscape=args.landscape)
    else:
        print("No se ha proporcionado (o está vacío) el CSV de muestras: "
              "se omiten las figuras 5-7, que dependen de la distribución "
              "completa por modelo/nivel.")

    print(f"\nFiguras guardadas en: {args.out.resolve()}")


if __name__ == "__main__":
    main()
    