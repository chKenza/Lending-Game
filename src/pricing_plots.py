"""Part 6 - plots of the pricing-rule tuning, used by analysis/pricing_rules.ipynb.

Colours follow the entity, never its rank: each tuning scenario T1-T6 always
has the same colour, the weighted profit is always dark ink, each lender
always the same colour.
"""

import numpy as np
import pandas as pd

from src.model_plots import INK, INK_SECONDARY, REFERENCE, SURFACE

# Categorical palette, fixed order (reference palette slots 1-6).
SCENARIO_COLORS = {"T1": "#2a78d6", "T2": "#eb6834", "T3": "#1baf7a",
                   "T4": "#eda100", "T5": "#e87ba4", "T6": "#008300"}
SCENARIO_LABELS = {"T1": "T1: margin 0", "T2": "T2: margin +2 pts",
                   "T3": "T3: margin +4 pts", "T4": "T4: markup 15%",
                   "T5": "T5: markup 35%", "T6": "T6: best-response cycle"}
LENDER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#6250d6"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot_best_response(table, curves, axes):
    """3a. Left: our profit against our margin, one line per competitor
    margin. Right: our best margin against the competitors' margin."""
    ax = axes[0]
    for color, mc in zip(SERIES_COLORS, curves.columns):
        ax.plot(100 * curves.index, curves[mc] / 1e3, color=color,
                label=f"competitors +{100 * mc:g} pts")
        best = table.loc[mc]
        ax.plot(100 * best["best_margin"], best["best_profit"] / 1e3, "o", ms=7,
                color=color, mec=SURFACE, mew=1.5)
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xlabel("our margin over break-even (points)")
    ax.set_ylabel("our profit (thousand EUR)")
    ax.set_title("R1: our profit against our margin")
    ax.legend(labelcolor=INK_SECONDARY)

    ax = axes[1]
    ax.plot(100 * table.index, 100 * table["best_margin"], "o-", color=LENDER_COLORS[3], ms=7,
            mec=SURFACE, mew=1.5)
    lim = 100 * max(table.index.max(), table["best_margin"].max()) + 1
    ax.plot([0, lim], [0, lim], color=REFERENCE, lw=1, ls="--", label="same margin as competitors")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("competitors' margin (points)")
    ax.set_ylabel("our best margin (points)")
    ax.set_title("Best-response curve")
    ax.legend(labelcolor=INK_SECONDARY)


def plot_equilibrium_path(history, ax, cycle_mean=None):
    """3b. Each lender's margin after each round of best responses; dashed
    lines at each lender's average over the cycle, if given."""
    for k in (1, 2, 3):
        label = "us (lender 3)" if k == 3 else f"lender {k}"
        ax.plot(history.index, 100 * history[f"m{k}"], "o-", color=LENDER_COLORS[k], ms=7,
                mec=SURFACE, mew=1.5, label=label)
        if cycle_mean is not None:
            ax.axhline(100 * cycle_mean[k - 1], color=LENDER_COLORS[k], lw=1, ls="--")
    ax.set_xticks(history.index)
    ax.set_xlabel("round")
    ax.set_ylabel("margin over break-even (points)")
    ax.set_title("Best-response dynamics" + ("" if cycle_mean is None else " (dashed: cycle averages)"))
    ax.legend(labelcolor=INK_SECONDARY)


def slice_results(results, param, fixed):
    """Rows of a grid_search result where every parameter except `param` is
    at its value in `fixed`, sorted by `param`."""
    mask = np.ones(len(results), dtype=bool)
    for k, v in fixed.items():
        if k != param and k in results.columns:
            mask &= np.isclose(results[k].astype(float), float(v))
    return results[mask].sort_values(param)


def plot_profit_curves(results, param, fixed, weights, ax, title, scale=100,
                       xlabel=None):
    """3e. Our profit against one parameter (the others at their tuned
    value), one line per scenario plus the weighted profit.

    scale : multiply the parameter for display (100 for points or percent).
    """
    sl = slice_results(results, param, fixed)
    x = scale * sl[param].astype(float)
    for name in weights:
        ax.plot(x, sl[f"profit_{name}"] / 1e3, color=SCENARIO_COLORS[name], lw=1.5,
                marker="o", ms=3, label=SCENARIO_LABELS[name])
    ax.plot(x, sl["weighted"] / 1e3, color=INK, lw=3, marker="o", ms=4, label="weighted")
    ax.axvline(scale * float(fixed[param]), color=INK_SECONDARY, lw=1, ls=":")
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xlabel(xlabel or param)
    ax.set_ylabel("our profit (thousand EUR)")
    ax.set_title(title)


def plot_bin_margins(table, margin_curve, ax, title="R4: best margin per risk decile"):
    """R4. Best margin of each decile and the smoothed increasing curve."""
    ax.plot(100 * table["centre"], 100 * table["best_margin"], "o", ms=8,
            color=LENDER_COLORS[3], mec=SURFACE, mew=1.5, label="best margin per decile")
    grid = np.linspace(table["p_low"].min(), table["p_high"].max(), 200)
    ax.plot(100 * grid, 100 * margin_curve(grid), color=INK, lw=2, label="smoothed curve m(p)")
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xlabel("our predicted default probability p_own (%)")
    ax.set_ylabel("margin over break-even (points)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)


def plot_heatmap(table, ax, title, kind="diverging", scale=1e3, unit="thousand EUR",
                 fmt="{:+,.0f}"):
    """Rules (rows) x scenarios (columns) heatmap with the value in each cell.

    kind "diverging"  : profit; two hues around a neutral 0 (red = loss, blue = gain)
    kind "sequential" : regret; one hue, light = small, dark = large
    """
    import matplotlib.colors as mcolors
    values = table.to_numpy(dtype=float) / scale
    if kind == "diverging":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "loss_gain", ["#c23b3a", "#f2b8b5", "#f3f2ef", "#9ec5f4", "#1c5cab"])
        vmax = np.nanmax(np.abs(values))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "regret", ["#f3f2ef", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
        norm = mcolors.Normalize(vmin=0, vmax=np.nanmax(values))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            dark = abs(norm(v) - (0.5 if kind == "diverging" else 0)) > 0.35
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)
    ax.set_title(title)
    return im


def plot_bootstrap_intervals(table, ax, title):
    """Forest plot of bootstrap differences: one row per comparison, point
    estimate with its 95% interval (thousand EUR)."""
    t = table.iloc[::-1]
    y = np.arange(len(t))
    ax.errorbar(t["estimate"] / 1e3, y,
                xerr=[(t["estimate"] - t["lower"]) / 1e3, (t["upper"] - t["estimate"]) / 1e3],
                fmt="o", ms=6, color=LENDER_COLORS[3], ecolor=LENDER_COLORS[3], elinewidth=1.5,
                capsize=3, mec=SURFACE)
    ax.axvline(0, color=REFERENCE, lw=1, ls="--")
    ax.set_yticks(y, t.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("profit difference (thousand EUR, 95% bootstrap interval)")
    ax.set_title(title)

