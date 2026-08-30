"""
Generate publication-ready Matplotlib figures from evaluation metrics.

Reads a ``summary_report.json`` dict and saves PNGs to an output directory.
All figures use a colorblind-safe palette, clear axis labels, and 300 DPI.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "conformbench_matplotlib"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from .metric_views import aggregate_metric_views


# ── Palette (Okabe-Ito, colorblind-safe) ────────────────────────────────

_C = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "red":       "#D55E00",
    "purple":    "#CC79A7",
    "cyan":      "#56B4E9",
    "yellow":    "#F0E442",
    "grey":      "#999999",
}


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.figsize": (8, 4.5),
    })


def _short_id(scenario_id: str, questionnaire: str | None = None) -> str:
    """Shorten scenario IDs for axis labels, optionally prefixed with questionnaire."""
    label = scenario_id.replace("_", " ").title()[:30]
    if questionnaire:
        q_short = questionnaire.replace("_", " ").title()[:14]
        return f"{q_short}\n{label}"
    return label


def _pct_label(v: float) -> str:
    """Format a 0-1 float as e.g. '82.5%'."""
    return f"{v * 100:.1f}%"


def _annotate_bars(ax, xs, vals, fontsize=7, offset=0.01):
    """Place percentage annotations above a series of bars."""
    for xi, v in zip(xs, vals):
        if v is not None and not np.isnan(v):
            ax.text(xi, v + offset, _pct_label(v),
                    ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "sparse": 3, "ablation": 4}


def _difficulty_key(u: dict) -> int:
    """Extract difficulty rank from scenario_id prefix (EASY_… → 0, MEDIUM_… → 1, HARD_… → 2)."""
    sid = u.get("scenario_id", "")
    prefix = sid.split("_")[0].lower() if sid else ""
    return _DIFFICULTY_ORDER.get(prefix, 99)


def _sort_by_difficulty(utts: list[dict]) -> list[dict]:
    """Return utterances sorted by questionnaire then easy → medium → hard."""
    return sorted(utts, key=lambda u: (u.get("questionnaire", ""), _difficulty_key(u)))


def _has_multiple_questionnaires(utts: list[dict]) -> bool:
    """Check if utterances span more than one questionnaire."""
    qs = {u.get("questionnaire", "") for u in utts}
    return len(qs) > 1


def _label_for(u: dict, multi_q: bool) -> str:
    """Return a display label for a scenario, prefixed with questionnaire if mixed."""
    return _short_id(u["scenario_id"], u.get("questionnaire") if multi_q else None)


# ── Figure generators ───────────────────────────────────────────────────


def _fig_prf1_by_scenario(utts: list[dict], out_dir: Path) -> None:
    """Grouped bar: Strict P, R, F1 per scenario."""
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    prec = [u["field_prf"]["strict"]["precision"] for u in utts]
    rec  = [u["field_prf"]["strict"]["recall"] for u in utts]
    f1   = [u["field_prf"]["strict"]["f1"] for u in utts]

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x - w, prec, w, label="Precision", color=_C["blue"])
    ax.bar(x,     rec,  w, label="Recall",    color=_C["orange"])
    ax.bar(x + w, f1,   w, label="F1",        color=_C["green"])

    _annotate_bars(ax, x - w, prec)
    _annotate_bars(ax, x, rec)
    _annotate_bars(ax, x + w, f1)

    ax.set_ylabel("Score")
    ax.set_title("Strict Field-Level Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "prf1_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def _fig_f1_comparison(utts: list[dict], out_dir: Path) -> None:
    """Grouped bar: Strict / Lenient / Weighted F1 per scenario."""
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    strict  = [u["field_prf"]["strict"]["f1"] for u in utts]
    lenient = [u["field_prf"]["lenient"]["f1"] for u in utts]
    weighted = [u["field_prf"]["weighted"]["f1"] for u in utts]

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x - w, strict,   w, label="Strict F1",   color=_C["blue"])
    ax.bar(x,     lenient,  w, label="Lenient F1",   color=_C["cyan"])
    ax.bar(x + w, weighted, w, label="Weighted F1",  color=_C["green"])

    _annotate_bars(ax, x - w, strict)
    _annotate_bars(ax, x, lenient)
    _annotate_bars(ax, x + w, weighted)

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Comparison: Strict vs Lenient vs Weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "f1_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _fig_error_taxonomy(utts: list[dict], agg: dict, out_dir: Path) -> None:
    """Stacked horizontal bars: correct / partial (by reason) / incorrect (split)."""
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    n = len(labels)

    correct_vals = []
    partial_halluc = []
    partial_overspec = []
    partial_omission = []
    partial_wrong = []
    partial_other = []
    incorrect_real = []      # agent filled but genuinely wrong
    incorrect_unmatched = [] # GT instances agent never attempted

    for u in utts:
        sc = u["scores"]
        ev = sc["evaluated"] or 1
        correct_vals.append(sc["correct"] / ev)

        # Split partial by reason
        ph, pos, po, pw, pother = 0, 0, 0, 0, 0
        # Split incorrect: unmatched GT instances vs real errors
        i_unmatched, i_real = 0, 0
        for m in u.get("mismatches", []):
            if m["verdict"] == "partially_correct":
                pr = m.get("partial_reason", "")
                if pr == "hallucination":
                    ph += 1
                elif pr == "over_specified":
                    pos += 1
                elif pr == "omission":
                    po += 1
                elif pr == "wrong_choice":
                    pw += 1
                else:
                    pother += 1
            elif m["verdict"] == "incorrect":
                # unmatched GT instance fields have "[?gt" in qid
                if "[?gt" in m["qid"]:
                    i_unmatched += 1
                else:
                    i_real += 1
        partial_halluc.append(ph / ev)
        partial_overspec.append(pos / ev)
        partial_omission.append(po / ev)
        partial_wrong.append(pw / ev)
        partial_other.append(pother / ev)
        incorrect_real.append(i_real / ev)
        incorrect_unmatched.append(i_unmatched / ev)

    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, max(3, n * 1.2)))
    left = np.zeros(n)

    segments = [
        (correct_vals,        "Correct",                 _C["green"]),
        (partial_halluc,      "Partial: hallucination",  _C["orange"]),
        (partial_overspec,    "Partial: over-specified", _C["cyan"]),
        (partial_omission,    "Partial: omission",       _C["yellow"]),
        (partial_wrong,       "Partial: wrong choice",   _C["purple"]),
        (partial_other,       "Partial: other",          _C["grey"]),
        (incorrect_real,      "Incorrect (real error)",  _C["red"]),
        (incorrect_unmatched, "Not attempted (GT miss)", "#440000"),
    ]

    for vals, label, color in segments:
        arr = np.array(vals)
        if arr.sum() == 0:  # skip empty segments from legend
            continue
        ax.barh(y, arr, left=left, label=label, color=color, height=0.55)
        left += arr

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Proportion of evaluated fields")
    ax.set_title("Error Taxonomy by Scenario")
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=7)
    fig.tight_layout(rect=[0, 0, 0.68, 1])
    fig.savefig(out_dir / "error_taxonomy.png", bbox_inches="tight")
    plt.close(fig)


def _fig_source_provenance(utts: list[dict], out_dir: Path) -> None:
    """Stacked bar per scenario for candidate-support provenance."""
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    series = {
        "extracted": [],
        "likely_inferred": [],
        "prior_state": [],
        "unsupported_inference": [],
        "fabricated": [],
    }

    for u in utts:
        sc = u.get("source_counts", {})
        if "made_up" in sc and "fabricated" not in sc:
            sc = {**sc, "fabricated": sc.get("made_up", 0)}
        total = sum(sc.values()) or 1
        for key in series:
            series[key].append(sc.get(key, 0) / total)

    x = np.arange(len(labels))
    w = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    bottoms = np.zeros(len(labels))
    labels_and_colors = [
        ("extracted", "Extracted", _C["green"]),
        ("likely_inferred", "Licensed inference", _C["cyan"]),
        ("prior_state", "Prior state", _C["blue"]),
        ("unsupported_inference", "Unsupported inference", _C["orange"]),
        ("fabricated", "Fabricated", _C["red"]),
    ]
    for key, label, color in labels_and_colors:
        vals = np.array(series[key])
        ax.bar(x, vals, w, bottom=bottoms, label=label, color=color)
        bottoms += vals

    ax.set_ylabel("Proportion of sourced fields")
    ax.set_title("Candidate Support Provenance by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "source_provenance.png", bbox_inches="tight")
    plt.close(fig)


def _fig_instance_alignment(utts: list[dict], agg: dict, out_dir: Path) -> None:
    """Grouped bar: instance P, R, F1 per scenario."""
    # Collect per-scenario combined instance metrics
    scenarios = []
    prec_vals = []
    rec_vals = []
    f1_vals = []

    for u in _sort_by_difficulty(utts):
        inst_list = u.get("instance_alignment", [])
        if not inst_list:
            continue
        # Combine across groups within this scenario
        m_total = sum(i["matched"] for i in inst_list)
        mi_total = sum(i["missed"] for i in inst_list)
        h_total = sum(i["hallucinated"] for i in inst_list)

        p = m_total / (m_total + h_total) if (m_total + h_total) else 0
        r = m_total / (m_total + mi_total) if (m_total + mi_total) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0

        multi_q = _has_multiple_questionnaires(utts)
        scenarios.append(_label_for(u, multi_q))
        prec_vals.append(p)
        rec_vals.append(r)
        f1_vals.append(f)

    if not scenarios:
        return

    x = np.arange(len(scenarios))
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.8), 4.5))
    ax.bar(x - w, prec_vals, w, label="Precision", color=_C["blue"])
    ax.bar(x,     rec_vals,  w, label="Recall",    color=_C["orange"])
    ax.bar(x + w, f1_vals,   w, label="F1",        color=_C["green"])

    _annotate_bars(ax, x - w, prec_vals)
    _annotate_bars(ax, x, rec_vals)
    _annotate_bars(ax, x + w, f1_vals)

    ax.set_ylabel("Score")
    ax.set_title("Instance Alignment: Precision / Recall / F1")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "instance_alignment.png", bbox_inches="tight")
    plt.close(fig)


# ── New diagnostic figures ──────────────────────────────────────────────


def _fig_completeness_vs_accuracy(utts: list[dict], out_dir: Path) -> None:
    """Scatter: field completion rate (X) vs accuracy on filled fields (Y).

    Separates 'agent didn't try' from 'agent tried and failed'.
    """
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]

    completion = []             # agent-filled / gt_expected
    accuracy_on_filled = []     # attempted accuracy (excl. unmatched GT instances)

    for u in utts:
        sc = u["scores"]
        gt_exp = sc.get("gt_expected", 1)
        att = u.get("attempted_scores", {})
        # completion = fields the agent actually filled / gt expected
        completion.append(sc["total"] / gt_exp if gt_exp else 0)
        # accuracy-when-attempted
        accuracy_on_filled.append(att.get("accuracy", sc.get("accuracy", 0)))

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [_C["green"], _C["cyan"], _C["red"]]
    for i, (cx, cy, lab) in enumerate(zip(completion, accuracy_on_filled, labels)):
        c = colors[i % len(colors)]
        ax.scatter(cx, cy, s=160, color=c, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(lab, (cx, cy), textcoords="offset points", xytext=(8, 6),
                    fontsize=7, ha="left")

    ax.set_xlabel("Field Completion Rate (evaluated / GT expected)")
    ax.set_ylabel("Accuracy on Filled Fields (attempted)")
    ax.set_title("Completeness vs. Accuracy")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(1.0, color=_C["grey"], ls="--", lw=0.7)
    ax.axvline(1.0, color=_C["grey"], ls="--", lw=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "completeness_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def _fig_scalar_vs_repeat(utts: list[dict], out_dir: Path) -> None:
    """Grouped bar: scalar accuracy vs repeat-group accuracy per scenario."""
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]

    scalar_acc = []
    repeat_acc = []

    for u in utts:
        svr = u.get("scalar_vs_repeat", {})
        sc = svr.get("scalar", {})
        rp = svr.get("repeat_group", {})
        sc_tot = sc.get("total", 0) or 1
        rp_tot = rp.get("total", 0) or 1
        scalar_acc.append(sc.get("correct", 0) / sc_tot)
        repeat_acc.append(rp.get("correct", 0) / rp_tot)

    x = np.arange(len(labels))
    w = 0.30

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x - w / 2, scalar_acc,  w, label="Scalar fields",       color=_C["blue"])
    ax.bar(x + w / 2, repeat_acc,  w, label="Repeat-group fields",  color=_C["orange"])

    _annotate_bars(ax, x - w / 2, scalar_acc)
    _annotate_bars(ax, x + w / 2, repeat_acc)

    ax.set_ylabel("Strict Accuracy")
    ax.set_title("Accuracy Split: Scalar vs. Repeat-Group Fields")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_dir / "scalar_vs_repeat.png", bbox_inches="tight")
    plt.close(fig)


def _fig_repeat_group_heatmap(utts: list[dict], out_dir: Path) -> None:
    """Per-group stacked bar: matched / missed / hallucinated instances.

    One cluster of bars per scenario, one bar per repeat group, stacked by
    matched (green), missed (red), hallucinated (orange).  Group names are
    shown inside a combined x-axis label below each bar.
    """
    utts = _sort_by_difficulty(utts)

    # Collect distinct group names (skip _combined)
    all_groups: list[str] = []
    for u in utts:
        for ia in u.get("instance_alignment", []):
            g = ia["group"]
            if g != "_combined" and g not in all_groups:
                all_groups.append(g)
    # If only _combined exists, use it
    if not all_groups:
        for u in utts:
            for ia in u.get("instance_alignment", []):
                g = ia["group"]
                if g not in all_groups:
                    all_groups.append(g)
    if not all_groups:
        return

    labels = [_short_id(u["scenario_id"], u.get("questionnaire") if _has_multiple_questionnaires(utts) else None) for u in utts]
    n_scen = len(labels)
    n_grp = len(all_groups)

    # Build data arrays: [scenario, group]
    matched = np.zeros((n_scen, n_grp))
    missed = np.zeros((n_scen, n_grp))
    halluc = np.zeros((n_scen, n_grp))

    for si, u in enumerate(utts):
        grp_map = {ia["group"]: ia for ia in u.get("instance_alignment", [])}
        for gi, g in enumerate(all_groups):
            ia = grp_map.get(g, {})
            matched[si, gi] = ia.get("matched", 0)
            missed[si, gi] = ia.get("missed", 0)
            halluc[si, gi] = ia.get("hallucinated", 0)

    # Each bar gets its own x-position; group bars within a scenario cluster
    bar_w = 0.30
    gap_between_groups = 0.08
    gap_between_scenarios = 0.7
    cluster_width = n_grp * bar_w + (n_grp - 1) * gap_between_groups

    # Compute x positions and tick labels
    xpositions = []  # flat list of x positions
    xtick_labels = []  # two-line labels: group name + scenario
    scenario_centers = []

    current_x = 0
    for si in range(n_scen):
        cluster_positions = []
        for gi in range(n_grp):
            pos = current_x + gi * (bar_w + gap_between_groups)
            cluster_positions.append(pos)
            xpositions.append(pos)
            xtick_labels.append(all_groups[gi])
        scenario_centers.append(np.mean(cluster_positions))
        current_x += cluster_width + gap_between_scenarios

    xpositions = np.array(xpositions)

    fig, ax = plt.subplots(figsize=(max(7, n_scen * 3), 5))

    # Draw bars
    idx = 0
    for si in range(n_scen):
        for gi in range(n_grp):
            xp = xpositions[idx]
            m = matched[si, gi]
            mi = missed[si, gi]
            h = halluc[si, gi]

            ax.bar(xp, m, bar_w, color=_C["green"], edgecolor="white", linewidth=0.5)
            ax.bar(xp, mi, bar_w, bottom=m, color=_C["red"], edgecolor="white", linewidth=0.5)
            ax.bar(xp, h, bar_w, bottom=m + mi, color=_C["orange"], edgecolor="white", linewidth=0.5)

            # Annotation above bar
            bar_top = m + mi + h
            if bar_top > 0:
                gt_total = int(m + mi)
                lbl = f"{int(m)}/{gt_total}"
                if h > 0:
                    lbl += f" +{int(h)}extra"
                ax.text(xp, bar_top + 0.12, lbl,
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

            idx += 1

    # X-axis: group names under each bar
    ax.set_xticks(xpositions)
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=30, ha="right")

    # Scenario labels as a second row using ax.text
    for si, center in enumerate(scenario_centers):
        ax.text(center, -0.08, labels[si],
                ha="center", va="top", fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel("Number of instances")
    ax.set_title("Repeat-Group Instance Matching  (matched / GT expected,  +extra = agent hallucinated)")
    y_max = (matched + missed + halluc).max() + 1.5
    ax.set_ylim(0, y_max)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_C["green"], label="Matched"),
        Patch(facecolor=_C["red"], label="Missed (GT not matched by agent)"),
        Patch(facecolor=_C["orange"], label="Extra (agent instance with no GT match)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
             bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=[0, 0.05, 0.76, 1])
    fig.savefig(out_dir / "repeat_group_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def _fig_conditional_accuracy(utts: list[dict], out_dir: Path) -> None:
    """Bar chart comparing overall accuracy to attempted-only accuracy.

    "Attempted" excludes placeholder fields for unmatched GT repeat-group
    instances (``[?gtN]``). This makes the comparison closer to
    "when the agent attempted an aligned field, how often was it right?"
    rather than a provenance-based conditional split.
    """
    utts = _sort_by_difficulty(utts)
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]

    full_acc = []
    cond_acc = []
    cond_lenient = []

    for u in utts:
        full_acc.append(u["scores"]["accuracy"])
        att = u.get("attempted_scores", {})
        cond_acc.append(att.get("accuracy", u["scores"]["accuracy"]))
        cond_lenient.append(att.get("lenient_accuracy", u["scores"]["lenient_accuracy"]))

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x - w, full_acc,      w, label="Overall accuracy",         color=_C["blue"])
    ax.bar(x,     cond_acc,      w, label="Attempted accuracy",       color=_C["green"])
    ax.bar(x + w, cond_lenient,  w, label="Attempted accuracy (lenient)", color=_C["cyan"])

    _annotate_bars(ax, x - w, full_acc)
    _annotate_bars(ax, x, cond_acc)
    _annotate_bars(ax, x + w, cond_lenient)

    ax.set_ylabel("Accuracy")
    ax.set_title("Overall vs Attempted Accuracy (excl. unmatched GT repeat fields)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    fig.savefig(out_dir / "conditional_accuracy.png", bbox_inches="tight")
    plt.close(fig)


# ── Ablation: single-agent recall vs vehicle count ──────────────────────

_ABLATION_RE = re.compile(r"ABLATION_(\d+)V", re.IGNORECASE)


def _extract_vehicle_count(scenario_id: str) -> int | None:
    """Extract vehicle count from ablation scenario IDs like 'ABLATION_3V_CHAIN'."""
    m = _ABLATION_RE.search(scenario_id)
    return int(m.group(1)) if m else None


def _fig_ablation_recall_vs_vehicles(utts: list[dict], out_dir: Path) -> None:
    """Line plot: repeat-group recall & F1 vs vehicle count (single agent).

    Only renders if ablation scenarios (ABLATION_*V_*) are present.
    """
    points: list[tuple[int, float, float, float]] = []
    for u in utts:
        vc = _extract_vehicle_count(u.get("scenario_id", ""))
        if vc is None:
            continue
        inst_list = u.get("instance_alignment", [])
        if not inst_list:
            continue
        m_total = sum(i["matched"] for i in inst_list)
        mi_total = sum(i["missed"] for i in inst_list)
        h_total = sum(i["hallucinated"] for i in inst_list)
        rec = m_total / (m_total + mi_total) if (m_total + mi_total) else 0
        prec = m_total / (m_total + h_total) if (m_total + h_total) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        points.append((vc, rec, prec, f1))

    points.sort()
    if len(points) < 2:
        return  # need at least 2 ablation points to plot a line

    vcounts = [p[0] for p in points]
    recalls = [p[1] for p in points]
    precs   = [p[2] for p in points]
    f1s     = [p[3] for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(vcounts, recalls, "o-", color=_C["orange"], linewidth=2.5,
            markersize=8, label="Recall", zorder=3)
    for v, r in zip(vcounts, recalls):
        ax.annotate(f"{r*100:.1f}%", (v, r),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=8, fontweight="bold", color=_C["orange"])

    ax.plot(vcounts, f1s, "s--", color=_C["green"], linewidth=1.5,
            markersize=6, label="F1", zorder=2)
    for v, f in zip(vcounts, f1s):
        ax.annotate(f"{f*100:.1f}%", (v, f),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=7.5, color=_C["green"])

    ax.plot(vcounts, precs, "^:", color=_C["blue"], linewidth=1.2,
            markersize=5, alpha=0.7, label="Precision", zorder=1)

    ax.set_xlabel("Number of vehicles (repeat-group instances)")
    ax.set_ylabel("Score")
    ax.set_title("Ablation: Instance Recall vs. Vehicle Count")
    ax.set_xticks(vcounts)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_recall_vs_vehicles.png", bbox_inches="tight")
    plt.close(fig)


def _fig_ablation_field_f1_vs_vehicles(utts: list[dict], out_dir: Path) -> None:
    """Line plot: field-level strict F1 vs vehicle count (single agent)."""
    points: list[tuple[int, float, float, float]] = []
    for u in utts:
        vc = _extract_vehicle_count(u.get("scenario_id", ""))
        if vc is None:
            continue
        fp = u["field_prf"]
        points.append((vc, fp["strict"]["f1"], fp["lenient"]["f1"], fp["weighted"]["f1"]))

    points.sort()
    if len(points) < 2:
        return

    vcounts  = [p[0] for p in points]
    strict   = [p[1] for p in points]
    weighted = [p[3] for p in points]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(vcounts, strict, "o-", color=_C["blue"], linewidth=2.5,
            markersize=8, label="Strict F1", zorder=3)
    for v, s in zip(vcounts, strict):
        ax.annotate(f"{s*100:.1f}%", (v, s),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=8, fontweight="bold", color=_C["blue"])

    ax.plot(vcounts, weighted, "s--", color=_C["green"], linewidth=1.5,
            markersize=6, label="Weighted F1", zorder=2)
    for v, w in zip(vcounts, weighted):
        ax.annotate(f"{w*100:.1f}%", (v, w),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=7.5, color=_C["green"])

    ax.set_xlabel("Number of vehicles (repeat-group instances)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation: Field-Level F1 vs. Vehicle Count")
    ax.set_xticks(vcounts)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_field_f1_vs_vehicles.png", bbox_inches="tight")
    plt.close(fig)


# ── View-first figures ──────────────────────────────────────────────────


def _fig_primary_fields_overview(report: dict[str, Any], out_dir: Path) -> None:
    views = (report.get("aggregate") or {}).get("metric_views", {})
    agg = views.get("primary_fields") or views.get("whole_form", {})
    macro = agg.get("macro", {})
    labels = [
        "Accuracy",
        "Lenient Acc.",
        "Strict F1",
        "Lenient F1",
        "Weighted F1",
        "Macro Strict F1",
    ]
    values = [
        agg.get("accuracy", 0),
        agg.get("lenient_accuracy", 0),
        (agg.get("strict") or {}).get("f1", 0),
        (agg.get("lenient") or {}).get("f1", 0),
        (agg.get("weighted") or {}).get("f1", 0),
        (macro.get("strict") or {}).get("f1", 0),
    ]
    colors = [_C["blue"], _C["cyan"], _C["green"], _C["orange"], _C["purple"], _C["grey"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.68)
    _annotate_bars(ax, x, values, fontsize=8, offset=0.02)
    ax.set_title("Primary Field Evaluation Overview")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "primary_fields_overview.png", bbox_inches="tight")
    plt.close(fig)


def _fig_whole_form_overview(report: dict[str, Any], out_dir: Path) -> None:
    views = (report.get("aggregate") or {}).get("metric_views", {})
    agg = views.get("whole_form") or views.get("primary_fields", {})
    exact = views.get("whole_record_exact_match") or {}
    changed = views.get("changed_fields") or {}
    preservation = views.get("preservation") or {}
    labels = [
        "All-field F1",
        "Accuracy",
        "Exact Match",
        "Lenient Exact",
        "Changed F1",
        "Preservation",
    ]
    values = [
        (agg.get("strict") or {}).get("f1", 0),
        agg.get("accuracy", 0),
        exact.get("exact_match_rate", 0),
        exact.get("lenient_exact_match_rate", 0),
        (changed.get("strict") or {}).get("f1", 0),
        preservation.get("preservation_success_rate", 0),
    ]
    colors = [_C["blue"], _C["cyan"], _C["green"], _C["yellow"], _C["orange"], _C["purple"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.68)
    _annotate_bars(ax, x, values, fontsize=8, offset=0.02)
    ax.set_title("Benchmark Run Overview (All Fields vs Updates)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "whole_form_overview.png", bbox_inches="tight")
    plt.close(fig)


def _fig_changed_fields_by_utterance(report: dict[str, Any], out_dir: Path) -> None:
    utts = _sort_by_difficulty(report.get("utterances") or [])
    if not utts:
        return
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    strict = [(u.get("metric_views", {}).get("changed_fields", {}).get("strict", {}) or {}).get("f1", 0) for u in utts]
    weighted = [(u.get("metric_views", {}).get("changed_fields", {}).get("weighted", {}) or {}).get("f1", 0) for u in utts]
    recall = [(u.get("metric_views", {}).get("changed_fields", {}).get("strict", {}) or {}).get("recall", 0) for u in utts]

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x - w, strict, width=w, color=_C["green"], label="Strict F1")
    ax.bar(x, recall, width=w, color=_C["orange"], label="Recall")
    ax.bar(x + w, weighted, width=w, color=_C["purple"], label="Weighted F1")
    _annotate_bars(ax, x - w, strict, fontsize=7)
    _annotate_bars(ax, x, recall, fontsize=7)
    _annotate_bars(ax, x + w, weighted, fontsize=7)
    ax.set_title("Changed-Field Performance by Scenario")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "changed_fields_by_utterance.png", bbox_inches="tight")
    plt.close(fig)


def _fig_preservation_errors(report: dict[str, Any], out_dir: Path) -> None:
    utts = _sort_by_difficulty(report.get("utterances") or [])
    if not utts:
        return
    multi_q = _has_multiple_questionnaires(utts)
    labels = [_label_for(u, multi_q) for u in utts]
    kept = [(u.get("metric_views", {}).get("preservation", {}) or {}).get("preservation_success_rate", 0) for u in utts]
    errors = [(u.get("metric_views", {}).get("preservation", {}) or {}).get("preservation_error_rate", 0) for u in utts]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5))
    ax.bar(x, kept, color=_C["green"], width=0.6, label="Preservation success")
    ax.bar(x, errors, bottom=kept, color=_C["red"], width=0.6, label="Collateral edit rate")
    ax.set_title("Preservation vs. Collateral Edits")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / "preservation_overview.png", bbox_inches="tight")
    plt.close(fig)


def _fig_task_diagnostics(report: dict[str, Any], out_dir: Path) -> None:
    agg = (report.get("aggregate") or {}).get("metric_views", {}).get("task_diagnostics", {})
    specs = [
        ("Correction", agg.get("correction_success") or {}),
        ("Retraction", agg.get("retraction_success") or {}),
        ("History", agg.get("history_recovery_success") or {}),
        ("Gate", agg.get("gate_execution_success") or {}),
        ("Repeat", agg.get("repeat_group_execution_success") or {}),
        ("Input incoh.", agg.get("input_incoherence_success") or {}),
    ]
    specs = [spec for spec in specs if spec[1].get("applicable", 0) or spec[1].get("success_rate") is not None]
    if not specs:
        return
    labels = [label for label, _ in specs]
    values = [block.get("success_rate", 0) for _, block in specs]
    applicable = [block.get("applicable", 0) for _, block in specs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        color=[_C["blue"], _C["orange"], _C["cyan"], _C["purple"], _C["green"], _C["yellow"]][:len(labels)],
        width=0.62,
    )
    for xi, value, count in zip(x, values, applicable):
        ax.text(
            xi,
            value + 0.03,
            f"{value * 100:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("Task-Specific Success Rates")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "task_diagnostics_success.png", bbox_inches="tight")
    plt.close(fig)


def _fig_repeat_group_diagnostics(report: dict[str, Any], out_dir: Path) -> None:
    repeat = (report.get("aggregate") or {}).get("metric_views", {}).get("repeat_groups", {})
    labels = [
        "Gold ops",
        "Candidate-created",
        "Missing",
        "Spurious",
        "Wrong-instance",
        "Merge",
        "Split",
    ]
    values = [
        repeat.get("gold_repeat_instance_operation_total", 0),
        repeat.get("candidate_created_repeat_instance_count", 0),
        repeat.get("missing_instance_count", 0),
        repeat.get("spurious_instance_count", 0),
        repeat.get("wrong_instance_update_count", 0),
        repeat.get("merge_error_count", 0),
        repeat.get("split_error_count", 0),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=[
            _C["blue"],
            _C["cyan"],
            _C["red"],
            _C["orange"],
            _C["purple"],
            _C["grey"],
            _C["yellow"],
        ],
        width=0.62,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("Repeat-Group Diagnostic Counts")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "repeat_group_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def _metric_views_for_bucket(utts: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_metric_views([u.get("metric_views") or {} for u in utts])


def _bucket_by_derived(
    report: dict[str, Any],
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for utterance in report.get("utterances") or []:
        if not isinstance(utterance, dict):
            continue
        value = (utterance.get("derived_variables") or {}).get(field)
        if isinstance(value, bool):
            key = "yes" if value else "no"
        else:
            key = str(value or "unknown")
        buckets.setdefault(key, []).append(utterance)
    return buckets


def _fig_delta_type_breakdown(report: dict[str, Any], out_dir: Path) -> None:
    buckets = _bucket_by_derived(report, "primary_delta_type")
    order = [value for value in ["add", "refine", "correct", "retract", "unknown"] if value in buckets]
    order.extend(sorted(key for key in buckets if key not in order))
    if len(order) < 2:
        return

    labels = []
    whole_f1 = []
    changed_f1 = []
    exact = []
    for key in order:
        views = _metric_views_for_bucket(buckets[key])
        whole = views.get("whole_form") or {}
        changed = views.get("changed_fields") or {}
        exact_view = views.get("whole_record_exact_match") or {}
        labels.append(f"{key}\n(n={len(buckets[key])})")
        whole_f1.append((whole.get("strict") or {}).get("f1", 0))
        changed_f1.append((changed.get("strict") or {}).get("f1", 0))
        exact.append(exact_view.get("exact_match_rate", 0))

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 1.3), 4.6))
    ax.bar(x - w, whole_f1, w, color=_C["blue"], label="All-field F1")
    ax.bar(x, changed_f1, w, color=_C["orange"], label="Changed-field F1")
    ax.bar(x + w, exact, w, color=_C["green"], label="Exact match")
    _annotate_bars(ax, x - w, whole_f1, fontsize=8)
    _annotate_bars(ax, x, changed_f1, fontsize=8)
    _annotate_bars(ax, x + w, exact, fontsize=8)
    ax.set_title("Performance by Task Type")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    fig.savefig(out_dir / "delta_type_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def _fig_repeat_pressure_breakdown(report: dict[str, Any], out_dir: Path) -> None:
    def coordination_total(utterance: dict[str, Any]) -> int:
        repeat = (utterance.get("metric_views") or {}).get("repeat_groups") or {}
        return int(
            repeat.get(
                "repeat_instance_coordination_total",
                repeat.get("gold_repeat_instance_total", 0),
            )
            or 0
        )

    def bucket_key(count: int) -> tuple[int, str]:
        if count <= 1:
            return 0, "0-1"
        if count == 2:
            return 1, "2"
        if count <= 4:
            return 2, "3-4"
        return 3, "5+"

    buckets: dict[str, list[dict[str, Any]]] = {}
    order_map: dict[str, int] = {}
    for utterance in report.get("utterances") or []:
        if not isinstance(utterance, dict):
            continue
        order_idx, label = bucket_key(coordination_total(utterance))
        buckets.setdefault(label, []).append(utterance)
        order_map[label] = order_idx

    order = sorted(buckets, key=lambda key: (order_map.get(key, 99), key))
    if len(order) < 2:
        return

    labels = []
    any_alignment_failure = []
    alignment_failures_per_instance = []
    spurious_instances_per_instance = []
    for key in order:
        utterances = buckets[key]
        n = len(utterances)
        coord_total = sum(coordination_total(utterance) for utterance in utterances)
        align_failure_total = sum(
            int(
                ((utterance.get("metric_views") or {}).get("repeat_groups") or {}).get(
                    "alignment_failure_count",
                    0,
                )
                or 0
            )
            for utterance in utterances
        )
        spurious_total = sum(
            int(
                ((utterance.get("metric_views") or {}).get("repeat_groups") or {}).get(
                    "spurious_instance_count",
                    0,
                )
                or 0
            )
            for utterance in utterances
        )
        items_with_failures = sum(
            1
            for utterance in utterances
            if int(
                ((utterance.get("metric_views") or {}).get("repeat_groups") or {}).get(
                    "alignment_failure_count",
                    0,
                )
                or 0
            )
            > 0
        )
        labels.append(f"{key}\nn={n}, inst={coord_total}")
        any_alignment_failure.append(items_with_failures / n if n else 0)
        alignment_failures_per_instance.append(
            align_failure_total / coord_total if coord_total else 0
        )
        spurious_instances_per_instance.append(
            spurious_total / coord_total if coord_total else 0
        )

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8.5, len(labels) * 1.9), 4.8))
    ax.bar(
        x - w,
        any_alignment_failure,
        w,
        color=_C["red"],
        label="Items with alignment failure",
    )
    ax.bar(
        x,
        alignment_failures_per_instance,
        w,
        color=_C["purple"],
        label="Alignment failures / instance",
    )
    ax.bar(
        x + w,
        spurious_instances_per_instance,
        w,
        color=_C["orange"],
        label="Spurious instances / instance",
    )
    _annotate_bars(ax, x - w, any_alignment_failure, fontsize=8)
    _annotate_bars(ax, x, alignment_failures_per_instance, fontsize=8)
    _annotate_bars(ax, x + w, spurious_instances_per_instance, fontsize=8)
    ax.set_title("Instance Alignment vs Coordination Load")
    ax.set_xlabel("Repeat instances to coordinate, bucketed by max(prior, gold)")
    ax.set_ylabel("Failure rate")
    ymax = max(
        1.0,
        *(any_alignment_failure + alignment_failures_per_instance + spurious_instances_per_instance),
    )
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    fig.savefig(out_dir / "repeat_pressure_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def _fig_transition_accuracy_overview(report: dict[str, Any], out_dir: Path) -> None:
    transition = (report.get("aggregate") or {}).get("metric_views", {}).get("transition_accuracy", {})
    by_transition = transition.get("by_transition") or {}
    order = [value for value in ["preserve", "set", "change", "clear"] if value in by_transition]
    if not order:
        return

    labels = []
    values = []
    counts = []
    for key in order:
        block = by_transition.get(key) or {}
        total = block.get("gold_total", 0)
        labels.append(key.title())
        values.append((block.get("correct", 0) / total) if total else 0)
        counts.append(total)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    colors = [_C["green"], _C["blue"], _C["orange"], _C["purple"]][:len(labels)]
    ax.bar(x, values, color=colors, width=0.62)
    for xi, value, count in zip(x, values, counts):
        ax.text(
            xi,
            value + 0.03,
            f"{value * 100:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("Transition Accuracy by Intended Field Action")
    ax.set_ylabel("Correct rate")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "transition_accuracy_overview.png", bbox_inches="tight")
    plt.close(fig)


def _fig_state_changed_f1_comparison(report: dict[str, Any], out_dir: Path) -> None:
    buckets = _bucket_by_derived(report, "prior_state_condition")
    order = [value for value in ["S1", "S2", "S3", "S4"] if value in buckets]
    order.extend(sorted(key for key in buckets if key not in order))
    if not order:
        return

    labels = []
    values = []
    counts = []
    for key in order:
        views = _metric_views_for_bucket(buckets[key])
        changed = views.get("changed_fields") or {}
        strict = changed.get("strict") or {}
        labels.append(f"{key}\n(n={len(buckets[key])})")
        values.append(strict.get("f1", 0))
        counts.append(changed.get("gold_changed_total", 0))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    colors = [_C["blue"], _C["orange"], _C["green"], _C["purple"]]
    ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(labels))], width=0.62)
    for xi, value, count in zip(x, values, counts):
        ax.text(
            xi,
            value + 0.03,
            f"{value * 100:.1f}%\nchanged={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("Changed-Field F1 by Prior-State Condition")
    ax.set_ylabel("Strict changed-field F1")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "state_changed_f1_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _fig_failure_budget(report: dict[str, Any], out_dir: Path) -> None:
    views = (report.get("aggregate") or {}).get("metric_views", {})
    task = views.get("task_diagnostics") or {}
    preservation = views.get("preservation") or {}
    repeat = views.get("repeat_groups") or {}
    auxiliary = views.get("auxiliary_unscored_notes") or {}
    labels = [
        "Unsupported\ncommits",
        "Collateral\nedits",
        "Repeat routing\nerrors",
        "Alignment\nfailures",
        "Failed\ncorrections",
        "Failed\nretractions",
        "Auxiliary note\nhallucinations",
    ]
    values = [
        task.get("unsupported_commit_count", 0),
        preservation.get("collateral_edit_count", task.get("collateral_edit_count", 0)),
        task.get("repeat_routing_error_count", 0),
        repeat.get("alignment_failure_count", 0),
        task.get("failed_correction_count", 0),
        task.get("failed_retraction_count", 0),
        auxiliary.get("hallucination_count", 0),
    ]
    if not any(values):
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    colors = [_C["red"], _C["orange"], _C["purple"], _C["cyan"], _C["blue"], _C["yellow"], _C["grey"]]
    bars = ax.bar(x, values, color=colors, width=0.65)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.1,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("Failure Budget: Where Errors Come From")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_dir / "failure_budget.png", bbox_inches="tight")
    plt.close(fig)


# ── Public API ──────────────────────────────────────────────────────────


def generate_figures(report: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate the curated Studio visualization set and return saved paths.

    Parameters
    ----------
    report : The dict returned by ``report_to_dict()``.
    out_dir : Directory where PNGs will be saved (created if needed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()
    _setup_style()

    def safe_call(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            # Figure generation should never make an evaluation run unusable.
            # The dashboard will simply omit figures that cannot be produced
            # from the available summary schema.
            return

    safe_call(_fig_whole_form_overview, report, out_dir)
    safe_call(_fig_task_diagnostics, report, out_dir)
    safe_call(_fig_transition_accuracy_overview, report, out_dir)
    safe_call(_fig_state_changed_f1_comparison, report, out_dir)

    return sorted(out_dir.glob("*.png"))
