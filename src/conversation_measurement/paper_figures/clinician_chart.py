import matplotlib.pyplot as plt
import pandas as pd

def plot_all_clinicians(
    segments,
    clinicians,
    anon=False,
    savepath=None,
):
    """
    Multi-panel clinician phase allocation plot with styling matched to make_seg_bars().
    - Consistent x-limits across clinicians
    - Shared y-axis labels
    - Thin bars + tight whitespace
    - Light spines (#444444) and no top/right spines
    """
    anon_map = {clinician: chr(ord("A") + i) for i, clinician in enumerate(clinicians)}

    phase_order = [
        "Opening", "History", "Exam", "Assessment",
        "Plan", "Education", "Closing", "Non-clinical",
    ][::-1]

    # --- Precompute per-clinician series + global max
    clinician_data = {}
    clinician_size = {}
    global_max = 0.0
    for clinician in clinicians:
        seg = segments[segments["clinician"] == clinician]
        overall = (
            seg.groupby("segment_title")["phase_pct"].sum()
            / seg["doc"].nunique()
            * 100
        ).reindex(phase_order).fillna(0)

        clinician_size[clinician] = seg.doc.nunique()

        clinician_data[clinician] = overall
        if overall.size > 0:
            global_max = max(global_max, float(overall.max()))

    # Annotation padding, matched to make_seg_bars
    pad = max(0.25, 0.03 * global_max)
    xlim_max = global_max + 4 * pad

    # --- Figure sizing: match density of make_seg_bars but across panels
    n_phases = len(phase_order)  # 8
    fig_h = max(2.0, 0.30 * n_phases)  # ~2.4 inches
    fig_w = 3.2 * len(clinicians)      # slightly tighter than 4*len()
    fig, axes = plt.subplots(
        1, len(clinicians),
        figsize=(fig_w, fig_h),
        sharey=True
    )
    if len(clinicians) == 1:
        axes = [axes]

    # --- Plot each panel
    for ax, clinician in zip(axes, clinicians):
        data = clinician_data[clinician]

        bars = ax.barh(
            data.index,
            data.values,
            edgecolor="white",
            height=0.55,
            linewidth=0.5
        )

        # Percent labels
        for bar, value in zip(bars, data.values):
            ax.text(
                value + pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=11
            )

        ax.set_xlim(0, xlim_max)

        # Tighten vertical whitespace (top/bottom margins)
        ax.margins(y=0)
        ax.set_ylim(-0.5, n_phases - 0.5)

        # Remove x ticks, tighten y padding, remove tick marks
        ax.set_xticks([])
        ax.tick_params(axis="y", pad=2, length=0, labelsize=11)

        # Title
        clinician_name = anon_map.get(clinician, clinician) if anon else clinician
        ax.set_title(f"{clinician_name} (n={clinician_size[clinician]})", fontsize=14, pad=4)

        # Clean spines like make_seg_bars
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left"]:
            ax.spines[spine].set_color("#444444")
            ax.spines[spine].set_linewidth(0.5)

    # Only leftmost subplot shows y labels (optional, but usually cleaner)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    # Shared xlabel (similar spacing control)
    fig.supxlabel("Average Phase Allocation", fontsize=14, y=0.08, x=(0.22 + 0.98)/2)
    

    # Tight layout / padding control (similar spirit to fig.subplots_adjust in make_seg_bars)
    # Increase left if y-labels clip; decrease wspace to tighten panels.
    fig.subplots_adjust(left=0.22, right=0.98, top=0.84, bottom=0.22, wspace=0.15)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", pad_inches=0.02)

    plt.show()
