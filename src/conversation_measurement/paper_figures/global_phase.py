import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_seg_bars(segments, clinician=None, barcolor=None, filename=None):
    
    print(clinician)
    print("NUM SEGMENTS ", len(segments))
    print("NUM DOCS ", segments.doc.nunique())

    if clinician is not None:
        segments = segments.copy(deep=True)
        segments = segments[segments['clinician'] == clinician]

    phase_order = [
        "Opening", "History", "Exam", "Assessment",
        "Plan", "Education", "Closing", "Non-clinical",
    ][::-1]

    overall_data = (
        segments.groupby('segment_title').phase_pct.agg('sum')
        / segments.doc.nunique() * 100
    ).reindex(phase_order).fillna(0).round(3)

    print(sum(overall_data.to_dict().values()))
    # --- tighter figure: height ~ number of bars * ~0.28-0.32 inches
    n = len(overall_data)
    fig_h = max(2.0, 0.30 * n)   # ~2.4 inches for 8 bars
    fig, ax = plt.subplots(figsize=(4.0, fig_h))

    bars = ax.barh(
        overall_data.index,
        overall_data.values,
        edgecolor="white",
        color=barcolor,
        height=0.55,             
        linewidth=0.5
    )

    # Percent labels
    xmax = float(overall_data.max())
    pad = max(0.25, 0.03 * xmax)  # small pad relative to data scale
    for bar, value in zip(bars, overall_data.values):
        ax.text(
            value + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=11
        )

    ax.set_xlim(0, xmax + 3 * pad)

    # Kill extra vertical whitespace (top/bottom margins)
    ax.margins(y=0)  # important
    ax.set_ylim(-0.5, n - 0.5)  # hard clamp to bars

    # Tighten label/tick padding
    
    ax.set_xlabel("Average Phase Allocation", labelpad=2, fontsize=12)
    ax.set_xticks([])  # no ticks
    ax.tick_params(axis='y', pad=2, length=0, labelsize=11)  # less left padding, remove tick marks

    # Clean spines
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    
    for spine in ["left"]:
        ax.spines[spine].set_color("#444444")
        ax.spines[spine].set_linewidth(0.5)

    # Make layout *really* tight (less outer padding than tight_layout default)
    fig.subplots_adjust(left=0.32, right=0.98, top=0.98, bottom=0.22)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=300)

    plt.show()
