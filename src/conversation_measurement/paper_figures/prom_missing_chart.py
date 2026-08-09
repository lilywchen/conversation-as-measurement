import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prom_item_missingness_matched(
    proms_df,
    tests=("Voice", "Cough", "Swallowing"),
    filename="survey.pdf",
    max_items=None,          # optionally limit to top-k most-missing per test
    left=None,               # optionally override left margin
):

    isna = proms_df[proms_df['score'].isna()]
    missing_totals = isna.groupby(["Test", "Question"]).doc.agg('count').reset_index()
    total_items = proms_df.groupby(["Test", "Question"]).doc.agg('count').reset_index(name="Total")
    missing_totals = missing_totals.merge(total_items, on=["Test", "Question"])
    missing_totals['missing_pct'] = missing_totals['doc']/missing_totals['Total']
    missing_totals = missing_totals.sort_values(by=["Test", "missing_pct"], ascending=[True, False])
    # --- figure sizing: similar density philosophy to make_seg_bars
    # more items -> taller figure

    n_panels = len(tests)
    fig_h = max(2.2, 2.2 * n_panels)  # ~2.2 inches per panel
    fig_w = 7.6                       # keep reasonably narrow for paper width
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h), sharex=True)

    if n_panels == 1:
        axes = [axes]

    # --- choose left margin based on longest label across all panels (rough heuristic)
    if left is None:
        max_len = 0
        for t in tests:
            sub = missing_totals[missing_totals["Test"] == t]
            if len(sub) == 0:
                continue
            max_len = max(max_len, int(sub["Question"].astype(str).str.len().max()))
        # map label length -> left margin (cap so it doesn't go crazy)
        left = min(0.62, max(0.28, 0.28 + 0.006 * max_len))

    # --- global x padding for annotations (percent scale 0..1)
    # keep consistent across panels
    global_max = 0.0
    for t in tests:
        sub = missing_totals[missing_totals["Test"] == t]
        if len(sub) == 0:
            continue
        global_max = max(global_max, float(sub["missing_pct"].max()))
    pad = max(0.02, 0.03 * global_max)      # like make_seg_bars: small relative pad
    xlim_max = min(1.0, global_max) + 9*pad # keep space for text to the right

    for ax, test in zip(axes, tests):
        subdf = missing_totals[missing_totals["Test"] == test].copy()
        subdf = subdf.sort_values("missing_pct", ascending=False)

        if max_items is not None:
            subdf = subdf.head(max_items)

        # y positions and values
        y = np.arange(len(subdf))
        vals = subdf["missing_pct"].to_numpy()

        bars = ax.barh(
            y,
            vals,
            height=0.55,          
            edgecolor="white",
            linewidth=0.5
        )

        # percentage annotations
        for bar, v in zip(bars, vals):
            ax.text(
                v + pad,
                bar.get_y() + bar.get_height()/2,
                f"{v:.1%}",          # 85% style; change to .1% if you want
                va="center",
                fontsize=10,
                clip_on=True
            )

        # y tick labels
        ax.set_yticks(y)
        ax.set_yticklabels(subdf["Question"], fontsize=10)
        ax.tick_params(axis="y", pad=2, length=0)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")

        # formatting like make_seg_bars
        ax.set_xticks([])  # no x ticks
        ax.set_xlim(0, xlim_max)

        ax.invert_yaxis()
        ax.set_title(test, loc="left", fontsize=12, pad=2)

        # spines
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left"]:
            ax.spines[spine].set_color("#444444")
            ax.spines[spine].set_linewidth(0.5)

        # remove per-axes xlabel; we'll add one shared label
        ax.set_xlabel("")

    # shared xlabel (centered over adjusted subplot region)
    right = 0.98
    top = 0.95
    bottom = 0.10

    
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    

    plt.tight_layout()

    x_left = axes[0].get_position().x0

    fig.supxlabel("Item Missingness", fontsize=13, x=x_left, ha="left", y=-0.02)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)

    plt.show()
    return missing_totals
