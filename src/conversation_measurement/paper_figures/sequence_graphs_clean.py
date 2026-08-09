import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def build_sequence_df_long(
    segment_df: pd.DataFrame,
    doc_col: str = "doc",
    seg_col: str = "segment_title",
    dur_col: str = "minutes_diff",
    order_col: str = "segment_start",
) -> pd.DataFrame:
    """
    Build long-form dataframe for sequence-index bars:
        columns: [doc, segment, start, end, width]

    Assumes:
        - order_col exists
        - durations are valid
    """

    df = segment_df.copy()

    # enforce stable ordering inside each document
    df = df.sort_values([doc_col, order_col], kind="mergesort")

    rows = []

    for doc, g in df.groupby(doc_col, sort=False):

        segs = list(zip(g[seg_col], g[dur_col]))
        total = float(sum(v for _, v in segs))

        # normalize durations
        segs = [(name, v / total if total > 0 else 0.0) for name, v in segs]

        start_pos = 0.0
        for name, w in segs:
            rows.append(
                {
                    doc_col: doc,
                    "segment": name,
                    "start": start_pos,
                    "end": start_pos + w,
                    "width": w,
                }
            )
            start_pos += w

    return pd.DataFrame(rows)



def cluster_docs_by_duration(
    df_long: pd.DataFrame,
    k_clusters: int,
    doc_col: str = "doc",
    seg_col: str = "segment",
    width_col: str = "width",
    random_state: int = 0,
) -> dict:

    mat = df_long.pivot_table(
        index=doc_col,
        columns=seg_col,
        values=width_col,
        aggfunc="sum",
        fill_value=0.0,
    )

    km = KMeans(n_clusters=k_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(mat.values)

    mat_with = mat.copy()
    mat_with["cluster"] = labels

    doc_order = mat_with.sort_values(["cluster"]).index.to_numpy()
    ordered_clusters = mat_with.loc[doc_order, "cluster"].to_numpy()

    cluster_blocks = []
    start = 0
    for i in range(1, len(ordered_clusters) + 1):
        if i == len(ordered_clusters) or ordered_clusters[i] != ordered_clusters[start]:
            end = i - 1
            cluster_blocks.append(
                {
                    "cluster": int(ordered_clusters[start]),
                    "start": start,
                    "end": end,
                    "center": (start + end) / 2.0,
                }
            )
            start = i

    cluster_blocks = pd.DataFrame(cluster_blocks)

    return {
        "doc_order": doc_order,
        "ordered_clusters": ordered_clusters,
        "cluster_blocks": cluster_blocks,
        "mat": mat,
        "model": km,
    }


def compute_cluster_layout(n_docs, cluster_blocks, cluster_gap_rows=2):
    y_positions = np.arange(n_docs, dtype=float)

    cluster_blocks = cluster_blocks.copy().reset_index(drop=True)

    # shift rows after each cluster start
    for block_idx, start in enumerate(cluster_blocks["start"][1:], start=1):
        y_positions[int(start):] += cluster_gap_rows

    # compute label positions
    cluster_blocks["label_y"] = (
        cluster_blocks["center"] + cluster_gap_rows * np.arange(len(cluster_blocks))
    )

    total_rows = int(y_positions[-1] + 1) if n_docs > 0 else 0

    return y_positions, cluster_blocks, total_rows


def _measure_text_width_px(fig, strings, fontsize):
    if not strings:
        return 0

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    temp_ax = fig.add_axes([0, 0, 1, 1])
    temp_ax.axis("off")
    texts = [temp_ax.text(0, 0, s, fontsize=fontsize, alpha=0.0) for s in strings]
    fig.canvas.draw()
    widths = [t.get_window_extent(renderer=renderer).width for t in texts]
    temp_ax.remove()

    return int(np.ceil(max(widths)))


def _compute_side_column_fracs(
    fig,
    cluster_label_strings,
    categories_legend,
    cluster_label_fontsize,
    legend_fontsize,
    label_pad_px=8,
    column_gap_px=12,
    legend_patch_pad_px=10,
):
    fig.canvas.draw()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi

    label_col_px = _measure_text_width_px(fig, cluster_label_strings, cluster_label_fontsize) + label_pad_px
    label_col_frac = label_col_px / fig_w_px if fig_w_px > 0 else 0.0

    legend_col_px = _measure_text_width_px(fig, categories_legend, legend_fontsize) + legend_patch_pad_px + 18
    legend_col_frac = legend_col_px / fig_w_px if fig_w_px > 0 else 0.0

    gap_frac = column_gap_px / fig_w_px if fig_w_px > 0 else 0.0

    return label_col_frac, legend_col_frac, gap_frac


def _draw_vertical_legend(
    ax,
    categories_legend,
    seg_to_color,
    palette,
    legend_fontsize,
    y_top=0.98,
    y_min=0.02,
    patch_height=0.06,
    patch_width=0.06,
    row_spacing=0.072,
):
    ax.axis("off")
    y = y_top

    for s in categories_legend:
        color = palette[seg_to_color[s]]
        ax.add_patch(
            mpatches.Rectangle(
                (0.0, y - patch_height),
                patch_width,
                patch_height * 0.8,
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(
            patch_width + 0.04,
            y - patch_height * 0.52,
            str(s),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=legend_fontsize,
            color="black",
        )
        y -= row_spacing
        if y < y_min:
            break


def plot_sequence_index(
    df_long: pd.DataFrame,
    cluster_meta: dict,
    cluster_labels=None,   # list[str] or dict[int, str] or None
    doc_col: str = "doc",
    seg_col: str = "segment",
    fig_w: float = 7.0,
    fig_h: float = 3.4,
    cluster_gap_rows: int = 2,
    cluster_label_fontsize: int = 8,
    legend_fontsize: int = 7,
    rasterize_bars: bool = True,
    savepath: str | None = None,
    dpi: int = 300,
):
    """
    Clustered sequence-index plot.

    Assumes cluster_meta contains:
      - "doc_order"
      - "cluster_blocks"
    """
    
    # ------------------------------------------------------------
    # Plot styling constants
    # ------------------------------------------------------------

    LABEL_PAD_PX = 8
    COLUMN_GAP_PX = 12
    LEGEND_PATCH_PAD_PX = 10
    LEGEND_Y_TOP = 0.98
    LEGEND_Y_MIN = 0.02
    LEGEND_PATCH_HEIGHT = 0.06
    LEGEND_PATCH_WIDTH = 0.06
    LEGEND_ROW_SPACING = 0.072

    # ------------------------------------------------------------
    # Phase color palette and legend ordering
    # ------------------------------------------------------------

    palette = [
        "#e377c2",
        "#7C3AED",
        "#d53333",
        "#1f77b4",
        "#9DE7E0",
        "#2ca02c",
        "#ff7f0e",
        "#8c564b",
    ]

    categories_legend = [
        "Opening", "History", "Exam", "Assessment",
        "Plan", "Education", "Closing", "Non-clinical",
    ]

    seg_to_color = {s: i for i, s in enumerate(categories_legend)}

    # ------------------------------------------------------------
    # Cluster metadata and vertical layout
    # ------------------------------------------------------------
    # docs are ordered by cluster; compute y positions with gaps

    docs = np.asarray(cluster_meta["doc_order"])
    cluster_blocks = cluster_meta["cluster_blocks"]

    n_docs = len(docs)
    y_positions, cluster_blocks, total_rows_with_gaps = compute_cluster_layout(
        n_docs=n_docs,
        cluster_blocks=cluster_blocks,
        cluster_gap_rows=cluster_gap_rows,
    )

    def _cluster_label(c: int) -> str:
        if cluster_labels is None:
            return f"C{c+1}"
        if isinstance(cluster_labels, dict):
            return str(cluster_labels.get(c, f"C{c+1}"))
        return str(cluster_labels[c])

    cluster_label_strings = [
        _cluster_label(int(c)) for c in cluster_blocks["cluster"]
    ]

    # ------------------------------------------------------------
    # Figure creation and dynamic column sizing
    # ------------------------------------------------------------
    # measure label/legend text to determine column widths

    fig = plt.figure(figsize=(fig_w, fig_h))

    label_col_frac, legend_col_frac, gap_frac = _compute_side_column_fracs(
        fig=fig,
        cluster_label_strings=cluster_label_strings,
        categories_legend=categories_legend,
        cluster_label_fontsize=cluster_label_fontsize,
        legend_fontsize=legend_fontsize,
        label_pad_px=LABEL_PAD_PX,
        column_gap_px=COLUMN_GAP_PX,
        legend_patch_pad_px=LEGEND_PATCH_PAD_PX,
    )

    # ------------------------------------------------------------
    # Axes layout (main plot | cluster labels | legend)
    # ------------------------------------------------------------

    right_cols = gap_frac + label_col_frac + gap_frac + legend_col_frac

    main_right = max(0.55, 1.0 - right_cols)
    main_ax_pos = [0.0, 0.0, main_right, 1.0]

    cur_left = main_right + gap_frac
    label_ax_pos = [cur_left, 0.0, label_col_frac, 1.0]

    cur_left += label_col_frac + gap_frac
    legend_ax_pos = [cur_left, 0.0, legend_col_frac, 1.0]

    ax = fig.add_axes(main_ax_pos)
    label_ax = fig.add_axes(label_ax_pos)
    leg_ax = fig.add_axes(legend_ax_pos)

    # ------------------------------------------------------------
    # Draw sequence-index bars (one horizontal bar per segment)
    # ------------------------------------------------------------

    doc_to_y = dict(zip(docs, y_positions))

    for doc in docs:
        sub = df_long[df_long[doc_col] == doc].sort_values("start", kind="mergesort")
        y = doc_to_y[doc]

        for _, r in sub.iterrows():
            ax.barh(
                y=y,
                width=r["width"],
                left=r["start"],
                height=1.0,
                color=palette[seg_to_color[r[seg_col]]],
                edgecolor="none",
                linewidth=0,
                antialiased=False,
                rasterized=rasterize_bars,
            )

    # ------------------------------------------------------------
    # Main axis styling (remove ticks/spines for dense display)
    # ------------------------------------------------------------

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, total_rows_with_gaps - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------
    # Draw cluster labels beside grouped documents
    # ------------------------------------------------------------

    label_ax.axis("off")
    label_ax.set_ylim(ax.get_ylim())
    for _, row in cluster_blocks.iterrows():
        label_ax.text(
            0.0,
            float(row["label_y"]),
            _cluster_label(int(row["cluster"])),
            va="center",
            ha="left",
            fontsize=cluster_label_fontsize,
            color="black",
            clip_on=False,
            transform=label_ax.transData,
        )

    _draw_vertical_legend(
    ax=leg_ax,
    categories_legend=categories_legend,
    seg_to_color=seg_to_color,
    palette=palette,
    legend_fontsize=legend_fontsize,
    y_top=LEGEND_Y_TOP,
    y_min=LEGEND_Y_MIN,
    patch_height=LEGEND_PATCH_HEIGHT,
    patch_width=LEGEND_PATCH_WIDTH,
    row_spacing=LEGEND_ROW_SPACING,
)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0, dpi=dpi)

    plt.show()
    return fig, ax
