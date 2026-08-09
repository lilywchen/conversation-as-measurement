import matplotlib.pyplot as plt
import pandas as pd


def make_question_phase_panel(
    questions,
    clinician=None,
    overall_barcolor="tab:blue",
    clinician_color="#6A3D9A",
    patient_color="#E6AB02",
    filename=None,
):
    """
    Two-panel figure:
      Left  = overall question share by phase
      Right = within-phase speaker share (100% stacked)

    Both are computed per encounter and averaged across encounters.
    """

    print(clinician)

    if clinician is not None:
        questions = questions.copy(deep=True)
        questions = questions[questions["clinician"] == clinician]

    print("NUM QUESTIONS ", len(questions))
    print("NUM DOCS ", questions.doc.nunique())

    phase_order = [
        "Opening", "History", "Exam", "Assessment",
        "Plan", "Education", "Closing", "Non-clinical",
    ][::-1]

    # ---------- Shared base counts ----------
    phase_doc_counts = (
        questions
        .groupby(["doc", "segment_title"])
        .size()
        .reset_index(name="num_qs")
    )

    # ---------- Left panel: overall question share by phase ----------
    phase_doc_counts["total_qs"] = phase_doc_counts.groupby("doc")["num_qs"].transform("sum")
    phase_doc_counts["qs_pct"] = phase_doc_counts["num_qs"] / phase_doc_counts["total_qs"]

    overall_data = (
        phase_doc_counts.groupby("segment_title")["qs_pct"].sum()
        / phase_doc_counts["doc"].nunique() * 100
    ).reindex(phase_order).fillna(0).round(3)

    print("SUM OVERALL QUESTION SHARE:", sum(overall_data.to_dict().values()))

    # ---------- Right panel: within-phase speaker share ----------
    speaker_groups = (
        questions
        .groupby(["doc", "segment_title", "speaker"])
        .size()
        .reset_index(name="num_qs")
    )

    speaker_groups["total_phase_doc_qs"] = speaker_groups.groupby(
        ["doc", "segment_title"]
    )["num_qs"].transform("sum")
    speaker_groups["role_qs_pct"] = (
        speaker_groups["num_qs"] / speaker_groups["total_phase_doc_qs"]
    )

    speaker_pivot = speaker_groups.pivot_table(
        index=["doc", "segment_title"],
        columns="speaker",
        values="role_qs_pct",
        fill_value=0,
    )

    role_phase_df = speaker_pivot.groupby("segment_title").mean()

    role_phase_df = (
        role_phase_df
        .reindex(phase_order)
        .fillna(0)
    )

    # safety normalization so rows sum to 1
    role_phase_df = role_phase_df.div(role_phase_df.sum(axis=1), axis=0).fillna(0)

    clinician_vals = role_phase_df["clinician"] * 100
    patient_vals = role_phase_df["patient"] * 100

    # ---------- Figure ----------
    n = len(phase_order)
    fig_h = max(2.0, 0.30 * n)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(8.2, fig_h),
        gridspec_kw={"width_ratios": [1, 1.05]}
    )

    # ---------- Left: overall question share ----------
    bars_left = ax1.barh(
        overall_data.index,
        overall_data.values,
        edgecolor="white",
        color=overall_barcolor,
        height=0.55,
        linewidth=0.5
    )

    xmax = float(overall_data.max())
    pad = max(0.25, 0.03 * xmax)

    for bar, value in zip(bars_left, overall_data.values):
        ax1.text(
            value + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=11
        )

    ax1.set_xlim(0, xmax + 3 * pad)
    ax1.margins(y=0)
    ax1.set_ylim(-0.5, n - 0.5)

    ax1.set_xlabel("Average Question Distribution", labelpad=2, fontsize=12)
    ax1.set_xticks([])
    ax1.tick_params(axis="y", pad=2, length=0, labelsize=11)

    for spine in ["top", "right", "bottom"]:
        ax1.spines[spine].set_visible(False)

    ax1.spines["left"].set_color("#444444")
    ax1.spines["left"].set_linewidth(0.5)

    # ---------- Right: speaker share stacked ----------
    bars1 = ax2.barh(
        role_phase_df.index,
        clinician_vals,
        color=clinician_color,
        edgecolor="white",
        height=0.55,
        linewidth=0.5,
        label="Clinician"
    )

    bars2 = ax2.barh(
        role_phase_df.index,
        patient_vals,
        left=clinician_vals,
        color=patient_color,
        edgecolor="white",
        height=0.55,
        linewidth=0.5,
        label="Patient"
    )

    for bar1, c, p in zip(bars1, clinician_vals, patient_vals):
        y = bar1.get_y() + bar1.get_height() / 2

        ax2.text(
            c / 2,
            y,
            f"{c:.1f}%",
            va="center",
            ha="center",
            fontsize=8,
            color="white"
        )

        x_patient = c + p / 2
        if p < 15:
            x_patient += 2.8

        ax2.text(
            x_patient,
            y,
            f"{p:.1f}%",
            va="center",
            ha="center",
            fontsize=8,
            color="black"
        )

    ax2.set_xlim(0, 100)
    ax2.margins(y=0)
    ax2.set_ylim(-0.5, n - 0.5)

    ax2.set_xlabel("Average Speaker Share", labelpad=2, fontsize=12)
    ax2.set_xticks([])
    ax2.tick_params(axis="y", pad=2, length=0, labelsize=0)  # no repeated phase labels
    ax2.set_yticklabels([])

    for spine in ["top", "right", "bottom"]:
        ax2.spines[spine].set_visible(False)

    ax2.spines["left"].set_color("#444444")
    ax2.spines["left"].set_linewidth(0.5)

    ax2.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10
    )

    # ---------- Layout ----------
    fig.subplots_adjust(left=0.22, right=0.88, top=0.98, bottom=0.22, wspace=0.28)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=300)

    plt.show()
    
