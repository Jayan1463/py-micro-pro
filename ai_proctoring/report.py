import os
from collections import Counter

import matplotlib.pyplot as plt


def _safe_violation_type(item):
    return str(item.get("violation", "Unknown"))


def _safe_timestamp(item):
    ts = item.get("timestamp", 0)
    try:
        return int(ts)
    except Exception:
        return 0


def generate_report(violations, report_path="violations_report.png"):
    violations = violations or []

    print("\n" + "=" * 40)
    print("SESSION SUMMARY")
    print("=" * 40)

    total_violations = len(violations)
    print(f"Total Violations: {total_violations}")

    if total_violations == 0:
        print("No violations detected. Great job!")
        print("=" * 40)

        # Still generate a clean placeholder report image for consistency.
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No Violations Detected",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#0f172a",
        )
        ax.text(
            0.5,
            0.40,
            "Session completed with a clean record.",
            ha="center",
            va="center",
            fontsize=12,
            color="#334155",
        )
        fig.savefig(report_path, dpi=180, bbox_inches="tight")
        print(f"Graph saved as '{report_path}'")
        plt.close(fig)
        return

    types = [_safe_violation_type(v) for v in violations]
    type_counts = Counter(types)

    print("\nViolation Breakdown:")
    for v_type, count in sorted(type_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {v_type}: {count}")
    print("=" * 40)

    timestamps = [_safe_timestamp(v) for v in violations]
    time_counts = Counter(timestamps)
    times = sorted(time_counts.keys())
    counts = [time_counts[t] for t in times]

    fig, (ax_types, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f8fafc")

    # Left chart: violation type distribution.
    sorted_types = sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))
    labels = [item[0] for item in sorted_types]
    values = [item[1] for item in sorted_types]
    bars = ax_types.bar(labels, values, color="#e11d48")
    ax_types.set_title("Violation Type Distribution", fontsize=12, fontweight="bold")
    ax_types.set_ylabel("Count")
    ax_types.grid(axis="y", alpha=0.25)
    ax_types.tick_params(axis="x", labelrotation=25)
    for bar, value in zip(bars, values):
        ax_types.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0f172a",
        )

    # Right chart: timeline of violation spikes.
    ax_time.bar(times, counts, width=0.8, color="#fb7185")
    ax_time.set_title("Violations Over Time", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Count")
    ax_time.grid(axis="y", alpha=0.25)

    total_text = f"Total Violations: {total_violations}"
    fig.suptitle(total_text, fontsize=15, fontweight="bold", color="#0f172a")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(report_path, dpi=180, bbox_inches="tight")
    print(f"Graph saved as '{report_path}'")
    plt.close(fig)
