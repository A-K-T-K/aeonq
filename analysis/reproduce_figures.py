"""Draw the evaluation figures from the stored results.

Data are read through the loaders in reproduce_tables.py; no experiment is
executed.

Usage, from the repository root:

    python analysis/reproduce_figures.py                 # write to analysis/figures
    python analysis/reproduce_figures.py --out DIRECTORY
    python analysis/reproduce_figures.py --only 11 14    # selected figures
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_tables as data  # noqa: E402

# Style ----------------------------------------------------------------------

TEXT_WIDTH = 7.0  # printed text width in inches
ASPECT = 16 / 9
FONT_SIZE = 9
SMALL = 8
INK = "#000000"
MUTED = "#6e6e6e"
GRID = "#e4e4e4"
# Categorical palette distinguishable under colour-vision deficiency; each
# series also has its own marker for greyscale reproduction.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#d4a000", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ("#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "legend.fontsize": SMALL,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figure(width_fraction: float = 1.0, **kwargs: Any) -> tuple[Any, Any]:
    width = TEXT_WIDTH * width_fraction
    return plt.subplots(figsize=(width, width / ASPECT), **kwargs)


def boxed(ax: Any, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)


def panel(ax: Any, index: int, title: str) -> None:
    ax.set_title(f"({'abcd'[index]}) {title}", loc="left", pad=3)


def nominal(ax: Any, y: float = 0.95) -> None:
    ax.axhline(y, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)


def save(fig: Any, out: Path, name: str, rect: Any = None) -> None:
    fig.tight_layout(pad=0.4, rect=rect)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=600)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def wilson_errors(cells: list[dict]) -> list[list[float]]:
    return [
        [max(0.0, c["coverage"] - c["wilson"][0]) for c in cells],
        [max(0.0, c["wilson"][1] - c["coverage"]) for c in cells],
    ]


# Figure 1: architecture -----------------------------------------------------


def figure_architecture(out: Path) -> None:
    counts = data.check_totals()
    fig, ax = figure()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    boxed(ax, grid=False)

    def box(x, y, w, h, title, lines, edge, fill, dashed=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.15",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=fill,
                linestyle="--" if dashed else "-",
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.35,
            title,
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + h - 1.05,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=SMALL,
            linespacing=1.45,
            color="#222222",
        )

    def arrow(start, end, dashed=False, label=None, offset=(0.15, 0)):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "linewidth": 0.9,
                "linestyle": "--" if dashed else "-",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 9,
            },
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2 + offset[0],
                (start[1] + end[1]) / 2 + offset[1],
                label,
                fontsize=SMALL,
                style="italic",
                color="#333333",
                ha="left" if offset[0] >= 0 else "right",
                va="center",
            )

    top, height = 5.3, 3.2
    box(
        0.3,
        top,
        2.9,
        height,
        "Problem\nspecification",
        ["", "State preparation $A$", "Objective predicate $\\chi$", "Register roles"],
        "#4d4d4d",
        "#f4f4f4",
    )
    box(
        3.55,
        top,
        3.5,
        height,
        "Semantic core",
        [
            "IR: $A$, $A^{\\dagger}$, $S_0$, $S_\\chi$, $Q^k$",
            "Explicit global phase",
            "Exact reference semantics",
            "Disjoint-cube $S_\\chi$ synthesis",
        ],
        COLOURS[0],
        "#edf3fb",
    )
    box(
        7.45,
        top,
        2.75,
        height,
        "Compiler\n(optional)",
        ["", "Exact local rewrites", "$Q^k$ kept symbolic", "Native option selection"],
        COLOURS[0],
        "#edf3fb",
        dashed=True,
    )
    box(
        10.6,
        top,
        5.1,
        height,
        "Backend adapters",
        [
            "Capabilities · validate · lower",
            "Compile · execute · normalize",
            "Resource metrics",
            "Qiskit, Cirq, PennyLane, Q#, reference",
        ],
        COLOURS[1],
        "#fcefe9",
    )
    box(
        0.3,
        0.3,
        9.4,
        2.9,
        "Conformance battery and mutation analysis",
        [
            "Operator: exact matrices, no free global phase "
            f"({counts['operator']} checks)",
            "Measurement: asymmetric computational basis states "
            f"({counts['measurement']} checks)",
            "Sampling: G-test of $Q^kA$ counts against exact probabilities "
            f"({counts['sampling']} checks)",
            f"{len(data.MUTANTS)} seeded lowering and readout faults injected "
            "into every adapter",
        ],
        "#4d4d4d",
        "#f4f4f4",
    )
    box(
        10.6,
        0.3,
        5.1,
        2.9,
        "Inference layer (SDK-independent)",
        [
            "Canonical, iterative, maximum-likelihood",
            "and faster estimators",
            "Interval, guarantee state and warnings",
        ],
        COLOURS[2],
        "#e8f6f0",
    )
    mid = top + height / 2
    arrow((3.2, mid), (3.55, mid))
    arrow((7.05, mid), (7.45, mid))
    arrow((10.2, mid), (10.6, mid))
    arrow(
        (5.3, top),
        (5.3, 3.2),
        dashed=True,
        label="Reference\nsemantics",
        offset=(-0.15, 0),
    )
    arrow((10.85, top), (9.7, 2.5), label="Adapters\nunder test", offset=(-0.35, 0.2))
    arrow((12.4, top), (12.4, 3.2), label="Normalized\ncounts", offset=(-0.15, 0))
    arrow(
        (14.5, top),
        (14.5, 3.2),
        label="Declared\nsampling\ncapabilities",
        offset=(-0.15, 0),
    )
    save(fig, out, "figure_01_architecture")


# Figure 6: compiler resources -----------------------------------------------


def figure_compiler(out: Path) -> None:
    rows = data.compiler_rows()
    pipelines = (
        ("naive_materialization", "Naive lowering (level 0)", False),
        ("generic_sdk", "General-purpose transpilation (level 3)", False),
        ("ae_aware", "Amplitude-estimation-aware pipeline", True),
    )
    cases = (
        ("bernoulli_0.25", "Bernoulli (one qubit)", "-"),
        ("uniform_threshold_n2_k2", "Threshold (two qubits)", "--"),
    )
    fig, axes = figure(0.75, ncols=2)
    for p, (pipeline, _, hollow) in enumerate(pipelines):
        for case, _, linestyle in cases:
            selected = sorted(
                (r for r in rows if r["case_id"] == case and r["pipeline"] == pipeline),
                key=lambda r: int(r["power"]),
            )
            powers = [int(r["power"]) for r in selected]
            for ax, column in zip(axes, ("depth", "two_qubit_gates")):
                ax.plot(
                    powers,
                    [int(r[column]) for r in selected],
                    color=COLOURS[p],
                    marker=MARKERS[p],
                    markersize=6.5 if hollow else 3.8,
                    markerfacecolor="none" if hollow else COLOURS[p],
                    markeredgewidth=0.9,
                    linestyle=linestyle,
                    zorder=3 if hollow else 2,
                )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Grover power $k$", ylabel="Transpiled depth")
    axes[1].set(xlabel="Grover power $k$", ylabel="Two-qubit gate count")
    for index, (ax, title) in enumerate(zip(axes, ("Depth", "Two-qubit gates"))):
        ax.set_xticks([0, 8, 16, 24, 32])
        boxed(ax)
        panel(ax, index, title)
    handles = [
        Line2D(
            [],
            [],
            color=COLOURS[p],
            marker=MARKERS[p],
            markerfacecolor="none" if hollow else COLOURS[p],
            markersize=6.5 if hollow else 3.8,
            linestyle="none",
            label=label,
        )
        for p, (_, label, hollow) in enumerate(pipelines)
    ] + [
        Line2D([], [], color=INK, linestyle=linestyle, label=label)
        for _, label, linestyle in cases
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
    )
    save(fig, out, "figure_06_compiler_resources", rect=(0, 0.2, 1, 1))


# Figure 11: baseline coverage -----------------------------------------------


def _problem_label(case: str) -> str:
    labels = {
        "bernoulli_1e-06": "$10^{-6}$",
        "bernoulli_0.999999": "$1-10^{-6}$",
        "lognormal_tail_n3_k5_flag": "Asset, $n=3$",
        "lognormal_tail_n4_k10_flag": "Asset, $n=4$",
    }
    if case in labels:
        return labels[case]
    return f"{float(case.split('_', 1)[1]):g}"


def figure_baseline(out: Path) -> None:
    cells = data.baseline_cells()
    cases = data.baseline_case_order(cells)
    methods = (
        ("iqae", "Iterative"),
        ("mlqae", "Maximum likelihood"),
        ("faster_ae", "Faster"),
        ("canonical_qae", "Canonical"),
    )
    series = (
        ("aeonq", "aeonq", False),
        ("qiskit_algorithms", "qiskit-algorithms", False),
        ("qiskit_algorithms_lr", "qiskit-algorithms, likelihood-ratio interval", True),
    )
    fig, grid = figure(nrows=2, ncols=2, sharex=True, sharey=True)
    axes = grid.ravel()
    x = range(len(cases))
    for index, (ax, (method, title)) in enumerate(zip(axes, methods)):
        for position in range(0, len(cases), 2):
            ax.axvspan(position - 0.5, position + 0.5, color="#f3f3f3", lw=0, zorder=0)
        present = [s for s in series if (method, s[0], cases[0]) in cells]
        for slot, (library, label, hollow) in enumerate(present):
            selected = [cells[(method, library, c)] for c in cases]
            offset = (slot - (len(present) - 1) / 2) * 0.26
            ax.errorbar(
                [i + offset for i in x],
                [c["coverage"] for c in selected],
                yerr=wilson_errors(selected),
                fmt=MARKERS[slot],
                color=COLOURS[slot],
                markerfacecolor="none" if hollow else COLOURS[slot],
                markersize=3.4,
                markeredgewidth=0.8,
                elinewidth=0.7,
                capsize=0,
                label=label,
                zorder=3,
            )
        nominal(ax)
        ax.set_xlim(-0.5, len(cases) - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        boxed(ax)
        ax.grid(False, axis="x")
        panel(ax, index, title)
    for ax in axes[2:]:
        ax.set_xticks(
            list(x),
            [_problem_label(c) for c in cases],
            rotation=50,
            ha="right",
            rotation_mode="anchor",
            fontsize=SMALL - 1,
        )
        ax.set_xlabel("Benchmark problem (Bernoulli amplitude $a$ or asset model)")
    for ax in axes[::2]:
        ax.set_ylabel("Coverage")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, handletextpad=0.3)
    save(fig, out, "figure_11_native_baseline_coverage", rect=(0, 0, 1, 0.94))


# Figure 12: noise -----------------------------------------------------------


def figure_noise(out: Path) -> None:
    cells = data.noise_cells()
    cases = (
        ("bernoulli_0.25", "Bernoulli (one qubit)"),
        ("lognormal_tail_n2_k2_flag", "Financial (three qubits)"),
        ("lognormal_tail_n3_k5_flag", "Financial (four qubits)"),
    )
    methods = (("iqae", "Iterative"), ("mlqae", "Maximum likelihood"))
    rates = sorted({rate for rate, _, _ in cells})
    noisy = [r for r in rates if r > 0]
    ideal = noisy[0] / 10
    fig, axes = figure(0.75, ncols=2, sharey=True)
    for index, (ax, (method, title)) in enumerate(zip(axes, methods)):
        ax.axvspan(ideal / 1.7, ideal * 1.7, color="#ececec", lw=0, zorder=0)
        for slot, (case, label) in enumerate(cases):
            selected = [cells[(rate, case, method)] for rate in rates]
            xs = [rate if rate > 0 else ideal for rate in rates]
            spread = 1 + 0.12 * (slot - 1)
            ax.errorbar(
                [value * spread for value in xs],
                [c["coverage"] for c in selected],
                yerr=wilson_errors(selected),
                color=COLOURS[slot],
                marker=MARKERS[slot],
                markersize=3.6,
                elinewidth=0.7,
                capsize=0,
                linewidth=0.9,
                label=label,
                zorder=3,
            )
        nominal(ax)
        ax.set_xscale("log")
        ax.set_xlim(ideal / 2.4, noisy[-1] * 1.8)
        ax.set_xticks(
            [ideal, *noisy],
            ["Ideal", *(_rate_label(r) for r in noisy)],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.minorticks_off()
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlabel("Single-qubit depolarizing rate")
        boxed(ax)
        panel(ax, index, title)
    axes[0].set_ylabel("Coverage")
    axes[1].legend(loc="lower left", handletextpad=0.3)
    save(fig, out, "figure_12_noise_coverage")


def _rate_label(rate: float) -> str:
    mantissa, exponent = f"{rate:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return f"${power}$" if mantissa == "1" else rf"${mantissa}\times{power}$"


# Figure 13: scaling ---------------------------------------------------------


def figure_scaling(out: Path) -> None:
    summary = {
        key: value for key, value in data.scaling_summary().items() if key[2] == 16
    }
    fig, ax = figure(0.75)
    for slot, (backend, label) in enumerate(data.SCALING_BACKENDS):
        widths = sorted(q for (b, q, _) in summary if b == backend)
        cells = [summary[(backend, q, 16)] for q in widths]
        ax.errorbar(
            widths,
            [average for average, _, _ in cells],
            yerr=[spread or 0.0 for _, spread, _ in cells],
            color=COLOURS[slot],
            marker=MARKERS[slot],
            markersize=4,
            capsize=2,
            label=label.replace("Microsoft Q# (QDK)", "Q# (QDK)"),
        )
    ax.set_yscale("log")
    ax.set_xticks(sorted({q for (_, q, _) in summary}))
    ax.set(
        xlabel="Register width (qubits)",
        ylabel="Wall-clock time (s)",
    )
    boxed(ax)
    ax.legend(loc="upper left", ncol=2)
    save(fig, out, "figure_13_scaling")


# Figure 14: mutation detection -----------------------------------------------


def figure_mutation(out: Path) -> None:
    profiles = data.mutation_profiles()
    counts = data.check_totals()
    totals = tuple(counts[level] for level in data.CHECK_LEVELS)
    columns = tuple(
        f"{level.capitalize()}\n({counts[level]} checks)" for level in data.CHECK_LEVELS
    )
    fig, ax = figure()
    for i, (mutant, _) in enumerate(data.MUTANTS):
        values = profiles[mutant]["failed"]
        for j, (value, total) in enumerate(zip(values, totals)):
            fraction = value / total
            ax.add_patch(
                Rectangle(
                    (j + 0.03, i + 0.06),
                    0.94,
                    0.88,
                    facecolor=SEQUENTIAL(fraction),
                    edgecolor=GRID if value == 0 else "none",
                    linewidth=0.5,
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                str(value),
                ha="center",
                va="center",
                fontsize=SMALL,
                color="white" if fraction > 0.55 else INK,
            )
        if values[1] == 0 and values[2] == 0:
            ax.add_patch(
                Rectangle(
                    (0.012, i + 0.035),
                    2.976,
                    0.93,
                    fill=False,
                    edgecolor=INK,
                    linestyle=(0, (3, 2)),
                    linewidth=0.8,
                )
            )
    ax.set_xlim(0, 3)
    ax.set_ylim(len(data.MUTANTS), 0)
    ax.set_xticks([0.5, 1.5, 2.5], columns)
    ax.xaxis.tick_top()
    ax.set_yticks(
        [i + 0.5 for i in range(len(data.MUTANTS))],
        [
            label.replace("R_Z", "$R_Z$")
            .replace("Q^(k+1) for Q^k", "$Q^{k+1}$ for $Q^k$")
            .replace("|1...1>", r"$|1\ldots1\rangle$")
            .replace("S_chi", r"$S_\chi$")
            for _, label in data.MUTANTS
        ],
    )
    ax.tick_params(length=0, top=False, right=False, bottom=False, left=False)
    boxed(ax, grid=False)
    colour_bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, 1)),
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    colour_bar.set_label("Fraction of category's checks failed")
    colour_bar.outline.set_linewidth(0.6)
    colour_bar.ax.tick_params(direction="in", labelsize=SMALL)
    save(fig, out, "figure_14_mutation_detection")


FIGURES = {
    1: figure_architecture,
    6: figure_compiler,
    11: figure_baseline,
    12: figure_noise,
    13: figure_scaling,
    14: figure_mutation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="output folder (default: analysis/figures)",
    )
    parser.add_argument("--only", type=int, nargs="*", help="figure numbers to draw")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    apply_style()
    for number, draw in FIGURES.items():
        if not arguments.only or number in arguments.only:
            draw(arguments.out)


if __name__ == "__main__":
    main()
