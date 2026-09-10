from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

FIGURE_SCALE = 5.0
PANEL_FONT_SIZE = 48.0
AXIS_LABEL_SIZE = 30.0
TICK_FONT_SIZE = 24.0
INTERNAL_FONT_SIZE = 24.0
OURS = "Frank-WT MF-PIRN"
ABLATION_RED = "#B64342"
ABLATION_BLUE = "#7884B4"
TEMPERATURE_LABEL = "Temperature (deg C)"
PLOT_LINE_WIDTH = 6.0
THIN_LINE_WIDTH = 4.0
MARKER_AREA = 320.0
BAR_CAP_SIZE = 10.0

METHODS = [
    OURS,
    "MF-HVKG",
    "qLogNEHVI",
    "MOMF",
    "PHSEA",
    "MSAEA-DM",
    "AS-SMEA",
    "B2 direct screening",
    "Original MF-BCL-SAEA",
    "ParEGO-GP",
]
METHOD_COLORS = {
    OURS: "#B64342",
    "MF-HVKG": "#484878",
    "qLogNEHVI": "#6677A8",
    "MOMF": "#8795BF",
    "PHSEA": "#75A6A0",
    "MSAEA-DM": "#8BB7AE",
    "AS-SMEA": "#B1B9D1",
    "B2 direct screening": "#A99AC1",
    "Original MF-BCL-SAEA": "#BEB4CE",
    "ParEGO-GP": "#A8A8A8",
}
METHOD_LABELS = {
    OURS: "Frank-WT\nMF-PIRN",
    "MF-HVKG": "MF-HVKG",
    "qLogNEHVI": "qLogNEHVI",
    "MOMF": "MOMF",
    "PHSEA": "PHSEA",
    "MSAEA-DM": "MSAEA-DM",
    "AS-SMEA": "AS-SMEA",
    "B2 direct screening": "B2 direct\nscreening",
    "Original MF-BCL-SAEA": "Original MF-\nBCL-SAEA",
    "ParEGO-GP": "ParEGO-GP",
}


def save_pdf_without_tight_crop(fig: plt.Figure, output_path: Path) -> None:
    stem = output_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def scaled_size(width, height):
    return width * FIGURE_SCALE, height * FIGURE_SCALE


def publication_style(**overrides):
    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": INTERNAL_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": INTERNAL_FONT_SIZE,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 3.5,
        "xtick.major.width": 3.0,
        "ytick.major.width": 3.0,
        "xtick.major.size": 17.5,
        "ytick.major.size": 17.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
    style.update(overrides)
    return style


def add_matplotlib_panel_label(ax, label, fontsize=PANEL_FONT_SIZE):
    """Add the author-specified 48-pt Times New Roman panel lettering."""
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        -0.085,
        1.015,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def regenerate_physics_validation_figure():
    """Redraw manuscript Fig. 3 with Times New Roman 48/30/24 pt typography."""
    hierarchy = pd.read_csv(DATA / "Table9_Frank_WT_hierarchy.csv")
    mesh = pd.read_csv(DATA / "mesh_convergence.csv")
    energy = pd.read_csv(DATA / "energy_performance_cases.csv")

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=scaled_size(7.20, 5.28))
        fig.subplots_adjust(
            left=0.105,
            right=0.905,
            bottom=0.155,
            top=0.94,
            wspace=0.34,
            hspace=0.42,
        )

        x = np.arange(len(hierarchy))
        width = 0.24
        axes[0, 0].bar(
            x - width,
            hierarchy.shape_MAE,
            width,
            label="shape MAE",
            color=ABLATION_RED,
        )
        axes[0, 0].bar(
            x,
            hierarchy.force_MAPE,
            width,
            label="force percentage error",
            color="#6677A8",
        )
        axes[0, 0].bar(
            x + width,
            hierarchy.stress_MAPE,
            width,
            label="stress percentage error",
            color="#75A6A0",
        )
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(
            hierarchy.model,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=TICK_FONT_SIZE,
            linespacing=0.9,
        )
        axes[0, 0].set_ylabel("Normalized error")
        axes[0, 0].legend(loc="upper right", fontsize=INTERNAL_FONT_SIZE)

        axes[0, 1].plot(
            mesh.mesh_size_mm,
            mesh.shape_error,
            "o-",
            color=ABLATION_RED,
            lw=PLOT_LINE_WIDTH,
            ms=18,
            label="shape error",
        )
        mesh_runtime_axis = axes[0, 1].twinx()
        mesh_runtime_axis.plot(
            mesh.mesh_size_mm,
            mesh.runtime_min,
            "s-",
            color="#6677A8",
            lw=PLOT_LINE_WIDTH,
            ms=18,
            label="runtime",
        )
        axes[0, 1].invert_xaxis()
        axes[0, 1].set_xlabel("Mesh size (mm)")
        axes[0, 1].set_ylabel("Shape error")
        mesh_runtime_axis.set_ylabel("Runtime (min)")
        mesh_runtime_axis.tick_params(
            labelsize=TICK_FONT_SIZE,
            width=3.0,
            length=15.0,
        )
        mesh_runtime_axis.spines["top"].set_visible(False)
        mesh_runtime_axis.spines["right"].set_linewidth(3.5)

        axes[1, 0].scatter(
            energy.WT_energy_mJ,
            energy.blocking_force_mN,
            c=energy.Frank_energy_mJ,
            cmap="viridis",
            s=MARKER_AREA,
            alpha=0.74,
            edgecolors="none",
        )
        axes[1, 0].set_xlabel("W-T energy (mJ)")
        axes[1, 0].set_ylabel("Blocking force (mN)")

        stress_scatter = axes[1, 1].scatter(
            energy.Frank_energy_mJ,
            energy.peak_stress_MPa,
            c=energy.shape_error,
            cmap="magma_r",
            s=MARKER_AREA,
            alpha=0.74,
            edgecolors="none",
        )
        axes[1, 1].set_xlabel("Frank energy (mJ)")
        axes[1, 1].set_ylabel("Peak stress (MPa)")
        colorbar = fig.colorbar(stress_scatter, ax=axes[1, 1], fraction=0.046, pad=0.04)
        colorbar.set_label("Shape error", fontsize=AXIS_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE, width=3.0, length=15.0)

        for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
            axis.grid(False)
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        output_path = OUTPUT / "Figure_3.pdf"
        save_pdf_without_tight_crop(fig, output_path)


def regenerate_device_validation_figure():
    """Redraw the material/device response plate with source-level 48/30/24 pt text."""
    thermal = pd.read_csv(DATA / "material_curves.csv")
    device = pd.read_csv(DATA / "Table11_device_validation.csv")
    cycles = pd.read_csv(DATA / "cycle_retention.csv")

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 3, figsize=scaled_size(7.20, 4.98))
        fig.subplots_adjust(
            left=0.085,
            right=0.985,
            bottom=0.105,
            top=0.945,
            wspace=0.34,
            hspace=0.42,
        )

        thermal_mean = thermal.groupby("temperature_C").mean(numeric_only=True)
        line_width = 6.0
        axes[0, 0].plot(
            thermal_mean.index,
            thermal_mean.DSC_heat_flow_au,
            color=ABLATION_RED,
            lw=line_width,
        )
        axes[0, 0].axvspan(80, 100, color="#FBE8E6")
        axes[0, 0].set(
            xlabel=TEMPERATURE_LABEL,
            ylabel="DSC heat flow (a.u.)",
        )

        axes[0, 1].semilogy(
            thermal_mean.index,
            thermal_mean.storage_modulus_MPa,
            color="#6677A8",
            lw=line_width,
        )
        axes[0, 1].set(
            xlabel=TEMPERATURE_LABEL,
            ylabel="Storage modulus (MPa)",
        )
        axes[0, 1].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        axes[0, 1].yaxis.set_minor_formatter(mticker.NullFormatter())

        axes[0, 2].plot(
            thermal_mean.index,
            thermal_mean.free_contraction_pct,
            color="#75A6A0",
            lw=line_width,
        )
        axes[0, 2].set(
            xlabel=TEMPERATURE_LABEL,
            ylabel="Free contraction (%)",
        )

        device_mean = (
            device.groupby(["design", "temperature_C"])
            .mean(numeric_only=True)
            .reset_index()
        )
        design_colors = {
            "shape-priority": "#6677A8",
            "balanced": ABLATION_RED,
            "force-priority": "#75A6A0",
        }
        for design in ("shape-priority", "balanced", "force-priority"):
            group = device_mean[device_mean.design == design]
            axes[1, 0].plot(
                group.temperature_C,
                group.shape_error,
                color=design_colors[design],
                lw=line_width,
                label=design,
            )
            axes[1, 1].plot(
                group.temperature_C,
                group.blocking_force_mN,
                color=design_colors[design],
                lw=line_width,
                label=design,
            )
        axes[1, 0].set(
            xlabel=TEMPERATURE_LABEL,
            ylabel="Shape error",
        )
        axes[1, 0].legend(loc="upper right", fontsize=INTERNAL_FONT_SIZE)
        axes[1, 1].set(
            xlabel=TEMPERATURE_LABEL,
            ylabel="Blocking force (mN)",
        )

        cycle_mean = cycles.groupby("cycle").mean(numeric_only=True)
        axes[1, 2].plot(
            cycle_mean.index,
            cycle_mean.force_retention * 100,
            color=ABLATION_RED,
            lw=line_width,
            label="force",
        )
        axes[1, 2].plot(
            cycle_mean.index,
            cycle_mean.shape_retention * 100,
            color="#6677A8",
            lw=line_width,
            label="shape",
        )
        axes[1, 2].set(xlabel="Thermal cycle", ylabel="Retention (%)")
        axes[1, 2].legend(loc="upper right", fontsize=INTERNAL_FONT_SIZE)

        for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
            axis.grid(False)
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        output_path = OUTPUT / "Figure_8.pdf"
        save_pdf_without_tight_crop(fig, output_path)


def generate_aligned_ablation_source():
    """Redraw the six ablation panels from the unchanged archived CSVs.

    A single 3 x 2 GridSpec gives panels (c) and (e), and panels (d) and (f),
    exactly identical axis boxes.  The first row remains slightly taller for
    the twelve ablation labels; all text retains the journal size.
    """
    ablation = pd.read_csv(DATA / "complete_ablation_runs.csv")
    factorial = pd.read_csv(DATA / "factorial_interaction_runs.csv")
    summary = (
        ablation.groupby("variant")
        .agg(
            HV=("normalized_HV", "mean"),
            sd=("normalized_HV", "std"),
            shape=("shape_MAE", "mean"),
            force=("blocking_force_MAPE", "mean"),
            calls=("HF_calls_to_target", "mean"),
            failure=("solver_failure_fraction", "mean"),
        )
        .sort_values("HV")
    )

    style = publication_style()

    with plt.rc_context(style):
        # One common six-panel canvas removes the sub-pixel discrepancy that
        # arose when the upper four and lower two panels were separate PDFs.
        fig, axes = plt.subplots(
            3,
            2,
            figsize=scaled_size(7.9, 8.15),
            gridspec_kw={"height_ratios": [1.18, 1.0, 1.0]},
        )
        fig.subplots_adjust(
            left=0.15, right=0.97, bottom=0.07, top=0.97, wspace=0.14, hspace=0.38
        )
        y = np.arange(len(summary))

        axes[0, 0].errorbar(
            summary.HV,
            y,
            xerr=summary.sd,
            fmt="o",
            color=ABLATION_RED,
            ecolor="#A8A8A8",
            capsize=BAR_CAP_SIZE,
            markersize=18,
            elinewidth=THIN_LINE_WIDTH,
        )
        axes[0, 0].set_yticks(y)
        display_variants = [
            "no material-\nmanufacturing fingerprint"
            if variant == "no material-manufacturing fingerprint"
            else "no elastic weight\nconsolidation"
            if variant == "no EWC"
            else variant
            for variant in summary.index
        ]
        axes[0, 0].set_yticklabels(
            display_variants, fontsize=TICK_FONT_SIZE, linespacing=0.86
        )
        axes[0, 0].set_xlabel("Normalized HV")

        axes[0, 1].barh(
            y,
            summary.calls,
            color=[
                ABLATION_RED if variant == OURS else "#B8C3D9"
                for variant in summary.index
            ],
        )
        axes[0, 1].set_yticks(y)
        axes[0, 1].set_yticklabels([])
        axes[0, 1].set_xlabel("HF calls to target")

        axes[1, 0].scatter(
            summary["shape"] * 100,
            summary["force"] * 100,
            s=MARKER_AREA,
            color=[
                ABLATION_RED if variant == OURS else ABLATION_BLUE
                for variant in summary.index
            ],
        )
        axes[1, 0].set_xlabel("Shape MAE (%)")
        axes[1, 0].set_ylabel("Blocking-force percentage error (%)")

        modules = ["Frank3", "fingerprint", "PIRN", "physics_gate"]
        full = factorial[(factorial[modules] == 1).all(axis=1)].normalized_HV.mean()
        matrix = np.zeros((4, 4))
        for i, first in enumerate(modules):
            for j, second in enumerate(modules):
                if i == j:
                    subset = factorial[factorial[first] == 0]
                else:
                    subset = factorial[
                        (factorial[first] == 0) & (factorial[second] == 0)
                    ]
                matrix[i, j] = full - subset.normalized_HV.mean()

        image = axes[1, 1].imshow(
            matrix,
            cmap="Reds",
            vmin=0,
            vmax=matrix.max(),
            aspect="auto",
            interpolation="nearest",
        )
        short = ["Frank-3", "fingerprint", "PIRN", "gate"]
        axes[1, 1].set_xticks(range(4))
        axes[1, 1].set_xticklabels(
            short, rotation=35, ha="right", fontsize=TICK_FONT_SIZE
        )
        axes[1, 1].set_yticks(range(4))
        axes[1, 1].set_yticklabels(short, fontsize=TICK_FONT_SIZE)
        for i in range(4):
            for j in range(4):
                axes[1, 1].text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=INTERNAL_FONT_SIZE,
                    color="white" if matrix[i, j] > matrix.max() * 0.55 else "black",
                )
        colorbar = fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
        colorbar.set_label("HV loss", fontsize=AXIS_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE, width=3.0, length=15.0)

        axes[2, 0].scatter(
            summary.failure * 100, summary.HV, color=ABLATION_RED, s=MARKER_AREA
        )
        axes[2, 0].set_xlabel("Solver failure (%)")
        axes[2, 0].set_ylabel("Normalized HV")
        axes[2, 1].scatter(summary.calls, summary.HV, color="#6677A8", s=MARKER_AREA)
        axes[2, 1].set_xlabel("HF calls")
        axes[2, 1].set_ylabel("Normalized HV")

        for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
            ax.grid(False)
            ax.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(ax, label)

        c_box = axes[1, 0].get_position()
        e_box = axes[2, 0].get_position()
        c_geometry = np.array([c_box.x0, c_box.x1, c_box.width, c_box.height])
        e_geometry = np.array([e_box.x0, e_box.x1, e_box.width, e_box.height])
        if not np.allclose(c_geometry, e_geometry, rtol=0.0, atol=1e-12):
            raise RuntimeError("Panels (c) and (e) do not share an identical axis box")

        source_path = OUTPUT / "Figure_6.pdf"
        save_pdf_without_tight_crop(fig, source_path)

    return source_path


def _set_method_ticks(
    axis, methods, *, show=True, rotation=80, fontsize=TICK_FONT_SIZE
):
    """Apply wrapped method labels without allowing adjacent names to collide."""
    axis.set_xticks(np.arange(len(methods)))
    if show:
        labels = [
            method if rotation >= 75 else METHOD_LABELS[method] for method in methods
        ]
        axis.set_xticklabels(
            labels,
            rotation=rotation,
            ha="right",
            rotation_mode="anchor",
            fontsize=fontsize,
            linespacing=0.86,
        )
    else:
        axis.set_xticklabels([])


def regenerate_main_comparison_figure():
    """Redraw Fig. 4 from the unchanged summary with non-overlapping labels."""
    summary = pd.read_csv(DATA / "main_SOTA_summary.csv")
    summary = summary.set_index("method").loc[METHODS]
    x = np.arange(len(METHODS))
    colors = [METHOD_COLORS[method] for method in METHODS]

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=scaled_size(7.35, 5.35),
            sharex=True,
            layout="constrained",
        )
        fig.set_constrained_layout_pads(
            w_pad=0.035, h_pad=0.035, wspace=0.07, hspace=0.05
        )

        axes[0, 0].bar(
            x, summary.HV_mean, yerr=summary.HV_ci95, color=colors, capsize=BAR_CAP_SIZE
        )
        axes[0, 0].set_ylabel("Normalized HV")
        axes[0, 0].set_ylim(0.82, 0.95)

        axes[0, 1].bar(x, summary.IGD_plus_mean, color=colors)
        axes[0, 1].set_ylabel(r"IGD$^{+}$")

        axes[1, 0].bar(x, summary.HF_calls_mean, color=colors)
        axes[1, 0].set_ylabel("HF calls to target")

        axes[1, 1].bar(x, summary.feasible_fraction_mean, color=colors)
        axes[1, 1].set_ylabel("Feasible fraction")

        for axis in axes[0]:
            _set_method_ticks(axis, METHODS, show=False)
        for axis in axes[1]:
            _set_method_ticks(axis, METHODS, show=True, rotation=80)
        for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        output_path = OUTPUT / "Figure_4.pdf"
        save_pdf_without_tight_crop(fig, output_path)


def compose_convergence_statistics_figure():
    """Redraw the six convergence panels from unchanged registered CSV data."""
    convergence = pd.read_csv(DATA / "convergence_runs.csv")
    main_runs = pd.read_csv(DATA / "main_SOTA_runs.csv")
    pairwise = pd.read_csv(DATA / "pairwise_statistics.csv")
    selected = METHODS[:6]

    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(
            3, 2, figsize=scaled_size(7.5, 7.85), layout="constrained"
        )
        fig.set_constrained_layout_pads(
            w_pad=0.035, h_pad=0.035, wspace=0.08, hspace=0.08
        )

        for method in selected:
            grouped = convergence[convergence.method == method].groupby("HF_calls")
            calls = np.array(sorted(grouped.groups))
            hv = grouped.normalized_HV.mean().reindex(calls).to_numpy()
            hv_sd = grouped.normalized_HV.std().reindex(calls).to_numpy()
            igd = grouped.IGD_plus.mean().reindex(calls).to_numpy()
            linewidth = PLOT_LINE_WIDTH if method == OURS else THIN_LINE_WIDTH
            axes[0, 0].plot(
                calls, hv, color=METHOD_COLORS[method], lw=linewidth, label=method
            )
            axes[0, 0].fill_between(
                calls,
                hv - hv_sd,
                hv + hv_sd,
                color=METHOD_COLORS[method],
                alpha=0.08,
                linewidth=0,
            )
            axes[0, 1].plot(calls, igd, color=METHOD_COLORS[method], lw=linewidth)

        axes[0, 0].set(xlabel="High-fidelity calls", ylabel="Normalized HV")
        axes[0, 1].set(xlabel="High-fidelity calls", ylabel=r"IGD$^{+}$")
        axes[0, 0].legend(ncol=2, loc="lower right")

        for method in selected[:4]:
            values = np.sort(
                main_runs[
                    main_runs.method == method
                ].HF_calls_to_90pct_oracle_HV.to_numpy()
            )
            axes[1, 0].step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                color=METHOD_COLORS[method],
                lw=PLOT_LINE_WIDTH,
                label=method,
            )
        axes[1, 0].set(xlabel="HF calls to 90% oracle HV", ylabel="Empirical CDF")
        axes[1, 0].legend(loc="lower right")

        anytime = []
        for method in METHODS:
            curve = (
                convergence[convergence.method == method]
                .groupby("HF_calls")
                .normalized_HV.mean()
            )
            anytime.append(
                np.trapezoid(curve.to_numpy(), curve.index.to_numpy())
                / (curve.index.max() - curve.index.min())
            )
        axes[1, 1].bar(
            np.arange(len(METHODS)),
            anytime,
            color=[METHOD_COLORS[method] for method in METHODS],
        )
        axes[1, 1].set_ylabel("Anytime HV area")
        _set_method_ticks(axes[1, 1], METHODS, rotation=80)

        task_means = (
            main_runs.groupby(["task", "method"])
            .normalized_HV.mean()
            .unstack()[METHODS]
        )
        comparators = METHODS[1:]
        gains = (
            task_means[OURS].to_numpy()[:, None] / task_means[comparators].to_numpy()
            - 1
        ) * 100
        heatmap = axes[2, 0].imshow(
            gains,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-1,
            vmax=max(5, float(np.nanmax(gains))),
        )
        axes[2, 0].set_yticks(np.arange(len(task_means)))
        axes[2, 0].set_yticklabels(task_means.index, fontsize=TICK_FONT_SIZE)
        _set_method_ticks(axes[2, 0], comparators, rotation=80)
        colorbar = fig.colorbar(heatmap, ax=axes[2, 0], fraction=0.045, pad=0.025)
        colorbar.set_label("HV gain (%)", fontsize=AXIS_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE, width=3.0, length=15.0)

        comparison_names = pairwise.comparison.str.replace(
            f"{OURS} vs ", "", regex=False
        )
        y = np.arange(len(pairwise))
        axes[2, 1].plot(pairwise.HV_gain_pct, y, "o", color=METHOD_COLORS[OURS], ms=18)
        axes[2, 1].axvline(0, color="#666666", ls="--", lw=THIN_LINE_WIDTH)
        axes[2, 1].set_yticks(y)
        axes[2, 1].set_yticklabels(comparison_names, fontsize=TICK_FONT_SIZE)
        axes[2, 1].set_xlabel("Paired HV gain (%)")
        axes[2, 1].invert_yaxis()

        for axis, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")):
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        output_path = OUTPUT / "Figure_5.pdf"
        save_pdf_without_tight_crop(fig, output_path)


def compose_transfer_robustness_figure():
    """Redraw all nine panels so no white mask or inherited label remains."""
    analytical = pd.read_csv(DATA / "analytical_generalization_runs.csv")
    ood = pd.read_csv(DATA / "material_topology_OOD_runs.csv")
    robustness = pd.read_csv(DATA / "robustness_runs.csv")

    analytical_methods = METHODS[:7]
    analytical_means = (
        analytical.groupby(["task", "method"])
        .normalized_HV.mean()
        .unstack()[analytical_methods]
    )
    ranks = analytical_means.rank(axis=1, ascending=False, method="average")
    ood_means = (
        ood.groupby(["task", "method"])
        .normalized_HV.mean()
        .unstack()[analytical_methods]
    )

    ood_label_map = {
        "double helix": "double helix",
        "fast-response actuator": "fast-response\nactuator",
        "graded-pitch helix": "graded-pitch\nhelix",
        "high-force actuator": "high-force\nactuator",
        "long compliant helix": "long compliant\nhelix",
        "low-stress actuator": "low-stress\nactuator",
        "manufacturing-tolerance shift": "manufacturing-\ntolerance shift",
        "short tight helix": "short tight helix",
        "torsional coil": "torsional coil",
        "unseen LCE formulation A": "unseen LCE\nformulation A",
        "unseen LCE formulation B": "unseen LCE\nformulation B",
        "variable-radius helix": "variable-radius\nhelix",
    }
    scenarios = [
        ("geometry_tolerance_pct", "Geometry tolerance (%)"),
        ("material_parameter_noise_pct", "Material noise (%)"),
        ("transition_temperature_shift_C", "Transition shift (°C)"),
        ("missing_fidelity_pairs_pct", "Missing fidelity pairs (%)"),
        ("HF_solver_failure_pct", "Failed HF solves (%)"),
        ("measurement_noise_pct", "Measurement noise (%)"),
    ]
    robust_methods = [OURS, "MF-HVKG", "qLogNEHVI", "PHSEA", "AS-SMEA"]

    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=scaled_size(7.5, 8.85))
        top_grid = fig.add_gridspec(
            1,
            3,
            left=0.055,
            right=0.985,
            bottom=0.675,
            top=0.965,
            width_ratios=(1.42, 0.78, 1.22),
            wspace=0.54,
        )
        robustness_grid = fig.add_gridspec(
            2,
            3,
            left=0.055,
            right=0.985,
            bottom=0.080,
            top=0.555,
            width_ratios=(1.0, 1.0, 1.0),
            wspace=0.42,
            hspace=0.36,
        )
        axes = [fig.add_subplot(top_grid[0, col]) for col in range(3)]
        axes.extend(
            fig.add_subplot(robustness_grid[row, col])
            for row in range(2)
            for col in range(3)
        )

        rank_image = axes[0].imshow(
            ranks.to_numpy(),
            aspect="auto",
            cmap="viridis_r",
            vmin=1,
            vmax=len(analytical_methods),
        )
        axes[0].set_xticks(np.arange(len(analytical_methods)))
        axes[0].set_xticklabels(
            analytical_methods,
            rotation=80,
            ha="right",
            rotation_mode="anchor",
            fontsize=TICK_FONT_SIZE,
            linespacing=0.84,
        )
        axes[0].set_yticks(np.arange(len(ranks)))
        axes[0].set_yticklabels(ranks.index, fontsize=TICK_FONT_SIZE)
        rank_bar = fig.colorbar(rank_image, ax=axes[0], fraction=0.042, pad=0.02)
        rank_bar.set_label("Rank", fontsize=AXIS_LABEL_SIZE)
        rank_bar.ax.tick_params(labelsize=TICK_FONT_SIZE, width=3.0, length=15.0)

        average_rank = ranks.mean().sort_values()
        y_rank = np.arange(len(average_rank))
        axes[1].barh(
            y_rank,
            average_rank,
            color=[METHOD_COLORS[method] for method in average_rank.index],
        )
        axes[1].set_yticks(y_rank)
        axes[1].set_yticklabels(
            [METHOD_LABELS[method] for method in average_rank.index],
            fontsize=TICK_FONT_SIZE,
            linespacing=0.82,
        )
        axes[1].tick_params(axis="y", pad=2)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Mean rank across 30 tasks")

        ood_gain = (
            ood_means[OURS] / ood_means.drop(columns=[OURS]).max(axis=1) - 1
        ) * 100
        y_ood = np.arange(len(ood_gain))
        axes[2].barh(
            y_ood,
            ood_gain,
            color=["#B64342" if value >= 0 else "#6677A8" for value in ood_gain],
        )
        axes[2].axvline(0, color="#555555", lw=THIN_LINE_WIDTH)
        axes[2].set_yticks(y_ood)
        axes[2].set_yticklabels(
            [ood_label_map.get(label, label) for label in ood_gain.index],
            fontsize=TICK_FONT_SIZE,
            linespacing=0.82,
        )
        axes[2].invert_yaxis()
        axes[2].set_xlabel("HV gain over strongest baseline (%)")

        for axis, (scenario, xlabel), label in zip(
            axes[3:], scenarios, ("(d)", "(e)", "(f)", "(g)", "(h)", "(i)")
        ):
            subset = robustness[robustness.scenario == scenario]
            for method in robust_methods:
                summary = (
                    subset[subset.method == method]
                    .groupby("level")
                    .normalized_HV.agg(["mean", "std"])
                )
                level = summary.index.to_numpy(dtype=float)
                mean = summary["mean"].to_numpy()
                std = summary["std"].to_numpy()
                axis.plot(
                    level,
                    mean,
                    "o-",
                    ms=18,
                    lw=PLOT_LINE_WIDTH if method == OURS else THIN_LINE_WIDTH,
                    color=METHOD_COLORS[method],
                    label=method,
                )
                axis.fill_between(
                    level,
                    mean - std,
                    mean + std,
                    color=METHOD_COLORS[method],
                    alpha=0.06,
                    linewidth=0,
                )
            axis.set(xlabel=xlabel, ylabel="Normalized HV")
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        axes[3].legend(ncol=2, loc="lower left", fontsize=INTERNAL_FONT_SIZE)
        for axis, label in zip(axes[:3], ("(a)", "(b)", "(c)")):
            axis.tick_params(
                labelsize=TICK_FONT_SIZE,
                width=3.0,
                length=15.0,
            )
            add_matplotlib_panel_label(axis, label)

        output_path = OUTPUT / "Figure_7.pdf"
        save_pdf_without_tight_crop(fig, output_path)


def main() -> None:
    regenerate_physics_validation_figure()
    regenerate_main_comparison_figure()
    compose_convergence_statistics_figure()
    generate_aligned_ablation_source()
    compose_transfer_robustness_figure()
    regenerate_device_validation_figure()


if __name__ == "__main__":
    main()
