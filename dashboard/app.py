import json
import io
import sys
import zipfile
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antennatwin.models import Room, Antenna, Receiver, Wall
from antennatwin.coverage import coverage_map
from antennatwin.optimizer import optimize_position
from antennatwin.pattern import radiation_pattern
from antennatwin.geometry import minimum_distance_to_walls_m


VERSION = "0.4.4.1"

st.set_page_config(
    page_title="AntennaTwin-Indoor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"📡 AntennaTwin-Indoor V{VERSION}")
st.caption("Interactive indoor antenna deployment analysis and optimization")


# -----------------------------------------------------------------------------
# Scenario/reproducibility helpers
# -----------------------------------------------------------------------------
def _scenario_payload(values):
    return {
        "software": "AntennaTwin-Indoor",
        "version": VERSION,
        "scenario": values,
    }

def _array_csv(X, Y, *arrays, names):
    data = {"x_m": np.asarray(X).ravel(), "y_m": np.asarray(Y).ravel()}
    for name, arr in zip(names, arrays):
        data[name] = np.asarray(arr).ravel()
    return pd.DataFrame(data).to_csv(index=False)

def _reproducibility_zip(report, scenario_json, files):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README_REPRODUCIBILITY.txt",
            "AntennaTwin-Indoor reproducibility package\n"
            "==========================================\n"
            f"Software version: {VERSION}\n"
            f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
            "Contents:\n"
            "- scenario.json: input parameters used for the run\n"
            "- simulation_report.json: numerical summary and derived metrics\n"
            "- initial_map.csv / optimized_map.csv: receiver-grid RSSI, wall loss and antenna gain\n"
            "- radiation_pattern.csv: initial and optimized directional pattern\n\n"
            "These results are model-based simulations. They do not constitute experimental or ray-tracing validation.\n")
        z.writestr("scenario.json", scenario_json)
        z.writestr("simulation_report.json", json.dumps(report, indent=2))
        for name, content in files.items():
            z.writestr(name, content)
    bio.seek(0)
    return bio.getvalue()

# -----------------------------------------------------------------------------
# Sidebar: model inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Scenario")

    uploaded_scenario = st.file_uploader(
        "Import scenario JSON",
        type=["json"],
        help="Load a previously exported AntennaTwin-Indoor scenario. The loaded values become the current input defaults.",
    )
    imported = {}
    if uploaded_scenario is not None:
        try:
            imported_doc = json.load(uploaded_scenario)
            imported = imported_doc.get("scenario", {})
            st.success("Scenario JSON loaded")
        except Exception as exc:
            st.error(f"Invalid scenario JSON: {exc}")

    L = st.number_input("Room length (m)", 5.0, 100.0, float(imported.get("room_length_m", 35.0)), 0.5)
    W = st.number_input("Room width (m)", 5.0, 100.0, float(imported.get("room_width_m", 24.0)), 0.5)
    H = st.number_input("Room height (m)", 2.0, 10.0, float(imported.get("room_height_m", 3.5)), 0.1)

    f = st.number_input("Frequency (GHz)", 0.1, 100.0, float(imported.get("frequency_ghz", 2.45)), 0.01)
    p = st.number_input("Tx power (dBm)", 0.0, 40.0, float(imported.get("tx_power_dbm", 12.0)), 1.0)
    g = st.number_input("Peak gain (dBi)", -10.0, 30.0, float(imported.get("peak_gain_dbi", 3.0)), 0.5)
    bw = st.number_input("3-dB beamwidth (deg)", 10.0, 180.0, float(imported.get("beamwidth_3db_deg", 90.0)), 5.0)
    fb = st.number_input("Front-to-back ratio (dB)", 0.0, 50.0, float(imported.get("front_to_back_db", 18.0)), 1.0)
    sens = st.number_input("Receiver sensitivity (dBm)", -120.0, -20.0, float(imported.get("receiver_sensitivity_dbm", -72.0)), 1.0)
    wl = st.number_input("Concrete wall loss (dB)", 0.0, 40.0, float(imported.get("wall_loss_db", 12.0)), 1.0)
    clr = st.number_input("Minimum wall clearance (m)", 0.0, 5.0, float(imported.get("minimum_wall_clearance_m", 0.5)), 0.1)

    st.subheader("Simulation resolution")
    grid = st.number_input(
        "Receiver grid resolution (m)",
        0.25,
        2.0,
        float(imported.get("receiver_grid_resolution_m", 0.50)),
        0.25,
        help="Spacing of receiver points used to calculate the RSSI and wall-loss maps.",
    )

    st.subheader("Optimization resolution")
    opt_spacing = st.number_input(
        "Position search spacing (m)",
        0.5,
        5.0,
        float(imported.get("optimization_position_spacing_m", 1.0)),
        0.5,
        help="Spacing between candidate transmitter positions during optimization.",
    )
    step = st.number_input(
        "Orientation step (deg)",
        5.0,
        90.0,
        float(imported.get("orientation_step_deg", 30.0)),
        5.0,
        help="Angular step used to search transmitter azimuth.",
    )

    st.subheader("Deployment strategy")
    deployment_mode = st.radio(
        "Placement mode",
        ["Automatic Optimization", "Manual Evaluation"],
        index=1 if imported.get("deployment_mode") == "Manual Evaluation" else 0,
        help="Automatic mode searches position and azimuth. Manual mode evaluates only the specified initial deployment.",
    )
    objective_labels = {
        "Coverage → Mean RSSI": "coverage_then_mean_rssi",
        "Mean RSSI": "mean_rssi",
        "Minimum RSSI": "minimum_rssi",
        "Coverage Margin": "coverage_margin",
    }
    objective_label = st.selectbox(
        "Optimization objective",
        list(objective_labels.keys()),
        index=list(objective_labels.values()).index(imported.get("optimization_objective", "coverage_then_mean_rssi")) if imported.get("optimization_objective", "coverage_then_mean_rssi") in objective_labels.values() else 0,
        disabled=(deployment_mode == "Manual Evaluation"),
        help="Coverage → Mean RSSI reproduces the validated default objective. Coverage Margin maximizes mean RSSI above receiver sensitivity.",
    )

    st.subheader("Antenna / receiver height")
    antenna_height = st.number_input(
        "Antenna height (m)", 0.5, max(0.5, float(H)), float(imported.get("antenna_height_m", min(2.5, max(0.5, float(H - 0.2))))), 0.1
    )
    receiver_height = st.number_input(
        "Receiver height (m)", 0.1, max(0.1, float(H)), float(imported.get("receiver_height_m", min(1.2, max(0.1, float(H - 0.2))))), 0.1
    )

    st.subheader("Initial deployment")
    initial_x = st.number_input(
        "Initial X position (m)",
        0.0,
        float(L),
        float(imported.get("initial_x_m", L / 2.0)),
        0.5,
        help="Baseline transmitter X coordinate before optimization.",
    )
    initial_y = st.number_input(
        "Initial Y position (m)",
        0.0,
        float(W),
        float(imported.get("initial_y_m", W / 2.0)),
        0.5,
        help="Baseline transmitter Y coordinate before optimization.",
    )
    initial_azimuth = st.number_input(
        "Initial azimuth (deg)",
        0.0,
        359.0,
        float(imported.get("initial_azimuth_deg", 0.0)),
        5.0,
        help="Baseline antenna boresight direction.",
    )

    st.subheader("Visualization")
    plot_size = st.selectbox(
        "Plot size",
        ["Compact", "Normal", "Large"],
        index=0,
        help="Controls the visualization footprint in the dashboard.",
    )

    run = st.button(
        "🚀 RUN OPTIMIZATION",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
if "R" not in st.session_state:
    st.session_state.R = None

if run:
    room = Room(L, W, H, grid)

    ant = Antenna(
        frequency_hz=f * 1e9,
        tx_power_dbm=p,
        gain_dbi=g,
        receiver_gain_dbi=0.0,
        height_m=float(antenna_height),
        min_distance_m=max(0.5, grid),
        azimuth_deg=float(initial_azimuth),
        beamwidth_3db_deg=bw,
        front_to_back_db=fb,
    )

    rx = Receiver(
        sensitivity_dbm=sens,
        height_m=float(receiver_height),
        gain_dbi=0.0,
    )

    # Demonstration indoor environment: two concrete partitions with finite openings.
    walls = [
        Wall(
            "Partition_A",
            L * 11.5 / 35.0,
            0.0,
            L * 11.5 / 35.0,
            W,
            "concrete",
            wl,
            [(W * 10.0 / 24.0, W * 14.0 / 24.0)],
        ),
        Wall(
            "Partition_B",
            L * 23.5 / 35.0,
            0.0,
            L * 23.5 / 35.0,
            W,
            "concrete",
            wl,
            [(W * 8.0 / 24.0, W * 12.0 / 24.0)],
        ),
    ]

    with st.spinner("Running deployment analysis..."):
        if deployment_mode == "Automatic Optimization":
            best, search_records = optimize_position(
                room,
                ant,
                rx,
                walls,
                spacing_m=opt_spacing,
                orientation_step_deg=step,
                wall_clearance_m=clr,
                objective=objective_labels[objective_label],
            )
        else:
            # Manual mode uses the specified initial deployment as the selected deployment.
            X0, Y0, rssi0, wall0, gain0 = coverage_map(
                room, ant, rx, initial_x, initial_y, walls
            )
            cov0 = float((rssi0 >= sens).mean() * 100.0)
            best = {
                "x_m": float(initial_x),
                "y_m": float(initial_y),
                "azimuth_deg": float(initial_azimuth),
                "coverage_percent": cov0,
                "dead_zone_percent": 100.0 - cov0,
                "min_rssi_dbm": float(rssi0.min()),
                "max_rssi_dbm": float(rssi0.max()),
                "mean_rssi_dbm": float(rssi0.mean()),
                "mean_wall_loss_db": float(wall0.mean()),
                "mean_antenna_gain_dbi": float(gain0.mean()),
                "mean_coverage_margin_db": float(rssi0.mean() - sens),
            }
            search_records = []

        optimized_ant = replace(
            ant,
            azimuth_deg=best["azimuth_deg"],
        )

        X, Y, rssi, wall, gain = coverage_map(
            room,
            optimized_ant,
            rx,
            best["x_m"],
            best["y_m"],
            walls,
        )

        baseline_ant = replace(
            ant,
            azimuth_deg=initial_azimuth,
        )

        BX, BY, brssi, bwall, bgain = coverage_map(
            room,
            baseline_ant,
            rx,
            initial_x,
            initial_y,
            walls,
        )

        base = {
            "x": float(initial_x),
            "y": float(initial_y),
            "azimuth": float(initial_azimuth),
            "coverage": float((brssi >= sens).mean() * 100.0),
            "dead": float((brssi < sens).mean() * 100.0),
            "min": float(brssi.min()),
            "max": float(brssi.max()),
            "mean": float(brssi.mean()),
            "wall": float(bwall.mean()),
        }

        st.session_state.R = {
            "room": room,
            "ant": ant,
            "rx": rx,
            "walls": walls,
            "best": best,
            "X": X,
            "Y": Y,
            "rssi": rssi,
            "wall": wall,
            "gain": gain,
            "BX": BX,
            "BY": BY,
            "brssi": brssi,
            "bwall": bwall,
            "bgain": bgain,
            "base": base,
            "sens": float(sens),
            "L": float(L),
            "W": float(W),
            "H": float(H),
            "frequency_ghz": float(f),
            "position_spacing": float(opt_spacing),
            "orientation_step": float(step),
            "objective": objective_labels[objective_label],
            "deployment_mode": deployment_mode,
            "search_records": search_records,
            "antenna_height_m": float(antenna_height),
            "receiver_height_m": float(receiver_height),
        }


R = st.session_state.R

if R is None:
    st.info("Set the scenario and initial deployment, then click RUN OPTIMIZATION.")
    st.markdown(
        "**Outputs:** compact RSSI map · wall-loss map · directional radiation pattern · "
        "3D RSSI surface · baseline/optimized comparison · downloadable JSON report."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Results references
# -----------------------------------------------------------------------------
b = R["best"]
base = R["base"]

# -----------------------------------------------------------------------------
# Derived performance metrics
# -----------------------------------------------------------------------------
selected_rssi = R["rssi"]
selected_wall = R["wall"]
selected_gain = R["gain"]
all_rssi = np.asarray(selected_rssi).ravel()
all_wall = np.asarray(selected_wall).ravel()
all_gain = np.asarray(selected_gain).ravel()
percentiles = {f"p{q}": float(np.percentile(all_rssi, q)) for q in [5, 25, 50, 75, 95]}
margin_thresholds = {
    f"coverage_plus_{m:g}db": float(np.mean(all_rssi >= (R["sens"] + m)) * 100.0)
    for m in [3, 6, 10]
}
wall_crossed_fraction = float(np.mean(all_wall > 1e-12) * 100.0)
positive_wall_points = int(np.sum(all_wall > 1e-12))
mean_clearance = float(minimum_distance_to_walls_m(b["x_m"], b["y_m"], R["walls"]))
search_evaluations = len(R.get("search_records", []))
search_feasible_positions = len({(round(x["x_m"], 9), round(x["y_m"], 9)) for x in R.get("search_records", [])})

# -----------------------------------------------------------------------------
# Results summary
# -----------------------------------------------------------------------------
coverage_delta = b["coverage_percent"] - base["coverage"]
dead_delta = b["dead_zone_percent"] - base["dead"]
min_delta = b["min_rssi_dbm"] - base["min"]
mean_delta = b["mean_rssi_dbm"] - base["mean"]
wall_delta = b["mean_wall_loss_db"] - base["wall"]

c = st.columns(4)
c[0].metric("Coverage", f'{b["coverage_percent"]:.2f}%', f"{coverage_delta:+.2f} pp")
c[1].metric("Dead zone", f'{b["dead_zone_percent"]:.2f}%', f"{dead_delta:+.2f} pp")
c[2].metric("Minimum RSSI", f'{b["min_rssi_dbm"]:.2f} dBm', f"{min_delta:+.2f} dB")
c[3].metric("Mean RSSI", f'{b["mean_rssi_dbm"]:.2f} dBm', f"{mean_delta:+.2f} dB")

st.success(
    f'Selected transmitter: ({b["x_m"]:.2f}, {b["y_m"]:.2f}) m · '
    f'Azimuth {b["azimuth_deg"]:.0f}°'
)

mc = st.columns(5)
mc[0].metric("RSSI P05", f"{percentiles['p5']:.2f} dBm")
mc[1].metric("RSSI Median", f"{percentiles['p50']:.2f} dBm")
mc[2].metric("RSSI P95", f"{percentiles['p95']:.2f} dBm")
mc[3].metric("Wall-crossed grid", f"{wall_crossed_fraction:.2f}%")
mc[4].metric("Tx-wall clearance", f"{mean_clearance:.2f} m")

# Initial -> optimized deployment summary.
left, arrow, right = st.columns([1.2, 0.35, 1.2])
with left:
    st.markdown("**Initial deployment**")
    st.write(
        f'Position: ({base["x"]:.2f}, {base["y"]:.2f}) m  '
        f'· Azimuth: {base["azimuth"]:.0f}°'
    )
with arrow:
    st.markdown("### →")
with right:
    st.markdown("**Optimized deployment**")
    st.write(
        f'Position: ({b["x_m"]:.2f}, {b["y_m"]:.2f}) m  '
        f'· Azimuth: {b["azimuth_deg"]:.0f}°'
    )


# -----------------------------------------------------------------------------
# Plot sizing helpers
# -----------------------------------------------------------------------------
plot_widths = {
    "Compact": 5.6,
    "Normal": 7.0,
    "Large": 8.6,
}
plot_width = plot_widths[plot_size]

# Center plots so they do not dominate the full browser width.
def centered_plot(fig):
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.pyplot(fig, use_container_width=True)


def close_fig(fig):
    plt.close(fig)


# -----------------------------------------------------------------------------
# Visualization tabs
# -----------------------------------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs(
    ["📍 RSSI", "🧱 Wall Loss", "📡 Radiation", "🧊 3D RSSI", "📈 Comparison"]
)

# Common side-by-side sizing.
if plot_size == "Compact":
    pair_figsize = (8.4, 3.8)
    three_d_figsize = (8.4, 4.2)
    single_figsize = (7.0, 4.4)
elif plot_size == "Normal":
    pair_figsize = (10.2, 4.5)
    three_d_figsize = (10.2, 5.0)
    single_figsize = (8.0, 5.0)
else:
    pair_figsize = (12.0, 5.2)
    three_d_figsize = (12.0, 5.8)
    single_figsize = (9.0, 5.8)


def centered_plot(fig):
    """Center a Matplotlib figure in the Streamlit content area."""
    _, center, _ = st.columns([0.55, 3.9, 0.55])
    with center:
        st.pyplot(fig, use_container_width=True)


def close_fig(fig):
    plt.close(fig)


with t1:
    # -------------------------------------------------------------------------
    # RSSI: initial vs optimized, identical color scale.
    # -------------------------------------------------------------------------
    vmin = float(min(R["brssi"].min(), R["rssi"].min()))
    vmax = float(max(R["brssi"].max(), R["rssi"].max()))

    fig, axes = plt.subplots(
        1, 2,
        figsize=pair_figsize,
        constrained_layout=True,
    )

    def draw_rssi_panel(ax, X, Y, data, title, x_pos, y_pos, marker, coverage, dead_zone):
        im = ax.pcolormesh(
            X, Y, data,
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )

        try:
            ax.contour(
                X, Y, data,
                levels=[R["sens"]],
                colors="black",
                linewidths=1.0,
            )
        except ValueError:
            pass

        for wall_obj in R["walls"]:
            ax.plot(
                [wall_obj.x1_m, wall_obj.x2_m],
                [wall_obj.y1_m, wall_obj.y2_m],
                "k-", lw=2.2,
            )

        ax.scatter(
            x_pos, y_pos,
            marker=marker,
            s=62 if marker == "o" else 78,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=6,
        )

        ax.set(
            xlabel="X (m)",
            ylabel="Y (m)",
            title=title,
        )
        ax.set_xlim(0, R["L"])
        ax.set_ylim(0, R["W"])
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.02, 0.03,
            f"Coverage: {coverage:.2f}%\nDead zone: {dead_zone:.2f}%",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                alpha=0.84,
                edgecolor="0.7",
            ),
        )
        return im

    draw_rssi_panel(
        axes[0], R["BX"], R["BY"], R["brssi"],
        "Initial Deployment",
        base["x"], base["y"], "o",
        base["coverage"], base["dead"],
    )

    im = draw_rssi_panel(
        axes[1], R["X"], R["Y"], R["rssi"],
        "Optimized Deployment",
        b["x_m"], b["y_m"], "^",
        b["coverage_percent"], b["dead_zone_percent"],
    )

    fig.suptitle("RSSI Coverage: Initial vs Optimized", fontsize=12)
    fig.colorbar(im, ax=axes, label="RSSI (dBm)", shrink=0.82, pad=0.025)
    centered_plot(fig)
    close_fig(fig)

    st.caption(
        "Same RSSI color scale and receiver-sensitivity contour are used in both panels. "
        "Circle = initial transmitter; triangle = optimized transmitter."
    )


with t2:
    # -------------------------------------------------------------------------
    # Wall loss: initial vs optimized, identical color scale.
    # -------------------------------------------------------------------------
    wall_vmin = float(min(R["bwall"].min(), R["wall"].min()))
    wall_vmax = float(max(R["bwall"].max(), R["wall"].max()))

    fig, axes = plt.subplots(
        1, 2,
        figsize=pair_figsize,
        constrained_layout=True,
    )

    def draw_wall_panel(ax, X, Y, data, title, x_pos, y_pos, marker):
        im = ax.pcolormesh(
            X, Y, data,
            shading="auto",
            cmap="magma",
            vmin=wall_vmin,
            vmax=wall_vmax,
        )

        for wall_obj in R["walls"]:
            ax.plot(
                [wall_obj.x1_m, wall_obj.x2_m],
                [wall_obj.y1_m, wall_obj.y2_m],
                "cyan", lw=2.2,
            )

        ax.scatter(
            x_pos, y_pos,
            marker=marker,
            s=62 if marker == "o" else 78,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.set(
            xlabel="X (m)",
            ylabel="Y (m)",
            title=title,
        )
        ax.set_xlim(0, R["L"])
        ax.set_ylim(0, R["W"])
        ax.set_aspect("equal", adjustable="box")
        return im

    draw_wall_panel(
        axes[0], R["BX"], R["BY"], R["bwall"],
        "Initial Deployment",
        base["x"], base["y"], "o",
    )

    im = draw_wall_panel(
        axes[1], R["X"], R["Y"], R["wall"],
        "Optimized Deployment",
        b["x_m"], b["y_m"], "^",
    )

    fig.suptitle("Wall Attenuation: Initial vs Optimized", fontsize=12)
    fig.colorbar(im, ax=axes, label="Wall loss (dB)", shrink=0.82, pad=0.025)
    centered_plot(fig)
    close_fig(fig)

    st.caption(
        f"Mean wall loss: {base['wall']:.2f} dB initial → "
        f"{b['mean_wall_loss_db']:.2f} dB optimized."
    )


with t3:
    # -------------------------------------------------------------------------
    # Radiation: initial azimuth vs optimized azimuth on the SAME polar axes.
    # -------------------------------------------------------------------------
    angles = np.arange(361)
    initial_ang, initial_gain = radiation_pattern(
        angles,
        R["ant"].gain_dbi,
        base["azimuth"],
        R["ant"].beamwidth_3db_deg,
        R["ant"].front_to_back_db,
    )
    optimized_ang, optimized_gain = radiation_pattern(
        angles,
        R["ant"].gain_dbi,
        b["azimuth_deg"],
        R["ant"].beamwidth_3db_deg,
        R["ant"].front_to_back_db,
    )

    fig, ax = plt.subplots(
        figsize=(4.8, 4.8),
        subplot_kw={"projection": "polar"},
    )
    ax.plot(
        np.deg2rad(initial_ang),
        initial_gain,
        linewidth=1.5,
        label=f"Initial ({base['azimuth']:.0f}°)",
    )
    ax.plot(
        np.deg2rad(optimized_ang),
        optimized_gain,
        linewidth=1.7,
        linestyle="--",
        label=f"Optimized ({b['azimuth_deg']:.0f}°)",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_title("Radiation Pattern: Initial vs Optimized", pad=14, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.10), fontsize=8)
    fig.tight_layout()
    centered_plot(fig)
    close_fig(fig)

    st.caption(
        "The antenna pattern shape is unchanged; optimization rotates the same directional pattern "
        f"from {base['azimuth']:.0f}° to {b['azimuth_deg']:.0f}°."
    )


with t4:
    # -------------------------------------------------------------------------
    # 3D RSSI: initial vs optimized, same z-axis range.
    # -------------------------------------------------------------------------
    zmin = float(min(R["brssi"].min(), R["rssi"].min()))
    zmax = float(max(R["brssi"].max(), R["rssi"].max()))

    fig = plt.figure(figsize=three_d_figsize, constrained_layout=True)
    axes = [
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    ]

    def draw_3d_panel(ax, X, Y, data, title, x_pos, y_pos):
        surf = ax.plot_surface(
            X, Y, data,
            cmap="turbo",
            linewidth=0,
            antialiased=True,
            vmin=zmin,
            vmax=zmax,
        )
        ax.scatter(
            [x_pos], [y_pos], [float(np.max(data))],
            marker="^",
            s=38,
            color="black",
            depthshade=False,
        )
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X (m)", labelpad=2)
        ax.set_ylabel("Y (m)", labelpad=2)
        ax.set_zlabel("RSSI (dBm)", labelpad=2)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=28, azim=-58)
        return surf

    draw_3d_panel(
        axes[0], R["BX"], R["BY"], R["brssi"],
        "Initial Deployment", base["x"], base["y"],
    )
    surf = draw_3d_panel(
        axes[1], R["X"], R["Y"], R["rssi"],
        "Optimized Deployment", b["x_m"], b["y_m"],
    )

    fig.suptitle("3D RSSI Surface: Initial vs Optimized", fontsize=12)
    fig.colorbar(surf, ax=axes, shrink=0.60, pad=0.04, label="RSSI (dBm)")
    centered_plot(fig)
    close_fig(fig)


with t5:
    # -------------------------------------------------------------------------
    # Comparison: side-by-side numerical table + compact improvement chart.
    # -------------------------------------------------------------------------
    comparison = pd.DataFrame(
        {
            "Metric": [
                "Coverage (%)",
                "Dead zone (%)",
                "Minimum RSSI (dBm)",
                "Mean RSSI (dBm)",
                "Mean wall loss (dB)",
            ],
            "Initial": [
                base["coverage"],
                base["dead"],
                base["min"],
                base["mean"],
                base["wall"],
            ],
            "Optimized": [
                b["coverage_percent"],
                b["dead_zone_percent"],
                b["min_rssi_dbm"],
                b["mean_rssi_dbm"],
                b["mean_wall_loss_db"],
            ],
            "Change": [
                coverage_delta,
                dead_delta,
                min_delta,
                mean_delta,
                wall_delta,
            ],
        }
    )

    st.markdown("**Initial vs optimized numerical comparison**")
    st.dataframe(
        comparison.style.format(
            {
                "Initial": "{:.2f}",
                "Optimized": "{:.2f}",
                "Change": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Coverage improvement", f"{coverage_delta:+.2f} pp")
        st.metric("Minimum RSSI improvement", f"{min_delta:+.2f} dB")
    with c2:
        st.metric("Mean RSSI improvement", f"{mean_delta:+.2f} dB")
        st.metric("Mean wall-loss change", f"{wall_delta:+.2f} dB")

    st.caption(
        f"Optimization mode: {R['deployment_mode']} · Objective: {R['objective']}. "
        "Wall loss enters the objective through the RSSI calculation."
    )

    st.markdown("**Improvement by physical metric group**")
    chart_cols = st.columns(3)
    chart_specs = [
        (chart_cols[0], "Coverage improvement", ["Coverage", "Dead zone"], [coverage_delta, dead_delta], "Percentage points"),
        (chart_cols[1], "RSSI improvement", ["Minimum RSSI", "Mean RSSI"], [min_delta, mean_delta], "dB"),
        (chart_cols[2], "Wall-loss change", ["Mean wall loss"], [wall_delta], "dB"),
    ]
    for col, title, labels, vals, ylabel in chart_specs:
        with col:
            fig, ax = plt.subplots(figsize=(3.2, 2.6))
            ax.bar(np.arange(len(labels)), vals, width=0.55)
            ax.axhline(0.0, linewidth=0.8)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            close_fig(fig)


    st.markdown("**Additional performance statistics**")
    stats_cols = st.columns(4)
    stats_cols[0].metric("Coverage +3 dB", f"{margin_thresholds['coverage_plus_3db']:.2f}%")
    stats_cols[1].metric("Coverage +6 dB", f"{margin_thresholds['coverage_plus_6db']:.2f}%")
    stats_cols[2].metric("Coverage +10 dB", f"{margin_thresholds['coverage_plus_10db']:.2f}%")
    stats_cols[3].metric("Mean antenna gain", f"{float(all_gain.mean()):.2f} dBi")

    st.markdown("**RSSI distribution (selected deployment)**")
    fig, ax = plt.subplots(figsize=(single_figsize[0], 3.0))
    xs = np.sort(all_rssi)
    cdf = np.arange(1, len(xs) + 1) / len(xs) * 100.0
    ax.plot(xs, cdf, linewidth=1.8)
    ax.axvline(R["sens"], linewidth=1.0, linestyle="--", label=f"Sensitivity {R['sens']:.1f} dBm")
    ax.set_xlabel("RSSI (dBm)")
    ax.set_ylabel("Cumulative percentage (%)")
    ax.set_title("RSSI CDF")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    centered_plot(fig)
    close_fig(fig)

    st.markdown("**Optimization search statistics**")
    if R["deployment_mode"] == "Automatic Optimization":
        st.write(
            f"Evaluated **{search_evaluations:,}** position-orientation combinations across "
            f"**{search_feasible_positions:,}** feasible transmitter positions."
        )
    else:
        st.write("Manual Evaluation mode: no optimization search was performed.")

# -----------------------------------------------------------------------------
# Model assumptions and limitations
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Model assumptions & limitations"):
    st.markdown(
        """
        **Propagation model:** free-space path loss with a minimum modeled distance, plus additive attenuation for internal wall crossings.

        **Antenna model:** analytic directional pattern defined by peak gain, 3-dB beamwidth and front-to-back ratio.

        **Geometry:** 2-D floor-plane receiver grid with user-defined transmitter/receiver heights; internal partitions are represented by line segments with finite openings.

        **Optimization:** grid search over transmitter position and azimuth subject to minimum wall-clearance constraints.

        **Interpretation:** results are deterministic, model-based simulations. They are intended for deployment studies and software verification; they are **not** experimental measurements, full-wave electromagnetic simulations, or ray-tracing validation.
        """
    )

# -----------------------------------------------------------------------------
# Downloadable report
# -----------------------------------------------------------------------------
report = {
    "software_version": VERSION,
    "scenario": {
        "room_length_m": R["L"],
        "room_width_m": R["W"],
        "room_height_m": R["H"],
        "frequency_ghz": R["frequency_ghz"],
        "tx_power_dbm": float(R["ant"].tx_power_dbm),
        "peak_gain_dbi": float(R["ant"].gain_dbi),
        "beamwidth_3db_deg": float(R["ant"].beamwidth_3db_deg),
        "front_to_back_db": float(R["ant"].front_to_back_db),
        "receiver_sensitivity_dbm": float(R["sens"]),
        "wall_loss_db": float(R["walls"][0].loss_db),
        "minimum_wall_clearance_m": float(clr),
        "receiver_grid_resolution_m": float(R["room"].grid_resolution_m),
        "optimization_position_spacing_m": float(R["position_spacing"]),
        "orientation_step_deg": float(R["orientation_step"]),
        "antenna_height_m": float(R["antenna_height_m"]),
        "receiver_height_m": float(R["receiver_height_m"]),
        "deployment_mode": R["deployment_mode"],
        "optimization_objective": R["objective"],
    },
    "initial_deployment": base,
    "optimized_deployment": {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in b.items()
    },
    "improvement": {
        "coverage_pp": float(coverage_delta),
        "dead_zone_pp": float(dead_delta),
        "minimum_rssi_db": float(min_delta),
        "mean_rssi_db": float(mean_delta),
        "mean_wall_loss_db": float(wall_delta),
    },
    "objective": {
        "selected": R["objective"],
        "wall_loss_role": "included in RSSI calculation and therefore affects coverage and RSSI statistics",
    },
    "additional_metrics": {
        "rssi_percentiles_dbm": percentiles,
        "coverage_margin_percent": margin_thresholds,
        "wall_crossed_grid_fraction_percent": wall_crossed_fraction,
        "wall_crossed_grid_points": positive_wall_points,
        "tx_wall_clearance_m": mean_clearance,
        "search_evaluations": search_evaluations,
        "feasible_positions": search_feasible_positions,
    },
}

scenario_values = {
    "room_length_m": R["L"], "room_width_m": R["W"], "room_height_m": R["H"],
    "frequency_ghz": R["frequency_ghz"], "tx_power_dbm": float(R["ant"].tx_power_dbm),
    "peak_gain_dbi": float(R["ant"].gain_dbi), "beamwidth_3db_deg": float(R["ant"].beamwidth_3db_deg),
    "front_to_back_db": float(R["ant"].front_to_back_db), "receiver_sensitivity_dbm": R["sens"],
    "wall_loss_db": float(R["walls"][0].loss_db), "minimum_wall_clearance_m": float(clr),
    "receiver_grid_resolution_m": float(R["room"].grid_resolution_m),
    "optimization_position_spacing_m": float(R["position_spacing"]),
    "orientation_step_deg": float(R["orientation_step"]),
    "antenna_height_m": float(R["antenna_height_m"]), "receiver_height_m": float(R["receiver_height_m"]),
    "deployment_mode": R["deployment_mode"], "optimization_objective": R["objective"],
    "initial_x_m": base["x"], "initial_y_m": base["y"], "initial_azimuth_deg": base["azimuth"],
}
scenario_json = json.dumps(_scenario_payload(scenario_values), indent=2)

initial_csv = _array_csv(R["BX"], R["BY"], R["brssi"], R["bwall"], R["bgain"], names=["rssi_dbm", "wall_loss_db", "antenna_gain_dbi"])
optimized_csv = _array_csv(R["X"], R["Y"], R["rssi"], R["wall"], R["gain"], names=["rssi_dbm", "wall_loss_db", "antenna_gain_dbi"])
rad_csv = pd.DataFrame({
    "angle_deg": initial_ang,
    "initial_gain_dbi": initial_gain,
    "optimized_gain_dbi": optimized_gain,
}).to_csv(index=False)

repro_zip = _reproducibility_zip(
    report, scenario_json,
    {
        "initial_map.csv": initial_csv,
        "optimized_map.csv": optimized_csv,
        "radiation_pattern.csv": rad_csv,
    },
)

st.markdown("### Export & reproducibility")
exp1, exp2, exp3 = st.columns(3)
with exp1:
    st.download_button(
        "⬇️ Simulation JSON", scenario_json,
        file_name="antennatwin_scenario.json", mime="application/json", use_container_width=True,
    )
with exp2:
    st.download_button(
        "⬇️ Numerical report", json.dumps(report, indent=2),
        file_name="antennatwin_simulation_report.json", mime="application/json", use_container_width=True,
    )
with exp3:
    st.download_button(
        "📦 Reproducibility package", repro_zip,
        file_name="antennatwin_reproducibility_package.zip", mime="application/zip", use_container_width=True,
    )

st.caption("The reproducibility ZIP contains the scenario, numerical report, receiver-grid CSVs and radiation-pattern CSV. The package records model-based results and assumptions used for the run.")
