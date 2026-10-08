import numpy as np

from .propagation import free_space_path_loss_db
from .pattern import directional_gain_db
from .coverage import _wall_loss_grid
from .geometry import minimum_distance_to_walls_m


def _make_receiver_grid(room):
    xs = np.arange(0.0, room.length_m + 1e-9, room.grid_resolution_m)
    ys = np.arange(0.0, room.width_m + 1e-9, room.grid_resolution_m)
    return np.meshgrid(xs, ys, indexing="xy")


def _evaluate_position(x, y, room, antenna, receiver, walls, orientation_angles):
    X, Y = _make_receiver_grid(room)
    dx, dy = X - x, Y - y
    dz = receiver.height_m - antenna.height_m
    distance = np.maximum(
        np.sqrt(dx**2 + dy**2 + dz**2),
        antenna.min_distance_m
    )
    wall_loss = _wall_loss_grid(X, Y, x, y, walls)
    fspl = free_space_path_loss_db(distance, antenna.frequency_hz)
    path_angle = np.degrees(np.arctan2(dy, dx))
    base_power = (
        antenna.tx_power_dbm
        + antenna.receiver_gain_dbi
        - fspl
        - wall_loss
    )

    records = []
    best = None
    for orientation in orientation_angles:
        gain = directional_gain_db(
            path_angle,
            antenna.gain_dbi,
            orientation,
            antenna.beamwidth_3db_deg,
            antenna.front_to_back_db,
        )
        rssi = base_power + gain
        covered = rssi >= receiver.sensitivity_dbm
        cov = float(np.mean(covered) * 100.0)
        result = {
            "x_m": float(x),
            "y_m": float(y),
            "azimuth_deg": float(orientation),
            "coverage_percent": cov,
            "dead_zone_percent": 100.0 - cov,
            "min_rssi_dbm": float(np.min(rssi)),
            "max_rssi_dbm": float(np.max(rssi)),
            "mean_rssi_dbm": float(np.mean(rssi)),
            "mean_wall_loss_db": float(np.mean(wall_loss)),
            "mean_antenna_gain_dbi": float(np.mean(gain)),
        }
        records.append(result)
        if best is None or (
            result["coverage_percent"],
            result["mean_rssi_dbm"]
        ) > (
            best["coverage_percent"],
            best["mean_rssi_dbm"]
        ):
            best = result
    return best, records


def optimize_position(
    room,
    antenna,
    receiver,
    walls,
    spacing_m=1.0,
    orientation_step_deg=30.0,
    spacing=None,
    margin=None,
    wall_clearance_m=0.5,
    minimum_wall_clearance_m=None,
    objective="coverage_then_mean_rssi",
):
    """
    Optimize antenna position and azimuth with a physical wall-clearance
    constraint.

    wall_clearance_m:
        Minimum horizontal distance from a candidate antenna location to
        every physical internal wall segment. Default: 0.5 m.

    minimum_wall_clearance_m:
        Explicit alias for wall_clearance_m.
    """
    if spacing is not None:
        spacing_m = spacing
    if minimum_wall_clearance_m is not None:
        wall_clearance_m = minimum_wall_clearance_m

    wall_clearance_m = max(float(wall_clearance_m), 0.0)

    x0 = float(margin) if margin is not None else 0.5
    y0 = float(margin) if margin is not None else 0.5

    xs = np.arange(x0, room.length_m - x0 + 1e-9, spacing_m)
    ys = np.arange(y0, room.width_m - y0 + 1e-9, spacing_m)
    orientations = np.arange(0.0, 360.0, orientation_step_deg)

    objective = str(objective).strip().lower()
    aliases = {
        "default": "coverage_then_mean_rssi",
        "coverage": "coverage_then_mean_rssi",
        "mean_rssi": "mean_rssi",
        "minimum_rssi": "minimum_rssi",
        "min_rssi": "minimum_rssi",
        "coverage_margin": "coverage_margin",
    }
    objective = aliases.get(objective, objective)
    valid_objectives = {
        "coverage_then_mean_rssi", "mean_rssi",
        "minimum_rssi", "coverage_margin"
    }
    if objective not in valid_objectives:
        raise ValueError(
            "objective must be one of: " + ", ".join(sorted(valid_objectives))
        )

    records = []
    best = None

    def score(result):
        if objective == "coverage_then_mean_rssi":
            return (result["coverage_percent"], result["mean_rssi_dbm"])
        if objective == "mean_rssi":
            return (result["mean_rssi_dbm"], result["coverage_percent"])
        if objective == "minimum_rssi":
            return (result["min_rssi_dbm"], result["coverage_percent"])
        # Mean RSSI above the receiver sensitivity is a simple coverage-margin metric.
        result["mean_coverage_margin_db"] = (
            result["mean_rssi_dbm"] - receiver.sensitivity_dbm
        )
        return (result["mean_coverage_margin_db"], result["coverage_percent"])

    for x in xs:
        for y in ys:
            clearance = minimum_distance_to_walls_m(
                float(x), float(y), walls
            )
            if clearance < wall_clearance_m:
                continue

            candidate, recs = _evaluate_position(
                x, y, room, antenna, receiver, walls, orientations
            )
            for result in recs:
                result["mean_coverage_margin_db"] = (
                    result["mean_rssi_dbm"] - receiver.sensitivity_dbm
                )
            records.extend(recs)

            # _evaluate_position returns the best orientation by the legacy
            # coverage/mean-RSSI rule. Re-select the orientation for the
            # requested dashboard objective.
            if objective != "coverage_then_mean_rssi":
                candidate = max(recs, key=score)

            if best is None or score(candidate) > score(best):
                best = candidate

    if best is None:
        raise ValueError(
            "No feasible antenna positions satisfy the requested "
            f"wall clearance of {wall_clearance_m:.3f} m."
        )

    return best, records
