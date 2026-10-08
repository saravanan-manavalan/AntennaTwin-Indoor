import math
import numpy as np
import matplotlib.pyplot as plt
from .propagation import free_space_path_loss_db
from .pattern import directional_gain_db

def _wall_loss_grid(X, Y, x, y, walls):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    total_loss = np.zeros_like(X, dtype=float)
    rx = X - x
    ry = Y - y

    for wall in walls:
        x1, y1 = float(wall.x1_m), float(wall.y1_m)
        x2, y2 = float(wall.x2_m), float(wall.y2_m)
        sx, sy = x2 - x1, y2 - y1
        wall_length = math.hypot(sx, sy)
        if wall_length < 1e-12:
            continue

        qx, qy = x1 - x, y1 - y
        denom = rx * sy - ry * sx
        valid = np.abs(denom) > 1e-12

        t = np.zeros_like(X, dtype=float)
        u = np.zeros_like(X, dtype=float)
        numerator_t = qx * sy - qy * sx
        numerator_u = qx * ry - qy * rx
        np.divide(numerator_t, denom, out=t, where=valid)
        np.divide(numerator_u, denom, out=u, where=valid)

        intersects = valid & (t > 0.0) & (t < 1.0) & (u >= 0.0) & (u <= 1.0)
        if not np.any(intersects):
            continue

        distance_along_wall = u * wall_length
        blocked = intersects.copy()
        for opening in wall.openings:
            start, end = sorted((float(opening[0]), float(opening[1])))
            inside = intersects & (distance_along_wall >= start - 1e-9) & (distance_along_wall <= end + 1e-9)
            blocked &= ~inside

        total_loss += np.where(blocked, float(wall.loss_db), 0.0)

    return total_loss

def coverage_map(room, antenna, receiver, antenna_x_m, antenna_y_m, walls):
    xs = np.arange(0.0, room.length_m + 1e-9, room.grid_resolution_m)
    ys = np.arange(0.0, room.width_m + 1e-9, room.grid_resolution_m)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    dx, dy = X - antenna_x_m, Y - antenna_y_m
    dz = receiver.height_m - antenna.height_m
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    distance = np.maximum(distance, antenna.min_distance_m)

    wall_loss = _wall_loss_grid(X, Y, antenna_x_m, antenna_y_m, walls)
    fspl = free_space_path_loss_db(distance, antenna.frequency_hz)
    path_angle = np.degrees(np.arctan2(dy, dx))
    gain = directional_gain_db(path_angle, antenna.gain_dbi, antenna.azimuth_deg, antenna.beamwidth_3db_deg, antenna.front_to_back_db)

    rssi = antenna.tx_power_dbm + gain + receiver.gain_dbi - fspl - wall_loss
    return X, Y, rssi, wall_loss, gain

def coverage_statistics(rssi, receiver):
    covered = np.asarray(rssi) >= receiver.sensitivity_dbm
    coverage_percent = float(np.mean(covered) * 100.0)
    return {
        "coverage_percent": coverage_percent,
        "dead_zone_percent": 100.0 - coverage_percent,
        "min_rssi_dbm": float(np.min(rssi)),
        "max_rssi_dbm": float(np.max(rssi)),
        "mean_rssi_dbm": float(np.mean(rssi)),
    }

def save_coverage_plot(X, Y, rssi, antenna_x, antenna_y, sensitivity_dbm, walls, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    contour = ax.contourf(X, Y, rssi, levels=30)
    fig.colorbar(contour, ax=ax, label="RSSI (dBm)")
    ax.scatter([antenna_x], [antenna_y], marker="^", s=100, label="Antenna")
    for wall in walls:
        ax.plot([wall.x1_m, wall.x2_m], [wall.y1_m, wall.y2_m], linewidth=3)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Indoor RSSI Coverage Map")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
