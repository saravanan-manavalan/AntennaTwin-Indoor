import json
from pathlib import Path
from dataclasses import replace

from antennatwin import Room, Antenna, Receiver, Wall
from antennatwin.optimizer import optimize_position
from antennatwin.coverage import coverage_map, save_coverage_plot
from antennatwin.pattern import radiation_pattern


BASE = Path(__file__).parent
CFG_PATH = BASE / "config" / "challenging_demo.json"
cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

room = Room(**cfg["room"])
antenna = Antenna(**cfg["antenna"])
receiver = Receiver(**cfg["receiver"])
walls = [Wall(**w) for w in cfg["walls"]]

opt = cfg["optimization"]
clearance = opt.get("minimum_wall_clearance_m", 0.5)

# Baseline: center of the room, boresight along 0 degrees.
baseline_x = room.length_m / 2.0
baseline_y = room.width_m / 2.0

baseline_antenna = replace(antenna, azimuth_deg=0.0)

BX, BY, baseline_rssi, baseline_wall, baseline_gain = coverage_map(
    room,
    baseline_antenna,
    receiver,
    baseline_x,
    baseline_y,
    walls,
)

baseline_coverage = float((baseline_rssi >= receiver.sensitivity_dbm).mean() * 100.0)

# Optimized position and azimuth.
best, records = optimize_position(
    room,
    antenna,
    receiver,
    walls,
    spacing_m=opt["candidate_spacing_m"],
    orientation_step_deg=opt["orientation_step_deg"],
    wall_clearance_m=clearance,
)

best_antenna = replace(
    antenna,
    azimuth_deg=best["azimuth_deg"]
)

X, Y, rssi, wall_loss, gain = coverage_map(
    room,
    best_antenna,
    receiver,
    best["x_m"],
    best["y_m"],
    walls,
)

out = BASE / "results" / "challenging_demo"
out.mkdir(parents=True, exist_ok=True)

save_coverage_plot(
    X,
    Y,
    rssi,
    best["x_m"],
    best["y_m"],
    receiver.sensitivity_dbm,
    walls,
    out / "optimized_coverage_map.png",
)

save_coverage_plot(
    BX,
    BY,
    baseline_rssi,
    baseline_x,
    baseline_y,
    receiver.sensitivity_dbm,
    walls,
    out / "baseline_coverage_map.png",
)

angles, pattern_gain = radiation_pattern(
    range(361),
    antenna.gain_dbi,
    best["azimuth_deg"],
    antenna.beamwidth_3db_deg,
    antenna.front_to_back_db,
)

result = {
    "scenario": cfg["name"],
    "baseline": {
        "x_m": baseline_x,
        "y_m": baseline_y,
        "azimuth_deg": 0.0,
        "coverage_percent": baseline_coverage,
        "dead_zone_percent": 100.0 - baseline_coverage,
        "min_rssi_dbm": float(baseline_rssi.min()),
        "max_rssi_dbm": float(baseline_rssi.max()),
        "mean_rssi_dbm": float(baseline_rssi.mean()),
        "mean_wall_loss_db": float(baseline_wall.mean()),
        "mean_antenna_gain_dbi": float(baseline_gain.mean()),
    },
    "optimized": best,
    "improvement": {
        "coverage_gain_percentage_points": best["coverage_percent"] - baseline_coverage,
        "mean_rssi_gain_db": best["mean_rssi_dbm"] - float(baseline_rssi.mean()),
        "minimum_rssi_gain_db": best["min_rssi_dbm"] - float(baseline_rssi.min()),
    },
    "minimum_wall_clearance_m": clearance,
}

(out / "comparison.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

(out / "radiation_pattern.json").write_text(
    json.dumps(
        {"angles_deg": angles.tolist(), "gain_dbi": pattern_gain.tolist()},
        indent=2
    ),
    encoding="utf-8"
)

print("\n=== AntennaTwin-Indoor V0.4.4.1: Challenging Demonstration ===")
print(f"Scenario                 : {cfg['name']}")
print(f"Room                     : {room.length_m:.1f} x {room.width_m:.1f} x {room.height_m:.1f} m")
print(f"Frequency                : {antenna.frequency_hz/1e9:.3f} GHz")
print(f"Tx power                 : {antenna.tx_power_dbm:.1f} dBm")
print(f"Receiver sensitivity     : {receiver.sensitivity_dbm:.1f} dBm")
print(f"Internal partitions      : {len(walls)}")
print("\n--- Baseline (room centre, 0 deg) ---")
print(f"Position                 : ({baseline_x:.2f}, {baseline_y:.2f}) m")
print(f"Coverage                 : {baseline_coverage:.2f}%")
print(f"Dead zone                : {100.0-baseline_coverage:.2f}%")
print(f"Minimum RSSI             : {baseline_rssi.min():.2f} dBm")
print(f"Mean RSSI                : {baseline_rssi.mean():.2f} dBm")
print(f"Mean wall loss           : {baseline_wall.mean():.2f} dB")
print("\n--- Optimized ---")
print(f"Position                 : ({best['x_m']:.2f}, {best['y_m']:.2f}) m")
print(f"Azimuth                  : {best['azimuth_deg']:.1f} deg")
print(f"Coverage                 : {best['coverage_percent']:.2f}%")
print(f"Dead zone                : {best['dead_zone_percent']:.2f}%")
print(f"Minimum RSSI             : {best['min_rssi_dbm']:.2f} dBm")
print(f"Mean RSSI                : {best['mean_rssi_dbm']:.2f} dBm")
print(f"Mean wall loss           : {best['mean_wall_loss_db']:.2f} dB")
print(f"Mean antenna gain        : {best['mean_antenna_gain_dbi']:.2f} dBi")
print("\n--- Improvement ---")
print(f"Coverage improvement     : {best['coverage_percent'] - baseline_coverage:+.2f} percentage points")
print(f"Mean RSSI improvement    : {best['mean_rssi_dbm'] - baseline_rssi.mean():+.2f} dB")
print(f"Minimum RSSI improvement : {best['min_rssi_dbm'] - baseline_rssi.min():+.2f} dB")
print(f"\nResults directory        : {out}")
