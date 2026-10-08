import json
from pathlib import Path
from dataclasses import replace

from antennatwin import Room, Antenna, Receiver, Wall
from antennatwin.optimizer import optimize_position
from antennatwin.coverage import coverage_map, save_coverage_plot
from antennatwin.pattern import radiation_pattern


BASE = Path(__file__).parent
cfg = json.loads(
    (BASE / "config" / "default.json").read_text(encoding="utf-8")
)

room = Room(**cfg["room"])
antenna = Antenna(**cfg["antenna"])
receiver = Receiver(**cfg["receiver"])
walls = [Wall(**w) for w in cfg["walls"]]

minimum_wall_clearance_m = cfg["optimization"].get(
    "minimum_wall_clearance_m", 0.5
)

best, records = optimize_position(
    room,
    antenna,
    receiver,
    walls,
    spacing_m=cfg["optimization"]["candidate_spacing_m"],
    orientation_step_deg=cfg["optimization"]["orientation_step_deg"],
    wall_clearance_m=minimum_wall_clearance_m,
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

out = BASE / "results"
out.mkdir(exist_ok=True)

save_coverage_plot(
    X,
    Y,
    rssi,
    best["x_m"],
    best["y_m"],
    receiver.sensitivity_dbm,
    walls,
    out / "coverage_map.png",
)

angles, pattern_gain = radiation_pattern(
    range(361),
    antenna.gain_dbi,
    best["azimuth_deg"],
    antenna.beamwidth_3db_deg,
    antenna.front_to_back_db,
)

result = {
    **best,
    "beamwidth_3db_deg": antenna.beamwidth_3db_deg,
    "front_to_back_db": antenna.front_to_back_db,
    "minimum_wall_clearance_m": minimum_wall_clearance_m,
}

(out / "optimization_result.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

(out / "radiation_pattern.json").write_text(
    json.dumps(
        {
            "angles_deg": angles.tolist(),
            "gain_dbi": pattern_gain.tolist(),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\n=== AntennaTwin-Indoor V0.3.3 ===")
print(
    f"Room                     : "
    f"{room.length_m:.1f} x {room.width_m:.1f} x {room.height_m:.1f} m"
)
print(f"Frequency                : {antenna.frequency_hz/1e9:.3f} GHz")
print(f"Tx power                 : {antenna.tx_power_dbm:.1f} dBm")
print(f"Peak antenna gain        : {antenna.gain_dbi:.1f} dBi")
print(f"3-dB beamwidth           : {antenna.beamwidth_3db_deg:.1f} deg")
print(f"Front-to-back ratio      : {antenna.front_to_back_db:.1f} dB")
print(
    f"Minimum wall clearance   : "
    f"{minimum_wall_clearance_m:.2f} m"
)
print(
    f"Optimized position       : "
    f"({best['x_m']:.2f}, {best['y_m']:.2f}) m"
)
print(f"Optimized azimuth        : {best['azimuth_deg']:.1f} deg")
print(f"Coverage                 : {best['coverage_percent']:.2f}%")
print(f"Dead zone                : {best['dead_zone_percent']:.2f}%")
print(f"Minimum RSSI             : {best['min_rssi_dbm']:.2f} dBm")
print(f"Maximum RSSI             : {best['max_rssi_dbm']:.2f} dBm")
print(f"Mean RSSI                : {best['mean_rssi_dbm']:.2f} dBm")
print(f"Mean wall loss           : {best['mean_wall_loss_db']:.2f} dB")
print(f"Mean antenna gain        : {best['mean_antenna_gain_dbi']:.2f} dBi")
print(f"Coverage map             : {out / 'coverage_map.png'}")
print(f"Radiation pattern data   : {out / 'radiation_pattern.json'}")
print("\nOptimization completed successfully.")
