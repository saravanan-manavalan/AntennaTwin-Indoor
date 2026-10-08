import json
from pathlib import Path
from antennatwin import Room, Antenna, Receiver, Wall
from antennatwin.optimizer import optimize_position
from antennatwin.geometry import minimum_distance_to_walls_m

def test_optimizer_returns_valid_solution():
    cfg = json.loads((Path(__file__).parents[1] / "config" / "default.json").read_text())
    room = Room(**cfg["room"])
    antenna = Antenna(**cfg["antenna"])
    receiver = Receiver(**cfg["receiver"])
    walls = [Wall(**w) for w in cfg["walls"]]
    best, _ = optimize_position(room, antenna, receiver, walls, spacing=2.0, orientation_step_deg=90)
    assert 0 <= best["x_m"] <= room.length_m
    assert 0 <= best["y_m"] <= room.width_m
    assert 0 <= best["azimuth_deg"] < 360
    assert 0 <= best["coverage_percent"] <= 100
