from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Room:
    length_m: float
    width_m: float
    height_m: float = 3.5
    grid_resolution_m: float = 0.25

@dataclass
class Antenna:
    frequency_hz: float
    tx_power_dbm: float
    gain_dbi: float
    receiver_gain_dbi: float = 0.0
    height_m: float = 2.5
    min_distance_m: float = 0.5
    azimuth_deg: float = 0.0
    beamwidth_3db_deg: float = 90.0
    front_to_back_db: float = 20.0

@dataclass
class Receiver:
    sensitivity_dbm: float = -67.0
    height_m: float = 1.2
    gain_dbi: float = 0.0

@dataclass
class Wall:
    name: str
    x1_m: float
    y1_m: float
    x2_m: float
    y2_m: float
    material: str = "concrete"
    loss_db: float = 8.0
    openings: List[Tuple[float, float]] = field(default_factory=list)
