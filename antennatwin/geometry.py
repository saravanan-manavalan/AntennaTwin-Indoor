import math
from typing import Iterable, Tuple


def _point_segment_distance(px: float, py: float,
                            x1: float, y1: float,
                            x2: float, y2: float) -> float:
    """Euclidean distance from a point to a finite line segment."""
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    denom = vx * vx + vy * vy
    if denom <= 0.0:
        return math.hypot(px - x1, py - y1)
    t = (wx * vx + wy * vy) / denom
    t = max(0.0, min(1.0, t))
    qx = x1 + t * vx
    qy = y1 + t * vy
    return math.hypot(px - qx, py - qy)


def _distance_along_wall(px: float, py: float,
                         x1: float, y1: float,
                         x2: float, y2: float) -> float:
    """Distance from wall start to the orthogonal projection of a point."""
    vx = x2 - x1
    vy = y2 - y1
    length = math.hypot(vx, vy)
    if length <= 0.0:
        return 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / (length * length)
    t = max(0.0, min(1.0, t))
    return t * length


def point_to_wall_distance_m(x: float, y: float, wall) -> float:
    """
    Minimum horizontal distance from a point to a physical wall segment.

    A declared opening is treated as a non-wall region for the clearance
    calculation.
    """
    distance_along = _distance_along_wall(
        x, y, wall.x1_m, wall.y1_m, wall.x2_m, wall.y2_m
    )
    vx = wall.x2_m - wall.x1_m
    vy = wall.y2_m - wall.y1_m
    length = math.hypot(vx, vy)

    if length > 0.0:
        for start, end in wall.openings:
            lo, hi = sorted((float(start), float(end)))
            if lo <= distance_along <= hi:
                return math.inf

    return _point_segment_distance(
        x, y, wall.x1_m, wall.y1_m, wall.x2_m, wall.y2_m
    )


def minimum_distance_to_walls_m(x: float, y: float, walls: Iterable) -> float:
    """Return the minimum distance from a point to all physical wall segments."""
    distances = [point_to_wall_distance_m(x, y, wall) for wall in walls]
    return min(distances) if distances else math.inf


def _crossing_parameters(x1, y1, x2, y2, x3, y3, x4, y4):
    r_x, r_y = x2 - x1, y2 - y1
    s_x, s_y = x4 - x3, y4 - y3
    denom = r_x * s_y - r_y * s_x
    if abs(denom) < 1e-12:
        return None
    qpx, qpy = x3 - x1, y3 - y1
    t = (qpx * s_y - qpy * s_x) / denom
    u = (qpx * r_y - qpy * r_x) / denom
    if 0.0 < t < 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


def segment_crosses_wall(tx_x, tx_y, rx_x, rx_y, wall) -> bool:
    result = _crossing_parameters(
        tx_x, tx_y, rx_x, rx_y,
        wall.x1_m, wall.y1_m, wall.x2_m, wall.y2_m
    )
    if result is None:
        return False

    _, u = result
    wall_length = math.hypot(
        wall.x2_m - wall.x1_m,
        wall.y2_m - wall.y1_m
    )
    crossing_distance = u * wall_length

    for start, end in wall.openings:
        lo, hi = sorted((float(start), float(end)))
        if lo <= crossing_distance <= hi:
            return False

    return True


def total_wall_loss_db(tx_x, tx_y, rx_x, rx_y, walls) -> Tuple[float, list]:
    total = 0.0
    names = []
    for wall in walls:
        if segment_crosses_wall(tx_x, tx_y, rx_x, rx_y, wall):
            total += float(wall.loss_db)
            names.append(wall.name)
    return total, names


def wall_crossed_by_path(tx_x, tx_y, rx_x, rx_y, wall) -> bool:
    """Backward-compatible alias for segment_crosses_wall."""
    return segment_crosses_wall(tx_x, tx_y, rx_x, rx_y, wall)
