from antennatwin.geometry import minimum_distance_to_walls_m
from antennatwin.models import Wall


def test_wall_clearance_distance():
    wall = Wall("Partition", 5.0, 0.0, 5.0, 10.0, loss_db=12.0)

    assert abs(
        minimum_distance_to_walls_m(4.5, 5.0, [wall]) - 0.5
    ) < 1e-12

    assert abs(
        minimum_distance_to_walls_m(5.0, 5.0, [wall])
    ) < 1e-12


def test_opening_is_not_treated_as_physical_wall_for_clearance():
    wall = Wall(
        "Partition",
        5.0,
        0.0,
        5.0,
        10.0,
        loss_db=12.0,
        openings=[(4.0, 6.0)],
    )

    assert minimum_distance_to_walls_m(5.0, 5.0, [wall]) == float("inf")
