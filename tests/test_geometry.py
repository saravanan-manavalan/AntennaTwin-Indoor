from antennatwin.models import Wall
from antennatwin.geometry import wall_crossed_by_path, total_wall_loss_db

def test_wall_crossing_applies_loss():
    wall = Wall("W", 5, 0, 5, 10, loss_db=12)
    assert wall_crossed_by_path(0, 5, 10, 5, wall)
    loss, names = total_wall_loss_db(0, 5, 10, 5, [wall])
    assert loss == 12
    assert names == ["W"]

def test_wall_opening_removes_loss():
    wall = Wall("W", 5, 0, 5, 10, loss_db=12, openings=[(4, 6)])
    assert not wall_crossed_by_path(0, 5, 10, 5, wall)
    loss, names = total_wall_loss_db(0, 5, 10, 5, [wall])
    assert loss == 0
    assert names == []
