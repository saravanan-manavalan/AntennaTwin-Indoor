import numpy as np
from antennatwin.models import Receiver
from antennatwin.coverage import coverage_statistics

def test_coverage_statistics():
    receiver = Receiver(sensitivity_dbm=-67, height_m=1.2)
    rssi = np.array([[-70.0, -60.0], [-50.0, -80.0]])
    s = coverage_statistics(rssi, receiver)
    assert s["coverage_percent"] == 50.0
    assert s["dead_zone_percent"] == 50.0
    assert s["min_rssi_dbm"] == -80.0
    assert s["max_rssi_dbm"] == -50.0
