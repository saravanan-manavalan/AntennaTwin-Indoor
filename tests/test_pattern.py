import numpy as np

from antennatwin.pattern import directional_gain_dbi, directional_gain_db

def test_peak_gain_at_boresight():
    g = float(directional_gain_dbi(0, 5, 90, 20))
    assert abs(g - 5.0) < 1e-9

def test_gain_lower_behind_antenna():
    front = float(directional_gain_dbi(0, 5, 90, 20))
    back = float(directional_gain_dbi(180, 5, 90, 20))
    assert front > back

def test_half_beamwidth_is_three_db_down():
    g0 = float(directional_gain_db(0, 5, 0, 90, 20))
    gh = float(directional_gain_db(45, 5, 0, 90, 20))
    expected_drop_db = 10.0 * np.log10(2.0)
    assert abs((g0 - gh) - expected_drop_db) < 1e-9
