import math
from antennatwin.propagation import free_space_path_loss_db

def test_fspl_matches_analytical_equation():
    d = 10.0
    f = 2.45e9
    expected = 20.0 * math.log10(4.0 * math.pi * d * f / 299792458.0)
    actual = float(free_space_path_loss_db(d, f))
    assert abs(actual - expected) < 1e-10
