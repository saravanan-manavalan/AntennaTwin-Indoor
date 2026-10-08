import numpy as np


def angular_difference_deg(angle_deg, reference_deg):
    """Smallest signed angular difference in degrees, in [-180, 180)."""
    difference = (
        np.asarray(angle_deg, dtype=float) - float(reference_deg) + 180.0
    ) % 360.0 - 180.0
    if np.ndim(angle_deg) == 0:
        return float(difference)
    return difference


def angular_offset_deg(angle_deg, reference_deg):
    """Smallest absolute angular offset in degrees, in [0, 180]."""
    return np.abs(angular_difference_deg(angle_deg, reference_deg))


def directional_gain_db(angle_deg, peak_gain_dbi, azimuth_deg, beamwidth_3db_deg, front_to_back_db):
    """Calculate directional antenna gain using a normalized Gaussian-like pattern."""
    angle = np.asarray(angle_deg, dtype=float)
    theta = angular_offset_deg(angle, azimuth_deg)
    half_bw = float(beamwidth_3db_deg) / 2.0
    if half_bw <= 0:
        raise ValueError("beamwidth_3db_deg must be greater than zero.")
    main_linear = np.exp(-np.log(2.0) * (theta / half_bw) ** 2)
    floor_linear = 10.0 ** (-float(front_to_back_db) / 10.0)
    pattern_linear = np.maximum(main_linear, floor_linear)
    gain_db = float(peak_gain_dbi) + 10.0 * np.log10(pattern_linear)
    if np.ndim(angle_deg) == 0:
        return float(gain_db)
    return gain_db


def directional_gain_dbi(angle_deg, peak_gain_dbi, beamwidth_3db_deg, front_to_back_db, azimuth_deg=0.0):
    """Backward-compatible public API used by earlier AntennaTwin releases."""
    return directional_gain_db(
        angle_deg,
        peak_gain_dbi,
        azimuth_deg,
        beamwidth_3db_deg,
        front_to_back_db,
    )


def radiation_pattern(angles_deg, peak_gain_dbi, azimuth_deg, beamwidth_3db_deg, front_to_back_db):
    """Generate radiation-pattern data for supplied angular coordinates."""
    angles = np.asarray(angles_deg, dtype=float)
    gains = directional_gain_db(angles, peak_gain_dbi, azimuth_deg, beamwidth_3db_deg, front_to_back_db)
    return angles, gains


def pattern_curve(angles_deg, peak_gain_dbi, azimuth_deg, beamwidth_3db_deg, front_to_back_db):
    """Backward-compatible alias for radiation_pattern."""
    return radiation_pattern(angles_deg, peak_gain_dbi, azimuth_deg, beamwidth_3db_deg, front_to_back_db)
