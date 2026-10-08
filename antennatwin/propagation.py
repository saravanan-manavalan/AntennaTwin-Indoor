import numpy as np

C = 299_792_458.0

def free_space_path_loss_db(distance_m, frequency_hz):
    d = np.maximum(np.asarray(distance_m, dtype=float), 1e-12)
    f = float(frequency_hz)
    return 20.0 * np.log10(4.0 * np.pi * d * f / C)
