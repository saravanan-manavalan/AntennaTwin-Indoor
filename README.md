## Quick start

```cmd
python -m pip install -r requirements.txt
python run.py
python -m pytest -q
python validation\run_validation.py
```

Results are written to `results/`.

## Model

The propagation baseline is free-space path loss with 3-D Tx/Rx separation, receiver sensitivity, directional antenna gain, and additive wall attenuation. Wall openings are specified as distances in metres measured along each wall.

This is an engineering propagation model, not a full-wave electromagnetic or ray-tracing solver.


