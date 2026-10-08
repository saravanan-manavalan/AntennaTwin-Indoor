# AntennaTwin-Indoor V0.4.4.1.1.1 Dashboard

Run from the project root:

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

## V0.4.4.1.1 additions

- Automatic Optimization or Manual Evaluation.
- Optimization objective selection:
  - Coverage → Mean RSSI (validated default)
  - Mean RSSI
  - Minimum RSSI
  - Coverage Margin
- Antenna and receiver height controls.
- Initial versus optimized side-by-side RSSI, wall-loss, radiation-pattern, 3D RSSI, and comparison views.
- RSSI P05/P25/P50/P75/P95 statistics.
- Coverage at sensitivity +3, +6, and +10 dB.
- Wall-crossed receiver-grid fraction.
- Transmitter-to-wall clearance.
- Optimization search statistics.
- Downloadable JSON report containing reproducibility metadata and performance metrics.

The propagation model remains the validated analytical/model-based indoor model from V0.3.x. Dashboard results should be described as simulation results unless independent measurement or ray-tracing validation is supplied.
