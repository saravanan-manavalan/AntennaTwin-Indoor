import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from antennatwin.models import Wall
from antennatwin.geometry import total_wall_loss_db
from antennatwin.propagation import free_space_path_loss_db
from antennatwin.pattern import directional_gain_db

out = ROOT / "results"
out.mkdir(exist_ok=True)

rows = []

d, f = 10.0, 2.45e9
expected = 20.0 * math.log10(4.0 * math.pi * d * f / 299792458.0)
actual = float(free_space_path_loss_db(d, f))
rows.append(("FSPL analytical", expected, actual, abs(expected-actual)))

wall = Wall("Concrete", 5, 0, 5, 10, loss_db=12)
loss, names = total_wall_loss_db(0, 5, 10, 5, [wall])
rows.append(("Single wall loss", 12.0, loss, abs(12.0-loss)))

opening_wall = Wall("Concrete", 5, 0, 5, 10, loss_db=12, openings=[(4,6)])
open_loss, _ = total_wall_loss_db(0, 5, 10, 5, [opening_wall])
rows.append(("Opening bypass", 0.0, open_loss, abs(open_loss)))

g0 = float(directional_gain_db(0, 5, 0, 90, 20))
g45 = float(directional_gain_db(45, 5, 0, 90, 20))
rows.append(("3-dB beamwidth", 3.0, g0-g45, abs(3.0-(g0-g45))))

report = {
    "release": "AntennaTwin-Indoor V0.3.2",
    "tests": [
        {"name": n, "expected": e, "actual": a, "absolute_error": err}
        for n,e,a,err in rows
    ],
    "all_pass": all(err < 1e-9 for _,_,_,err in rows)
}
(out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("=== AntennaTwin-Indoor V0.3.2 Validation ===")
for n,e,a,err in rows:
    print(f"{n:24s}: expected={e:.6f}, actual={a:.6f}, error={err:.3e}")
print(f"Overall validation       : {'PASS' if report['all_pass'] else 'FAIL'}")
print(f"Report                   : {out / 'validation_report.json'}")
