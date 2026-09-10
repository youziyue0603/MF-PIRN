# FWT-MF-PIRN

Code for the Frank--Warner--Terentjev multi-fidelity physics-informed residual
network (FWT-MF-PIRN) for inverse design of LCE actuators.

## Files

- `src/frank_wt_mf_pirn/schema.py`: 14-variable design domain and thresholds.
- `src/frank_wt_mf_pirn/mechanics.py`: Frank and Warner--Terentjev mechanics.
- `src/frank_wt_mf_pirn/residual.py`: physics-informed residual model and replay buffer.
- `src/frank_wt_mf_pirn/reliability.py`: calibration and reliability gate.
- `src/frank_wt_mf_pirn/acquisition.py`: multi-fidelity acquisition.
- `src/frank_wt_mf_pirn/records.py`: evaluation records and fidelity repository.
- `src/frank_wt_mf_pirn/engine.py`: main optimization loop.
- `src/frank_wt_mf_pirn/objectives.py`: objective projection.
- `src/frank_wt_mf_pirn/fixtures.py`: packaged validation backend.
- `scripts/run_structural_fixture.py`: validation run.
- `plotting/`: fixed figure data and final plotting code.

## Run

```powershell
python -m pip install -e ".[dev]"
python .\scripts\run_structural_fixture.py
```

To reproduce the quantitative figures:

```powershell
python -m pip install -r .\plotting\requirements.txt
python .\plotting\scripts\plot_all.py
```
