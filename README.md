# Frank-WT-MF-PINN

Code for multi-fidelity physics-informed inverse design of LCE actuators.

## Files

- `src/frank_wt_mf_pinn/schema.py`: 14-variable design domain and thresholds.
- `src/frank_wt_mf_pinn/mechanics.py`: Frank and Warner--Terentjev mechanics.
- `src/frank_wt_mf_pinn/residual.py`: residual PINN and replay buffer.
- `src/frank_wt_mf_pinn/reliability.py`: calibration and reliability gate.
- `src/frank_wt_mf_pinn/acquisition.py`: multi-fidelity acquisition.
- `src/frank_wt_mf_pinn/records.py`: evaluation records and fidelity repository.
- `src/frank_wt_mf_pinn/engine.py`: main optimization loop.
- `src/frank_wt_mf_pinn/objectives.py`: objective projection.
- `src/frank_wt_mf_pinn/fixtures.py`: packaged validation backend.
- `scripts/run_structural_fixture.py`: validation run.

## Run

```powershell
python -m pip install -e ".[dev]"
python .\scripts\run_structural_fixture.py
```
