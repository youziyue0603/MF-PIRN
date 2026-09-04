from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> int:
    from frank_wt_mf_pinn.engine import AlgorithmConfig, FrankWTMFPINNOptimizer
    from frank_wt_mf_pinn.fixtures import (
        DeterministicFixtureEvaluator,
        FixtureObjectiveProjector,
        FixtureReliabilityEstimator,
        fitted_fixture_calibrator,
    )
    from frank_wt_mf_pinn.residual import AdjacentResidualPINN
    from frank_wt_mf_pinn.schema import DesignDomain

    domain = DesignDomain()
    response_scales = np.asarray([1.0, 8.0, 1.0, 2.0, 1.0, 20.0, 3.0, 1.0])
    config = AlgorithmConfig(
        seed=20260827,
        initial_counts=(198, 198, 198, 198),
        candidate_pool_size=4,
        response_scales=response_scales,
        objective_scales=np.asarray([1.0, 50.0, 1.0]),
        reference_point=np.asarray([1.0, 0.0, 1.2]),
        energy_scale=4.0,
        high_fidelity_budget=200,
        replay_capacity=24,
        structural_fixture=True,
    )
    residual = AdjacentResidualPINN(
        response_scales=response_scales,
        energy_scale=4.0,
        hidden=(8,),
        ensemble_size=1,
        epochs=1,
        seed=config.seed,
    )
    optimizer = FrankWTMFPINNOptimizer(
        config,
        domain,
        DeterministicFixtureEvaluator(domain),
        residual,
        FixtureReliabilityEstimator(),
        FixtureObjectiveProjector(),
        calibrator=fitted_fixture_calibrator(),
    )
    result = optimizer.run(np.asarray([1.0, 1.0, 1.0]), np.asarray([20.0, 140.0]))
    checks = {
        "exactly_200_hf_calls": result.repository.high_fidelity_count == 200,
        "at_least_one_sequential_iteration": len(result.iterations) > 0,
        "archive_is_hf_only_and_feasible": all(
            record.fidelity == 3 and record.feasible for record in result.archive
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "structural validation workflow",
        "checks": checks,
        "hf_calls": result.repository.high_fidelity_count,
        "iterations": len(result.iterations),
        "archive_size": len(result.archive),
        "all_archive_records_are_feasible_F3": all(
            record.fidelity == 3 and record.feasible for record in result.archive
        ),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "structural_fixture_run.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
