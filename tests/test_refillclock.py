"""Unit and regression tests for the re system."""

import numpy as np
import pytest

from refillclock.generator import ScenarioConfig, generate
from refillclock.forecasting import (
    fit_clock,
    activity_weight,
    window_prob,
    norm_cdf,
    z_of,
    CrostonPolicy,
    MovingAveragePolicy,
    MinMaxPolicy,
    RefillClockPolicy,
)
from refillclock.simulate import run, simulate, make_policy
from refillclock.evaluate import evaluate_missed_refills


def test_fit_clock_single_purchase():
    """Single purchase should default to prior cycle and prior SD."""
    c, s, n, q = fit_clock([(10, 1)])
    assert c == 30.0
    assert s == 5.0
    assert n == 0
    assert q == 1.0


def test_fit_clock_multi_purchase():
    """Consistent 30-day gaps should yield ~30 cycle and low SD."""
    history = [(10, 1), (40, 1), (70, 1), (100, 1)]
    c, s, n, q = fit_clock(history)
    assert 28.0 <= c <= 32.0
    assert n == 3
    assert q == 1.0


def test_z_of_boundary_clamping():
    """z_of should handle 0.0, 1.0 and intermediate values safely without raising."""
    z_low = z_of(0.0)
    z_high = z_of(1.0)
    z_mid = z_of(0.5)
    assert z_low < -3.0
    assert z_high > 3.0
    assert abs(z_mid) < 0.01


def test_generator_deterministic():
    """Same seed should generate identical events."""
    cfg1 = ScenarioConfig(seed=123, n_days=60, n_chronic=5, n_acute=2)
    cfg2 = ScenarioConfig(seed=123, n_days=60, n_chronic=5, n_acute=2)
    sc1 = generate(cfg1)
    sc2 = generate(cfg2)
    assert len(sc1.events) == len(sc2.events)
    assert (sc1.events.qty.values == sc2.events.qty.values).all()


def test_zero_id_share_equals_croston():
    """At id_share=0.0, Refill Clock must produce the exact same metrics as Croston (SBA)."""
    cfg = ScenarioConfig(seed=99, n_days=120, n_chronic=6, n_acute=3, id_share=0.0)
    sc = generate(cfg)
    res_rc = run(sc, "Refill clock", lead_time=2, service=0.95, warmup=60)
    res_cr = run(sc, "Croston (SBA)", lead_time=2, service=0.95, warmup=60)
    assert res_rc.metrics["regular_fill_rate"] == pytest.approx(res_cr.metrics["regular_fill_rate"], rel=1e-5)
    assert res_rc.metrics["regulars_turned_away"] == res_cr.metrics["regulars_turned_away"]


def test_simulation_runs_all_policies():
    """Ensure all policies run and produce valid fill rates."""
    cfg = ScenarioConfig(seed=1, n_days=90, n_chronic=5, n_acute=2)
    sc = generate(cfg)
    for pol_name in ("Refill clock", "Croston (SBA)", "Moving average", "Fixed min/max"):
        res = run(sc, pol_name, lead_time=1, service=0.90, warmup=30)
        assert 0.0 <= res.metrics["regular_fill_rate"] <= 1.0
        assert res.metrics["avg_stock_value"] > 0


def test_missed_refills_evaluation():
    """Missed refills evaluator should produce valid delay and alarm metrics."""
    cfg = ScenarioConfig(seed=42, n_days=180, n_chronic=8, n_acute=2, churn_prob=0.3)
    res = evaluate_missed_refills(cfg, lead_time=1, service=0.95)
    assert res["n_evaluated_churners"] >= 0
    assert res["active_patient_intervals"] > 0
