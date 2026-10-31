"""Evaluation suite and sensitivity sweeps for refill.

Provides functions to compute:
1. Trade-off curves: Fill rate vs. Inventory value across service levels
2. Regularity noise sweep: cycle_noise_sd from 1.0 to 10.0 days
3. Identification share sweep: id_share from 0.0 to 1.0
4. Lead time sweep: lead_time 0, 1, 2, 5 days
5. Missed-refill evaluation (The Pivot): detection delay and false alarms
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .generator import ScenarioConfig, generate
from .simulate import run, WARMUP


def evaluate_tradeoff(
    cfg: ScenarioConfig | None = None,
    service_levels: Sequence[float] = (0.75, 0.85, 0.90, 0.95, 0.98),
    lead_time: int = 2,
    warmup: int = WARMUP,
    policies: Sequence[str] = ("Refill clock", "Croston (SBA)", "Moving average", "Fixed min/max"),
) -> pd.DataFrame:
    """Sweep service levels across policies and collect fill rate vs. stock value."""
    if cfg is None:
        cfg = ScenarioConfig()
    sc = generate(cfg)
    rows = []
    for pol in policies:
        for s in service_levels:
            res = run(sc, pol, lead_time=lead_time, service=s, warmup=warmup)
            m = res.metrics
            rows.append({
                "policy": pol,
                "service": s,
                "regular_fill_rate": m["regular_fill_rate"],
                "unit_fill_rate": m["unit_fill_rate"],
                "avg_stock_value": m["avg_stock_value"],
                "regulars_turned_away": m["regulars_turned_away"],
                "regular_visits": m["regular_visits"],
            })
    return pd.DataFrame(rows)


def sweep_regularity(
    noise_levels: Sequence[float] = (1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0),
    service: float = 0.95,
    lead_time: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep cycle_noise_sd to find the crossover point where Refill Clock loses its edge."""
    rows = []
    for noise in noise_levels:
        cfg = ScenarioConfig(seed=seed, cycle_noise_sd=noise)
        sc = generate(cfg)
        for pol in ("Refill clock", "Croston (SBA)", "Moving average"):
            res = run(sc, pol, lead_time=lead_time, service=service)
            m = res.metrics
            rows.append({
                "cycle_noise_sd": noise,
                "policy": pol,
                "regular_fill_rate": m["regular_fill_rate"],
                "avg_stock_value": m["avg_stock_value"],
                "regulars_turned_away": m["regulars_turned_away"],
            })
    return pd.DataFrame(rows)


def sweep_id_share(
    id_shares: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    service: float = 0.95,
    lead_time: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep id_share from 0% to 100%. At 0%, Refill Clock should equal baseline."""
    rows = []
    for share in id_shares:
        cfg = ScenarioConfig(seed=seed, id_share=share)
        sc = generate(cfg)
        for pol in ("Refill clock", "Croston (SBA)"):
            res = run(sc, pol, lead_time=lead_time, service=service)
            m = res.metrics
            rows.append({
                "id_share": share,
                "policy": pol,
                "regular_fill_rate": m["regular_fill_rate"],
                "avg_stock_value": m["avg_stock_value"],
                "regulars_turned_away": m["regulars_turned_away"],
            })
    return pd.DataFrame(rows)


def sweep_lead_time(
    lead_times: Sequence[int] = (0, 1, 2, 5),
    service: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep lead time: 0 (same day), 1, 2, 5 days to demonstrate objection (a)."""
    rows = []
    cfg = ScenarioConfig(seed=seed)
    sc = generate(cfg)
    for lt in lead_times:
        for pol in ("Refill clock", "Croston (SBA)", "Moving average"):
            res = run(sc, pol, lead_time=lt, service=service)
            m = res.metrics
            rows.append({
                "lead_time": lt,
                "policy": pol,
                "regular_fill_rate": m["regular_fill_rate"],
                "avg_stock_value": m["avg_stock_value"],
                "regulars_turned_away": m["regulars_turned_away"],
            })
    return pd.DataFrame(rows)


def evaluate_missed_refills(
    cfg: ScenarioConfig | None = None,
    lead_time: int = 2,
    service: float = 0.95,
    fixed_rule_days: int = 45,
) -> dict:
    """Evaluate missed-refill (churn / lapse) detection for the Pivot.

    Compares:
    1. Refill Clock alert: flagged when overdue > 2 * sd (personalized)
    2. Fixed rule alert: flagged when days since last purchase > fixed_rule_days
    Metrics:
    - True churn count
    - Detection delay (days between true churn and alert)
    - False alarms (active patients alerted prematurely)
    """
    if cfg is None:
        cfg = ScenarioConfig()
    sc = generate(cfg)
    res = run(sc, "Refill clock", lead_time=lead_time, service=service)
    rc_policy = res.policy

    # Evaluate churners who churned between warmup (day 90) and day 330
    churners = sc.patients[
        (sc.patients.churn_day >= WARMUP)
        & (sc.patients.churn_day <= cfg.n_days - 35)
        & (sc.patients.identified)
    ].copy()

    clock_delays = []
    fixed_delays = []

    for _, pat in churners.iterrows():
        pid = pat.patient_id
        c_day = pat.churn_day
        if pid not in rc_policy.hist or len(rc_policy.hist[pid]) < 2:
            continue
        cycle = rc_policy.cycle[pid]
        sd = rc_policy.sd[pid]
        last_d = rc_policy.last[pid]
        # True expected next refill day
        expected = last_d + cycle

        # Refill clock flag day: when overdue > 2 * sd
        # overdue = d - expected > 2*sd => d = expected + 2*sd
        rc_flag_day = expected + 2.0 * sd
        rc_delay = max(0.0, rc_flag_day - c_day)
        clock_delays.append(rc_delay)

        # Fixed rule flag day: d = last_d + fixed_rule_days
        fixed_flag_day = last_d + float(fixed_rule_days)
        fixed_delay = max(0.0, fixed_flag_day - c_day)
        fixed_delays.append(fixed_delay)

    # False alarms: count active patients flagged when they were just slightly late
    # An active patient buys again. We check if an alert fired before their actual purchase.
    active_pat_count = 0
    rc_false_alarms = 0
    fixed_false_alarms = 0

    for pid in rc_policy.hist:
        h = rc_policy.hist[pid]
        if len(h) < 3:
            continue
        active_pat_count += 1
        cycle = rc_policy.cycle[pid]
        sd = rc_policy.sd[pid]
        for i in range(1, len(h)):
            prev_d, prev_q = h[i - 1]
            curr_d, curr_q = h[i]
            # Days between purchases
            gap = curr_d - prev_d
            expected_gap = cycle * (prev_q / rc_policy.qtyp[pid])
            if gap > expected_gap + 2.0 * sd:
                rc_false_alarms += 1
            if gap > fixed_rule_days:
                fixed_false_alarms += 1

    return {
        "n_evaluated_churners": len(clock_delays),
        "refill_clock_avg_delay_days": float(np.mean(clock_delays)) if clock_delays else 0.0,
        "fixed_rule_avg_delay_days": float(np.mean(fixed_delays)) if fixed_delays else 0.0,
        "active_patient_intervals": active_pat_count,
        "refill_clock_false_alarms": rc_false_alarms,
        "fixed_rule_false_alarms": fixed_false_alarms,
    }


def plot_tradeoff_curve(df_tradeoff: pd.DataFrame, output_path: str = "tradeoff_curve.png") -> None:
    """Plot Fill Rate vs. Average Stock Value trade-off curve."""
    plt.figure(figsize=(8, 5.5), dpi=150)
    colors = {
        "Refill clock": "#1f77b4",
        "Croston (SBA)": "#ff7f0e",
        "Moving average": "#2ca02c",
        "Fixed min/max": "#d62728",
    }
    markers = {
        "Refill clock": "o",
        "Croston (SBA)": "s",
        "Moving average": "^",
        "Fixed min/max": "D",
    }
    for pol in df_tradeoff.policy.unique():
        sub = df_tradeoff[df_tradeoff.policy == pol].sort_values("avg_stock_value")
        plt.plot(
            sub.avg_stock_value,
            sub.regular_fill_rate * 100.0,
            label=pol,
            color=colors.get(pol, "grey"),
            marker=markers.get(pol, "x"),
            linewidth=2.2,
            markersize=7,
        )

    plt.xlabel("Average Inventory Value Held (₹)", fontsize=11, fontweight="bold")
    plt.ylabel("Regular Patient Fill Rate (%)", fontsize=11, fontweight="bold")
    plt.title("Fill Rate vs. Inventory Trade-off Curve (Higher & Left is Superior)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sweeps(
    df_reg: pd.DataFrame,
    df_id: pd.DataFrame,
    output_path: str = "sensitivity_sweeps.png",
) -> None:
    """Plot Regularity Noise and ID Share sweeps side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Regularity noise sweep
    for pol in df_reg.policy.unique():
        sub = df_reg[df_reg.policy == pol].sort_values("cycle_noise_sd")
        ax1.plot(
            sub.cycle_noise_sd,
            sub.regular_fill_rate * 100.0,
            label=pol,
            marker="o",
            linewidth=2,
        )
    ax1.set_xlabel("Refill Cycle Noise Std Dev (days)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Regular Patient Fill Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Regularity Sweep: Advantage vs. Patient Jitter", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)

    # 2. ID share sweep
    for pol in df_id.policy.unique():
        sub = df_id[df_id.policy == pol].sort_values("id_share")
        ax2.plot(
            sub.id_share * 100.0,
            sub.regular_fill_rate * 100.0,
            label=pol,
            marker="s",
            linewidth=2,
        )
    ax2.set_xlabel("% of Bills with Customer ID", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Regular Patient Fill Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title("ID Coverage Sweep: Baseline Equivalence at 0%", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_full_evaluation(output_dir: str = "artifacts") -> dict:
    """Run full benchmarking pipeline and save plots and tables."""
    os.makedirs(output_dir, exist_ok=True)

    print("Running Tradeoff Curve sweep...")
    df_tradeoff = evaluate_tradeoff()
    tradeoff_csv = os.path.join(output_dir, "tradeoff_curve.csv")
    tradeoff_png = os.path.join(output_dir, "tradeoff_curve.png")
    df_tradeoff.to_csv(tradeoff_csv, index=False)
    plot_tradeoff_curve(df_tradeoff, tradeoff_png)

    print("Running Sensitivity: Regularity Noise sweep...")
    df_reg = sweep_regularity()
    df_reg.to_csv(os.path.join(output_dir, "sweep_regularity.csv"), index=False)

    print("Running Sensitivity: ID Share sweep...")
    df_id = sweep_id_share()
    df_id.to_csv(os.path.join(output_dir, "sweep_id_share.csv"), index=False)

    sweeps_png = os.path.join(output_dir, "sensitivity_sweeps.png")
    plot_sweeps(df_reg, df_id, sweeps_png)

    print("Running Sensitivity: Lead Time sweep...")
    df_lt = sweep_lead_time()
    df_lt.to_csv(os.path.join(output_dir, "sweep_lead_time.csv"), index=False)

    print("Evaluating Missed Refills (Pivot)...")
    missed_res = evaluate_missed_refills()
    pd.DataFrame([missed_res]).to_csv(os.path.join(output_dir, "missed_refills_eval.csv"), index=False)

    print(f"Evaluation complete! All artifacts saved to {output_dir}/")
    return {
        "tradeoff_csv": tradeoff_csv,
        "tradeoff_png": tradeoff_png,
        "sweeps_png": sweeps_png,
        "missed_metrics": missed_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate refill forecasting against baselines")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to save evaluation artifacts")
    args = parser.parse_args()
    run_full_evaluation(args.output_dir)
