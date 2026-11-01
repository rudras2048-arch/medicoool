"""Day-by-day inventory replay of one synthetic shop under one reorder policy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecasting import POLICIES, Policy, RefillClockPolicy
from .generator import Scenario

WARMUP = 90  # days excluded from metrics while policies build history


def make_policy(name: str, sc: Scenario, lead_time: int, service: float) -> Policy:
    cls = POLICIES[name]
    args = (sc.n_meds, sc.config.n_days, lead_time, service)
    if cls is RefillClockPolicy:
        return cls(*args, n_ids=len(sc.patients))
    return cls(*args)


def initial_stock(sc: Scenario, lead_time: int) -> np.ndarray:
    """Same opening stock for every policy: about (lead time + 7) days of early demand."""
    ev = sc.events[sc.events.day < 60]
    daily = np.bincount(ev.med_id, ev.qty, sc.n_meds) / 60.0
    return np.ceil(daily * (lead_time + 7))


@dataclass
class SimResult:
    metrics: dict
    daily: pd.DataFrame          # day, regulars_turned_away, walkins_turned_away, stock_value
    on_hand: np.ndarray          # on-hand at the *start* of the stop day (after deliveries)
    pipeline: np.ndarray         # pending deliveries indexed by arrival day
    policy: Policy
    lost: pd.DataFrame           # every turned-away request
    stop_day: int | None


def simulate(sc: Scenario, policy: Policy, delivery_lead_time_days: int, warmup: int = WARMUP,
             stop_day: int | None = None) -> SimResult:
    n, M, L = sc.config.n_days, sc.n_meds, delivery_lead_time_days
    price = sc.meds.price.to_numpy()
    by_day = sc.events_by_day()
    on_hand = initial_stock(sc, L)
    pipeline = np.zeros((n + L + 2, M))
    reg_away = np.zeros(n, int)
    walk_away = np.zeros(n, int)
    stock_val = np.zeros(n)
    lost_rows = []
    served_units = demand_units = 0.0
    reg_visits = reg_served = 0
    stopped = None

    for d in range(n):
        on_hand += pipeline[d]
        if stop_day is not None and d == stop_day:
            stopped = d
            break
        ip = on_hand + pipeline[d + 1:d + L + 1].sum(0)
        q = np.maximum(0.0, np.round(policy.order_qty(d, ip)))
        if L == 0:
            on_hand += q
        else:
            pipeline[d + L] += q

        meds, qtys, vids, pids, isreg = by_day[d]
        total = np.zeros(M)
        resid = np.zeros(M)
        purchases = []
        for m, qty, vid, pid, reg in zip(meds, qtys, vids, pids, isreg):
            ok = on_hand[m] >= qty
            if d >= warmup:
                demand_units += qty
                if reg:
                    reg_visits += 1
            if ok:
                on_hand[m] -= qty
                total[m] += qty
                if vid >= 0:
                    purchases.append((int(vid), int(m), int(qty)))
                else:
                    resid[m] += qty
                if d >= warmup:
                    served_units += qty
                    reg_served += int(reg)
            else:
                if reg:
                    reg_away[d] += 1
                else:
                    walk_away[d] += 1
                lost_rows.append((d, int(m), int(qty), int(pid), bool(reg)))
        policy.observe(d, total, resid, purchases)
        stock_val[d] = float(on_hand @ price)

    end = stopped if stopped is not None else n
    days = np.arange(end)
    daily = pd.DataFrame({
        "day": days, "regulars_turned_away": reg_away[:end],
        "walkins_turned_away": walk_away[:end], "stock_value": stock_val[:end],
    })
    lost = pd.DataFrame(lost_rows, columns=["day", "med_id", "qty", "patient_id", "is_regular"])
    post = daily[daily.day >= warmup]
    metrics = {
        "policy": policy.name,
        "service": policy.service,
        "regular_fill_rate": reg_served / reg_visits if reg_visits else float("nan"),
        "unit_fill_rate": served_units / demand_units if demand_units else float("nan"),
        "avg_stock_value": float(post.stock_value.mean()) if len(post) else float("nan"),
        "regulars_turned_away": int(post.regulars_turned_away.sum()),
        "regular_visits": reg_visits,
    }
    return SimResult(metrics, daily, on_hand.copy(), pipeline.copy(), policy, lost, stopped)


def run(sc: Scenario, policy_name: str, lead_time: int, service: float, **kw) -> SimResult:
    return simulate(sc, make_policy(policy_name, sc, lead_time, service), lead_time, **kw)
