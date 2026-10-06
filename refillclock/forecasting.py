"""Reorder policies: three baselines and the refill-clock method.

Every policy is an order-up-to policy reviewed daily:
    order = max(0, S_d - inventory_position)
where S_d is a quantile of demand over the protection period (lead time + 1 day).
Policies only ever see *served* sales (lost demand is invisible, as in a real shop).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

PRIOR_CYCLE = 30.0   # days; population prior for a monthly refill, updated from data
PRIOR_SD = 5.0       # days
K0 = 2.0             # pseudo-observations for shrinking per-patient estimates to the prior
MIN_SD = 1.5

_erf = np.vectorize(math.erf, otypes=[float])


def norm_cdf(x) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def z_of(service: float) -> float:
    p = min(max(float(service), 0.001), 0.999)
    return NormalDist().inv_cdf(p)


# --------------------------------------------------------------------------- clocks
def fit_clock(history, prior_cycle=PRIOR_CYCLE, prior_sd=PRIOR_SD):
    """Fit one patient's refill clock from [(day, qty), ...] at this shop.

    Gaps are normalised by quantity (a 2-strip purchase should last twice as long),
    the centre is a median (robust to a refill bought elsewhere -> one double gap),
    and both centre and spread are shrunk towards the population prior.
    Returns (cycle, sd, n_gaps, typical_qty) where cycle = days per typical purchase.
    """
    days = np.array([h[0] for h in history], dtype=float)
    qtys = np.array([h[1] for h in history], dtype=float)
    qtyp = float(np.median(qtys))
    if len(days) < 2:
        return prior_cycle, prior_sd, 0, qtyp
    gaps = np.diff(days) * qtyp / qtys[:-1]
    ng = len(gaps)
    med = float(np.median(gaps))
    cycle = (ng * med + K0 * prior_cycle) / (ng + K0)
    raw_sd = 1.4826 * float(np.median(np.abs(gaps - med))) if ng >= 3 else prior_sd
    sd = math.sqrt((ng * raw_sd ** 2 + K0 * prior_sd ** 2) / (ng + K0))
    return cycle, max(MIN_SD, sd), ng, qtyp


def activity_weight(overdue, sd):
    """Belief that a late patient is still a regular here. 1 until 2 sd late, then decays;
    0 once clearly lapsed (> max(15 days, 4 sd) late)."""
    over = overdue - 2.0 * sd
    w = np.where(over <= 0, 1.0, np.exp(-np.clip(over, 0, None) / 10.0))
    return np.where(overdue > np.maximum(15.0, 4.0 * sd), 0.0, w)


def window_prob(d, horizon, mu, sd):
    """P(next purchase falls in days [d, d+horizon) | no purchase yet), times activity weight."""
    w = activity_weight(d - mu, sd)
    surv = np.maximum(1.0 - norm_cdf((d - 0.5 - mu) / sd), 1e-5)
    hi = norm_cdf((d + horizon - 0.5 - mu) / sd)
    lo = norm_cdf((d - 0.5 - mu) / sd)
    p = np.where(w > 0, np.clip((hi - lo) / surv, 0.0, 1.0) * w, 0.0)
    return p


# --------------------------------------------------------------------------- base
class SBA:
    """Croston's method with the Syntetos-Boylan bias correction, vectorised over medicines.
    Also tracks a smoothed one-day MSE for the spread of protection-period demand."""

    def __init__(self, n_meds: int, alpha: float = 0.1):
        self.a = alpha
        self.z = np.ones(n_meds)          # smoothed non-zero demand size
        self.p = np.full(n_meds, 7.0)     # smoothed interval between demands
        self.q = np.zeros(n_meds)         # days since last demand
        self.mse = np.full(n_meds, 0.5)

    def rate(self) -> np.ndarray:
        return (1.0 - self.a / 2.0) * self.z / self.p

    def update(self, y: np.ndarray) -> None:
        f = self.rate()
        self.mse = (1 - self.a) * self.mse + self.a * (y - f) ** 2
        self.q += 1
        nz = y > 0
        self.z[nz] = (1 - self.a) * self.z[nz] + self.a * y[nz]
        self.p[nz] = (1 - self.a) * self.p[nz] + self.a * self.q[nz]
        self.q[nz] = 0


class Policy:
    name = "base"

    def __init__(self, n_meds: int, n_days: int, lead_time: int, service: float):
        self.M, self.N, self.L, self.service = n_meds, n_days, lead_time, service
        self.P = lead_time + 1
        self.served = np.zeros((n_days, n_meds))

    def observe(self, d, total, resid, purchases):
        self.served[d] = total

    def target(self, d: int) -> np.ndarray:
        raise NotImplementedError

    def order_qty(self, d: int, ip: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.target(d) - ip)


class MovingAveragePolicy(Policy):
    name = "Moving average"

    def target(self, d):
        if d < 7:
            mean, sd = np.full(self.M, 0.2), np.full(self.M, 0.5)
        else:
            w = self.served[max(0, d - 28):d]
            mean, sd = w.mean(0), w.std(0)
        return np.ceil(mean * self.P + z_of(self.service) * sd * math.sqrt(self.P))


class MinMaxPolicy(Policy):
    """Stand-in for a chemist / billing-software min-max rule: levels set once from the
    first 30 days and never updated. `service` is a multiplier on the min level."""
    name = "Fixed min/max"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lo = self.hi = None

    def order_qty(self, d, ip):
        if d < 30:
            return MovingAveragePolicy.target(self, d).clip(0) - ip if d >= 7 else np.maximum(0, 1 - ip)
        if self.lo is None:
            mu0 = self.served[:30].mean(0)
            self.lo = np.ceil(mu0 * self.P * self.service)
            self.hi = self.lo + np.maximum(1, np.ceil(mu0 * 7))
        return np.where(ip <= self.lo, self.hi - ip, 0.0)


class CrostonPolicy(Policy):
    name = "Croston (SBA)"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sba = SBA(self.M)

    def observe(self, d, total, resid, purchases):
        super().observe(d, total, resid, purchases)
        self.sba.update(total)

    def target(self, d):
        mean = self.sba.rate() * self.P
        var = self.sba.mse * self.P
        return np.ceil(mean + z_of(self.service) * np.sqrt(var))


class RefillClockPolicy(Policy):
    """Demand = sum over identified regulars of P(they come in during the protection
    period) x their usual quantity, plus a Croston (SBA) forecast for everything that
    can't be tied to a person (walk-ins, regulars without a phone number on the bill)."""
    name = "Refill clock"

    def __init__(self, n_meds, n_days, lead_time, service, n_ids: int):
        super().__init__(n_meds, n_days, lead_time, service)
        self.resid = SBA(n_meds)
        self.hist: dict[int, list] = {}
        self.seen = np.zeros(n_ids, bool)
        self.med = np.zeros(n_ids, int)
        self.last = np.zeros(n_ids)
        self.lastq = np.ones(n_ids)
        self.qtyp = np.ones(n_ids)
        self.cycle = np.full(n_ids, PRIOR_CYCLE)
        self.sd = np.full(n_ids, PRIOR_SD)
        self.ngaps = np.zeros(n_ids, int)
        self.prior_cycle, self.prior_sd = PRIOR_CYCLE, PRIOR_SD

    def observe(self, d, total, resid, purchases):
        super().observe(d, total, resid, purchases)
        self.resid.update(resid)
        for vid, m, q in purchases:
            h = self.hist.setdefault(vid, [])
            h.append((d, q))
            self.seen[vid], self.med[vid], self.last[vid], self.lastq[vid] = True, m, d, q
            self._refit(vid)
        if d % 30 == 29:
            good = self.seen & (self.ngaps >= 3)
            if good.sum() >= 10:
                self.prior_cycle = float(np.median(self.cycle[good]))
                self.prior_sd = float(np.median(self.sd[good]))

    def _refit(self, vid):
        c, s, ng, qt = fit_clock(self.hist[vid], self.prior_cycle, self.prior_sd)
        self.cycle[vid], self.sd[vid], self.ngaps[vid], self.qtyp[vid] = c, s, ng, qt

    def next_due(self) -> np.ndarray:
        return self.last + self.cycle * self.lastq / self.qtyp

    def target(self, d):
        idx = np.nonzero(self.seen)[0]
        mean = self.resid.rate() * self.P
        var = self.resid.mse * self.P
        if len(idx):
            mu, sd, q = self.next_due()[idx], self.sd[idx], self.qtyp[idx]
            p = window_prob(d, self.P, mu, sd)
            mean = mean + np.bincount(self.med[idx], p * q, self.M)
            var = var + np.bincount(self.med[idx], p * (1 - p) * q * q, self.M)
        return np.ceil(mean + z_of(self.service) * np.sqrt(var))

    # ---- explanation API for the chemist screen
    def regulars_table(self, d: int, horizon: int = 7) -> pd.DataFrame:
        idx = np.nonzero(self.seen)[0]
        mu, sd = self.next_due()[idx], self.sd[idx]
        return pd.DataFrame({
            "vid": idx,
            "med_id": self.med[idx],
            "last_day": self.last[idx].astype(int),
            "due_day": mu,
            "sd": sd,
            "cycle": self.cycle[idx],
            "qty": self.qtyp[idx],
            "n_buys": self.ngaps[idx] + 1,
            "overdue": d - mu,
            "p_window": window_prob(d, horizon, mu, sd),
            "active": activity_weight(d - mu, sd),
        })


POLICIES = {
    "Fixed min/max": MinMaxPolicy,
    "Moving average": MovingAveragePolicy,
    "Croston (SBA)": CrostonPolicy,
    "Refill clock": RefillClockPolicy,
}
