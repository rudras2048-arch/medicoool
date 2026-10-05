"""Synthetic chemist-shop generator.

EVERYTHING HERE IS SYNTHETIC. The parameters are knobs for sensitivity
analysis, not estimates of real Indian pharmacy behaviour. Medicine names are
generic salts + strengths (no brands); prices are made up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

CHRONIC_MEDS = [
    "Metformin 500 mg", "Metformin 1000 mg", "Glimepiride 1 mg", "Glimepiride 2 mg",
    "Sitagliptin 100 mg", "Vildagliptin 50 mg", "Teneligliptin 20 mg", "Gliclazide 80 mg",
    "Amlodipine 5 mg", "Amlodipine 10 mg", "Telmisartan 40 mg", "Telmisartan 80 mg",
    "Losartan 50 mg", "Cilnidipine 10 mg", "Metoprolol 25 mg", "Metoprolol 50 mg",
    "Atenolol 50 mg", "Ramipril 5 mg", "Torsemide 10 mg", "Atorvastatin 10 mg",
    "Atorvastatin 20 mg", "Rosuvastatin 10 mg", "Levothyroxine 25 mcg", "Levothyroxine 50 mcg",
    "Levothyroxine 75 mcg", "Levothyroxine 100 mcg", "Clopidogrel 75 mg", "Aspirin 75 mg",
    "Pantoprazole 40 mg", "Montelukast 10 mg",
]
ACUTE_MEDS = [
    "Amoxicillin 500 mg", "Azithromycin 500 mg", "Paracetamol 650 mg", "Ibuprofen 400 mg",
    "Cetirizine 10 mg", "Levocetirizine 5 mg", "Ondansetron 4 mg", "Ofloxacin 200 mg",
    "Diclofenac 50 mg", "Domperidone 10 mg", "Metronidazole 400 mg", "Famotidine 20 mg",
    "Cefixime 200 mg", "ORS sachet", "Dextromethorphan syrup 100 ml",
]
SURNAMES = [
    "Verma", "Iyer", "Khan", "Sharma", "Patel", "Reddy", "Das", "Nair", "Gupta", "Singh",
    "Mehta", "Joshi", "Rao", "Banerjee", "Kulkarni", "Menon", "Pillai", "Chopra", "Malhotra",
    "Bhat", "Saxena", "Mishra", "Pandey", "Ghosh", "Sen", "Fernandes", "D'Souza", "Ali",
    "Qureshi", "Agarwal",
]


@dataclass(frozen=True)
class ScenarioConfig:
    """Knobs for one synthetic shop-year. Frozen so it can be cached/hashed."""

    seed: int = 7
    n_days: int = 365
    n_chronic: int = 30
    n_acute: int = 15
    min_patients: int = 4            # regulars per chronic medicine
    max_patients: int = 14
    cycle_noise_sd: float = 3.0      # day-level jitter on each refill gap ("regularity")
    id_share: float = 0.7            # share of regulars whose bills carry a phone number
    double_buy_prob: float = 0.08    # refill covers two cycles
    leak_prob: float = 0.08          # this refill is bought at another shop (unseen)
    churn_prob: float = 0.15         # regular stops buying here during the year
    chronic_walkin: tuple = (0.02, 0.12)   # per-day walk-in rate range, chronic meds
    acute_rate: tuple = (0.05, 0.8)        # per-day rate range, acute meds


@dataclass
class Scenario:
    config: ScenarioConfig
    meds: pd.DataFrame       # med_id, name, chronic, price
    patients: pd.DataFrame   # ground truth for every regular
    events: pd.DataFrame     # latent demand: day, med_id, qty, patient_id, visible_id, is_regular
    _by_day: list = field(default=None, repr=False)

    @property
    def n_meds(self) -> int:
        return len(self.meds)

    def events_by_day(self) -> list:
        """List indexed by day of tuples (med, qty, visible_id, patient_id, is_regular)."""
        if self._by_day is None:
            n = self.config.n_days
            out = [None] * n
            ev = self.events
            groups = {d: g for d, g in ev.groupby("day", sort=True)}
            empty = tuple(np.array([], dtype=int) for _ in range(5))
            for d in range(n):
                g = groups.get(d)
                if g is None:
                    out[d] = empty
                else:
                    out[d] = (
                        g.med_id.to_numpy(), g.qty.to_numpy(), g.visible_id.to_numpy(),
                        g.patient_id.to_numpy(), g.is_regular.to_numpy(),
                    )
            self._by_day = out
        return self._by_day


def _make_meds(cfg: ScenarioConfig, rng: np.random.Generator) -> pd.DataFrame:
    chronic = CHRONIC_MEDS[: cfg.n_chronic]
    acute = ACUTE_MEDS[: cfg.n_acute]
    names = chronic + acute
    price = np.concatenate([
        rng.integers(6, 52, len(chronic)) * 5,   # Rs 30-255 per strip (synthetic)
        rng.integers(3, 30, len(acute)) * 5,     # Rs 15-145 per unit (synthetic)
    ])
    return pd.DataFrame({
        "med_id": np.arange(len(names)),
        "name": names,
        "chronic": [True] * len(chronic) + [False] * len(acute),
        "price": price.astype(float),
    })


def generate(cfg: ScenarioConfig) -> Scenario:
    rng = np.random.default_rng(cfg.seed)
    meds = _make_meds(cfg, rng)
    rows, patients = [], []
    pid = 0
    for m in meds.med_id[meds.chronic]:
        for _ in range(int(rng.integers(cfg.min_patients, cfg.max_patients + 1))):
            qty = int(rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.2, 0.15]))
            cycle = float(rng.uniform(26, 34))
            identified = bool(rng.random() < cfg.id_share)
            churn_day = float(rng.uniform(60, cfg.n_days)) if rng.random() < cfg.churn_prob else np.inf
            first = float(rng.uniform(0, cycle))
            name = f"{rng.choice(['Mr.', 'Mrs.', 'Ms.'])} {chr(65 + int(rng.integers(0, 26)))}. {rng.choice(SURNAMES)}"
            phone = f"••{int(rng.integers(1000, 10000))}"
            t, missed = first, np.nan
            while t < cfg.n_days:
                if t >= churn_day:
                    missed = t
                    break
                mult = 2 if rng.random() < cfg.double_buy_prob else 1
                if rng.random() >= cfg.leak_prob:
                    rows.append((int(t), int(m), qty * mult, pid, pid if identified else -1, True))
                t += max(1.0, cycle * mult + rng.normal(0.0, cfg.cycle_noise_sd))
            patients.append(dict(
                patient_id=pid, name=name, phone=phone, med_id=int(m), qty=qty, cycle=cycle,
                identified=identified, churn_day=churn_day, missed_day=missed, first_day=first,
            ))
            pid += 1

    # Walk-in / one-off demand (never identified as a regular).
    for m, chronic in zip(meds.med_id, meds.chronic):
        lo, hi = cfg.chronic_walkin if chronic else cfg.acute_rate
        rate = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        counts = rng.poisson(rate, cfg.n_days)
        for d in np.nonzero(counts)[0]:
            for _ in range(int(counts[d])):
                q = 1 if chronic else int(rng.choice([1, 2], p=[0.7, 0.3]))
                rows.append((int(d), int(m), q, -1, -1, False))

    events = pd.DataFrame(rows, columns=["day", "med_id", "qty", "patient_id", "visible_id", "is_regular"])
    events = events.iloc[rng.permutation(len(events))].sort_values("day", kind="stable").reset_index(drop=True)
    return Scenario(cfg, meds, pd.DataFrame(patients), events)


@lru_cache(maxsize=64)
def generate_cached(cfg: ScenarioConfig) -> Scenario:
    return generate(cfg)
