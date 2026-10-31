# re

> **Forecasting the People, Not the Curve.**  
> A patient-level refill cycle forecaster and inventory simulator for independent retail chemists in India.

---

## 🎯 The Core Problem & Insight

- **The Problem:** Independent chemists in India experience chronic-medicine stockouts (BP, diabetes, thyroid) because reordering is based on memory or aggregate moving averages. For slow-moving, intermittent medicines, sales look like unpredictable spikes separated by long strings of zeros.
- **The Core Insight:** For chronic therapies, demand is not an aggregate curve—it is a small group of identifiable people on biological refill clocks. **Forecast the people, not the curve.**
- **The Data Science Edge:** Moving averages and Croston's intermittent demand methods throw away the timing information latent in individual customer refill intervals. RefillClock estimates Bayesian individual refill intervals $\mathcal{N}(\mu_i, \sigma_i^2)$ and evaluates purchase probabilities within distributor lead times, achieving a superior fill-rate vs. working capital trade-off.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
cd refillclock
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run the 60-Second Demo (Streamlit)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Panel 1 (Regulars Due):** Explains impending demand by naming patients due this week.
- **Panel 2 (Order Today):** Generates an explainable reorder list tied to lead-time delivery windows.
- **Panel 3 (Overdue Regulars):** Identifies silent patient churn and triggers privacy-safe WhatsApp reminders (DPDP Act compliant—never mentions medicine name).
- **⚔️ Live 60s Replay:** Side-by-side simulation comparing Sharma Medical vs. a baseline shop, demonstrating exact stockout avoidance.

### 3. Run Benchmark Sweeps & Generate Evidence
```bash
python -m refillclock.evaluate --output-dir artifacts
```
Generates publication-quality charts and CSVs in `artifacts/`:
- `tradeoff_curve.png`: Regular Patient Fill Rate (%) vs. Average Inventory Value Held (₹) against Croston (SBA), Moving Average, and Fixed Min/Max.
- `sensitivity_sweeps.png`:
  - **Regularity Jitter Sweep:** Crossover point where cycle noise degrades advantage.
  - **Customer ID Coverage Sweep:** Proves mathematical convergence to Croston at 0% ID share.
- `missed_refills_eval.csv`: Quantifies that RefillClock flags lapsed patients **7 days earlier** with **27% fewer false alarms** than static 45-day thresholds.

### 4. Run Test Suite
```bash
pytest -v tests/
```

---

## 📊 Evaluation & Empirical Results

| Metric | RefillClock | Croston (SBA) | Moving Average |
| :--- | :---: | :---: | :---: |
| **Regular Patient Fill Rate (95% service)** | **83.4%** | 82.2% | 82.0% |
| **Patients Turned Away (Losses)** | **318** | 342 | 345 |
| **0% ID Share Equivalence (Bug Check)** | **81.92%** | **81.92%** | N/A |
| **Lapse Detection Delay (Pivot)** | **17.4 days** | N/A | 24.4 days (fixed 45d) |
| **False Alarms on Active Patients** | **183** | N/A | 251 (fixed 45d) |

---

## ⚖️ Hackathon Defense & Kill List Preparedness

1. **"What if distributor delivers daily in metros?"**
   - *Defense:* The lead-time sweep shows that with same-day deliveries, stockout urgency shrinks. In urban areas, we pivot the lead pitch to **Missed-Refill & Churn Alert**: retaining chronic patients drifting to e-pharmacies.
2. **"What if the patient accepts a substitute brand?"**
   - *Defense:* We model demand at the generic **salt + strength** level (e.g., Metformin 500mg), not brand level. A stockout means zero substitutable options remain.
3. **"Is purchase history sensitive health data?"**
   - *Defense:* Designed around the Digital Personal Data Protection (DPDP) Act 2023: zero cloud storage (local SQLite on chemist PC), hashed phone IDs, only 4 operational fields, and WhatsApp alerts never mention the medical condition or drug name.
4. **"Does this improve health outcomes?"**
   - *Defense:* We do not claim a clinical adherence cure without clinical trials. We measure *lapsed refill detection delay* as an operational metric.


---

## 📖 API Reference

### `simulate()`

Simulates pharmacy refill cycles and inventory depletion over a time horizon.

| Parameter | Description |
| :--- | :--- |
| `sc` | Scenario configuration object |
| `policy` | Reordering policy (Croston or RefillClock) |
| `lead_time` | Distributor delivery lead time in days |
| `timeout_secs` | Obsolete connection timeout parameter |
