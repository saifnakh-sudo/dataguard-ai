# DataGuard AI — Training Data Firewall

A three-phase defensive pipeline against data poisoning attacks on AI/ML models.

**Prevention • Detection • Correction**

---

## Quick Start (5 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python src/generate_dataset.py

# 3. Inject poison (simulates the attack)
python src/poison_injector.py

# 4. Run the pipeline (optional — can also run inside the dashboard)
python src/phase1_prevention.py
python src/phase2_detection.py
python src/phase3_correction.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser and upload `data/poisoned.csv`.

---

## Pipeline

```
[Raw CSV]
    |
    v
[Phase 1: k-NN Prevention Filter]  ---->  rejected.csv (audit log)
    |
    v
[Phase 2: Isolation Forest Detector]  ---->  flagged.csv (dashboard)
    |
    v
[Phase 3: Quarantine + Retrain]
    |
    v
[Clean model + before/after report]
```

## File Structure

```
dataguard/
├── data/              # datasets (generated at runtime)
├── src/               # pipeline modules
│   ├── generate_dataset.py
│   ├── poison_injector.py
│   ├── train_baseline.py
│   ├── phase1_prevention.py
│   ├── phase2_detection.py
│   └── phase3_correction.py
├── dashboard/
│   └── app.py         # Streamlit UI
├── reports/           # auto-generated outputs
├── requirements.txt
└── README.md
```
