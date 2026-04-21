# Grade = Predictor

An app that estimates a student's likely final IB grade from current performance and shows what is needed to improve.

## Features
- Input test scores (multiple assessments)
- Input IA progress and current IA quality estimate
- Predict likely final grade (1-7)
- Show score targets needed to reach the next grade band
- Quick trajectory guidance with simple forecasting logic

## Model 
The prediction combines:
- **Tests** (recent weighted average)
- **IA** (scaled by progress so incomplete IA contributes less)

The combined score is mapped to IB-style grade boundaries:

- 85-100 → 7
- 75-84 → 6
- 65-74 → 5
- 55-64 → 4
- 45-54 → 3
- 35-44 → 2
- 0-34 → 1

==
## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

==
