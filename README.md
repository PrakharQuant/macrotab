# MacroPulse

**Global macro regimes · crisis early warning · transparent country macro tilts**

MacroPulse turns an annual country-year macro panel into an exploratory research workflow: feature engineering, historical-only regime clustering, next-year crisis scoring, a disclosed country macro rank, and an interactive Streamlit dashboard. It is intended as a reproducible **low-frequency macro-finance research project**, not a high-frequency trading system.

## What it does

- **Data validation:** checks the GMD schema, records exclusions and duplicate country-years, and restricts the analysis window to 1960–2031.
- **Macro features:** computes real GDP growth as the within-country annual log change and REER change only where consecutive years exist.
- **Regime discovery:** fits a four-cluster K-means model using observations through the selected historical cutoff. Descriptive cluster names are assigned from standardized growth and inflation centroids; they are not calibrated regime probabilities.
- **Crisis radar:** aligns predictors at year *t* to crisis labels at *t + 1*, requires consecutive calendar years, preserves missing labels, uses regularized logistic regression, and reports out-of-time ROC-AUC, PR-AUC, and Brier score.
- **Country macro tilts:** creates a cross-sectional, equal-weight heuristic from growth, absolute inflation, public debt, current-account balance, and unemployment. A softmax converts the top ranks into clearly labeled illustrative weights plus a cash reserve.
- **Dashboard:** global view, country lab, crisis radar, macro tilts, and a data coverage audit.

## Quick start

Python 3.10 or later is required.

```bash
git clone https://github.com/PrakharQuant/macroplulse.git
cd macroplulse
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Place your authorized copy of the source file at `data/GMD.csv`, then launch:

```bash
streamlit run app.py
```

To use a different file location:

```bash
MACROPULSE_DATA=/path/to/GMD.csv streamlit run app.py
```

The dashboard is designed around the supplied CSV column names. The loader requires `countryname`, `ISO3`, `year`, `rGDP`, and `infl`; optional macro or crisis columns are handled as missing.

### Streamlit Community Cloud

After the code is pushed to GitHub, create a Streamlit Community Cloud app using repository `PrakharQuant/macroplulse`, branch `main`, and file path `app.py`. The CSV is not bundled; the deployed app prompts the viewer to upload their authorized copy. For local use, place it at `data/GMD.csv`. Do not add private Streamlit secrets for this file.

## Data handling and period assumptions

The supplied file contains rows dated **1086–2031**, with 57,392 country-year rows and 162 columns. Its provided brief describes a practical analysis span of roughly 1960–2031. MacroPulse therefore restricts analysis to **1960–2031** and records how many source rows fall outside that window.

The application uses **1960–2024** as the historical research period and displays **2025–2031** as a forward/forecast-period section. This is a conservative project convention: the CSV does not provide a complete vintage/provenance field that certifies every row after 2024 as a forecast, and the UI states that limitation. Crisis outcomes after the historical cutoff are never used for training or testing. If you have a better source-vintage cutoff, update the constants in `src/macropulse/data.py` and document the evidence.

The original CSV is **not committed**. The file's source attribution and redistribution licence were not included with the upload; `.gitignore` excludes local CSVs. Only publish the data if you have verified permission to redistribute it.

## Method and limitations

### Temporal discipline

- GDP growth and REER changes are calculated only for consecutive country-year observations; gaps are not bridged.
- Regime scaling and cluster fitting use rows no later than the selected historical cutoff. Future-period rows are classified by that historical model, not used to refit it.
- Crisis labels at *t + 1* are shifted within country and accepted only for a one-year calendar step. A missing label remains missing rather than becoming a zero.
- The crisis model has a chronological training/holdout split. The live country scores are exploratory logistic model scores; they have not been independently calibrated or validated for operational decisions.

### What it does not claim

The GMD file contains annual macro variables and crisis indicators, not security total-return series. MacroPulse **does not produce CAGR, Sharpe ratio, volatility, drawdown, transaction costs, or an asset-allocation backtest**. The country weights are heuristic research tilts, not investable securities, portfolio advice, or evidence of predictive alpha. Country scores are not causal estimates. Definitions, units, revisions, missingness and forecast vintages remain subject to the upstream dataset.

## Repository layout

```text
app.py                         Streamlit dashboard
src/macropulse/data.py         validation and coverage audit
src/macropulse/features.py     annual macro feature engineering
src/macropulse/portfolio.py    regime model, country scores, illustrative tilts
src/macropulse/crisis.py       t+1 crisis labels and temporal model evaluation
tests/test_engine.py           unit tests
data/README.md                 local data instructions
```

## Tests

```bash
pytest
```

## License and use

Project code is provided under the MIT License. The dataset is excluded and may have separate terms. This is educational research software, not investment advice.
