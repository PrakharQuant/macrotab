# MacroTab

**Global macro regimes · crisis early warning · transparent country macro tilts**

**Built by Prakhar Gupta** · [LinkedIn](https://www.linkedin.com/in/prakhar-gupta-5b7250372/) · [X / @PrakharQuant](https://x.com/PrakharQuant)

MacroPulse turns an annual country-year macro panel into an exploratory research workflow: feature engineering, historical-only regime clustering, next-year crisis scoring, a disclosed country macro rank, and an interactive Streamlit dashboard. It is intended as a reproducible **low-frequency macro-finance research project**, not a high-frequency trading system.

## Data source and citation

MacroPulse was built using the Global Macro Database (GMD). The dataset is maintained by Karsten Müller, Chenzi Xu, Mohamed Lehbib, and Ziliang Chen. Please cite the source as:

```bibtex
@techreport{GMD2025,
  title       = {The Global Macro Database: A New International Macroeconomic Dataset},
  author      = {M{"u}ller, Karsten and Xu, Chenzi and Lehbib, Mohamed and Chen, Ziliang},
  institution = {National Bureau of Economic Research},
  type        = {Working Paper},
  series      = {Working Paper Series},
  number      = {33714},
  year        = {2025},
  month       = {April},
  doi         = {10.3386/w33714},
  URL         = {http://www.nber.org/papers/w33714}
}
```

See the [GMD research paper](https://www.globalmacrodata.com/research-paper.html) and [Research Use Terms](https://www.globalmacrodata.com/license.html). The GMD terms govern the data separately from this project's MIT-licensed code. They restrict republishing the data or derived data on another website or service without written approval. Because a public dashboard may expose GMD-derived results, confirm the permitted-use scope with the maintainers before serving those results publicly. Citation alone does not grant redistribution or service rights. The app therefore does not bundle or automatically download the GMD dataset.

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

The repository's entry point is `app.py`. The current mode prompts the viewer to upload their own authorized CSV; it does not call Anansi MCP. GMD documents the OAuth-protected AI-agent endpoint [`https://mcp.anansidata.com/mcp`](https://mcp.anansidata.com/mcp) and the direct Python package [`global-macro-data`](https://github.com/KMueller-Lab/Global-Macro-Database-Python). A Manus MCP connection does not pass credentials to Streamlit Cloud. Any live GMD-backed public service still needs to comply with the Research Use Terms and any required written permission; this app will not automatically fetch or serve GMD data until that is resolved.

## Data handling and period assumptions

The supplied file contains rows dated **1086–2031**, with 57,392 country-year rows and 162 columns. Its provided brief describes a practical analysis span of roughly 1960–2031. MacroPulse therefore restricts analysis to **1960–2031** and records how many source rows fall outside that window.

The application uses **1960–2024** as the historical research period and displays **2025–2031** as a forward/forecast-period section. This is a conservative project convention: the CSV does not provide a complete vintage/provenance field that certifies every row after 2024 as a forecast, and the UI states that limitation. Crisis outcomes after the historical cutoff are never used for training or testing. If you have a better source-vintage cutoff, update the constants in `src/macropulse/data.py` and document the evidence.

The original CSV is **not committed**. `.gitignore` excludes local CSVs. The source is credited above; its separate Research Use Terms prohibit re-hosting or distributing GMD data or derived data on another site/service without the required permission. Do not include the CSV or activate public live-data serving unless the use is authorized.

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

Project code is provided under the MIT License. The GMD dataset is excluded and is governed by its separate [Research Use Terms](https://www.globalmacrodata.com/license.html). This is research software, not investment advice, and the GMD authors do not endorse it.
