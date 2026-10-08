# Local dataset setup

For local development, place the user-provided file at `data/GMD.csv`, or set `MACROPULSE_DATA` to its path. A deployed Streamlit instance without a local copy prompts the viewer to upload their own authorized `GMD.csv`. The raw CSV is intentionally git-ignored: its redistribution licence/provenance was not included with the supplied file. Do not commit or publish it unless you have verified you have redistribution rights.

The loader retains rows from 1960 through 2031. Years before 1960 are excluded from this project's analysis window. Years through 2024 are treated as the historical research sample; 2025–2031 are displayed as a forward/forecast period and are not used as crisis-model outcomes. These cutoffs are explicit project assumptions, not dataset-provided provenance claims.
