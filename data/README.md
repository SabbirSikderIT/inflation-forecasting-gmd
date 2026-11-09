# Data

## Source

The data comes from the **Global Macro Database (GMD)**:
- Website: https://www.globalmacrodata.com/
- Paper: Müller, K., Xu, C., Lehbib, M., & Chen, Z. (2025). 
  The Global Macro Database. NBER Working Paper No. 33714.

## Download Instructions

1. Visit https://www.globalmacrodata.com/data
2. Download the CSV file (`gmd.csv`)
3. Place it in this `data/` folder

## Why isn't the data committed?

The full GMD dataset is ~50MB and contains 57,392 rows × 162 columns. 
To keep the repository lightweight, we don't commit it. Users must download 
it manually.

## Cached data

If you want to avoid re-downloading, save intermediate files:
- `df_long_backup.csv` (52 countries, 100+ years) — ~5 MB
- `df_model_1970.csv` (1970–2025 filtered) — ~2 MB

These are still ignored by git but can be regenerated from `gmd.csv`.