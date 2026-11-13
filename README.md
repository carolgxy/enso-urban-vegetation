# Global urban vegetation responses to lagged ENSO-related atmospheric drying

Analysis code accompanying the Brief Communication *"Global urban vegetation responses to lagged
ENSO-related atmospheric drying"*. The repository reproduces the main figures (Fig. 1, Fig. 2) and the
focused Supplementary figures (Figs S1–S4) from the derived input tables, and re-fits the subregional
panel models on the same quality-screened sample used in the manuscript.

The scientific logic implemented here is: historical (1981–2018) ENSO–hydroclimate relationships are
projected onto the 2019–2026 RONI record to obtain two exposure indices (HV for atmospheric drying via
VPD, HS for soil moisture), which are then related to Sentinel-2 urban vegetation-area anomalies under
GUB and year–month fixed effects. Coefficients are reported as associations, not as causal ENSO effects.

---

## 1. Repository contents

| File | Role |
| --- | --- |
| `maincode.py` | Driver. Rebuilds the Q3 quality-screened sample, validates it against the regression audit, then draws Fig. 1, Fig. 2 and Supplementary Figs S1–S4. Also re-fits the lag-3 subregional models when `--fit-regions` is set. |
| `enso_formal_interaction_models.py` | Shared foundation: panel reader, GUB and year–month fixed effects via alternating within transforms (no dummy matrices), 5° spatial-block CR1 inference, season lookup, geometry lookup. |
| `enso_hydroclimate_panel_models.py` | H0/H1/H2 model definitions, RONI series handling, exposure attachment and standardization, two-way (5° block × year–month) clustering, Benjamini–Hochberg adjustment, per-subregion fitting. |
| `scl_Q3_common_sample_lag_robustness.py` | Sentinel-2 quality screens (Q1/Q2), reconstruction of the historical Q3 reference, construction of the common lag 0–3 sample, reference-sensitivity regressions. |
| `requirements.txt` | Python dependencies. |

---

## 2. Requirements

- Python 3.9 or newer
- `numpy`, `pandas`, `scipy >= 1.9`, `matplotlib >= 3.7`

`scipy >= 1.9` is needed for `spearmanr(...).statistic`; `matplotlib >= 3.7` is needed for
`legend(loc='outside lower center')`. Figures are typeset in Arial — if Arial is unavailable, matplotlib
falls back to DejaVu Sans and emits a warning; the layout is unaffected but text metrics differ slightly.

Optional: `geopandas` is only required if you pass `--boundaries-shp` to draw a coastline/country-outline
background under the maps.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The two largest tables total ≈ 1 GB and are expanded in memory during sample construction; 16 GB of RAM
is recommended. Peak memory occurs while the panel, quality flags and exposure table are joined, before
the intermediate frames are released with `gc.collect()`.

---

## 3. Running the analysis

```bash
bash run_analysis.sh
```

which expands to:

```bash
python maincode.py \
  --panel "GUB_Final_Analysis_Dataset.csv" \
  --quality "SCL_quality_all72.csv" \
  --exposure "enso_exposure_long_all16.csv" \
  --geo "GUB_Country_Region_Lookup.csv" \
  --roni "RONI.csv" \
  --q3-summary "H_main_comparison_for_SI.csv" \
  --q3-audit "Q3_reference_sample_audit.csv" \
  --quality-monthly "01_monthly_quality_summary.csv" \
  --cluster-comparison "single_vs_twoway_comparison.csv" \
  --fit-regions \
  --out "Final_Results"
```

Windows (PowerShell), single line:

```powershell
python maincode.py --panel "GUB_Final_Analysis_Dataset.csv" --quality "SCL_quality_all72.csv" --exposure "enso_exposure_long_all16.csv" --geo "GUB_Country_Region_Lookup.csv" --roni "RONI.csv" --q3-summary "H_main_comparison_for_SI.csv" --q3-audit "Q3_reference_sample_audit.csv" --quality-monthly "01_monthly_quality_summary.csv" --cluster-comparison "single_vs_twoway_comparison.csv" --fit-regions --out "Final_Results"
```

### Arguments

| Argument | Required | Meaning |
| --- | --- | --- |
| `--panel` | yes | Full 2019–2026 GUB panel (`greenspace area`, `temp`, `precip`, `soil_moisture`, `vpd`). |
| `--quality` | yes | Per GUB–year–month SCL quality flags. Must be the unfiltered table (`SCL_quality_all72.csv`), not a pre-filtered subset. |
| `--exposure` | yes | Historical ENSO β coefficients by GUB × season × lag (all lags in one long table). |
| `--geo` | yes | GUB centroids, country, continent and UN subregion. |
| `--roni` | yes | Relative Oceanic Niño Index, monthly. |
| `--q3-summary` | yes | Q3 coefficient table used for the Fig. 2 lag panel. Requires columns `lag, model, term, estimate_Q3_reference, ci_low_Q3_reference, ci_high_Q3_reference`. |
| `--q3-audit` | no | One-row audit table; if given, `n_q3_obs`, `n_q3_gub` and `n_q3_blocks` must match the reconstructed sample exactly. |
| `--quality-monthly` | no | Monthly quality summary over the full GUB universe (Fig. S1). |
| `--cluster-comparison` | no | One-way versus two-way clustered comparison (Fig. S4). |
| `--fit-regions` | no* | Re-estimate lag-3 H0/H1 per UN subregion on the Q3 sample, two-way clustered. |
| `--region-results` | no* | Existing Q3 lag-3 regional CSV instead of re-fitting (must be `lag == 3` and `reference == 'Q3'`). |
| `--boundaries-shp` | no | World outline (shapefile or GeoJSON) drawn beneath the maps; requires `geopandas`. |
| `--out` | yes | Output directory (created if absent). |
| `--min-clear` / `--min-clear2` / `--max-snow` | no | Q2 screen: clear coverage ≥ 80 %, twice-clear ≥ 50 %, snow ≤ 5 % (defaults match the manuscript). |
| `--min-hist` | no | Minimum number of Q2-qualified 2019–2025 years per GUB × calendar month for the Q3 reference (default 5). |
| `--min-reference-area` | no | Minimum reference area in m² (default 10 000). |
| `--min-region-blocks` | no | Minimum number of occupied 5° blocks for a subregion to be fitted (default 30). |
| `--map-point-size` | no | Marker size for the centroid maps (default 1.2). |

\* One of `--fit-regions` or `--region-results` is required, because Fig. 2b must come from the same Q3
sample. `--q3-m` is accepted for backward compatibility and is not used by this driver.

---

## 4. Input data

| File | Size | Key columns |
| --- | --- | --- |
| `GUB_Final_Analysis_Dataset.csv` | ≈ 577 MB | `ORIG_FID, year, month, greenspace area, temp, precip, soil_moisture, vpd` |
| `SCL_quality_all72.csv` | ≈ 407 MB | `ORIG_FID, year, month, clear_coverage_pct, clear2_coverage_pct, snow_any_pct, veg_area_m2, Q1, Q2` |
| `enso_exposure_long_all16.csv` | ≈ 66 MB | `ORIG_FID, season, enso_lag, ENSO_VPD_beta, ENSO_SM_beta` |
| `GUB_Country_Region_Lookup.csv` | ≈ 5.7 MB | `ORIG_FID, country_iso3, country_name, longitude, latitude, continent, subregion` |
| `RONI.csv` | 16 KB | `YR, MTH, ANOM` |
| `H_main_comparison_for_SI.csv` | 8 KB | Q3 vs original-reference coefficients, all lags, HV and HS |
| `Q3_reference_sample_audit.csv` | 1 KB | Sample audit (`n_q3_obs = 2,972,017`, `n_q3_gub = 58,882`, `n_q3_blocks = 526`) |
| `01_monthly_quality_summary.csv` | 8 KB | `year, month, frac_Q1, frac_Q2, clear_mean, snow_mean, ...` |
| `single_vs_twoway_comparison.csv` | 4 KB | `model, term, inference, estimate, se, ci_low, ci_high, p` |

**Sample definition.** For each GUB and calendar month, the Q3 reference is the mean vegetation area over
2019–2025 keeping only Q2-qualified records, and only if at least `--min-hist` qualified years exist. The
anomaly is `100 × (A − A_ref) / A_ref`. Only Q2-qualified GUB–months enter the regressions, which yields an
unbalanced panel of 2,972,017 records across 58,882 units within 526 5° blocks.

**Large files.** GitHub rejects files above 100 MB, so the three large tables are archived on Zenodo
rather than committed here: https://doi.org/10.5281/zenodo.24188060.

Dataset provenance: Sentinel-2 Level-2A Scene Classification Layer (ESA), ERA5-Land reanalysis
(Copernicus C3S), Global Urban Boundary polygons, and the Relative Oceanic Niño Index (NOAA CPC). Please
cite the original data providers in addition to the paper.

---

## 5. Outputs (`Final_Results/`)

| File | Content |
| --- | --- |
| `Figure1_Q3.png` / `.pdf` | Fig. 1: 2026 anomaly maps (Feb/Apr/Jun/Aug), monthly medians and IQR, anomaly-direction shares, subregion heatmap. |
| `Figure2_Q3.png` / `.pdf` | Fig. 2: HV lag curves, subregional forest plot, predicted lag-3 exposure heatmap, observed spatial correlations. |
| `Supplementary_FigS1_SCL_quality.png` / `.pdf` | Quality diagnostics over the full GUB universe. |
| `Supplementary_FigS2_sample_retention.png` / `.pdf` | Retained Q3 records by year. |
| `Supplementary_FigS3_reference_sensitivity.png` / `.pdf` | Original versus Q3 reference, all lags, HV and HS. |
| `Supplementary_FigS4_clustering.png` / `.pdf` | One-way versus two-way clustered intervals (needs `--cluster-comparison`). |
| `Fig1_2026_monthly_Q3.csv`, `Fig1_region_monthly_Q3.csv`, `Fig1_unit_2026_mean_Q3.csv` | Underlying values for Fig. 1e–g. |
| `Fig2_region_Q3_lag3_H0_H1_twoway.csv`, `Fig2_region_sample_audit.csv` | Subregional estimates and the blocks/units audit behind Fig. 2b. |
| `Fig2_2026_subregion_expected_HV_lag3.csv`, `Fig2_2026_observed_climate_vegetation_spatial_correlations.csv` | Values behind Fig. 2c and 2d. |
| `FigS2_Q3_retention_by_year_month.csv`, `FigS2_Q3_retention_by_subregion.csv`, `Figure_data_sample_audit.csv` | Sample retention and audit tables. |

---

## 6. Notes and limitations

- **Maps are centroid maps.** The geography table stores centroids only. Point colour encodes the GUB-level
  anomaly at a fixed marker size (`--map-point-size`), but the original GUB polygons are not reconstructed.
  `--boundaries-shp` adds a background outline for orientation and changes nothing about the data.
- **Regional estimates are re-fitted**, not reused: `--fit-regions` estimates lag-3 H0/H1 per UN subregion
  on the Q3 sample with 5° block × year–month clustering and BH adjustment over the nine eligible regions.
  Subregions with fewer than `--min-region-blocks` occupied blocks are reported as excluded.
- **The lag panel in Fig. 2a** is read from `--q3-summary`, i.e. the coefficient table produced by the
  modelling script, so the same numbers as the manuscript are plotted without re-running four lag models.
- **Interpretation.** HV and HS are expectations derived from historical relationships, standardized by
  lag-dependent historical scales; adjacent RONI lags are statistically dependent, and the covariance of
  the historical β surfaces is not propagated into the second-stage intervals. Coefficients are
  percentage points of vegetation anomaly per historical standard deviation of exposure.
- **Figure sizes** are set in inches at 300 dpi PNG plus vector PDF (`pdf.fonttype = 42`), so the PDFs keep
  editable text for journal production.

---

## 7. Citation

If you use this code, please cite the paper (full reference to be added upon publication) and this
repository.
Questions about the code: please contact the corresponding author listed in the paper.
