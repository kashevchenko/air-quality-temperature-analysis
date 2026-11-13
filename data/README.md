# Dataset access and provenance

Place the authorized input at `data/air_quality_dataset.csv`, or pass `--data PATH`.
The supplied file was named `air_quality_dataset(2).csv` and contains 500 records
and ten numeric columns: Temperature, PM2.5, PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4.

The source, collection period, geography, measurement units, sampling design,
and whether the records are measured or simulated were not supplied. No claim
about any of these properties is made. Redistribution permission has not been
established, so the raw CSV is intentionally excluded from version control.
An upload fallback is provided in Colab. Exact reproduction requires the same
input; its SHA-256 is recorded in `results/data_quality.json`.

The analysis never rewrites the input. Complete cases are selected across all ten
variables. Rows with negative pollutant values are excluded as an explicit
convention, since their meaning is unknown. Duplicate analysis rows are removed.
IQR extremes are flagged, retained, and counted. A separate correlation table
retains negative values to assess sensitivity to that decision.
