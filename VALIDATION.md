# Validation record

Validated on 2026-11-13 with Python 3.12 and the exact dependencies in
`requirements.txt`, using the authorized local input identified by the SHA-256
in `results/data_quality.json`.

- All required Python imports succeeded.
- `python src/analysis.py` completed successfully.
- Six automated tests passed: input integrity/cleaning, missing input/schema,
  correlations, reproducible split and coefficient transformations, robust
  intervals, and output artifacts.
- All nine notebook code cells executed sequentially through IPython using
  `python tests/execute_notebook.py`; notebook schema validation passed. Saved
  notebook outputs contain text and tables; PNGs are in `results/` and display
  inline on an interactive rerun.
- A separate empty virtual environment was created and installed solely from
  `requirements.txt`; `pip check` reported no broken requirements. The analysis
  was rerun there with `--output` pointing to a fresh directory. All 14 output
  files (six PNGs, seven CSVs and one JSON) were byte-identical to the first run.
- Every CSV parsed successfully, every PNG decoded successfully, and each PNG
  reported 300 DPI (allowing PNG metadata rounding). All six plots were visually
  inspected for titles, labels, legends and layout.
- README relative file/image links were checked. The Colab badge image, Colab
  notebook URL and author URL returned HTTP 200. The notebook URL targets the
  published repository path; HTTP success alone does not test a hosted runtime.
- Raw input is ignored by Git; no license was added. The historical notebook is
  unchanged. Changed source, notebook outputs and documentation were reviewed
  for credentials and unexpected local paths.

## Environment limitations

A conventional `nbclient` kernel launch failed because this execution environment
blocks socket binding, including TCP and IPC. This was not a notebook-cell error.
The IPython execution runner was used to execute every cell without sockets.
The hosted Google Colab runtime and browser upload dialog were not exercised;
the notebook includes portable cloning, dependency installation and upload code.

The raw CSV cannot be supplied through the repository until redistribution rights
are established. The clean-environment test used the supplied local CSV, not a
public download. Exact data reproduction remains conditional on obtaining that file.
