"""Reproducible temperature–pollutant analysis; never modifies input data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
POLLUTANTS = ["PM2.5", "PM10", "NOx", "NO2", "SO2", "VOCs", "CO", "CO2", "CH4"]
COLUMNS = ["Temperature", *POLLUTANTS]
SEED = 42


def load_and_clean(path):
    """Audit original records; use complete, nonnegative, unique records.

    Negative values may be corrected instrument readings, but their meaning is
    unknown here. Exclusion is an explicit analysis convention, not proof of error.
    IQR flags describe extremes; they are retained without instrument metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Supply your authorized CSV with --data PATH; see data/README.md.")
    raw = pd.read_csv(path)
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    numeric = raw[COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() | ~np.isfinite(numeric)
    negative = numeric[POLLUTANTS].lt(0).any(axis=1)
    valid = ~bad.any(axis=1) & ~negative
    eligible = numeric.loc[valid].copy()
    duplicate = eligible.duplicated()
    clean = eligible.loc[~duplicate].copy()
    if len(clean) < 20 or (clean.nunique() < 3).any():
        raise ValueError("At least 20 eligible rows and three distinct values per variable are required.")
    q1, q3 = clean.quantile(.25), clean.quantile(.75)
    iqr = q3 - q1
    outliers = (clean.lt(q1 - 1.5 * iqr) | clean.gt(q3 + 1.5 * iqr))
    audit = {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_shape": list(raw.shape), "raw_dtypes": raw.dtypes.astype(str).to_dict(),
        "missing_by_column": raw.isna().sum().astype(int).to_dict(),
        "coercion_failures": (numeric.isna() & raw[COLUMNS].notna()).sum().astype(int).to_dict(),
        "nonfinite_or_missing_by_column": bad.sum().astype(int).to_dict(),
        "raw_duplicate_records": int(raw.duplicated().sum()),
        "negative_by_pollutant": numeric[POLLUTANTS].lt(0).sum().astype(int).to_dict(),
        "rows_excluded_missing_or_nonfinite": int(bad.any(axis=1).sum()),
        "rows_excluded_negative_after_missing": int((negative & ~bad.any(axis=1)).sum()),
        "duplicate_analysis_rows_removed": int(duplicate.sum()),
        "clean_rows": len(clean),
        "iqr_outliers_by_column": outliers.sum().astype(int).to_dict(),
        "rows_with_any_iqr_outlier_retained": int(outliers.any(axis=1).sum()),
        "policy": "Complete cases across all ten variables; exclude negative pollutant rows; deduplicate analysis columns; retain IQR extremes. Raw file unchanged."
    }
    return raw, clean, audit


def correlations(data):
    rows = []
    for pollutant in POLLUTANTS:
        r, p = pearsonr(data.Temperature, data[pollutant])
        rho, sp = spearmanr(data.Temperature, data[pollutant])
        rows.append(dict(pollutant=pollutant, n=len(data), pearson_r=r,
                         pearson_p=p, spearman_r=rho, spearman_p=sp))
    result = pd.DataFrame(rows)
    # Separate prespecified families: nine primary Pearson, nine secondary Spearman.
    for method in ["pearson", "spearman"]:
        result[f"{method}_q"] = multipletests(result[f"{method}_p"], method="fdr_bh")[1]
        result[f"{method}_significant"] = result[f"{method}_q"] < .05
    return result


def regressions(data):
    """Full-sample OLS with HC3 covariance and t-based 95% intervals."""
    rows = []
    for pollutant in POLLUTANTS:
        fit = sm.OLS(data[pollutant], sm.add_constant(data[["Temperature"]])).fit(cov_type="HC3", use_t=True)
        intervals = fit.conf_int()
        for term in fit.params.index:
            rows.append(dict(pollutant=pollutant, term=term, coefficient=fit.params[term],
                             robust_se=fit.bse[term], p_value=fit.pvalues[term],
                             ci_low=intervals.loc[term, 0], ci_high=intervals.loc[term, 1],
                             in_sample_r2=fit.rsquared, n=len(data), covariance="HC3"))
    result = pd.DataFrame(rows)
    slopes = result.term.eq("Temperature")
    result["slope_q"] = np.nan
    result.loc[slopes, "slope_q"] = multipletests(result.loc[slopes, "p_value"], method="fdr_bh")[1]
    return result


def evaluate_models(data):
    """Identical held-out observations for every model; train-only transforms."""
    train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=SEED)
    x = data[["Temperature"]]
    models, rows, coefficients = {}, [], []
    for pollutant in POLLUTANTS:
        y = data[pollutant]
        for name, model in {
            "Mean baseline": DummyRegressor(strategy="mean"),
            "Linear": LinearRegression(),
            "Polynomial (degree 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
        }.items():
            model.fit(x.iloc[train], y.iloc[train])
            pred = model.predict(x.iloc[test])
            rows.append(dict(pollutant=pollutant, model=name, n_train=len(train), n_test=len(test),
                             test_r2=r2_score(y.iloc[test], pred),
                             test_rmse=np.sqrt(mean_squared_error(y.iloc[test], pred)),
                             test_mae=mean_absolute_error(y.iloc[test], pred)))
            models[pollutant, name] = model
            if name == "Linear":
                b0, b1, b2 = model.intercept_, model.coef_[0], 0.
            elif name == "Polynomial (degree 2)":
                scaler, reg = model[0], model[-1]
                mu, scale = scaler.mean_[0], scaler.scale_[0]
                a1, a2 = reg.coef_
                b2 = a2 / scale**2
                b1 = a1 / scale - 2 * a2 * mu / scale**2
                b0 = reg.intercept_ - a1 * mu / scale + a2 * mu**2 / scale**2
            else:
                continue
            coefficients.append(dict(pollutant=pollutant, model=name, beta0=b0, beta1=b1, beta2=b2))
    split = pd.DataFrame({"source_row_0based": data.index, "partition": ["train" if i in set(train) else "test" for i in range(len(data))]})
    return pd.DataFrame(rows), models, split, pd.DataFrame(coefficients)


def figures(data, corr, comparison, models, output):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    def save(fig, name):
        fig.savefig(output / name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    sns.heatmap(data[COLUMNS].corr(), annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0, square=True, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix | cleaned observations", pad=16)
    save(fig, "correlation_matrix.png")

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ordered = corr.assign(magnitude=corr.pearson_r.abs()).sort_values("magnitude")
    ypos = np.arange(len(ordered))
    ax.barh(ypos-.18, ordered.pearson_r, .36, label="Pearson", color="#24788e")
    ax.barh(ypos+.18, ordered.spearman_r, .36, label="Spearman", color="#df9850")
    ax.set(yticks=ypos, yticklabels=ordered.pollutant, xlim=(-1, 1), xlabel="Correlation coefficient", title="Temperature associations | sorted by absolute Pearson r")
    ax.axvline(0, color=".3", lw=1); ax.legend(loc="lower right")
    save(fig, "temperature_correlations.png")

    fig, axes = plt.subplots(4, 3, figsize=(13, 12), layout="constrained")
    for col, ax in zip(COLUMNS, axes.flat):
        sns.histplot(data[col], bins=25, ax=ax, color="#24788e")
        ax.set_title(col); ax.set_xlabel("Recorded value (unit unspecified)")
    for ax in axes.flat[len(COLUMNS):]: ax.set_visible(False)
    fig.suptitle("Distributions after cleaning | IQR extremes retained")
    save(fig, "distributions.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for pollutant, ax in zip(POLLUTANTS, axes.flat):
        sns.regplot(data=data, x="Temperature", y=pollutant, ci=None, ax=ax,
                    scatter_kws={"s": 10, "alpha": .4}, line_kws={"color": "#b54d42"})
        ax.set_title(pollutant)
    fig.suptitle("Full-sample linear fits | units unspecified; lines are not causal effects")
    save(fig, "regression_analysis.png")

    grid = pd.DataFrame({"Temperature": np.linspace(data.Temperature.min(), data.Temperature.max(), 200)})
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for pollutant, ax in zip(POLLUTANTS, axes.flat):
        ax.scatter(data.Temperature, data[pollutant], s=9, alpha=.25, color="gray", label="All observations")
        for name, color in [("Linear", "#24788e"), ("Polynomial (degree 2)", "#b54d42")]:
            ax.plot(grid.Temperature, models[pollutant, name].predict(grid), color=color, label=name)
        ax.set(title=pollutant, xlabel="Temperature", ylabel=pollutant)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Train-only fitted curves | descriptive overlay on all observations")
    save(fig, "polynomial_comparison.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for pollutant, ax in zip(POLLUTANTS, axes.flat):
        subset = comparison[comparison.pollutant.eq(pollutant)]
        ax.bar(["Mean", "Linear", "Quadratic"], subset.test_rmse, color=["#a2adb3", "#24788e", "#b54d42"])
        ax.set(title=pollutant, ylabel="Test RMSE (recorded units)")
    fig.suptitle("Held-out prediction error | lower is better; scales differ by pollutant")
    save(fig, "model_comparison.png")


def run(data_path=ROOT / "data/air_quality_dataset.csv", output=ROOT / "results"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    raw, clean, audit = load_and_clean(data_path)
    corr = correlations(clean)
    regression = regressions(clean)
    comparison, models, split, coefficients = evaluate_models(clean)
    descriptive = clean.describe().T.rename(columns={"50%": "median"})
    for name, frame in {"descriptive_statistics": descriptive, "correlation_results": corr,
                        "linear_regression_results": regression, "model_comparison": comparison,
                        "model_coefficients": coefficients, "split_assignments": split}.items():
        frame.to_csv(output / f"{name}.csv", index=name == "descriptive_statistics")
    # Sensitivity analysis: retain negative readings, but still require complete unique rows.
    sensitivity = raw[COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    correlations(sensitivity).to_csv(output / "correlation_sensitivity_keep_negatives.csv", index=False)
    (output / "data_quality.json").write_text(json.dumps(audit, indent=2) + "\n")
    figures(clean, corr, comparison, models, output)
    print(json.dumps(audit, indent=2))
    print(corr.to_string(index=False))
    print(comparison.to_string(index=False))
    return {"data": clean, "audit": audit, "correlations": corr, "regressions": regression, "comparison": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/air_quality_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.data, args.output)
