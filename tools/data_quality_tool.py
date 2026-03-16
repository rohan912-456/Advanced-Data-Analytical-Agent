from langchain_core.tools import tool
import pandas as pd
import numpy as np
import io
from io import StringIO

@tool
def profile_data_quality(data_json: str) -> str:
    """
    Performs a comprehensive data quality assessment on a dataset.
    Run this FIRST before any analysis to understand the data's health.
    
    Detects:
      - Missing values per column (count + percentage)
      - Duplicate rows
      - Outliers using IQR method for numeric columns
      - Data type summary
      - Overall data quality score (0–100)
    
    data_json: A JSON string of records '[{"col": val, ...}, ...]'
    
    Returns a detailed data quality report.
    """
    try:
        df = pd.read_json(StringIO(data_json), orient="records")
    except Exception as e:
        return f"Error parsing data: {str(e)}"

    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return "Error: Dataset is empty."

    report = [
        "## 🔬 Data Quality Report",
        f"**Shape:** {rows:,} rows × {cols} columns",
        ""
    ]

    # ── MISSING VALUES ──────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / rows * 100).round(1)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

    if missing_df.empty:
        report.append("### ✅ Missing Values\nNo missing values found — dataset is complete.")
    else:
        report.append(f"### ⚠️ Missing Values ({len(missing_df)} columns affected)")
        for col in missing_df.index:
            cnt = int(missing_df.loc[col, "Missing Count"])
            pct = missing_df.loc[col, "Missing %"]
            severity = "🔴 Critical" if pct > 30 else "🟡 Moderate" if pct > 10 else "🟢 Minor"
            report.append(f"  - **{col}**: {cnt:,} missing ({pct}%) — {severity}")

    report.append("")

    # ── DUPLICATES ──────────────────────────────────────────────────────────
    dupe_count = int(df.duplicated().sum())
    dupe_pct = round(dupe_count / rows * 100, 1)
    if dupe_count == 0:
        report.append("### ✅ Duplicate Rows\nNo duplicate rows detected.")
    else:
        report.append(
            f"### ⚠️ Duplicate Rows\n"
            f"  - **{dupe_count:,}** duplicate rows ({dupe_pct}% of dataset)\n"
            f"  - Recommendation: Remove duplicates before analysis."
        )

    report.append("")

    # ── OUTLIERS ────────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            pct = round(len(outliers) / len(series) * 100, 1)
            outlier_results.append(
                f"  - **{col}**: {len(outliers)} outliers ({pct}%) "
                f"[range: {series.min():.2f} – {series.max():.2f}, "
                f"IQR bounds: {lower:.2f} – {upper:.2f}]"
            )

    if not outlier_results:
        report.append("### ✅ Outliers\nNo significant outliers detected in numeric columns.")
    else:
        report.append(f"### ⚠️ Outliers Detected ({len(outlier_results)} columns)")
        report.extend(outlier_results)

    report.append("")

    # ── DATA TYPES ──────────────────────────────────────────────────────────
    report.append("### 📋 Column Summary")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample_vals = ", ".join([str(v) for v in df[col].dropna().head(3)])
        report.append(
            f"  - **{col}** `{dtype}` — {nunique} unique values (e.g. {sample_vals})"
        )

    report.append("")

    # ── QUALITY SCORE ────────────────────────────────────────────────────────
    missing_score = max(0, 100 - (missing.sum() / (rows * cols) * 100) * 3)
    dupe_score    = max(0, 100 - dupe_pct * 2)
    outlier_score = max(0, 100 - len(outlier_results) * 5)
    quality_score = round((missing_score * 0.4 + dupe_score * 0.3 + outlier_score * 0.3), 1)

    grade = (
        "A (Excellent)" if quality_score >= 90 else
        "B (Good)"      if quality_score >= 75 else
        "C (Fair)"      if quality_score >= 60 else
        "D (Poor)"      if quality_score >= 40 else
        "F (Critical)"
    )

    report.append(
        f"### 🏆 Overall Data Quality Score\n"
        f"  **{quality_score}/100 — Grade {grade}**\n\n"
        f"  - Completeness:  {missing_score:.0f}/100\n"
        f"  - Uniqueness:    {dupe_score:.0f}/100\n"
        f"  - Consistency:   {outlier_score:.0f}/100\n\n"
        f"  *{'Data is ready for analysis.' if quality_score >= 75 else 'Consider cleaning the data before running analysis.'}*"
    )

    return "\n".join(report)
