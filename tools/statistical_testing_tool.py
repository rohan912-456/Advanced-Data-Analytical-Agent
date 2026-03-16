from langchain_core.tools import tool
import pandas as pd
import numpy as np
import json

@tool
def statistical_test(test_type: str, data_json: str, params_json: str = "{}") -> str:
    """
    Performs rigorous statistical hypothesis tests on a dataset.
    
    test_type options:
      - 'ttest_ind'   : Independent samples t-test (compare means of 2 groups)
      - 'ttest_1samp' : One-sample t-test (test if mean equals a target value)
      - 'chi_square'  : Chi-square test of independence (categorical variables)
      - 'anova'       : One-way ANOVA (compare means across 3+ groups)
      - 'normality'   : Shapiro-Wilk normality test on a column
    
    data_json: JSON records string '[{"col": val, ...}, ...]'
    params_json: JSON dict of test-specific parameters:
      - ttest_ind:   {"col": "revenue", "group_col": "segment"}
      - ttest_1samp: {"col": "score", "popmean": 50}
      - chi_square:  {"col1": "region", "col2": "churn"}
      - anova:       {"value_col": "sales", "group_col": "product"}
      - normality:   {"col": "price"}
    
    Returns hypothesis test result with p-value, conclusion, and business interpretation.
    """
    try:
        from scipy import stats
    except ImportError:
        return "Error: 'scipy' is not installed. Run: pip install scipy"
    
    try:
        df = pd.read_json(data_json, orient="records")
        params = json.loads(params_json) if params_json else {}
    except Exception as e:
        return f"Error parsing input: {str(e)}"

    ALPHA = 0.05

    def significance_verdict(p_val):
        if p_val < 0.001:
            return f"Highly significant (p={p_val:.4f} < 0.001) ✅"
        elif p_val < ALPHA:
            return f"Statistically significant (p={p_val:.4f} < {ALPHA}) ✅"
        else:
            return f"NOT significant (p={p_val:.4f} ≥ {ALPHA}) ❌"

    test = test_type.lower().strip()

    # ── INDEPENDENT T-TEST ──────────────────────────────────────────────────
    if test == "ttest_ind":
        col       = params.get("col")
        group_col = params.get("group_col")
        if not col or not group_col:
            return "Error: 'ttest_ind' requires params {'col': '...', 'group_col': '...'}"
        
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return f"Error: group_col '{group_col}' has fewer than 2 unique groups."
        
        g1, g2 = groups[0], groups[1]
        a = df[df[group_col] == g1][col].dropna()
        b = df[df[group_col] == g2][col].dropna()
        
        stat, p = stats.ttest_ind(a, b)
        effect_size = (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)
        
        return (
            f"## Independent Samples t-Test\n"
            f"**H₀:** Mean {col} is equal between '{g1}' and '{g2}'\n"
            f"**H₁:** Means are significantly different\n\n"
            f"| Group | N | Mean | Std Dev |\n|---|---|---|---|\n"
            f"| {g1} | {len(a)} | {a.mean():,.2f} | {a.std():,.2f} |\n"
            f"| {g2} | {len(b)} | {b.mean():,.2f} | {b.std():,.2f} |\n\n"
            f"**t-statistic:** {stat:.4f}\n"
            f"**Result:** {significance_verdict(p)}\n"
            f"**Cohen's d (effect size):** {effect_size:.3f} "
            f"({'Large' if abs(effect_size) >= 0.8 else 'Medium' if abs(effect_size) >= 0.5 else 'Small'})\n\n"
            f"**Conclusion:** {'The difference in ' + col + ' between groups is statistically significant.' if p < ALPHA else 'No significant difference detected between groups.'}"
        )

    # ── ONE-SAMPLE T-TEST ───────────────────────────────────────────────────
    elif test == "ttest_1samp":
        col     = params.get("col")
        popmean = params.get("popmean")
        if not col or popmean is None:
            return "Error: 'ttest_1samp' requires params {'col': '...', 'popmean': value}"
        
        sample = df[col].dropna()
        stat, p = stats.ttest_1samp(sample, float(popmean))
        
        return (
            f"## One-Sample t-Test\n"
            f"**H₀:** Mean {col} = {popmean}\n"
            f"**H₁:** Mean {col} ≠ {popmean}\n\n"
            f"**Sample mean:** {sample.mean():,.4f} (n={len(sample)})\n"
            f"**Test value (μ₀):** {popmean}\n"
            f"**t-statistic:** {stat:.4f}\n"
            f"**Result:** {significance_verdict(p)}\n\n"
            f"**Conclusion:** {'The sample mean is significantly different from ' + str(popmean) + '.' if p < ALPHA else 'No evidence that the sample mean differs from ' + str(popmean) + '.'}"
        )

    # ── CHI-SQUARE ──────────────────────────────────────────────────────────
    elif test == "chi_square":
        col1 = params.get("col1")
        col2 = params.get("col2")
        if not col1 or not col2:
            return "Error: 'chi_square' requires params {'col1': '...', 'col2': '...'}"
        
        contingency = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
        
        return (
            f"## Chi-Square Test of Independence\n"
            f"**H₀:** '{col1}' and '{col2}' are independent\n"
            f"**H₁:** '{col1}' and '{col2}' are associated\n\n"
            f"**Contingency table shape:** {contingency.shape[0]} × {contingency.shape[1]}\n"
            f"**Chi-square statistic:** {chi2:.4f}\n"
            f"**Degrees of freedom:** {dof}\n"
            f"**Result:** {significance_verdict(p)}\n"
            f"**Cramér's V (association strength):** {cramers_v:.3f} "
            f"({'Strong' if cramers_v >= 0.5 else 'Moderate' if cramers_v >= 0.3 else 'Weak'})\n\n"
            f"**Conclusion:** {'Significant association found between ' + col1 + ' and ' + col2 + '.' if p < ALPHA else 'No significant association between ' + col1 + ' and ' + col2 + '.'}"
        )

    # ── ONE-WAY ANOVA ───────────────────────────────────────────────────────
    elif test == "anova":
        value_col = params.get("value_col")
        group_col = params.get("group_col")
        if not value_col or not group_col:
            return "Error: 'anova' requires params {'value_col': '...', 'group_col': '...'}"
        
        groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
        if len(groups) < 3:
            return "Error: ANOVA requires at least 3 groups. Use 'ttest_ind' for 2 groups."
        
        stat, p = stats.f_oneway(*groups)
        group_stats = df.groupby(group_col)[value_col].agg(["count", "mean", "std"]).round(2)
        
        table_rows = "\n".join([
            f"| {idx} | {int(row['count'])} | {row['mean']:,.2f} | {row['std']:,.2f} |"
            for idx, row in group_stats.iterrows()
        ])
        
        return (
            f"## One-Way ANOVA\n"
            f"**H₀:** All group means of '{value_col}' are equal\n"
            f"**H₁:** At least one group mean is different\n\n"
            f"| Group | N | Mean | Std Dev |\n|---|---|---|---|\n{table_rows}\n\n"
            f"**F-statistic:** {stat:.4f}\n"
            f"**Result:** {significance_verdict(p)}\n\n"
            f"**Conclusion:** {'At least one group has a significantly different mean.' if p < ALPHA else 'No significant difference across group means.'}"
        )

    # ── NORMALITY TEST ──────────────────────────────────────────────────────
    elif test == "normality":
        col = params.get("col")
        if not col:
            return "Error: 'normality' requires params {'col': '...'}"
        
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        
        stat, p = stats.shapiro(sample)
        skewness = float(stats.skew(sample))
        kurt     = float(stats.kurtosis(sample))
        
        return (
            f"## Shapiro-Wilk Normality Test\n"
            f"**Column:** {col} (n={len(sample)})\n"
            f"**H₀:** The data follows a normal distribution\n\n"
            f"**W-statistic:** {stat:.4f}\n"
            f"**Result:** {significance_verdict(p)}\n"
            f"**Skewness:** {skewness:.3f} "
            f"({'Right-skewed ↗' if skewness > 0.5 else 'Left-skewed ↙' if skewness < -0.5 else 'Approximately symmetric'})\n"
            f"**Kurtosis:** {kurt:.3f} "
            f"({'Heavy tails (leptokurtic)' if kurt > 1 else 'Light tails (platykurtic)' if kurt < -1 else 'Normal tails'})\n\n"
            f"**Conclusion:** {'Data is NOT normally distributed. Consider non-parametric tests.' if p < ALPHA else 'Cannot reject normality — data appears normally distributed.'}"
        )

    else:
        return (
            f"Error: Unknown test_type '{test_type}'. "
            "Valid options: 'ttest_ind', 'ttest_1samp', 'chi_square', 'anova', 'normality'"
        )
