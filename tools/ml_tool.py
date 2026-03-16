from langchain_core.tools import tool
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Ensure directories exist
os.makedirs("output_graphs", exist_ok=True)
os.makedirs(os.path.join("output_graphs", "interactive"), exist_ok=True)

def _save_professional_visuals(fig, plotly_fig, base_filename, title):
    """Internal helper to save PNG and HTML and return markers."""
    png_path = os.path.join("output_graphs", f"{base_filename}.png")
    html_name = f"{base_filename}.html"
    html_path = os.path.join("output_graphs", "interactive", html_name)
    
    # Save PNG
    fig.savefig(png_path, dpi=140, bbox_inches="tight", facecolor="#080c14")
    plt.close(fig)
    
    # Save Plotly
    plotly_theme = {
        "paper_bgcolor": "#080c14",
        "plot_bgcolor": "#0d1420",
        "font": {"color": "#e2e8f0"},
        "xaxis": {"gridcolor": "#1e2d45", "zerolinecolor": "#1e2d45"},
        "yaxis": {"gridcolor": "#1e2d45", "zerolinecolor": "#1e2d45"},
    }
    plotly_fig.update_layout(**plotly_theme)
    plotly_fig.write_html(html_path, include_plotlyjs="cdn")
    
    return f"\n\n[[CHART:{base_filename}.png]][[PLOTLY:{html_name}]]"

@tool
def advanced_ml_analysis(task: str, data_json: str, params_json: str = "{}") -> str:
    """
    Runs advanced machine learning analysis on a dataset and auto-generates professional visuals.
    
    task options:
      - 'cluster'   : KMeans clustering — finds natural customer/data segments
      - 'forecast'  : Time-series forecasting using Exponential Smoothing (ARIMA-like)
      - 'correlate' : Pearson correlation matrix — identifies strongest variable relationships
      - 'anomaly'   : Detects multivariate outliers/anomalies using Isolation Forest
      - 'predict'   : Simple predictive model (Random Forest) for regression/classification
    """
    try:
        df = pd.read_json(data_json, orient="records")
        params = json.loads(params_json) if params_json else {}
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Professional Seaborn Theme
        sns.set_theme(style="darkgrid", rc={
            "axes.facecolor": "#0d1420",
            "figure.facecolor": "#080c14",
            "axes.edgecolor": "#1e2d45",
            "grid.color": "#1e2d45",
            "text.color": "#e2e8f0",
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
        })
    except Exception as e:
        return f"Error parsing input data: {str(e)}"
    
    task = task.lower().strip()

    def get_numeric_data(features=None):
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
        if features:
            numeric_df = numeric_df[[c for c in features if c in numeric_df.columns]]
        return numeric_df

    # ─────────────────────────── CLUSTERING ────────────────────────────────
    if task == "cluster":
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA

            features = params.get("features", None)
            n_clusters = int(params.get("n_clusters", 3))
            numeric_df = get_numeric_data(features)
            
            if numeric_df.shape[1] < 2:
                return "Error: Need at least 2 numeric columns for clustering."

            X = scaler.fit_transform(numeric_df)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            df["Cluster"] = labels.astype(str)

            # Visual: PCA Scatter Plot
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
            pca_df["Cluster"] = df["Cluster"]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="viridis", ax=ax)
            ax.set_title(f"Cluster Segments (PCA Projection)", color="white", fontweight="bold")
            
            plotly_fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", 
                                    title=f"Segment Distribution for {n_clusters} Clusters",
                                    template="plotly_dark")

            visual_markers = _save_professional_visuals(fig, plotly_fig, "ml_clusters", "Clustering Analysis")

            sil_score = silhouette_score(X, labels)
            segment_summaries = []
            for cluster_id in range(n_clusters):
                segment = df[df["Cluster"] == str(cluster_id)]
                stats = segment[numeric_df.columns].mean().round(2).to_dict()
                segment_summaries.append(f"  **Segment {cluster_id + 1}** ({len(segment)} records):\n" + 
                                         "\n".join([f"    - Avg {k}: {v}" for k, v in stats.items()]))

            return (f"## KMeans Clustering Analysis\n"
                    f"**Number of clusters:** {n_clusters} | **Silhouette Score:** {sil_score:.3f}\n\n"
                    f"### Segment Profiles\n" + "\n\n".join(segment_summaries) + visual_markers)
        except Exception as e:
            return f"Clustering error: {str(e)}"

    # ─────────────────────────── FORECASTING ───────────────────────────────
    elif task == "forecast":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            target_col = params.get("target_col", df.select_dtypes(include=[np.number]).columns[0])
            periods = int(params.get("periods", 6))
            series = df[target_col].dropna().reset_index(drop=True).astype(float)
            
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True, disp=False)
            forecast = fit.forecast(periods)
            full_series = pd.concat([series, forecast]).reset_index(drop=True)
            
            # Visual: Line Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(series.index, series.values, label="Historical", color="#3b82f6", linewidth=2)
            ax.plot(range(len(series), len(series)+periods), forecast.values, label="Forecast", color="#00d4aa", linestyle="--", linewidth=2)
            ax.set_title(f"Future Forecast: {target_col}", color="white", fontweight="bold")
            ax.legend()

            plotly_df = pd.DataFrame({
                "Period": range(len(full_series)),
                "Value": full_series.values,
                "Type": ["Actual"]*len(series) + ["Forecast"]*len(forecast)
            })
            plotly_fig = px.line(plotly_df, x="Period", y="Value", color="Type", title=f"Trend Forecast for {target_col}")

            visual_markers = _save_professional_visuals(fig, plotly_fig, "ml_forecast", "Forecast Analysis")

            forecast_lines = "\n".join([f"  Period {i + 1}: {v:,.2f}" for i, v in enumerate(forecast)])
            return (f"## Time-Series Forecast: {target_col}\n"
                    f"**Forecast horizon:** {periods} periods\n\n"
                    f"### Predicted Values\n{forecast_lines}" + visual_markers)
        except Exception as e:
            return f"Forecasting error: {str(e)}"

    # ─────────────────────────── ANOMALY ───────────────────────────────────
    elif task == "anomaly":
        try:
            from sklearn.ensemble import IsolationForest
            numeric_df = get_numeric_data(params.get("features"))
            contamination = float(params.get("contamination", 0.05))
            X = scaler.fit_transform(numeric_df)
            model = IsolationForest(contamination=contamination, random_state=42).fit(X)
            df["Is_Anomaly"] = model.predict(X) == -1
            
            # Visual: 2D Anomaly Plot (first two numeric cols)
            cols = numeric_df.columns[:2]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=cols[0], y=cols[1], hue="Is_Anomaly", palette={True: "#ef4444", False: "#3b82f6"}, ax=ax)
            ax.set_title(f"Anomaly Detection Profile", color="white", fontweight="bold")

            plotly_fig = px.scatter(df, x=cols[0], y=cols[1], color="Is_Anomaly", title="Anomalous Data Point Identification")
            visual_markers = _save_professional_visuals(fig, plotly_fig, "ml_anomalies", "Anomaly Analysis")

            anomalies = df[df["Is_Anomaly"]]
            return (f"## Anomaly Detection Report\n"
                    f"**Anomalies detected:** {len(anomalies)} ({contamination*100}% threshold)\n\n"
                    f"Found {len(anomalies)} suspicious records." + visual_markers)
        except Exception as e:
            return f"Anomaly detection error: {str(e)}"

    # ─────────────────────────── PREDICT ───────────────────────────────────
    elif task == "predict":
        try:
            from sklearn.ensemble import RandomForestRegressor
            target = params.get("target_col")
            X = get_numeric_data(params.get("features")).drop(columns=[target], errors="ignore").dropna()
            y = df.loc[X.index, target]
            model = RandomForestRegressor(n_estimators=100).fit(X, y)
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            
            # Visual: Feature Importance Bar
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances.values, y=importances.index, palette="mako", ax=ax)
            ax.set_title("Critical Feature Importance", color="white", fontweight="bold")

            plotly_fig = px.bar(x=importances.values, y=importances.index, orientation='h', title="Key Driver Analysis")
            visual_markers = _save_professional_visuals(fig, plotly_fig, "ml_importance", "Prediction Drivers")

            return (f"## Predictive Driver Analysis\n"
                    f"Identify key factors influencing **{target}**." + visual_markers)
        except Exception as e:
            return f"Prediction error: {str(e)}"

    # ─────────────────────────── CORRELATION ───────────────────────────────
    elif task == "correlate":
        try:
            numeric_df = get_numeric_data()
            corr = numeric_df.corr()
            
            # Visual: Heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Variable Relationship Network", color="white", fontweight="bold")

            plotly_fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Multi-Matrix")
            visual_markers = _save_professional_visuals(fig, plotly_fig, "ml_correlation", "Network Analysis")

            return (f"## Correlation Network Analysis\n"
                    f"Analyzing linear relationships between {len(numeric_df.columns)} variables." + visual_markers)
        except Exception as e:
            return f"Correlation error: {str(e)}"

    return f"Error: Unknown task '{task}'."
