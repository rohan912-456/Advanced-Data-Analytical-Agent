from langchain_core.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Must be before plt import for headless server
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Ensure directories exist for saving outputs
os.makedirs("output_graphs", exist_ok=True)
os.makedirs(os.path.join("output_graphs", "interactive"), exist_ok=True)


@tool
def generate_standard_chart(
    chart_type: str,
    title: str,
    x_col: str,
    y_col: str,
    data_json: str,
    filename: str
) -> str:
    """
    Generates ultra-professional static and interactive charts for data analytics.
    
    chart_type: 'line', 'bar', 'scatter', 'pie', 'histogram', 'heatmap', 'regression', 'dist'
    title: Professional title for the visualization.
    x_col: Data column for X-axis.
    y_col: Data column for Y-axis.
    data_json: JSON string of data records.
    filename: Final filename (without extension).
    """
    try:
        df = pd.read_json(data_json, orient="records")
        base_name = os.path.splitext(filename)[0]
        png_path = os.path.join("output_graphs", f"{base_name}.png")
        html_name = f"{base_name}.html"
        html_path = os.path.join("output_graphs", "interactive", html_name)

        chart_type = chart_type.lower().strip()
        
        # ── PREMIUM BRAND PALETTE ────────────────────────────────────────────
        BRAND_TEAL = "#00d4aa"
        BRAND_BLUE = "#3b82f6"
        BRAND_PURPLE = "#8b5cf6"
        BRAND_PINK = "#ec4899"
        BG_DARK = "#080c14"
        BG_CARD = "#0d1420"
        TEXT_MUTE = "#94a3b8"

        # ── STATIC PROFESSIONAL RENDER ───────────────────────────────────────
        plt.figure(figsize=(12, 7), dpi=140)
        sns.set_theme(style="dark", rc={
            "axes.facecolor": BG_CARD,
            "figure.facecolor": BG_DARK,
            "axes.edgecolor": "#1e2d45",
            "grid.color": "#1e2d45",
            "text.color": "#f8fafc",
            "axes.labelcolor": TEXT_MUTE,
            "xtick.color": TEXT_MUTE,
            "ytick.color": TEXT_MUTE,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        if chart_type in ("line", "trend"):
            sns.lineplot(data=df, x=x_col, y=y_col, color=BRAND_TEAL, linewidth=3, marker="o", markersize=6)
        elif chart_type == "bar":
            sns.barplot(data=df, x=x_col, y=y_col, palette="mako", hue=x_col, legend=False)
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col, color=BRAND_BLUE, alpha=0.8, s=100, edgecolor="#ffffff", linewidth=0.5)
        elif chart_type == "regression":
            sns.regplot(data=df, x=x_col, y=y_col, color=BRAND_TEAL, scatter_kws={'alpha':0.5, 's':60}, line_kws={'color':BRAND_PINK, 'linewidth':3})
        elif chart_type in ("dist", "distribution"):
            sns.kdeplot(data=df, x=x_col or y_col, fill=True, color=BRAND_BLUE, alpha=0.4, linewidth=2)
            sns.histplot(data=df, x=x_col or y_col, color=BRAND_TEAL, alpha=0.2, kde=False)
        elif chart_type == "pie":
            plt.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%", colors=[BRAND_TEAL, BRAND_BLUE, BRAND_PURPLE, BRAND_PINK], textprops={'color': 'white'})
        elif chart_type == "heatmap":
            sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="mako", fmt=".2f", linewidths=1)
        else:
            return f"Error: Chart type '{chart_type}' is not professionally supported yet."

        plt.title(title, fontsize=16, fontweight="bold", pad=20, color="white")
        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight", facecolor=BG_DARK)
        plt.close()

        # ── INTERACTIVE PREMIUM PLOTLY ───────────────────────────────────────
        plotly_theme = {
            "template": "plotly_dark",
            "paper_bgcolor": BG_DARK,
            "plot_bgcolor": BG_CARD,
            "font": {"family": "Inter, sans-serif", "color": "#f1f5f9"},
        }
        
        if chart_type in ("line", "trend"):
            fig = px.line(df, x=x_col, y=y_col, title=title, render_mode="svg")
            fig.update_traces(line=dict(width=4, color=BRAND_TEAL))
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title, color=y_col, color_continuous_scale="mako")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, size_max=15)
        elif chart_type == "regression":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, trendline="ols", trendline_color_override=BRAND_PINK)
        elif chart_type == "heatmap":
            fig = px.imshow(df.select_dtypes(include="number").corr(), text_auto=True, color_continuous_scale="mako")
        else:
            fig = px.histogram(df, x=x_col or y_col, title=title, color_discrete_sequence=[BRAND_BLUE])

        fig.update_layout(**plotly_theme)
        fig.write_html(html_path, include_plotlyjs="cdn")

        return f"Professional chart '{title}' generated.\n[[CHART:{base_name}.png]][[PLOTLY:{html_name}]]"

    except Exception as e:
        return f"Visualization error: {str(e)}"

    except Exception as e:
        return f"Error generating chart: {str(e)}"
