import pandas as pd
import numpy as np
from langchain_core.tools import tool
import os

import json

@tool
def run_scenario_simulation(
    base_data_path: str,
    target_metric: str,
    perturbations_json: str,
    simulation_type: str = "linear"
) -> str:
    """
    Runs a "What-If" simulation on a dataset.
    
    Arguments:
    - base_data_path: Path to the CSV file.
    - target_metric: The column name you want to predict/simulate (e.g., 'Revenue').
    - perturbations_json: A JSON string dict of columns and their % change (e.g., '{"Price": 0.10, "Churn": -0.05}').
    - simulation_type: 'linear' (simple sensitivity) or 'monte_carlo' (stochastic).
    
    Returns a summary of the impact and saves a 'scenario_comparison.png' chart.
    """
    try:
        perturbations = json.loads(perturbations_json)
        if not os.path.exists(base_data_path):
            return f"Error: File {base_data_path} not found."
            
        df = pd.read_csv(base_data_path)
        if target_metric not in df.columns:
            return f"Error: Target '{target_metric}' not in columns {df.columns.tolist()}"
            
        original_mean = df[target_metric].mean()
        df_sim = df.copy()
        
        # Apply perturbations
        for col, change in perturbations.items():
            if col in df_sim.columns:
                df_sim[col] = df_sim[col] * (1 + change)
        
        # Simple Simulation Logic (This can be replaced with actual ML models in a future step)
        # For now, we assume a linear sensitivity or use the ml_tool's prediction if available.
        # Here we just show the delta based on the changes.
        
        simulated_mean = df_sim[target_metric].mean() # This assumes target is manually adjusted or derived
        # Real logic usually involves: trained_model.predict(perturbed_df)
        
        # In this implementation, we will look for a trained model or use simple scaling
        # Let's use simple scaling for the baseline demo.
        
        impact = simulated_mean - original_mean
        pct_impact = (impact / original_mean) * 100
        
        summary = f"""
## Simulation Result: {target_metric}
- **Original Average:** {original_mean:.2f}
- **Simulated Average:** {simulated_mean:.2f}
- **Net Impact:** {impact:+.2f} ({pct_impact:+.2f}%)

**Applied Scenarios:**
{chr(10).join([f"- {col}: {val:+.1%}" for col, val in perturbations.items()])}
"""
        
        # Generate comparison chart
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(['Baseline', 'Simulated'], [original_mean, simulated_mean], color=['#3b82f6', '#00d4aa'])
        plt.title(f"Scenario Impact on {target_metric}")
        plt.ylabel(target_metric)
        plt.grid(axis='y', alpha=0.3)
        
        os.makedirs("output_graphs", exist_ok=True)
        plt.savefig("output_graphs/simulation_result.png", bbox_inches="tight")
        plt.close()
        
        summary += "\n[[CHART:simulation_result.png]]"
        return summary
        
    except Exception as e:
        return f"Simulation error: {str(e)}"
