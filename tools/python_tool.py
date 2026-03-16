import os
import sys
import io
import pandas as pd
import numpy as np
from langchain_core.tools import tool

@tool
def python_data_analyzer(code: str) -> str:
    """
    Executes Python code for data analysis. 
    Available: pd, np, plt (matplotlib), px (plotly express), save_chart, save_plotly.
    
    To show charts in the UI:
    - For static: `save_chart(plt, "name")`
    - For interactive: `save_plotly(fig, "name")`
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        import matplotlib.pyplot as plt
        import plotly.express as px
        import plotly.io as pio

        # Helpers
        def save_chart(plt_obj, name):
            os.makedirs("output_graphs", exist_ok=True)
            path = os.path.join("output_graphs", f"{name}.png")
            plt_obj.savefig(path, bbox_inches="tight")
            print(f"\n[[CHART:{name}.png]]")

        def save_plotly(fig, name):
            os.makedirs("output_graphs/interactive", exist_ok=True)
            path = os.path.join("output_graphs/interactive", f"{name}.html")
            pio.write_html(fig, file=path, auto_open=False)
            print(f"\n[[PLOTLY:{name}.html]]")

        safe_globals = {
            "pd": pd, "np": np, "plt": plt, "px": px,
            "save_chart": save_chart, "save_plotly": save_plotly,
            "__builtins__": __builtins__
        }
        
        exec(code, safe_globals)
        output = redirected_output.getvalue()
        return output if output else "Executed successfully."
        
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
