from langchain_core.tools import tool
import pandas as pd
import os

@tool
def read_excel_or_csv(file_path: str) -> str:
    """
    Reads a CSV or Excel file from the given file path and returns a summary 
    of its contents (first few rows and column information).
    Useful for understanding the structure of a raw dataset before analysis.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            return "Error: Unsupported file format. Only .csv, .xls, and .xlsx are supported."
            
        summary = f"File loaded successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n\n"
        summary += f"Columns and Data Types:\n{df.dtypes.to_string()}\n\n"
        summary += f"First 5 rows preview:\n{df.head(5).to_string()}"
        
        return summary
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def process_excel_dataset(
    file_path: str, 
    action: str = "clean",
    strategy: str = "auto",
    output_name: str = "cleaned_data.xlsx"
) -> str:
    """
    Professional-grade automated data cleaning and processing for Excel/CSV.
    
    Actions:
    - 'clean': Handles missing values, duplicates, and normalizes column names.
    - 'summarize': Generates detailed statistical summaries for each column.
    - 'export': Processes and saves the file to a professional Excel format.
    
    Strategies (for 'clean'):
    - 'auto': Fills numeric nulls with mean and categorical with mode.
    - 'drop': Removes all rows with any null values.
    - 'strict': Only removes duplicates and normalizes names.
    """
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
        
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        initial_rows = len(df)
        
        if action == "clean":
            # 1. Normalize Column Names (Snake Case)
            df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]
            
            # 2. Remove Duplicates
            df = df.drop_duplicates()
            dupe_count = initial_rows - len(df)
            
            # 3. Handle Nulls
            if strategy == "drop":
                df = df.dropna()
            elif strategy == "auto":
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
            final_rows = len(df)
            
            # Save the processed file
            out_dir = "output_graphs/reports"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, output_name)
            df.to_excel(out_path, index=False)
            
            return (f"✅ Data cleaning successful.\n"
                    f"- Initial rows: {initial_rows}\n"
                    f"- Final rows: {final_rows}\n"
                    f"- Duplicates removed: {dupe_count}\n"
                    f"- Columns normalized: {', '.join(df.columns)}\n"
                    f"- Result saved to: {out_path}")
                    
        elif action == "summarize":
            summary = df.describe(include='all').transpose().to_markdown()
            return f"## 📊 Professional Data Summary\n{summary}"
            
        return "Invalid action selected."
        
    except Exception as e:
        return f"Excel processing error: {str(e)}"
