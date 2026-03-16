import os
from langchain_core.tools import tool
from sqlalchemy import create_engine, MetaData, inspect
import pandas as pd

# Global engine with environment variable fallback
DB_URL = os.getenv("DATABASE_URL", "sqlite:///analytics.db")
engine = create_engine(DB_URL)

@tool
def get_database_schema(tables: str = "all") -> str:
    """
    Retrieves a professional-grade schema overview of the connected SQL database.
    """
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        schema_info = ["## 🗄️ Database Schema Architecture\n"]
        for table_name in table_names:
            if tables != "all" and table_name not in tables.split(','):
                continue
            
            cols = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name).get('constrained_columns', [])
            
            # Simple row count for context
            try:
                count_df = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", engine)
                row_count = count_df['count'].iloc[0]
            except:
                row_count = "Unknown"
            
            column_desc = [f"  - `{c['name']}` ({c['type']})" + (" **[PK]**" if c['name'] in pk else "") for c in cols]
            schema_info.append(f"### Table: `{table_name}` ({row_count} rows)")
            schema_info.extend(column_desc)
            schema_info.append("")
            
        return "\n".join(schema_info) if len(schema_info) > 1 else "No tables found."
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"

@tool
def profile_database_table(table_name: str) -> str:
    """
    Professional database profiler: Analyzes a table for nulls, cardinality, and data health.
    """
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5000", engine)
        if df.empty: return f"Table {table_name} is empty or not found."
        
        stats = []
        for col in df.columns:
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()
            dtype = str(df[col].dtype)
            stats.append({
                "Column": col,
                "Type": dtype,
                "Nulls": f"{nulls} ({round(nulls/len(df)*100, 1)}%)",
                "Unique": unique
            })
        
        report = pd.DataFrame(stats).to_markdown(index=False)
        return f"## 🔬 Data Health Profile: `{table_name}`\n\n{report}"
    except Exception as e:
        return f"Profiling error: {str(e)}"

@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a SELECT or WITH SQL query and returns a professional Markdown table.
    """
    try:
        clean_query = query.strip().upper()
        if not (clean_query.startswith("SELECT") or clean_query.startswith("WITH")):
            return "Error: Read-only tool. Only 'SELECT' and 'WITH' statements are allowed."

        if "LIMIT" not in clean_query:
            query = f"SELECT * FROM ({query}) AS subquery LIMIT 1000"

        df = pd.read_sql(query, engine)
        if df.empty:
            return "Query executed successfully. 0 rows returned."
            
        table = df.to_markdown(index=False)
        return f"### 📊 Query Results ({len(df)} rows)\n\n{table}"
    except Exception as e:
        return f"Error executing SQL: {str(e)}"
