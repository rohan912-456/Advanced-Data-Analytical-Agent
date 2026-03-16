import streamlit as st
import os
import sys

# Import the core engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.analytics_agent import run_analytics_request

def render_dashboard():
    # Example format requested by user
    st.set_page_config(page_title="InsightForge AI", page_icon="📊", layout="wide")
    
    st.title("InsightForge AI")

    st.markdown("""
        Welcome to your autonomous **Data Analytics Agent**. I am capable of acting as your Manager, Data Engineer, 
        Data Analyst, Data Scientist, and BI Analyst. Upload data or ask me to query a database to get started.
    """)
    
    # Sidebar for data ingestion / settings
    with st.sidebar:
        st.header("Upload Context")
        uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            # Save the file temporarily so tools can access it
            os.makedirs("data", exist_ok=True)
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded to {save_path}!")
            st.session_state["dataset_path"] = save_path
            
        st.header("Database Connection")
        db_string = st.text_input("SQLAlchemy URI (e.g. sqlite:///test.db)", key="db_uri")
        if st.button("Connect to DB"):
            from tools.sql_tool import init_db
            init_db(db_string)
            st.success("Database connected!")

    # Chat Interface using user's exact requested style
    user_query = st.text_input("Ask your data anything")

    if user_query:
        if "dataset_path" in st.session_state and "dataset" not in user_query.lower():
            # Automatically inject context if a file is uploaded but not explicitly mentioned
            modified_query = f"Using the dataset at {st.session_state['dataset_path']}, {user_query}"
        else:
            modified_query = user_query
            
        with st.spinner("Analyzing..."):
            result = run_analytics_request(modified_query)
            st.write(result)
                    
    # Visualization Display Area
    st.subheader("📊 Output Visualizations")
    st.write("Generated charts will appear below if requested.")
    
    if os.path.exists("output_graphs"):
        charts = [f for f in os.listdir("output_graphs") if f.endswith(('.png', '.html'))]
        if charts:
            cols = st.columns(2)
            for i, chart in enumerate(charts):
                with cols[i % 2]:
                    if chart.endswith('.png'):
                        st.image(os.path.join("output_graphs", chart), caption=chart)
                    elif chart.endswith('.html'):
                        # Using an iframe to render the html plotly chart
                        with open(os.path.join("output_graphs", chart), 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        st.components.v1.html(html_data, height=400)
        else:
            st.info("No charts generated yet. Ask InsightForge to create a visualization!")
