import os
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global placeholders for lazy loading
_agent_executor = None

def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        print("\n[SYSTEM]: Initializing Specialist Agent Team (Lazy Loading)...", flush=True)
        
        # Internal imports to avoid module-level hang
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent
        
        # Tools
        from tools.python_tool import python_data_analyzer
        from tools.sql_tool import get_database_schema, execute_sql_query
        from tools.excel_tool import read_excel_or_csv, process_excel_dataset
        from tools.visualization_tool import generate_standard_chart
        from tools.web_search_tool import web_search
        from tools.pdf_tool import read_pdf
        from tools.ml_tool import advanced_ml_analysis
        from tools.data_quality_tool import profile_data_quality
        from tools.statistical_testing_tool import statistical_test
        from tools.report_export_tool import export_consulting_report
        from memory.vector_memory import store_insight, recall_past_insights
        from tools.simulation_tool import run_scenario_simulation
        from tools.knowledge_rag_tool import search_knowledge_library, ingest_knowledge_document
        from tools.dashboard_builder_tool import build_interactive_dashboard
        from tools.pptx_export_tool import export_presentation_deck

        tools = [
            python_data_analyzer,
            get_database_schema,
            execute_sql_query,
            read_excel_or_csv,
            process_excel_dataset,
            read_pdf,
            generate_standard_chart,
            web_search,
            advanced_ml_analysis,
            profile_data_quality,
            statistical_test,
            export_consulting_report,
            store_insight,
            recall_past_insights,
            run_scenario_simulation,
            search_knowledge_library,
            ingest_knowledge_document,
            build_interactive_dashboard,
            export_presentation_deck,
        ]

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            convert_system_message_to_human=True,
            streaming=True
        )

        SYSTEM_PROMPT = """You are 'InsightForge AI', an ultra-advanced data analytics super-agent specializing in deep business intelligence and visual strategy.

[ROLES]
1. [DATA SCIENTIST]: Expert in SQL, Python models (ML), and statistical rigor.
2. [VISUAL ANALYST]: Can 'see' screenshots/images. Interprets trends, UI layouts, and suggests improvements.
3. [STRATEGIC CONSULTANT]: Uses past insights (Deep Brain), RAG knowledge, and Scenario-Simulations to give 'What-If' advice.

[GUIDELINES]
- **UNIVERSAL PROFILING**: You can automatically profile CSV, SQL, and Python files. RELY on the front-end 'Auto Data Profile' for instant context.
- **CLEANING-FIRST**: Before performing ML or complex analysis on raw Excel/CSV datasets, use `process_excel_dataset(action='clean')` to ensure data cleanliness.
- **SQL PROFILING**: Before querying a new SQL table, use `get_database_schema` followed by `profile_database_table` to investigate data health.
- **VISUAL-FIRST**: NEVER provide a purely text-based answer for trends, clustering, or forecasting. Any ML/Statistical tool call MUST be followed by a visualization call (`generate_standard_chart` or `build_interactive_dashboard`).
- **HIGH-END DASHBOARDS**: Use `build_interactive_dashboard` for multi-chart summaries; it now supports premium Glassmorphism aesthetics.
- Use `[[CHART:filename.png]]` for static charts and `[[PLOTLY:filename.html]]` for interactive ones.
- When an image is provided, analyze it visually and relate it to the data.
- Use `export_presentation_deck` for PPTX and `process_excel_dataset(action='export')` for professional workbooks.
"""
        memory = MemorySaver()
        _agent_executor = create_react_agent(
            llm,
            tools,
            checkpointer=memory,
            prompt=SYSTEM_PROMPT
        )
        print("[SYSTEM]: Agent Team Ready.\n", flush=True)
    
    return _agent_executor

# ── Agent Runners ────────────────────────────────────────────────────────────

def run_analytics_request(message_str: str, session_id: str, image_data: str = None, files: list = None):
    """Sync runner for the agent with multimodal and file support."""
    from langchain_core.messages import HumanMessage
    agent = get_agent_executor()
    config = {"configurable": {"thread_id": session_id}}
    
    content = [{"type": "text", "text": message_str}]
    
    if image_data:
        content.append({"type": "image_url", "image_url": {"url": image_data}})
        
    if files:
        file_ctx = "\n\n[ATTACHED FILES]:\n"
        for f in files:
            file_ctx += f"--- {f['filename']} ---\n{f['content'][:5000]}\n"
        content[0]["text"] += file_ctx

    response = agent.invoke({"messages": [HumanMessage(content=content)]}, config)
    return response["messages"][-1].content


async def stream_analytics_request(message_str: str, session_id: str, image_data: str = None, files: list = None):
    """Async generator for streaming agent responses with multimodal and file support."""
    from langchain_core.messages import HumanMessage
    agent = get_agent_executor()
    config = {"configurable": {"thread_id": session_id}}
    
    content = [{"type": "text", "text": message_str}]
    
    if image_data:
        content.append({"type": "image_url", "image_url": {"url": image_data}})
        
    if files:
        file_ctx = "\n\n[ATTACHED FILES]:\n"
        for f in files:
            file_ctx += f"--- {f['filename']} ---\n{f['content'][:5000]}\n"
        content[0]["text"] += file_ctx

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=content)]},
        config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content"):
                yield chunk.content
        elif event["event"] == "on_tool_start":
            yield f"\n\n[SYSTEM]: Calling {event['name']} AGENT...\n"
