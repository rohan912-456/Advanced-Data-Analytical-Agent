import os
from dotenv import load_dotenv
load_dotenv()
import sys
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent.analytics_agent import run_analytics_request, stream_analytics_request

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="InsightForge AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    print(f"ERROR: 422 Unprocessable Content: {exc_str}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

# Serve static chart files at /charts/
CHARTS_DIR = Path("output_graphs")
CHARTS_DIR.mkdir(exist_ok=True)
(CHARTS_DIR / "interactive").mkdir(exist_ok=True)
(CHARTS_DIR / "reports").mkdir(exist_ok=True)
(CHARTS_DIR / "dashboards").mkdir(exist_ok=True) # Added for dashboards
app.mount("/charts", StaticFiles(directory="output_graphs"), name="charts")
app.mount("/dashboards", StaticFiles(directory="output_graphs/dashboards"), name="dashboards")


from typing import List, Optional

# ── Request models ────────────────────────────────────────────────────────────
class FileItem(BaseModel):
    filename: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    image_data: Optional[str] = None
    files: List[FileItem] = []

class EDARequest(BaseModel):
    csv_text: str
    filename: str = "dataset"


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = Path(__file__).parent / "frontend.html"
    return frontend_path.read_text(encoding="utf-8")


# ── Standard (non-streaming) chat ────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        reply = run_analytics_request(
            request.message, 
            request.session_id, 
            image_data=request.image_data, 
            files=[f.dict() if hasattr(f, 'dict') else f for f in request.files]
        )
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming chat (SSE) ──────────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        full_response = ""
        try:
            async for chunk in stream_analytics_request(
                request.message, 
                request.session_id, 
                image_data=request.image_data,
                files=[f.dict() if hasattr(f, 'dict') else f for f in request.files]
            ):
                # Coerce chunk to string if it's a list (common in multimodal responses)
                if isinstance(chunk, list):
                    # Robustly extract text from multimodal parts
                    text_parts = []
                    for part in chunk:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            text_parts.append(part.get("text", ""))
                        else:
                            text_parts.append(str(part))
                    chunk = "".join(text_parts)
                
                full_response += str(chunk)
                payload = json.dumps({"chunk": chunk, "done": False})
                yield f"data: {payload}\n\n"
        except Exception as e:
            err = json.dumps({"chunk": f"\n\n⚠️ Error: {str(e)}", "done": False})
            yield f"data: {err}\n\n"
        finally:
            # Signal completion with the accumulated full text (for chart parsing)
            done_payload = json.dumps({"chunk": "", "done": True, "full": full_response})
            yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ── Universal Auto-Profiling endpoint ─────────────────────────────────────────
@app.post("/api/auto-eda")
async def auto_eda(request: EDARequest):
    """
    Instantly profiles a CSV, SQL, or Python dataset as soon as the user uploads it.
    """
    try:
        import pandas as pd
        import io
        import re

        filename = request.filename.lower()
        content = request.csv_text  # Reusing field for raw content
        
        # ── HANDLE CSV / EXCEL ──────────────────────────────────────────────
        if filename.endswith(".csv") or filename.endswith(".xlsx"):
            df = pd.read_csv(io.StringIO(content)) if filename.endswith(".csv") else pd.read_excel(io.BytesIO(content.encode('utf-8')))
            records_json = df.head(500).to_json(orient="records")

            from tools.data_quality_tool import profile_data_quality
            profile = profile_data_quality.func(records_json)
            
            summary = {
                "rows": len(df),
                "cols": len(df.columns),
                "columns": list(df.columns),
                "numeric_cols": df.select_dtypes(include="number").columns.tolist()
            }
            return {"profile": profile, "summary": summary}

        # ── HANDLE SQL ──────────────────────────────────────────────────────
        elif filename.endswith(".sql"):
            tables = re.findall(r"CREATE\s+TABLE\s+(\w+)", content, re.I)
            columns = re.findall(r"(\w+)\s+(?:TEXT|INTEGER|REAL|BLOB|VARCHAR|TIMESTAMP)", content, re.I)
            unique_cols = list(set(columns))
            
            profile = f"## 🗄️ SQL Schema Analysis\nDetected **{len(tables)}** table definitions.\n\n### Tables Found\n" + \
                      "\n".join([f"  - `{t}`" for t in tables]) + \
                      f"\n\n### Detected Columns\n" + ", ".join([f"`{c}`" for c in unique_cols[:20]])
            
            return {"profile": profile, "summary": {"rows": "N/A", "cols": len(unique_cols), "columns": unique_cols}}

        # ── HANDLE PYTHON ───────────────────────────────────────────────────
        elif filename.endswith(".py"):
            imports = re.findall(r"(?:import|from)\s+(\w+)", content)
            funcs = re.findall(r"def\s+(\w+)", content)
            
            profile = f"## 🐍 Python Script Profile\n**Imports:** {', '.join(set(imports))}\n" + \
                      f"**Functions:** {len(funcs)} detected.\n\n" + \
                      "### Structure Analysis\n" + \
                      "\n".join([f"  - Method: `{f}()`" for f in funcs[:10]])
            
            return {"profile": profile, "summary": {"rows": "Code", "cols": len(funcs), "columns": imports}}

        return {"profile": "Unsupported file type for auto-profiling.", "summary": None}

    except Exception as e:
        return {"profile": f"Profile Error: {str(e)}", "summary": None}


# ── Export report download (PDF) ───────────────────────────────────────────────
@app.get("/api/export-report")
async def export_report():
    try:
        from tools.report_export_tool import get_last_report_path
        path = get_last_report_path()
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No report has been generated yet.")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=os.path.basename(path)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Export presentation download (PPTX) ───────────────────────────────────────
@app.get("/api/export-pptx")
async def export_pptx():
    try:
        from tools.pptx_export_tool import get_last_pptx_path
        path = get_last_pptx_path()
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No presentation has been generated yet.")
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=os.path.basename(path)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
