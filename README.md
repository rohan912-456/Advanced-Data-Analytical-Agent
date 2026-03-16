# 📊 InsightForge AI: Advanced Data Analytical Agent (v2.5)

**InsightForge AI** is a professional-grade strategic intelligence platform powered by **Gemini 3.1 Flash Lite**. It transforms messy datasets (CSV, Excel, SQL) into executive-level dashboards and ML-driven insights with a single prompt.

---

## 🚀 Key Features

- **💎 Glassmorphism UI**: A premium, dark-themed dashboard builder with interactive Plotly visualizations and blur-effects.
- **🔬 Automated ML Intelligence**: Built-in specialized tools for K-Means Clustering, Forecasting, Anomaly Detection, and ANOVA testing.
- **🗄️ Universal Auto-Profiling**: Instant schema and data quality analysis for `.csv`, `.sql`, and `.py` files upon upload.
- **🧹 Professional Data Cleaning**: Intelligent "Cleaning-First" workflow that normalizes schemas and handles missing values automatically.
- **👔 Executive Reporting**: One-click export to professional PowerPoint (.pptx) decks for boardroom presentations.

---

## 🛠️ Technology Stack

- **Core Engine**: LangGraph, LangChain, Gemini 3.1 Flash Lite
- **Data Science**: Pandas, NumPy, Scikit-learn, Scipy, Statsmodels
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Backend/API**: FastAPI, Uvicorn, Python-pptx
- **Frontend**: Custom React UI (Streaming SSE Integration)

---

## 🎬 Quick Start (Recruiter Demo Guide)

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Advanced-Data-Analytical-Agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
DATABASE_URL=sqlite:///analytics.db
```

### 4. Launch
```bash
# Run the FastAPI backend and start the application
python main.py
```
*The app will be accessible at `http://localhost:8000`*

---

## 📈 Demo Use Case: Workforce Talent Analysis
Included in the repo is `Workforce_Data.csv`. 
1. **Upload** the file.
2. **Watch**: The Auto-Profiling engine will instantly map the talent tiers.
3. **Ask**: *"Perform a salary clustering analysis and generate an executive dashboard."*
4. **Result**: InsightForge will run K-Means, generate a Glassmorphism UI, and provide a downloadable PPTX.

---
*Developed as a showcase of Advanced Agentic Coding and Strategic Data Engineering.*
