import sys
import os

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import render_dashboard
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    render_dashboard()
