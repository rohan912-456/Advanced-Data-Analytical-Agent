from langchain_core.tools import tool
import os
from datetime import datetime

# Directory where PDFs will be saved
REPORTS_DIR = os.path.join("output_graphs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Store the last generated report path so the download endpoint can serve it
_last_report_path = {"path": None}


def get_last_report_path() -> str | None:
    return _last_report_path["path"]


@tool
def export_consulting_report(title: str, content: str, filename: str = "") -> str:
    """
    Exports a consulting report as a professional PDF document.
    Call this when the user requests a formal report, executive summary, or
    wants to download the analysis results.
    
    Args:
        title: The report title shown on the cover page.
        content: The full report body in plain text / markdown-like format.
                 Use ## for section headers, - for bullets.
        filename: Optional custom filename (without extension). 
                  Auto-generated with timestamp if not provided.
    
    Returns a confirmation string with the saved file path and download hint.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
            Table, TableStyle
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        return "Error: 'reportlab' is not installed. Run: pip install reportlab"

    try:
        # ── File path ────────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = (filename or title).replace(" ", "_").replace("/", "-")[:40]
        pdf_filename = f"{safe_name}_{timestamp}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

        # ── Styles ───────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()
        teal   = colors.HexColor("#00d4aa")
        dark   = colors.HexColor("#080c14")
        light  = colors.HexColor("#e2e8f0")
        slate  = colors.HexColor("#64748b")

        cover_title = ParagraphStyle(
            "CoverTitle", parent=styles["Title"],
            fontSize=28, textColor=dark, spaceAfter=6, alignment=TA_CENTER,
            fontName="Helvetica-Bold"
        )
        cover_sub = ParagraphStyle(
            "CoverSub", parent=styles["Normal"],
            fontSize=12, textColor=slate, alignment=TA_CENTER, spaceAfter=4
        )
        section_header = ParagraphStyle(
            "SectionHeader", parent=styles["Heading1"],
            fontSize=14, textColor=colors.HexColor("#1e40af"),
            spaceBefore=16, spaceAfter=4, fontName="Helvetica-Bold",
            borderPad=(0, 0, 0, 6)
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, textColor=dark, spaceAfter=4, leading=15
        )
        bullet_style = ParagraphStyle(
            "Bullet", parent=styles["Normal"],
            fontSize=10, textColor=dark, spaceAfter=3,
            leftIndent=16, bulletIndent=6, leading=14
        )

        # ── Build document ──────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm
        )

        story = []

        # Cover block
        story.append(Spacer(1, 1.5 * cm))
        story.append(Paragraph("InsightForge AI", cover_sub))
        story.append(Paragraph(title, cover_title))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            cover_sub
        ))
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=2, color=teal))
        story.append(Spacer(1, 0.8 * cm))

        # Parse the content line by line
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 0.2 * cm))
                continue

            if stripped.startswith("## "):
                story.append(Paragraph(stripped[3:], section_header))
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#e2e8f0")))
            elif stripped.startswith("### "):
                h3 = ParagraphStyle(
                    "H3", parent=styles["Heading2"],
                    fontSize=11, textColor=colors.HexColor("#3b82f6"),
                    spaceBefore=10, spaceAfter=2, fontName="Helvetica-Bold"
                )
                story.append(Paragraph(stripped[4:], h3))
            elif stripped.startswith(("- ", "• ", "* ")):
                text = stripped[2:].replace("**", "<b>", 1).replace("**", "</b>", 1)
                story.append(Paragraph(f"• {text}", bullet_style))
            elif stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                story.append(Paragraph(stripped, bullet_style))
            elif stripped.startswith("**") and stripped.endswith("**"):
                bold_style = ParagraphStyle(
                    "Bold", parent=body_style,
                    fontName="Helvetica-Bold", fontSize=10
                )
                story.append(Paragraph(stripped.replace("**", ""), bold_style))
            else:
                # Inline bold via reportlab markup
                text = stripped.replace("**", "<b>", 1).replace("**", "</b>", 1)
                story.append(Paragraph(text, body_style))

        # Footer
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "Generated by InsightForge AI — Autonomous Data Analytics Agent",
            cover_sub
        ))

        doc.build(story)

        # Store last report path for the download endpoint
        _last_report_path["path"] = pdf_path

        return (
            f"✅ PDF report exported successfully!\n"
            f"**File:** {pdf_filename}\n"
            f"**Path:** {pdf_path}\n\n"
            f"🔗 **Download:** Click the 'Export Report' button in the top bar, "
            f"or navigate to `/api/export-report` in your browser.\n"
            f"[[REPORT:{pdf_filename}]]"
        )

    except Exception as e:
        return f"Error generating PDF report: {str(e)}"
