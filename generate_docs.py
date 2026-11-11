# generate_docs.py
"""Utility script to generate project documentation PDF and a PowerPoint deck.

- Reads ``StyleFinder_Documentation.md`` (the markdown source).
- Creates ``StyleFinder_Documentation.pdf`` using **FPDF** (simple text layout).
- Creates ``StyleFinder_Presentation.pptx`` using **python-pptx** with a few
  auto‑generated slides (title, overview, key files, tools, deployment options).

Run with ``python generate_docs.py`` from the project root.
"""
import os
from pathlib import Path

# ---- PDF generation -----------------------------------------------------
try:
    from fpdf import FPDF
except ImportError:
    raise RuntimeError("fpdf is not installed. Add it to requirements.txt and install it.")

md_path = Path(__file__).parent / "StyleFinder_Documentation.md"
pdf_path = Path(__file__).parent / "StyleFinder_Documentation.pdf"

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Using default Helvetica font; custom TTF not required.
        # self.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)  # removed to avoid missing font error

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "StyleFinder - AI Clothing Retrieval", ln=True, align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def chapter_body(self, text):
        self.set_font("Helvetica", "", 11)
        for line in text.splitlines():
            # simple handling – wrap long lines automatically
            self.multi_cell(0, 5, line)
        self.ln()

pdf = MarkdownPDF()
pdf.add_page()

if not md_path.is_file():
    raise FileNotFoundError(f"Markdown source not found at {md_path}")

with open(md_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Very naive markdown parsing – headings become chapter titles
buffer = []
for line in lines:
    # Replace non‑ASCII characters with ASCII equivalents for PDF compatibility
    line = line.replace('–', '-').replace('—', '-').replace('‑', '-').encode('ascii', 'ignore').decode()
    stripped = line.strip()
    if stripped.startswith("# ") and not stripped.startswith("## "):
        # top‑level title – already used in header, skip adding a new page
        continue
    if stripped.startswith("## "):
        # flush previous paragraph
        if buffer:
            pdf.chapter_body("\n".join(buffer))
            buffer = []
        title = stripped.replace("## ", "")
        pdf.chapter_title(title)
    else:
        buffer.append(line.rstrip())
# flush remaining text
if buffer:
    pdf.chapter_body("\n".join(buffer))

pdf.output(str(pdf_path))
print(f"PDF generated at {pdf_path}")

# ---- PowerPoint generation ---------------------------------------------
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except ImportError:
    raise RuntimeError("python-pptx is not installed. Add it to requirements.txt and install it.")

ppt_path = Path(__file__).parent / "StyleFinder_Presentation.pptx"
prs = Presentation()

# Helper to add a title slide
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "StyleFinder – AI Clothing Retrieval"
subtitle.text = "Visual fashion search demo"

# Overview slide
bullet_slide_layout = prs.slide_layouts[1]
slide = prs.slides.add_slide(bullet_slide_layout)
shapes = slide.shapes
shapes.title.text = "Project Overview"
body = shapes.placeholders[1].text_frame
body.text = "• FastAPI backend serving a ViT‑B/32 model"
for pt in ["• Cloudflare Tunnel for public HTTPS", "• Docker container ready for cloud hosts", "• Google Colab notebook for free demo"]:
    p = body.add_paragraph()
    p.text = pt
    p.level = 0

# Key files slide
slide = prs.slides.add_slide(bullet_slide_layout)
shapes = slide.shapes
shapes.title.text = "Key Files & Roles"
body = shapes.placeholders[1].text_frame
files = [
    "cir_server.py – FastAPI app + UI",
    "Dockerfile – container definition",
    "requirements.txt – Python deps",
    "prepare_cloud_package.py – build deploy zip",
    "upload_to_hf.py – HF upload helper",
    "start_background_server.vbs – launch server & tunnel",
    "stop_background_server.bat – stop processes",
    "StyleFinder_Colab.ipynb – Colab demo",
    "generate_docs.py – PDF & PPT generator",
]
for f in files:
    p = body.add_paragraph()
    p.text = f"• {f}"
    p.level = 0

# Tools slide
slide = prs.slides.add_slide(bullet_slide_layout)
shapes = slide.shapes
shapes.title.text = "Tools & Libraries"
body = shapes.placeholders[1].text_frame
tools = [
    "FastAPI, Uvicorn",
    "torch, transformers, timm",
    "cloudflared (Cloudflare Tunnel)",
    "python‑pptx (PPT generation)",
    "fpdf (PDF generation)",
    "Docker",
    "Google Colab",
]
for t in tools:
    p = body.add_paragraph()
    p.text = f"• {t}"
    p.level = 0

# Deployment slide
slide = prs.slides.add_slide(bullet_slide_layout)
shapes = slide.shapes
shapes.title.text = "Deployment Options"
body = shapes.placeholders[1].text_frame
options = [
    "Local + Cloudflare Tunnel (requires laptop on)",
    "Docker container (any host)",
    "Google Colab notebook (free, temporary)",
    "Hugging Face Space – Pro plan required",
]
for o in options:
    p = body.add_paragraph()
    p.text = f"• {o}"
    p.level = 0

prs.save(str(ppt_path))
print(f"PowerPoint generated at {ppt_path}")
