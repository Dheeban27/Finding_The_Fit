# Finding The Fit (StyleFinder)

> **Composed Image Retrieval (CIR) for Fashion Discovery**  
> Multimodal fashion search powered by fine-tuned SigLIP vision-language models.

---

## 👔 Overview
Finding The Fit is an AI-powered visual fashion search system. Traditional image search only finds visual duplicates, while text-only search struggles with subtle visual details like cut, silhouette, and texture. 

StyleFinder combines **both modalities**:
1. **Query Photo:** Extracts the visual identity and silhouette of a garment ("same style as this").
2. **Text Condition:** Guides the modification ("...but in red color", "...with long sleeves").
3. **Composed Retrieval:** Scores and retrieves matching garments from a pre-indexed 9,872-item catalog.

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Dheeban27/Finding_The_Fit.git
cd Finding_The_Fit

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Locally (Windows 1-Click)
Double-click `start_app.bat` or run:
```bash
python cir_server.py --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in any browser.

---

## 🏗️ Architecture & Pipeline

* **Stage 1 (Attribute Alignment):** Fine-tunes SigLIP (`google/siglip-base-patch16-512`) on multi-label fashion attributes (color, pattern, material, sleeve length, collar type).
* **Stage 2 (Street-to-Shop Fine-Tuning):** Weighted Multi-Similarity / NT-Xent metric learning on consumer-to-catalog triplets.
* **Inference Engine:** FastAPI server embedding image queries and natural language conditions on CUDA GPU, computing cosine similarity and z-score blending against the precomputed 9,872 gallery index.

---

## 📂 Repository Structure

```
├── cir_server.py            # Main application server & modern UI
├── requirements.txt         # Core dependencies
├── start_app.bat            # 1-click launcher for server & permanent tunnel
├── run_local.bat            # Local server launcher
├── train_stage1_v2.py       # Stage 1 attribute alignment training
├── train_stage2.py          # Stage 2 street-to-shop metric learning
├── make_triplets.py         # Triplet generation pipeline
├── Dockerfile               # Container deployment specification
├── docker-compose.yml       # Docker Compose setup
├── DEPLOYMENT.md            # Deployment guide
├── StyleFinder_Documentation.md # Detailed project documentation
└── stage1*.json, triplets*.json # Dataset splits and metadata
```

---

## 📜 License
MIT License.
