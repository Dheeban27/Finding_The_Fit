# StyleFinder – AI Clothing Retrieval

## Project Overview
StyleFinder is a FastAPI‑based web service that enables visual fashion search using a pretrained **ViT‑B/32** model from the **transformers** library. Users can upload images or capture photos via the web UI to retrieve similar clothing items from a local gallery.

## Architecture
```mermaid
flowchart TD
    subgraph Client
        UI[Web UI (HTML/JS)] -->|HTTP| API[FastAPI Server]
    end
    subgraph Server
        API --> Model[ViT‑B/32 Model]
        Model --> Gallery[Image Gallery (2 GB)]
    end
    subgraph Deployment
        API -->|tunnel| Cloudflare[Cloudflare Tunnel]
        Cloudflare --> Internet[Public URL]
    end
```

## Key Files & Their Roles
| File | Purpose |
|------|---------|
| `cir_server.py` | FastAPI application, serves UI and API endpoints, includes new camera UI button. |
| `Dockerfile` | Container definition for deployment; respects `PORT` env var. |
| `requirements.txt` | Python dependencies (fastapi, uvicorn, torch, transformers, timm, python‑pptx, fpdf). |
| `prepare_cloud_package.py` | Collects only required assets (model checkpoints, selected images) into `deploy_package/`. |
| `upload_to_hf.py` | Helper script to create a Hugging Face repo and upload the deployment package. |
| `start_background_server.vbs` | Windows VBS script that starts the FastAPI server **and** the Cloudflare tunnel, writes the public URL to `tunnel_url.log` and creates a clickable shortcut. |
| `stop_background_server.bat` | Terminates the server and tunnel processes. |
| `StyleFinder_Colab.ipynb` | Google Colab notebook that launches the service in the cloud (free alternative to a 24/7 tunnel). |
| `StyleFinder_Documentation.md` | This markdown file – source for the PDF documentation. |
| `generate_docs.py` | Python script that converts the markdown to a PDF (`StyleFinder_Documentation.pdf`) and creates a PowerPoint deck (`StyleFinder_Presentation.pptx`). |

## Tools & Libraries Used
- **FastAPI** – web framework for the API.
- **Uvicorn** – ASGI server.
- **torch & transformers** – model inference.
- **timm** – vision model utilities.
- **cloudflared** – provides a public HTTPS URL via Cloudflare Tunnel.
- **python‑pptx** – programmatic PowerPoint generation.
- **fpdf** – lightweight PDF generation from text.
- **Docker** – containerization for reproducible deployment.
- **Google Colab** – free compute environment for a 24/7‑like demo.

## Deployment Options
1. **Local + Cloudflare Tunnel** – Run `start_background_server.vbs`; a public URL is created while the laptop stays on.
2. **Docker Container** – Build with `docker build . -t stylefinder` and run with `docker run -p 8000:8000 stylefinder`.
3. **Google Colab Notebook** – Execute `StyleFinder_Colab.ipynb` to launch a temporary server with a tunnel.
4. **Future Cloud Hosting** – Upload `deploy_package/` to Hugging Face Spaces (requires a Pro plan) or other cloud providers.

## Generating Documentation & Presentation
Running the following command will produce both artifacts:
```bash
python generate_docs.py
```
- `StyleFinder_Documentation.pdf` – compiled PDF version of this markdown.
- `StyleFinder_Presentation.pptx` – a ready‑to‑present slide deck.

## Next Steps
- Consider a paid Hugging Face Space for true 24/7 hosting.
- Replace the static image gallery with a cloud‑based storage bucket for scalability.
- Extend the UI to support mobile cameras directly (e.g., via `<input type="file" capture="environment">`).

