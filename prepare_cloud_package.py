"""
Prepare a clean, lightweight, self-contained deployment package for 24/7 Cloud hosting
(e.g., Hugging Face Spaces, Cloud VPS, or Docker container).
"""
import os, shutil, torch

OUT_DIR = "deploy_package"
print(f"Creating clean deployment package in ./{OUT_DIR}...")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load index to find only referenced image paths
print("Analyzing cir_index.pt for referenced gallery images...")
d = torch.load("cir_index.pt", map_location="cpu")
unique_paths = list(set(d["paths"]))
print(f"Found {len(unique_paths)} unique image paths in gallery index.")

# 2. Copy referenced images preserving relative directories
copied_count = 0
for p in unique_paths:
    if os.path.exists(p):
        target_path = os.path.join(OUT_DIR, p)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(p, target_path)
        copied_count += 1
print(f"Copied {copied_count} gallery images into {OUT_DIR}.")

# 3. Copy models
for ckpt in ["ckpt_stage1", "ckpt_stage2_b"]:
    target_ckpt = os.path.join(OUT_DIR, ckpt)
    print(f"Copying {ckpt}...")
    shutil.copytree(ckpt, target_ckpt, dirs_exist_ok=True)

# 4. Copy index and core code files
for f in ["cir_index.pt", "cir_server.py", "requirements.txt", "Dockerfile", ".dockerignore"]:
    print(f"Copying {f}...")
    shutil.copy2(f, os.path.join(OUT_DIR, f))

# 5. Create Hugging Face Space README.md
hf_readme = """---
title: StyleFinder - Composed Fashion Retrieval
emoji: 👗
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# StyleFinder — AI Composed Fashion Retrieval
Multimodal clothing search with image queries and text modification conditions.
"""
with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(hf_readme)

print(f"\n[DONE] Deployment package ready in ./{OUT_DIR}!")

