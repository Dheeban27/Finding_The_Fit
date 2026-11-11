# StyleFinder Deployment Guide

This project can be deployed in multiple ways depending on your target environment.

---

### 1. Instant Public Access (No Cloud Fees, Live Public HTTPS)
If you want to share your running application with others or test it on your mobile phone without uploading multi-gigabyte models to cloud servers:

1. **Start the local server**:
   ```powershell
   .\run_local.bat
   # or manually:
   # .venv\Scripts\activate
   # python cir_server.py --host 0.0.0.0 --port 8000
   ```
2. **Open a public tunnel**:
   ```powershell
   .\share_public.bat
   # or manually:
   # npx localtunnel --port 8000
   ```
   This gives you an instant `https://*.loca.lt` URL accessible globally.

---

### 2. Local Network (LAN / Wi-Fi) Deployment
To allow any device connected to the same Wi-Fi router (phones, tablets, other PCs) to access the app:

1. Run the server bound to `0.0.0.0`:
   ```powershell
   .venv\Scripts\activate
   python cir_server.py --host 0.0.0.0 --port 8000
   ```
2. Find your local IP address (e.g. `10.215.130.236` or run `ipconfig`).
3. Open `http://<your-local-ip>:8000` in any mobile browser or laptop on the same network.

---

### 3. Docker Deployment (Local or Remote Server)
The project includes a ready-to-use `Dockerfile` and `docker-compose.yml`:

```bash
docker compose up --build -d
```
The container mounts `ckpt_stage1`, `ckpt_stage2_b`, `dataset`, and `cir_index.pt` as volumes so you don't need to rebuild the image when updating dataset assets.

---

### 4. Cloud Deployment (Hugging Face Spaces / GPU Cloud / VPS)
- **Hugging Face Spaces**:
  - Create a new Docker Space on [huggingface.co/spaces](https://huggingface.co).
  - Push the repository files, `Dockerfile`, and `requirements.txt`.
  - Checkpoints (`ckpt_stage1` and `ckpt_stage2_b`) can be uploaded using Git LFS or Hugging Face Hub model repositories.
- **GPU VPS (RunPod, Vast.ai, AWS EC2, GCP)**:
  - Clone the repo, install NVIDIA Docker, and run:
    ```bash
    docker compose up --build -d
    ```

