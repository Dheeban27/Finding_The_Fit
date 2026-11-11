import json, random, argparse
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

NAME = "google/siglip-base-patch16-224"
DEV = "cuda"

def ok(v): return bool(v) and str(v).upper() not in ("NONE", "FALSE")

def attrs_to_texts(a):
    """One short text per attribute (the multi-positive set for an image)."""
    sub = a["subcategory"].lower()
    t = [sub]
    if ok(a.get("color")): t.append(f'{a["color"].lower()} {sub}')
    if ok(a.get("fabric")): t.append(f'{a["fabric"].lower()} {sub}')
    sl = a.get("sleeve_length")
    if ok(sl): t.append(f"sleeveless {sub}" if sl.upper() == "SLEEVELESS" else f"{sl.lower()}-sleeve {sub}")
    if ok(a.get("printing")): t.append(f'{a["printing"].lower()} pattern {sub}')
    if ok(a.get("neckline")): t.append(f'{a["neckline"].lower()} neckline {sub}')
    if ok(a.get("top_length")): t.append(f'{a["top_length"].lower()} length {sub}')
    if a.get("collar") == "TRUE": t.append(f"{sub} with collar")
    if a.get("hood") == "TRUE": t.append(f"{sub} with hood")
    if a.get("pocket") == "TRUE": t.append(f"{sub} with pocket")
    if ok(a.get("zip_up")): t.append(f'{a["zip_up"].lower()} zip {sub}')
    if ok(a.get("button")): t.append(f'{a["button"].lower()} button {sub}')
    return t

class DS(Dataset):
    def __init__(self, path, proc):
        self.items = json.load(open(path, encoding="utf-8")); self.proc = proc
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; texts = attrs_to_texts(it["attrs"])
        img = Image.open(it["image"]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px, random.choice(texts), texts

def make_collate(proc):
    def collate(batch):
        px = torch.stack([b[0] for b in batch]); texts = [b[1] for b in batch]
        valid = [set(b[2]) for b in batch]
        # +1 if text j is a valid description of image i (multi-positive), else -1
        y = torch.tensor([[1. if t in valid[i] else -1. for t in texts] for i in range(len(texts))])
        ids = proc.tokenizer(texts, padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt")["input_ids"]
        return px, ids, y
    return collate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="stage1.json"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--freeze_text", action="store_true")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME).to(DEV)
    if a.freeze_text:
        for p in model.text_model.parameters(): p.requires_grad = False
    dl = DataLoader(DS(a.data, proc), batch_size=a.bs, shuffle=True, drop_last=True,
                    num_workers=0, collate_fn=make_collate(proc))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(a.epochs):
        run = 0.0
        for step, (px, ids, y) in enumerate(tqdm(dl, desc=f"epoch {ep+1}/{a.epochs}")):
            px, ids, y = px.to(DEV), ids.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixel_values=px, input_ids=ids).logits_per_image.float()
            loss = -F.logsigmoid(y * logits).sum() / px.size(0)      # SigLIP loss, Eq. 2/4
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if step % 50 == 0: print(f"  step {step} loss {loss.item():.4f}")
        print(f"epoch {ep+1} mean loss {run/len(dl):.4f}")
        model.save_pretrained("ckpt_stage1"); proc.save_pretrained("ckpt_stage1")
    print("saved to ckpt_stage1")