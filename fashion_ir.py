"""
Fashion Image Retrieval: SigLIP attribute pre-training (Stage 1)
+ Weighted Multi NT-Xent street-to-shop fine-tuning (Stage 2)
Reimplementation sketch of Kim et al., IEEE Access 2025.

pip install torch torchvision transformers pillow matplotlib tqdm
Usage:
  python fashion_ir.py stage1   --data stage1.json
  python fashion_ir.py stage2   --data triplets.json --ckpt ckpt_stage1
  python fashion_ir.py index    --ckpt ckpt_stage2 --shop_dir shop_images/
  python fashion_ir.py query    --ckpt ckpt_stage2 --query q.jpg --out result.png
"""
import argparse, json, random, os, glob
NW = 0 if os.name == "nt" else 4   # Windows: no worker processes
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SiglipModel, SiglipProcessor
from tqdm import tqdm

BASE = os.environ.get("SIGLIP_BASE", "google/siglip-base-patch16-224")  # paper: 512 model; 224 fits a 6GB GPU
def _t(o):
    """transformers>=5 returns an output object; older versions return a tensor"""
    return o if hasattr(o, "shape") else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# 0. VLM ATTRIBUTE ANNOTATION (data preparation)
# ----------------------------------------------------------------------
ANNOTATION_PROMPT = """Look at the main garment and answer ONLY with JSON:
{"subcategory":"", "color":"", "top_length":"", "sleeve_length":"",
 "fabric":"", "printing":"", "neckline":"", "collar":"TRUE/FALSE",
 "hood":"TRUE/FALSE", "zip_up":"FULL/HALF/NONE", "button":"FULL/HALF/NONE",
 "pocket":"TRUE/FALSE"}"""
# Send this prompt + image to any strong VLM (Claude, GPT-4o, Gemini...) and
# save results as stage1.json:
# [{"image": "img/0001.jpg", "attrs": {...12 attributes...}}, ...]

def attrs_to_texts(a):
    """One short text per attribute (multi-positive pairs)."""
    sub = a["subcategory"].lower()
    t = [f'{a["color"].lower()} {sub}',
         f'{a["fabric"].lower()} {sub}',
         f'{a["sleeve_length"].lower()}-sleeve {sub}',
         f'{a["printing"].lower()} pattern {sub}',
         f'{a["neckline"].lower()} neckline {sub}',
         f'{a["top_length"].lower()} length {sub}']
    if a["collar"] == "TRUE": t.append(f"{sub} with collar")
    if a["hood"] == "TRUE": t.append(f"{sub} with hood")
    if a["zip_up"] != "NONE": t.append(f'{a["zip_up"].lower()} zip {sub}')
    if a["button"] != "NONE": t.append(f'{a["button"].lower()} button {sub}')
    if a["pocket"] == "TRUE": t.append(f"{sub} with pocket")
    return [x for x in t if "none" not in x]

# ----------------------------------------------------------------------
# 1. STAGE 1 : multi-positive SigLIP pre-training
# ----------------------------------------------------------------------
class AttrDataset(Dataset):
    def __init__(self, path):
        self.items = json.load(open(path))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        text = random.choice(attrs_to_texts(it["attrs"]))   # sample one attribute text
        all_texts = attrs_to_texts(it["attrs"])
        return img, text, all_texts

def collate_stage1(batch, proc):
    imgs, texts, alls = zip(*batch)
    enc = proc(text=list(texts), images=list(imgs), padding="max_length",
               max_length=64, truncation=True, return_tensors="pt")
    # y[i,j]=1 if text j is ANY valid description of image i  -> multi-positive
    y = torch.tensor([[1. if texts[j] in alls[i] else -1. for j in range(len(texts))]
                      for i in range(len(texts))])
    return enc, y

def sigmoid_loss(img_emb, txt_emb, scale, bias, y):
    """SigLIP loss: independent binary decision for each (image,text) pair."""
    logits = scale * img_emb @ txt_emb.T + bias
    return -F.logsigmoid(y * logits).sum() / img_emb.size(0)

def train_stage1(a):
    model = SiglipModel.from_pretrained(BASE).to(DEV)
    proc = SiglipProcessor.from_pretrained(BASE)
    dl = DataLoader(AttrDataset(a.data), batch_size=a.bs, shuffle=True, num_workers=NW,
                    collate_fn=lambda b: collate_stage1(b, proc))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(a.epochs):
        for enc, y in tqdm(dl, desc=f"stage1 ep{ep}"):
            enc = {k: v.to(DEV) for k, v in enc.items()}; y = y.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16):
                out = model(**enc)
                ie = F.normalize(out.image_embeds, dim=-1)
                te = F.normalize(out.text_embeds, dim=-1)
                loss = sigmoid_loss(ie, te, model.logit_scale.exp(), model.logit_bias, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        print(f"epoch {ep} loss {loss.item():.4f}")
    model.save_pretrained("ckpt_stage1"); proc.save_pretrained("ckpt_stage1")

# ----------------------------------------------------------------------
# 2. STAGE 2 : Weighted Multi NT-Xent (street -> user / shop)
# ----------------------------------------------------------------------
class TripletDS(Dataset):
    """json: [{"item_id":..,"user":[paths...],"shop":[paths...]}]  (Street2Shop style)
    Needs >=2 user photos per item (query + hard positive) and >=1 shop photo."""
    def __init__(self, path, proc):
        self.items = [x for x in json.load(open(path)) if len(x["user"]) >= 2 and x["shop"]]
        self.proc = proc
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        q, h = random.sample(it["user"], 2)
        p = random.choice(it["shop"])
        load = lambda f: self.proc(images=Image.open(f).convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return load(q), load(h), load(p)

def nt_xent(za, zb, tau):
    """Anchor i's positive is zb[i]; every other sample in the batch is a negative
    (no hard-negative mining needed)."""
    za, zb = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
    N = za.size(0)
    z = torch.cat([za, zb], 0)                               # 2N samples
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                        # exclude self (1[k!=i])
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)

def train_stage2(a):
    model = SiglipModel.from_pretrained(a.ckpt).to(DEV)
    proc = SiglipProcessor.from_pretrained(a.ckpt)
    for p in model.text_model.parameters(): p.requires_grad = False   # freeze text tower
    enc = model.vision_model
    dl = DataLoader(TripletDS(a.data, proc), batch_size=a.bs, shuffle=True, num_workers=NW, drop_last=True)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    TAU, WP = 0.5, 0.5
    embed = lambda x: _t(model.get_image_features(pixel_values=x))
    for ep in range(a.epochs):
        for q, h, p in tqdm(dl, desc=f"stage2 ep{ep}"):
            q, h, p = q.to(DEV), h.to(DEV), p.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16):
                za, zh, zp = embed(q), embed(h), embed(p)
                loss = nt_xent(za, zh, TAU) + WP * nt_xent(za, zp, TAU)   # Eq. (6)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step(); print(f"epoch {ep} loss {loss.item():.4f}")
    model.save_pretrained("ckpt_stage2"); proc.save_pretrained("ckpt_stage2")

# ----------------------------------------------------------------------
# 3. INDEXING + RETRIEVAL + VISUALISATION
# ----------------------------------------------------------------------
@torch.no_grad()
def embed_paths(model, proc, paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
        with torch.autocast(DEV, dtype=torch.float16):
            out.append(F.normalize(_t(model.get_image_features(pixel_values=x)), dim=-1).float().cpu())
    return torch.cat(out)

def build_index(a):
    model = SiglipModel.from_pretrained(a.ckpt).to(DEV).eval()
    proc = SiglipProcessor.from_pretrained(a.ckpt)
    paths = sorted(glob.glob(os.path.join(a.shop_dir, "*.jpg")) + glob.glob(os.path.join(a.shop_dir, "*.png")))
    torch.save({"paths": paths, "emb": embed_paths(model, proc, paths)}, "shop_index.pt")
    print(f"indexed {len(paths)} shop images")

def query(a, k=5):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    model = SiglipModel.from_pretrained(a.ckpt).to(DEV).eval()
    proc = SiglipProcessor.from_pretrained(a.ckpt)
    idx = torch.load("shop_index.pt")
    q = embed_paths(model, proc, [a.query])
    scores = (q @ idx["emb"].T)[0]
    top = scores.topk(k)
    fig, ax = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    ax[0].imshow(Image.open(a.query)); ax[0].set_title("INPUT (street photo)")
    for j, (s, i) in enumerate(zip(top.values, top.indices)):
        ax[j + 1].imshow(Image.open(idx["paths"][i])); ax[j + 1].set_title(f"#{j+1}  cos={s:.3f}")
    for x in ax: x.axis("off")
    plt.tight_layout(); plt.savefig(a.out, dpi=150); print("saved", a.out)

def text_query(a, k=5):
    """Stage-1 bonus: text-to-image search, e.g. 'brown wool blazer'."""
    model = SiglipModel.from_pretrained(a.ckpt).to(DEV).eval()
    proc = SiglipProcessor.from_pretrained(a.ckpt)
    idx = torch.load("shop_index.pt")
    t = proc(text=[a.text], padding="max_length", max_length=64, return_tensors="pt").to(DEV)
    with torch.no_grad(): te = F.normalize(_t(model.get_text_features(**t)), dim=-1).cpu()
    print([idx["paths"][i] for i in (te @ idx["emb"].T)[0].topk(k).indices])

def evaluate(model_ckpt, queries, gt, idx_path="shop_index.pt", ks=(1, 10, 20, 50)):
    """queries: list of paths; gt: list of sets of correct shop paths. Returns Acc@k."""
    model = SiglipModel.from_pretrained(model_ckpt).to(DEV).eval()
    proc = SiglipProcessor.from_pretrained(model_ckpt); idx = torch.load(idx_path)
    q = embed_paths(model, proc, queries); rank = (q @ idx["emb"].T).argsort(1, descending=True)
    return {f"Acc@{k}": sum(any(idx["paths"][j] in gt[i] for j in rank[i, :k]) for i in range(len(queries))) / len(queries)
            for k in ks}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage1", "stage2", "index", "query", "text"])
    ap.add_argument("--data"); ap.add_argument("--ckpt", default="ckpt_stage2")
    ap.add_argument("--shop_dir"); ap.add_argument("--query"); ap.add_argument("--text")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)   # paper Stage 1: 5e-4 (use ~1e-5 if your batch is small)
    a = ap.parse_args()
    {"stage1": train_stage1, "stage2": train_stage2, "index": build_index,
     "query": query, "text": text_query}[a.mode](a)
