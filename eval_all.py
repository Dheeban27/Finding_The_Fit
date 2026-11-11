"""
Evaluation for every experiment in the paper.  Works for our checkpoints AND the baselines:
  --model ckpt_stage2                        (ours)
  --model google/siglip-base-patch16-512     (vanilla SigLIP)
  --model openai/clip-vit-base-patch32       (CLIP)
  --model patrickjohncyh/fashion-clip        (FashionCLIP)

python eval_all.py split     --data triplets.json                      # makes train/val/test json
python eval_all.py hit5      --model M --data stage1_test.json         # text -> image, Hit@5
python eval_all.py zeroshot  --model M                                 # Fashion-MNIST P/R/F1
python eval_all.py retrieval --model M --data triplets_test.json       # mAP, Acc@1/10/20/50
"""
import argparse, json, random
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

def _t(o):
    """transformers>=5 returns an output object; older versions return a tensor"""
    return o if hasattr(o, "shape") else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load(name):
    m = AutoModel.from_pretrained(name).to(DEV).eval()
    p = AutoProcessor.from_pretrained(name)
    return m, p, ("siglip" in name.lower() or "stage" in name.lower())

@torch.no_grad()
def img_feats(m, p, paths, bs=64):
    out = []
    for i in tqdm(range(0, len(paths), bs), desc="images"):
        ims = [Image.open(x).convert("RGB") if isinstance(x, str) else x.convert("RGB") for x in paths[i:i+bs]]
        x = p(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
        with torch.autocast(DEV, dtype=torch.float16):
            out.append(F.normalize(_t(m.get_image_features(pixel_values=x)), dim=-1).float().cpu())
    return torch.cat(out)

@torch.no_grad()
def txt_feats(m, p, texts, sig, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        kw = dict(padding="max_length", max_length=64) if sig else dict(padding=True)
        t = p(text=texts[i:i+bs], truncation=True, return_tensors="pt", **kw).to(DEV)
        out.append(F.normalize(_t(m.get_text_features(**t)), dim=-1).float().cpu())
    return torch.cat(out)

# ---------------- split (by item, never by image) ----------------
def split(a):
    items = json.load(open(a.data)); random.Random(0).shuffle(items)
    n = len(items); cut = {"train": items[:int(.8*n)], "val": items[int(.8*n):int(.9*n)], "test": items[int(.9*n):]}
    for k, v in cut.items():
        json.dump(v, open(a.data.replace(".json", f"_{k}.json"), "w")); print(k, len(v))

# ---------------- text -> image, Hit@5 ----------------
def hit5(a):
    from fashion_ir import attrs_to_texts
    m, p, sig = load(a.model)
    items = json.load(open(a.data))                     # stage1-format test set
    valid = [set(attrs_to_texts(x["attrs"])) for x in items]
    I = img_feats(m, p, [x["image"] for x in items])
    rng = random.Random(0); qs = [rng.choice(sorted(v)) for v in valid if v]
    keep = [i for i, v in enumerate(valid) if v]
    T = txt_feats(m, p, qs, sig)
    top = (T @ I.T).topk(5, dim=1).indices
    # hit if ANY of the top-5 images is truly described by the query text (multi-positive)
    hits = [any(qs[r] in valid[j] for j in top[r].tolist()) for r in range(len(qs))]
    print(f"Hit@5 = {np.mean(hits):.4f}  ({len(qs)} queries)")

# ---------------- zero-shot classification ----------------
def zeroshot(a):
    from torchvision.datasets import FashionMNIST
    from sklearn.metrics import precision_recall_fscore_support
    m, p, sig = load(a.model)
    cls = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    T = txt_feats(m, p, [f"a photo of a {c}" for c in cls], sig)
    ds = FashionMNIST("data", train=False, download=True)
    imgs = [ds[i][0].convert("RGB").resize((224, 224)) for i in range(len(ds))]
    I = img_feats(m, p, imgs); pred = (I @ T.T).argmax(1).numpy()
    y = np.array([ds[i][1] for i in range(len(ds))])
    P, R, F1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    print(f"Precision {P:.4f}  Recall {R:.4f}  F1 {F1:.4f}")
    # DeepFashion-MultiModal: same procedure with its 24 category names + image paths.

# ---------------- street -> shop retrieval ----------------
def retrieval(a):
    m, p, sig = load(a.model)
    items = json.load(open(a.data))
    gal, gal_id, qry, qry_id = [], [], [], []
    for it in items:
        for u in it["user"]: qry.append(u); qry_id.append(it["item_id"])   # every user/street crop is a query
        for s in it["shop"]: gal.append(s); gal_id.append(it["item_id"])
    Q, G = img_feats(m, p, qry), img_feats(m, p, gal)
    order = (Q @ G.T).argsort(1, descending=True).numpy(); gal_id = np.array(gal_id)
    ks, acc, APs = (1, 10, 20, 50), {1: 0, 10: 0, 20: 0, 50: 0}, []
    for i in range(len(qry)):
        rel = (gal_id[order[i]] == qry_id[i])
        for k in ks: acc[k] += rel[:k].any()
        pos = np.where(rel)[0]
        APs.append(np.mean([(j + 1) / (r + 1) for j, r in enumerate(pos)]) if len(pos) else 0)   # Eq. 7
    print(f"mAP {np.mean(APs):.4f} | " + " ".join(f"Acc@{k} {acc[k]/len(qry):.4f}" for k in ks))   # Eq. 8

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "hit5", "zeroshot", "retrieval"])
    ap.add_argument("--model", default="ckpt_stage2"); ap.add_argument("--data")
    a = ap.parse_args()
    {"split": split, "hit5": hit5, "zeroshot": zeroshot, "retrieval": retrieval}[a.mode](a)
