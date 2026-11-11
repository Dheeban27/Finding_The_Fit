import argparse, collections, json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ckpt_stage2_b")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip_retrieval", action="store_true")
ap.add_argument("--retrieval_data", default="s2s_test.json")
ap.add_argument("--out", default="verify_retrieval.png")
a = ap.parse_args()
rng = random.Random(a.seed)

def count(d):
    if not os.path.isdir(d): return None
    return sum(1 for _, _, fs in os.walk(d) for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png")))

def show(ax, img, title="", fs=8):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=fs)

# ---------------- A. datasets ----------------
print("\n=== A. DATASETS ===")
for name, d in [("DeepFashion-MultiModal", "dataset/DeepFashion_MultiModal/images"),
                ("DeepFashion2 train images", "dataset/DeepFashion2/train/image"),
                ("male_fashion", "dataset/male_fashion"), ("female_fashion", "dataset/female_fashion")]:
    print(f"{name:28s} {count(d)} images  ({d})")
s2s = json.load(open("s2s_test.json"))
print(f"Street2Shop sample           {len(s2s)} products, {sum(len(x['user']) for x in s2s)} street crops, "
      f"{sum(len(x['shop']) for x in s2s)} shop photos")
tri = json.load(open("triplets_test.json"))
print(f"DeepFashion2 test split      {len(tri)} products")
stage1 = json.load(open("stage1.json", encoding="utf-8"))
print(f"stage1.json                  {len(stage1)} annotated images")

from torchvision.datasets import FashionMNIST
fm = FashionMNIST("data", train=False, download=True)
print(f"Fashion-MNIST test           {len(fm)} images")

fig, axs = plt.subplots(1, 6, figsize=(18, 3.6))
show(axs[0], Image.open(stage1[0]["image"]), "DeepFashion-MultiModal")
show(axs[1], Image.open(tri[0]["user"][0]), "DeepFashion2: consumer crop")
show(axs[2], Image.open(tri[0]["shop"][0]), "DeepFashion2: shop crop")
show(axs[3], Image.open(s2s[0]["user"][0]), "Street2Shop: street crop")
show(axs[4], Image.open(s2s[0]["shop"][0]), "Street2Shop: shop photo")
show(axs[5], fm[0][0].resize((140, 140)), f"Fashion-MNIST (class {fm[0][1]})")
plt.tight_layout(); plt.savefig("verify_datasets.png", dpi=110); plt.close()
print("saved verify_datasets.png")

# ---------------- B. 12 attributes ----------------
print("\n=== B. 12 ATTRIBUTES ===")
keys = ["subcategory", "color", "top_length", "sleeve_length", "fabric", "printing", "neckline",
        "collar", "hood", "zip_up", "button", "pocket"]
SRC = {"sleeve_length", "fabric", "printing", "neckline"}
for k in keys:
    c = collections.Counter(str(d["attrs"].get(k)) for d in stage1)
    empty = sum(v for x, v in c.items() if x in ("NONE", "FALSE", "None"))
    tag = "dataset labels + SigLIP" if k in SRC else "SigLIP zero-shot     "
    print(f"{k:14s} [{tag}] empty/false {100 * empty / len(stage1):5.1f}%  top: {c.most_common(4)}")

fig, axs = plt.subplots(2, 3, figsize=(13, 12))
for ax, d in zip(axs.ravel(), rng.sample(stage1, 6)):
    show(ax, Image.open(d["image"]), "")
    ax.set_xlabel("\n".join(f"{k}: {d['attrs'].get(k)}" for k in keys), fontsize=8, loc="left")
plt.tight_layout(); plt.savefig("verify_attributes.png", dpi=100); plt.close()
print("saved verify_attributes.png  (check each label against its photo)")

# ---------------- C. retrieval ----------------
if not a.skip_retrieval:
    print("\n=== C. RETRIEVAL (street query -> shop gallery, correct = same product) ===")
    from eval_all import load, img_feats
    gal, gid, qry, qid = [], [], [], []
    for it in json.load(open(a.retrieval_data)):
        for u in it["user"]: qry.append(u); qid.append(it["item_id"])
        for s in it["shop"]: gal.append(s); gid.append(it["item_id"])
    gid = np.array(gid); orders = {}
    for name, tag in (("google/siglip-base-patch16-224", "vanilla SigLIP"), (a.model, "ours")):
        m, p, _ = load(name)
        Q, Gm = img_feats(m, p, qry), img_feats(m, p, gal)
        orders[tag] = (Q @ Gm.T).argsort(1, descending=True).numpy()
        acc1 = np.mean([gid[orders[tag][i, 0]] == qid[i] for i in range(len(qry))])
        print(f"{tag:16s} Acc@1 = {acc1:.4f}  over {len(qry)} queries")
        del m; torch.cuda.empty_cache()

    picks = rng.sample(range(len(qry)), a.n)
    fig, axs = plt.subplots(a.n, 7, figsize=(17, 2.7 * a.n))
    for r, qi in enumerate(picks):
        show(axs[r, 0], Image.open(qry[qi]), "QUERY (street)")
        for c in range(6):
            tag, j = ("vanilla SigLIP", c) if c < 3 else ("ours", c - 3)
            gi = orders[tag][qi, j]; ok = gid[gi] == qid[qi]
            ax = axs[r, c + 1]
            show(ax, Image.open(gal[gi]), f"{tag} #{j + 1} {'OK' if ok else 'wrong'}")
            for sp in ax.spines.values():
                sp.set_edgecolor("green" if ok else "red"); sp.set_linewidth(4)
    plt.tight_layout(); plt.savefig(a.out, dpi=95); plt.close()
    print("saved verify_retrieval.png  (green = correct product, red = wrong)")