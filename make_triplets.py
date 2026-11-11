import argparse, json, os, glob
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="crops")
ap.add_argument("--json", default="triplets.json")
ap.add_argument("--min_size", type=int, default=64)
ap.add_argument("--max_files", type=int, default=20000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

groups = defaultdict(lambda: {"user": [], "shop": []})
files = sorted(glob.glob(f"{a.root}/annos/*.json"))[: a.max_files]
for path in tqdm(files):
    try:
        anno = json.load(open(path))
        name = os.path.splitext(os.path.basename(path))[0]
        img_path = f"{a.root}/image/{name}.jpg"
        src, pid = anno["source"], anno["pair_id"]
        items = [(k, it) for k, it in anno.items()
                 if k.startswith("item") and it.get("style", 0) > 0]
        if not items or not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        for k, it in items:
            x1, y1, x2, y2 = it["bounding_box"]
            if x2 - x1 < a.min_size or y2 - y1 < a.min_size:
                continue
            out = f"{a.out}/{name}_{k}.jpg"
            img.crop((x1, y1, x2, y2)).save(out, quality=90)
            groups[f'{pid}_{it["style"]}'][src].append(out)
        del img
    except MemoryError:
        print("memory error at", path, "- saving what we have"); break
    except Exception as e:
        print("skip", path, e)

items = [{"item_id": k, "user": v["user"], "shop": v["shop"]}
         for k, v in groups.items() if len(v["user"]) >= 2 and len(v["shop"]) >= 1]
json.dump(items, open(a.json, "w"))
print(f"{len(items)} usable items (need >=2 user crops and >=1 shop crop)")