"""
Turn dataset/Street2Shop/street2shop_test_sample.parquet into an evaluation set.
Street photos are cropped with the given box (left, top, width, height).

  python prepare_street2shop_eval.py
  python eval_all.py retrieval --model ckpt_stage2 --data s2s_test.json
"""
import argparse, io, json, os
from collections import defaultdict
import pandas as pd
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=r"dataset\Street2Shop\street2shop_test_sample.parquet")
ap.add_argument("--out", default="s2s_eval")
ap.add_argument("--json", default="s2s_test.json")
a = ap.parse_args()
os.makedirs(f"{a.out}/street", exist_ok=True); os.makedirs(f"{a.out}/shop", exist_ok=True)

d = pd.read_parquet(a.parquet)
groups = defaultdict(lambda: {"user": [], "shop": [], "category": None})
seen_shop = {}
for i, r in tqdm(d.iterrows(), total=len(d)):
    try:
        st = Image.open(io.BytesIO(r["street_photo_image"]["bytes"])).convert("RGB")
        sh = Image.open(io.BytesIO(r["shop_photo_image"]["bytes"])).convert("RGB")
    except Exception as e:
        print("skip", i, e); continue
    l, t, w, h = r["left"], r["top"], r["width"], r["height"]
    box = (max(0, int(l)), max(0, int(t)), min(st.width, int(l + w)), min(st.height, int(t + h)))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16: continue            # skip tiny/invalid boxes
    sp = f'{a.out}/street/{r["street_photo_id"]}_{i}.jpg'; st.crop(box).save(sp)
    shp = seen_shop.get(r["shop_photo_id"])
    if shp is None:
        shp = f'{a.out}/shop/{r["shop_photo_id"]}.jpg'; sh.save(shp); seen_shop[r["shop_photo_id"]] = shp
    g = groups[str(r["product_id"])]
    g["user"].append(sp); g["category"] = r["category"]
    if shp not in g["shop"]: g["shop"].append(shp)

items = [{"item_id": k, **v} for k, v in groups.items() if v["user"] and v["shop"]]
json.dump(items, open(a.json, "w"))
print(f"{len(items)} products | {sum(len(x['user']) for x in items)} street queries | {len(seen_shop)} shop images")
