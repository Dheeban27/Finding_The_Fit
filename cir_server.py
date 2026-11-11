"""
StyleFinder - composed fashion retrieval (image + REQUIRED text condition), full-stack demo.

  pip install fastapi uvicorn python-multipart
  python cir_server.py --build --limit 1500     # embed the gallery once (two models)
  python cir_server.py                          # open http://127.0.0.1:8000

How it works
  1. The query PHOTO is embedded with the street-to-shop model (--img_model, default ckpt_stage2_b)
     and the TOP-N most visually similar garments are taken as candidates ("same kind of clothes").
  2. The text CONDITION is embedded with a text-aligned model (--txt_model, default ckpt_stage1)
     and every candidate is scored against it ("...but red", "...with long sleeves").
  3. Both scores are z-normalised inside the candidate pool and mixed with the "text influence" slider.
"""
import argparse, glob, io, json, os, random, time
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from transformers import AutoModel, AutoProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"img_model": "ckpt_stage2_b", "txt_model": "ckpt_stage1",
       "gallery": ["s2s_test.json", "triplets_test.json", "dataset/male_fashion", "dataset/female_fashion"],
       "limit": 1500, "index": "cir_index.pt"}
M = {}
G = {"paths": [], "A": None, "B": None, "gender": [], "est": []}


def _t(o):
    """transformers>=5 returns an output object; older versions return a tensor"""
    return o if hasattr(o, "shape") else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])


def get(name):
    if name not in M:
        M[name] = (AutoModel.from_pretrained(name).to(DEV).eval(), AutoProcessor.from_pretrained(name))
    return M[name]


@torch.no_grad()
def img_emb(name, images):
    m, p = get(name)
    x = p(images=images, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
        f = _t(m.get_image_features(pixel_values=x))
    return F.normalize(f.float(), dim=-1).cpu()


@torch.no_grad()
def txt_emb(name, texts):
    m, p = get(name)
    ids = p.tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")["input_ids"].to(DEV)
    f = _t(m.get_text_features(input_ids=ids))
    return F.normalize(f.float(), dim=-1).cpu()


# ------------------------------------------------------------------ gallery
def gender_from_path(p):
    parts = [x.lower() for x in p.replace("\\", "/").split("/")]
    if "male_fashion" in parts: return "men"
    if "female_fashion" in parts: return "women"
    fn = parts[-1]
    if fn.startswith("men-"): return "men"
    if fn.startswith("women-"): return "women"
    return None


def collect(sources, limit):
    rng, out, seen = random.Random(0), [], set()
    for s in sources:
        paths = []
        if s.endswith(".json") and os.path.exists(s):
            for it in json.load(open(s, encoding="utf-8")):
                paths += it.get("shop", [])
        elif os.path.isdir(s):
            for ext in ("jpg", "jpeg", "png"):
                paths += glob.glob(os.path.join(s, "**", "*." + ext), recursive=True)
        else:
            print("skipping missing source:", s)
        paths = [p for p in dict.fromkeys(paths) if os.path.exists(p)]
        if limit and not s.endswith(".json") and len(paths) > limit:
            paths = rng.sample(paths, limit)
        print(f"{s}: {len(paths)} images")
        for p in paths:
            if p not in seen:
                seen.add(p); out.append(p)
    return out


FULL_BODY = ("men-", "women-")
KEY_TAG = "parts-v1"


def entries_for(path, im):
    """whole image + (for full-body model photos) approximate top and bottom garment regions"""
    out = [(im, None, "full")]
    if os.path.basename(path).lower().startswith(FULL_BODY) and im.height > 1.2 * im.width:
        w, h = im.size
        for part, (y0, y1) in (("top", (0.14, 0.55)), ("bottom", (0.45, 0.93))):
            box = (0.05, y0, 0.95, y1)
            out.append((im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))), box, part))
    return out


def build_index():
    paths = collect(CFG["gallery"], CFG["limit"])
    if not paths:
        raise RuntimeError("No gallery images found.")
    good, boxes, parts, A, B = [], [], [], [], []
    same = CFG["img_model"] == CFG["txt_model"]
    for i in range(0, len(paths), 10):
        ims, meta = [], []
        for p in paths[i:i + 10]:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            for crop, box, part in entries_for(p, im):
                ims.append(crop); meta.append((p, box, part))
        if not ims:
            continue
        a = img_emb(CFG["img_model"], ims)
        A.append(a); B.append(a if same else img_emb(CFG["txt_model"], ims))
        for p, box, part in meta:
            good.append(p); boxes.append(box); parts.append(part)
        print(f"indexed {len(good)} entries ({min(i + 10, len(paths))}/{len(paths)} images)", end="\r")
    A, B = torch.cat(A), torch.cat(B)
    gender = [gender_from_path(p) for p in good]
    est = [False] * len(good)
    unk = [i for i, g in enumerate(gender) if g is None]
    if unk:
        t = txt_emb(CFG["txt_model"], ["a photo of men's clothing", "a photo of women's clothing"])
        sc = B[unk] @ t.T
        for j, i in enumerate(unk):
            gender[i] = "men" if sc[j, 0] > sc[j, 1] else "women"; est[i] = True
    G.update(paths=good, box=boxes, part=parts, A=A, B=B, gender=gender, est=est)
    torch.save({"paths": good, "box": boxes, "part": parts, "A": A, "B": B, "gender": gender, "est": est,
                "key": [CFG["img_model"], CFG["txt_model"], CFG["gallery"], CFG["limit"], KEY_TAG]}, CFG["index"])
    print(f"\nindex saved: {len(good)} entries ({parts.count('top')} top, {parts.count('bottom')} bottom crops) -> {CFG['index']}")


def init():
    get(CFG["img_model"]); get(CFG["txt_model"])
    key = [CFG["img_model"], CFG["txt_model"], CFG["gallery"], CFG["limit"], KEY_TAG]
    if os.path.exists(CFG["index"]):
        d = torch.load(CFG["index"])
        if d.get("key") == key:
            G.update(paths=d["paths"], box=d["box"], part=d["part"], A=d["A"], B=d["B"], gender=d["gender"], est=d["est"]); return
        print("index settings changed, rebuilding")
    build_index()


# ------------------------------------------------------------------ search
import re


def product_key(p):
    fn = os.path.basename(p)
    m = re.match(r"^(.*?-id_\d+-\d+)_", fn)
    return m.group(1) if m else fn


TOP_RE = re.compile(r"\b(?:shirt|t-?shirt|tee|top|blouse|sweater|jumper|jacket|coat|hoodie|cardigan|polo|vest|tank|blazer|sweatshirt)s?\b")
BOTTOM_RE = re.compile(r"\b(?:pant|trouser|jean|short|skirt|legging|jogger|chino|bottom)s?\b")


def infer_part(text):
    t = re.sub(r"short[- ]?sleeve\w*", " ", text.lower())
    top, bot = bool(TOP_RE.search(t)), bool(BOTTOM_RE.search(t))
    return "top" if top and not bot else "bottom" if bot and not top else "any"


def zs(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)


def search(qimg, cond, k, gender, alpha, pool, part="auto"):
    idx = [i for i, g in enumerate(G["gender"]) if gender == "all" or (g == gender and not G["est"][i])]
    if not idx:
        raise HTTPException(404, "No gallery images for this filter")
    used = infer_part(cond) if part == "auto" else part
    if used in ("top", "bottom"):
        idx = [i for i in idx if G["part"][i] == used]
        if not idx:
            raise HTTPException(404, "No gallery images for this filter")
    else:
        used = "any"
    idx_t = torch.tensor(idx)
    qa = img_emb(CFG["img_model"], [qimg])[0]
    qt = txt_emb(CFG["txt_model"], [cond])[0]
    s_img = G["A"][idx_t] @ qa
    pool = max(1, min(pool, len(idx)))
    top = s_img.topk(pool)
    cand, si = idx_t[top.indices], top.values            # step 1: visually similar candidates
    st = G["B"][cand] @ qt
    all_t = G["B"][idx_t] @ qt                           # text score of EVERY gallery item (for the percentile)
    final = (1 - alpha) * zs(si) + alpha * zs(st)
    order = final.topk(pool).indices.tolist()
    out, seen = [], set()
    for o in order:
        i = int(cand[o])
        key = product_key(G["paths"][i])
        if key in seen:                                  # skip other views of the same product
            continue
        seen.add(key)
        out.append({"id": i, "url": f"/img/{i}", "name": os.path.basename(G["paths"][i]),
                    "gender": G["gender"][i], "est": G["est"][i],
                    "img_score": round(float(si[o]), 3), "txt_score": round(float(st[o]), 4),
                    "cond_pct": int(round(100 * float((all_t < st[o]).float().mean())))})
        if len(out) >= k:
            break
    return out, used


@asynccontextmanager
async def lifespan(app):
    init()
    yield


app = FastAPI(title="StyleFinder", lifespan=lifespan)

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">👔</text></svg>"""


@app.get("/favicon.ico")
def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/status")
def status():
    return {"img_model": CFG["img_model"], "txt_model": CFG["txt_model"], "device": DEV,
            "gallery": len(G["paths"]), "men": G["gender"].count("men"), "women": G["gender"].count("women")}


@app.post("/api/search")
async def api_search(file: UploadFile = File(...), text: str = Form(""), k: int = Form(6),
                     gender: str = Form("all"), alpha: float = Form(0.5), pool: int = Form(100),
                     part: str = Form("auto")):
    if not text.strip():
        raise HTTPException(400, "Please describe the condition (text is required)")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that file as an image")
    t0 = time.time()
    res, used = search(img, text.strip(), k, gender if gender in ("all", "men", "women") else "all",
                 min(max(alpha, 0.0), 1.0), pool, part)
    return {"results": res, "part": used, "ms": round((time.time() - t0) * 1000, 1)}


@app.get("/img/{i}")
def gallery_image(i: int):
    if i < 0 or i >= len(G["paths"]):
        raise HTTPException(404, "No such image")
    box = G["box"][i]
    if box is None:
        return FileResponse(G["paths"][i])
    im = Image.open(G["paths"][i]).convert("RGB")
    w, h = im.size
    im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/api/reindex")
def reindex():
    build_index()
    return {"gallery": len(G["paths"])}


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StyleFinder</title>
<style>
:root{--bg:#f5f6fa;--fg:#161a23;--card:#fff;--line:#e2e5ee;--accent:#4f46e5;--accent2:#7c3aed;--mut:#667085;--ok:#0e9f6e}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#eceef4;--card:#181b22;--line:#2a2e38;--accent:#818cf8;--accent2:#a78bfa;--mut:#9aa3b5}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:1120px;margin:0 auto;padding:0 22px}
.hero{background:linear-gradient(120deg,var(--accent),var(--accent2));color:#fff;padding:30px 0 26px}
.hero h1{margin:0;font-size:30px;letter-spacing:-.5px}.hero p{margin:6px 0 10px;opacity:.92}
#status{font-size:12.5px;opacity:.85}
.layout{display:grid;grid-template-columns:380px 1fr;gap:22px;margin:22px auto}
@media (max-width:900px){.layout{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.step{display:flex;align-items:center;gap:9px;font-weight:600;margin:16px 0 8px}.step:first-child{margin-top:0}
.step>span:first-child{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--accent);color:#fff;font-size:13px}
.req{color:#e11d48;font-size:12px;font-weight:600}
#drop{border:2px dashed var(--line);border-radius:12px;padding:20px;text-align:center;cursor:pointer;color:var(--mut);transition:.15s}
#drop.over,#drop:hover{border-color:var(--accent);background:rgba(79,70,229,.05)}
#preview{display:none;max-width:100%;max-height:230px;border-radius:10px;margin:0 auto}
textarea{width:100%;min-height:78px;border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit;background:var(--bg);color:var(--fg);resize:vertical}
textarea:focus{outline:2px solid var(--accent);border-color:transparent}
#chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{border:1px solid var(--line);background:var(--bg);color:var(--fg);padding:4px 10px;border-radius:999px;font-size:12.5px;cursor:pointer}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.seg{display:flex;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{flex:1;padding:8px;border:0;background:var(--bg);color:var(--fg);cursor:pointer;font:inherit}
.seg button.on{background:var(--accent);color:#fff}
.sl{margin-top:12px;font-size:13px}.sl label{display:flex;justify-content:space-between;color:var(--mut)}
input[type=range]{width:100%;accent-color:var(--accent)}
#go{width:100%;margin-top:18px;padding:12px;border:0;border-radius:10px;font:600 15px system-ui;color:#fff;cursor:pointer;background:linear-gradient(120deg,var(--accent),var(--accent2))}
#go:disabled{opacity:.45;cursor:not-allowed}
#hint{font-size:12px;color:var(--mut);margin-top:6px;text-align:center}
.qbar{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}
.qbar img{height:56px;border-radius:8px}.qbar .q{background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:13.5px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}
figure{margin:0;position:relative;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;transition:.15s}
figure:hover{transform:translateY(-3px);box-shadow:0 8px 22px rgba(0,0,0,.12)}
figure img{width:100%;height:250px;object-fit:cover;display:block;cursor:zoom-in}
.badge{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.72);color:#fff;border-radius:999px;padding:2px 9px;font-size:12px}
.tag{position:absolute;top:8px;right:8px;border-radius:999px;padding:2px 9px;font-size:11.5px;color:#fff}
.tag.men{background:#2563eb}.tag.women{background:#db2777}
figcaption{padding:10px 11px;font-size:12.5px}
.bar{display:grid;grid-template-columns:62px 1fr 38px;gap:6px;align-items:center;margin:4px 0;color:var(--mut)}
.bar div{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.bar.c i{background:var(--ok)}.bar b{color:var(--fg);font-weight:600;text-align:right}
small{color:var(--mut);word-break:break-all}
.empty{text-align:center;color:var(--mut);padding:60px 10px}
.sk{height:330px;border-radius:12px;background:linear-gradient(90deg,var(--line),var(--card),var(--line));background-size:200% 100%;animation:sh 1.1s infinite}
@keyframes sh{to{background-position:-200% 0}}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;place-items:center;z-index:9;cursor:zoom-out}
#modal img{max-width:92vw;max-height:92vh;border-radius:10px}
#err{color:#e11d48;font-size:13px;margin-top:8px}
</style></head><body>
<div class="hero"><div class="wrap"><h1>StyleFinder</h1>
<p>Upload a garment, say what you want different, and find matching clothes for men and women.</p>
<div id="status">loading...</div></div></div>
<div class="wrap layout">
 <div class="card">
  <div class="step"><span>1</span> Upload a clothing photo</div>
  <div id="drop"><div id="dropmsg">Drop a photo here or click to choose</div><img id="preview" alt="preview"><input id="file" type="file" accept="image/*" hidden></div>
  <div class="step"><span>2</span> Your condition <span class="req">required</span></div>
  <textarea id="cond" placeholder="e.g. same style but in red color, with long sleeves"></textarea>
  <div id="chips"></div>
  <div class="step"><span>3</span> Refine</div>
  <div class="seg" id="seg"><button data-g="all" class="on">All</button><button data-g="men">Men</button><button data-g="women">Women</button></div>
  <div class="sl"><label><span>Garment type</span></label><select id="part" style="width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)"><option value="auto">Auto (from my text)</option><option value="top">Top (shirt, jacket...)</option><option value="bottom">Bottom (pants, skirt...)</option><option value="any">Any</option></select></div>
  <div class="sl"><label><span>Text influence</span><b id="av">50%</b></label><input id="alpha" type="range" min="0" max="100" value="50"></div>
  <div class="sl"><label><span>Look-alike pool (high = more text freedom)</span><b id="pv">600</b></label><input id="pool" type="range" min="20" max="3000" step="20" value="600"></div>
  <div class="sl"><label><span>Results</span><b id="kv">6</b></label><input id="k" type="range" min="3" max="12" value="6"></div>
  <button id="go" disabled>Find clothes</button>
  <div id="hint">Add a photo and a condition to search</div><div id="err"></div>
 </div>
 <div class="card"><div id="qbar" class="qbar" style="display:none"></div>
  <div id="grid"><div class="empty">Your results will appear here.<br>The photo picks the kind of garment, your text picks what changes.</div></div></div>
</div>
<div id="modal"><img id="big" alt=""></div>
<script>
const $=s=>document.querySelector(s);let file=null,gender='all';
const EX=['red color','black','white','long sleeves','short sleeves','with hood','striped pattern','checked pattern','denim','cotton','with pockets','formal style'];
$('#chips').innerHTML=EX.map(t=>'<button class="chip">'+t+'</button>').join('');
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{const t=$('#cond');t.value=(t.value.trim()?t.value.trim()+', ':'')+c.textContent;upd()});
fetch('/api/status').then(r=>r.json()).then(j=>{$('#status').textContent='Gallery: '+j.gallery+' items ('+j.men+' men, '+j.women+' women)  |  look model: '+j.img_model+'  |  text model: '+j.txt_model+'  |  '+j.device});
function upd(){const ok=file&&$('#cond').value.trim().length>0;$('#go').disabled=!ok;$('#hint').textContent=ok?'Ready':(!file?'Add a photo to continue':'Describe your condition to continue')}
function setFile(f){if(!f)return;file=f;const p=$('#preview');p.src=URL.createObjectURL(f);p.style.display='block';$('#dropmsg').style.display='none';upd()}
$('#drop').onclick=()=>$('#file').click();$('#file').onchange=e=>setFile(e.target.files[0]);
$('#drop').ondragover=e=>{e.preventDefault();$('#drop').classList.add('over')};
$('#drop').ondragleave=()=>$('#drop').classList.remove('over');
$('#drop').ondrop=e=>{e.preventDefault();$('#drop').classList.remove('over');setFile(e.dataTransfer.files[0])};
$('#cond').oninput=upd;
document.querySelectorAll('#seg button').forEach(b=>b.onclick=()=>{gender=b.dataset.g;document.querySelectorAll('#seg button').forEach(x=>x.classList.toggle('on',x===b))});
for(const [id,out,suf] of [['alpha','av','%'],['pool','pv',''],['k','kv','']])$('#'+id).oninput=()=>$('#'+out).textContent=$('#'+id).value+suf;
$('#go').onclick=async()=>{
  $('#err').textContent='';$('#go').disabled=true;
  $('#grid').innerHTML=Array(+$('#k').value).fill('<div class="sk"></div>').join('');
  const fd=new FormData();fd.append('file',file);fd.append('text',$('#cond').value.trim());fd.append('k',$('#k').value);
  fd.append('gender',gender);fd.append('part',$('#part').value);fd.append('alpha',$('#alpha').value/100);fd.append('pool',$('#pool').value);
  try{const r=await fetch('/api/search',{method:'POST',body:fd});const j=await r.json();if(!r.ok)throw new Error(j.detail||'error');show(j)}
  catch(e){$('#err').textContent='Error: '+e.message;$('#grid').innerHTML='<div class="empty">No results.</div>'}
  upd()};
function show(j){
  const q=$('#qbar');q.style.display='flex';
  q.innerHTML='<img src="'+$('#preview').src+'"><span>+</span><span class="q">'+$('#cond').value.trim().replace(/</g,'&lt;')+'</span><span style="color:var(--mut);font-size:12.5px">'+j.results.length+' results ('+(j.part||'any')+') in '+j.ms+' ms</span>';
  $('#grid').innerHTML=j.results.map((r,i)=>'<figure><div class="badge">#'+(i+1)+'</div><div class="tag '+r.gender+'">'+r.gender+(r.est?' ~':'')+'</div><img src="'+r.url+'" loading="lazy">'+
   '<figcaption><div class="bar"><label>Look</label><div><i style="width:'+Math.round(r.img_score*100)+'%"></i></div><b>'+r.img_score.toFixed(2)+'</b></div>'+
   '<div class="bar c"><label title="Matches your text better than this % of the whole gallery">Condition</label><div><i style="width:'+r.cond_pct+'%"></i></div><b>'+r.cond_pct+'%</b></div><small>'+r.name+'</small></figcaption></figure>').join('');
  document.querySelectorAll('#grid figure img').forEach(im=>im.onclick=()=>{$('#big').src=im.src;$('#modal').style.display='grid'})}
$('#modal').onclick=()=>$('#modal').style.display='none';
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_model", default=CFG["img_model"])
    ap.add_argument("--txt_model", default=CFG["txt_model"])
    ap.add_argument("--gallery", nargs="+", default=CFG["gallery"], help="folders or json files with a 'shop' list")
    ap.add_argument("--limit", type=int, default=CFG["limit"], help="max images sampled per folder source")
    ap.add_argument("--host", default="0.0.0.0", help="host to bind the server (e.g. 0.0.0.0 or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--build", action="store_true", help="only build the gallery index and exit")
    a = ap.parse_args()
    CFG.update(img_model=a.img_model, txt_model=a.txt_model, gallery=a.gallery, limit=a.limit)
    if a.build:
        get(CFG["img_model"]); get(CFG["txt_model"]); build_index()
    else:
        uvicorn.run(app, host=a.host, port=a.portHTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>StyleFinder — AI Fashion Discovery</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #f8fafc;
  --bg-gradient: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
                 radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.06) 0px, transparent 50%), #f8fafc;
  --fg: #0f172a;
  --card: #ffffff;
  --card-subtle: #f1f5f9;
  --line: #e2e8f0;
  --line-light: #f1f5f9;
  --accent: #4f46e5;
  --accent-hover: #4338ca;
  --accent-light: rgba(79, 70, 229, 0.08);
  --accent2: #7c3aed;
  --pink: #db2777;
  --pink-light: rgba(219, 39, 119, 0.1);
  --blue: #2563eb;
  --blue-light: rgba(37, 99, 235, 0.1);
  --grad: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  --grad-hover: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
  --mut: #64748b;
  --ok: #059669;
  --danger: #e11d48;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
  --shadow-md: 0 8px 24px -4px rgba(15, 23, 42, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
  --shadow-lg: 0 20px 40px -8px rgba(15, 23, 42, 0.12);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #090d16;
    --bg-gradient: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                   radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.10) 0px, transparent 50%), #090d16;
    --fg: #f8fafc;
    --card: #111827;
    --card-subtle: #1a2234;
    --line: #1e293b;
    --line-light: #253349;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-light: rgba(99, 102, 241, 0.16);
    --accent2: #a855f7;
    --grad: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    --grad-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
    --mut: #94a3b8;
    --ok: #10b981;
    --danger: #f43f5e;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 8px 28px -4px rgba(0,0,0,0.45);
    --shadow-lg: 0 20px 40px -8px rgba(0,0,0,0.6);
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  background: var(--bg-gradient);
  color: var(--fg);
  min-height: 100vh;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

/* HERO HEADER */
.hero {
  background: var(--grad);
  color: #fff;
  padding: 44px 0 40px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.4);
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.18), transparent 45%);
  pointer-events: none;
}
.hero-wrap { position: relative; z-index: 1; text-align: center; }
.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.18);
  backdrop-filter: blur(8px);
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.3px;
  margin-bottom: 12px;
  border: 1px solid rgba(255, 255, 255, 0.25);
}
.hero h1 {
  font-size: 38px;
  font-weight: 800;
  letter-spacing: -1px;
  line-height: 1.15;
  margin-bottom: 10px;
}
.hero p {
  max-width: 680px;
  margin: 0 auto;
  font-size: 15.5px;
  opacity: 0.94;
  line-height: 1.6;
}

/* MAIN LAYOUT */
.layout {
  display: grid;
  grid-template-columns: 430px 1fr;
  gap: 28px;
  margin: 32px auto 48px;
  align-items: start;
}
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; gap: 24px; margin: 20px auto; }
}

/* CARDS */
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: 24px;
  box-shadow: var(--shadow-md);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

/* STEP SECTIONS */
.step-section { margin-bottom: 24px; }
.step-section:last-of-type { margin-bottom: 16px; }
.step-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 15px;
  margin-bottom: 12px;
}
.step-num {
  width: 26px;
  height: 26px;
  display: inline-grid;
  place-items: center;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  font-size: 12.5px;
  font-weight: 800;
}
.req-tag {
  margin-left: auto;
  color: var(--danger);
  font-size: 11.5px;
  font-weight: 700;
  background: rgba(225, 29, 72, 0.1);
  padding: 2px 8px;
  border-radius: 6px;
}

/* ACTION BUTTONS (CAMERA & UPLOAD) */
.photo-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 12px;
}
.btn-tab {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line);
  background: var(--card-subtle);
  color: var(--fg);
  font-size: 13.5px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.18s ease;
}
.btn-tab:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-light);
}
.btn-tab.primary-action {
  background: var(--accent-light);
  border-color: var(--accent);
  color: var(--accent);
}

/* DROPZONE */
#drop {
  border: 2px dashed var(--line);
  border-radius: var(--radius-md);
  padding: 24px 16px;
  text-align: center;
  cursor: pointer;
  color: var(--mut);
  background: var(--card-subtle);
  transition: all 0.2s ease;
  position: relative;
}
#drop:hover, #drop.over {
  border-color: var(--accent);
  background: var(--accent-light);
  color: var(--accent);
}
.drop-icon {
  margin-bottom: 8px;
  color: var(--accent);
}
.drop-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--fg);
  margin-bottom: 2px;
}
.drop-subtitle {
  font-size: 12px;
  color: var(--mut);
}

/* PREVIEW BOX */
#preview-wrap {
  display: none;
  position: relative;
  text-align: center;
}
#preview {
  max-width: 100%;
  max-height: 250px;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  display: block;
  margin: 0 auto;
  border: 2px solid var(--accent);
  object-fit: cover;
}
.preview-controls {
  margin-top: 10px;
  display: flex;
  justify-content: center;
  gap: 8px;
}
.btn-retake {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--card);
  border: 1px solid var(--line);
  color: var(--danger);
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.btn-retake:hover {
  background: var(--danger);
  color: #fff;
  border-color: var(--danger);
}

/* TEXTAREA & CHIPS */
textarea {
  width: 100%;
  min-height: 84px;
  border: 1.5px solid var(--line);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  font: inherit;
  font-size: 14px;
  background: var(--card-subtle);
  color: var(--fg);
  resize: vertical;
  transition: all 0.18s ease;
}
textarea:focus {
  outline: none;
  border-color: var(--accent);
  background: var(--card);
  box-shadow: 0 0 0 3px var(--accent-light);
}
.chips-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--mut);
  margin: 10px 0 6px;
}
#chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.chip {
  border: 1px solid var(--line);
  background: var(--card-subtle);
  color: var(--fg);
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  user-select: none;
}
.chip:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-light);
  transform: translateY(-1px);
}

/* SEGMENTED GENDER BUTTONS */
.seg {
  display: flex;
  background: var(--card-subtle);
  padding: 4px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line);
  gap: 4px;
}
.seg button {
  flex: 1;
  padding: 8px 12px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--mut);
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.18s ease;
}
.seg button.on {
  background: var(--card);
  color: var(--fg);
  box-shadow: var(--shadow-sm);
}

/* CONTROLS & SLIDERS */
.ctrl-row { margin-top: 14px; }
.ctrl-row label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  font-weight: 600;
  color: var(--fg);
  margin-bottom: 6px;
}
.ctrl-val {
  color: var(--accent);
  font-weight: 700;
  font-size: 12.5px;
}
select.styled-select {
  width: 100%;
  padding: 9px 12px;
  border: 1.5px solid var(--line);
  border-radius: var(--radius-md);
  background: var(--card-subtle);
  color: var(--fg);
  font: inherit;
  font-size: 13.5px;
  font-weight: 500;
  outline: none;
  cursor: pointer;
  transition: border-color 0.15s;
}
select.styled-select:focus {
  border-color: var(--accent);
}
input[type=range] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: var(--line);
  outline: none;
  accent-color: var(--accent);
  cursor: pointer;
}
.slider-caption {
  font-size: 11.5px;
  color: var(--mut);
  margin-top: 4px;
}

/* SEARCH BUTTON */
#go {
  width: 100%;
  margin-top: 22px;
  padding: 14px;
  border: 0;
  border-radius: var(--radius-md);
  font: 700 15px 'Plus Jakarta Sans', system-ui;
  color: #fff;
  cursor: pointer;
  background: var(--grad);
  box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35);
  transition: all 0.18s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
#go:hover:not(:disabled) {
  background: var(--grad-hover);
  box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
  transform: translateY(-1px);
}
#go:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}
#hint {
  font-size: 12px;
  color: var(--mut);
  margin-top: 8px;
  text-align: center;
}
#err {
  color: var(--danger);
  font-size: 13px;
  font-weight: 600;
  margin-top: 10px;
  text-align: center;
  background: rgba(225, 29, 72, 0.08);
  padding: 8px;
  border-radius: 8px;
  display: none;
}

/* QUERY BAR */
.qbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
  flex-wrap: wrap;
}
.qbar img {
  height: 58px;
  width: 58px;
  border-radius: var(--radius-sm);
  object-fit: cover;
  border: 2px solid var(--accent);
}
.qbar-plus {
  color: var(--mut);
  font-weight: 800;
  font-size: 18px;
}
.qbar-badge {
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 13.5px;
  font-weight: 600;
}
.qbar-meta {
  color: var(--mut);
  font-size: 12.5px;
  margin-left: auto;
}

/* RESULTS GRID */
#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 18px;
}
figure {
  position: relative;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  overflow: hidden;
  transition: all 0.22s ease;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
}
figure:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color: var(--accent);
}
figure img {
  width: 100%;
  height: 260px;
  object-fit: cover;
  display: block;
  cursor: zoom-in;
  background: var(--card-subtle);
  transition: transform 0.3s ease;
}
figure:hover img {
  transform: scale(1.03);
}
.card-badge {
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(15, 23, 42, 0.85);
  backdrop-filter: blur(4px);
  color: #fff;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.3px;
  z-index: 2;
}
.card-tag {
  position: absolute;
  top: 10px;
  right: 10px;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  z-index: 2;
  text-transform: capitalize;
}
.card-tag.men { background: var(--blue); }
.card-tag.women { background: var(--pink); }

figcaption {
  padding: 12px 14px;
  font-size: 12.5px;
  background: var(--card);
  border-top: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-grow: 1;
}
.score-bar {
  display: grid;
  grid-template-columns: 68px 1fr 38px;
  gap: 8px;
  align-items: center;
  color: var(--mut);
  font-size: 11.5px;
}
.score-track {
  height: 6px;
  background: var(--line);
  border-radius: 3px;
  overflow: hidden;
}
.score-fill {
  display: block;
  height: 100%;
  border-radius: 3px;
  background: var(--accent);
}
.score-bar.condition .score-fill {
  background: var(--ok);
}
.score-num {
  font-weight: 700;
  color: var(--fg);
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.item-filename {
  font-size: 11px;
  color: var(--mut);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-top: 2px;
}

/* EMPTY STATE */
.empty {
  grid-column: 1 / -1;
  text-align: center;
  color: var(--mut);
  padding: 60px 20px;
}
.empty-icon {
  font-size: 42px;
  margin-bottom: 12px;
}
.empty-title {
  font-size: 17px;
  font-weight: 700;
  color: var(--fg);
  margin-bottom: 6px;
}
.empty-desc {
  font-size: 13.5px;
  max-width: 440px;
  margin: 0 auto;
  line-height: 1.5;
}

/* SHIMMER SKELETON */
.sk {
  height: 350px;
  border-radius: var(--radius-md);
  background: linear-gradient(90deg, var(--line), var(--card), var(--line));
  background-size: 200% 100%;
  animation: sh 1.2s infinite;
}
@keyframes sh { to { background-position: -200% 0; } }

/* LIGHTBOX MODAL */
#modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(8px);
  display: none;
  place-items: center;
  z-index: 100;
  cursor: zoom-out;
  padding: 20px;
}
#modal img {
  max-width: 90vw;
  max-height: 90vh;
  border-radius: var(--radius-md);
  box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}

/* CAMERA MODAL */
#cam-modal {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.88);
  backdrop-filter: blur(8px);
  display: none;
  place-items: center;
  z-index: 110;
  padding: 16px;
}
.cam-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  width: 100%;
  max-width: 540px;
  overflow: hidden;
  box-shadow: var(--shadow-lg);
  display: flex;
  flex-direction: column;
}
.cam-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
}
.cam-head-title {
  font-size: 16px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cam-close-btn {
  background: transparent;
  border: 0;
  color: var(--mut);
  font-size: 20px;
  cursor: pointer;
  padding: 4px;
  border-radius: 6px;
  transition: all 0.15s;
}
.cam-close-btn:hover {
  color: var(--danger);
  background: rgba(225, 29, 72, 0.1);
}
.cam-view {
  position: relative;
  background: #000;
  aspect-ratio: 4 / 3;
  width: 100%;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
#cam-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.cam-reticle {
  position: absolute;
  inset: 20px;
  border: 2px dashed rgba(255, 255, 255, 0.4);
  border-radius: var(--radius-md);
  pointer-events: none;
}
.cam-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 24px;
  background: var(--card);
  border-top: 1px solid var(--line);
}
.btn-cam-ctrl {
  background: var(--card-subtle);
  border: 1px solid var(--line);
  color: var(--fg);
  padding: 10px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.btn-cam-ctrl:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.btn-shutter {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  border: 4px solid var(--accent);
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}
.btn-shutter:hover {
  transform: scale(1.06);
}
.btn-shutter:active {
  transform: scale(0.94);
}
.shutter-core {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: var(--accent);
}
</style>
</head>
<body>

<header class="hero">
  <div class="wrap hero-wrap">
    <div class="hero-chip">✨ AI Multi-Modal Fashion Retrieval</div>
    <h1>StyleFinder</h1>
    <p>Snap or upload any clothing photo, specify what you want changed, and discover matching outfits tailored to your desired style.</p>
  </div>
</header>

<main class="wrap layout">
  <!-- LEFT: CONTROLS & QUERY -->
  <div class="card">
    <!-- STEP 1: GARMENT PHOTO -->
    <div class="step-section">
      <div class="step-header">
        <span class="step-num">1</span>
        <span>Clothing Photo</span>
      </div>

      <div class="photo-actions">
        <button type="button" class="btn-tab primary-action" id="btn-open-cam">
          📸 Open Camera
        </button>
        <button type="button" class="btn-tab" id="btn-browse-file">
          📁 Choose File
        </button>
      </div>

      <div id="drop">
        <div id="drop-ui">
          <div class="drop-icon">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242M12 12v9m-4-4 4-4 4 4"/>
            </svg>
          </div>
          <div class="drop-title">Drop a clothing photo here</div>
          <div class="drop-subtitle">or click to browse from device</div>
        </div>

        <div id="preview-wrap">
          <img id="preview" alt="Selected garment">
          <div class="preview-controls">
            <button type="button" class="btn-retake" id="btn-clear-photo">✕ Change / Retake</button>
          </div>
        </div>

        <input id="file" type="file" accept="image/*" hidden>
        <input id="file-cam-fallback" type="file" accept="image/*" capture="environment" hidden>
      </div>
    </div>

    <!-- STEP 2: CONDITION -->
    <div class="step-section">
      <div class="step-header">
        <span class="step-num">2</span>
        <span>Your Condition</span>
        <span class="req-tag">Required</span>
      </div>
      <textarea id="cond" placeholder="e.g. same style but in navy blue color, with long sleeves and cotton texture..."></textarea>
      <div class="chips-label">Quick suggestions:</div>
      <div id="chips"></div>
    </div>

    <!-- STEP 3: REFINE -->
    <div class="step-section">
      <div class="step-header">
        <span class="step-num">3</span>
        <span>Refine Parameters</span>
      </div>

      <div class="ctrl-row">
        <label><span>Department</span></label>
        <div class="seg" id="seg">
          <button type="button" data-g="all" class="on">All</button>
          <button type="button" data-g="men">Men 👔</button>
          <button type="button" data-g="women">Women 👗</button>
        </div>
      </div>

      <div class="ctrl-row">
        <label><span>Garment Piece</span></label>
        <select id="part" class="styled-select">
          <option value="auto">Auto (detected from your text)</option>
          <option value="top">Top (shirt, jacket, hoodie, sweater)</option>
          <option value="bottom">Bottom (pants, jeans, skirt, shorts)</option>
          <option value="any">Any piece</option>
        </select>
      </div>

      <div class="ctrl-row">
        <label><span>Text Influence</span><span class="ctrl-val" id="av">50%</span></label>
        <input id="alpha" type="range" min="0" max="100" value="50">
      </div>

      <div class="ctrl-row">
        <label><span>Look-alike Candidate Pool</span><span class="ctrl-val" id="pv">600</span></label>
        <input id="pool" type="range" min="20" max="3000" step="20" value="600">
        <div class="slider-caption">High = more freedom for text conditions to modify the look</div>
      </div>

      <div class="ctrl-row">
        <label><span>Number of Results</span><span class="ctrl-val" id="kv">6</span></label>
        <input id="k" type="range" min="3" max="12" value="6">
      </div>

      <button type="button" id="go" disabled>
        <span>🔍 Find Clothes</span>
      </button>
      <div id="hint">Select a photo and specify changes to begin</div>
      <div id="err"></div>
    </div>
  </div>

  <!-- RIGHT: RESULTS GALLERY -->
  <div class="card">
    <div id="qbar" class="qbar" style="display:none"></div>
    <div id="grid">
      <div class="empty">
        <div class="empty-icon">✨</div>
        <div class="empty-title">Visual Results Will Appear Here</div>
        <div class="empty-desc">Your photo selects the silhouette and cut, while your text description tailors the color, texture, and style details.</div>
      </div>
    </div>
  </div>
</main>

<!-- LIGHTBOX MODAL -->
<div id="modal">
  <img id="big" alt="Enlarged photo">
</div>

<!-- CAMERA VIEW MODAL -->
<div id="cam-modal">
  <div class="cam-card">
    <div class="cam-head">
      <div class="cam-head-title">
        <span>📸</span> <span>Capture Garment</span>
      </div>
      <button type="button" class="cam-close-btn" id="cam-close" title="Close">✕</button>
    </div>
    <div class="cam-view">
      <video id="cam-video" autoplay playsinline muted></video>
      <div class="cam-reticle"></div>
      <canvas id="cam-canvas" style="display:none"></canvas>
    </div>
    <div class="cam-bar">
      <button type="button" class="btn-cam-ctrl" id="cam-flip">🔄 Flip</button>
      <button type="button" class="btn-shutter" id="cam-snap" title="Take Photo">
        <span class="shutter-core"></span>
      </button>
      <button type="button" class="btn-cam-ctrl" id="cam-cancel">Cancel</button>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let file = null, gender = 'all', camStream = null, facingMode = 'environment';

// QUICK CHIPS
const EX = [
  'red color', 'black', 'white', 'navy blue', 'beige',
  'long sleeves', 'short sleeves', 'with hood', 'striped pattern',
  'checked pattern', 'denim', 'cotton', 'formal style', 'casual style'
];
$('#chips').innerHTML = EX.map(t => '<button type="button" class="chip">' + t + '</button>').join('');
document.querySelectorAll('.chip').forEach(c => c.onclick = () => {
  const t = $('#cond');
  t.value = (t.value.trim() ? t.value.trim() + ', ' : '') + c.textContent;
  upd();
});

// VALIDATION & STATUS
function upd() {
  const ok = file && $('#cond').value.trim().length > 0;
  $('#go').disabled = !ok;
  $('#hint').textContent = ok ? 'Ready to search!' : (!file ? 'Add a photo to continue' : 'Describe what to change to continue');
}

// FILE HANDLING
function setFile(f) {
  if (!f) return;
  file = f;
  const p = $('#preview');
  p.src = URL.createObjectURL(f);
  $('#preview-wrap').style.display = 'block';
  $('#drop-ui').style.display = 'none';
  upd();
}

function clearFile() {
  file = null;
  $('#preview').src = '';
  $('#preview-wrap').style.display = 'none';
  $('#drop-ui').style.display = 'block';
  $('#file').value = '';
  $('#file-cam-fallback').value = '';
  upd();
}

$('#btn-clear-photo').onclick = e => { e.stopPropagation(); clearFile(); };
$('#btn-browse-file').onclick = () => $('#file').click();
$('#file').onchange = e => setFile(e.target.files[0]);
$('#file-cam-fallback').onchange = e => setFile(e.target.files[0]);

$('#drop').onclick = () => { if (!file) $('#file').click(); };
$('#drop').ondragover = e => { e.preventDefault(); $('#drop').classList.add('over'); };
$('#drop').ondragleave = () => $('#drop').classList.remove('over');
$('#drop').ondrop = e => {
  e.preventDefault();
  $('#drop').classList.remove('over');
  setFile(e.dataTransfer.files[0]);
};

$('#cond').oninput = upd;

// SEGMENTED GENDER
document.querySelectorAll('#seg button').forEach(b => b.onclick = () => {
  gender = b.dataset.g;
  document.querySelectorAll('#seg button').forEach(x => x.classList.toggle('on', x === b));
});

// SLIDER LABELS
for (const [id, out, suf] of [['alpha', 'av', '%'], ['pool', 'pv', ''], ['k', 'kv', '']]) {
  $('#' + id).oninput = () => $('#' + out).textContent = $('#' + id).value + suf;
}

// CAMERA ACCESS
async function startCamera() {
  stopCamera();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // Fallback directly to native camera capture
    $('#file-cam-fallback').click();
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    const v = $('#cam-video');
    v.srcObject = camStream;
    await v.play();
    $('#cam-modal').style.display = 'grid';
  } catch (err) {
    console.warn("Camera ideal constraint failed, retrying basic:", err);
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const v = $('#cam-video');
      v.srcObject = camStream;
      await v.play();
      $('#cam-modal').style.display = 'grid';
    } catch (e) {
      alert("Camera access was not permitted or not supported. Falling back to photo upload.");
      $('#file-cam-fallback').click();
    }
  }
}

function stopCamera() {
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  const v = $('#cam-video');
  if (v) v.srcObject = null;
  $('#cam-modal').style.display = 'none';
}

function snapPhoto() {
  const v = $('#cam-video');
  const c = $('#cam-canvas');
  if (!v || !v.videoWidth) return;
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(v, 0, 0, c.width, c.height);
  c.toBlob(blob => {
    if (!blob) return;
    const snapFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
    setFile(snapFile);
    stopCamera();
  }, 'image/jpeg', 0.92);
}

$('#btn-open-cam').onclick = startCamera;
$('#cam-close').onclick = stopCamera;
$('#cam-cancel').onclick = stopCamera;
$('#cam-snap').onclick = snapPhoto;
$('#cam-flip').onclick = () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
};

// SEARCH EXECUTION
$('#go').onclick = async () => {
  $('#err').style.display = 'none';
  $('#err').textContent = '';
  $('#go').disabled = true;
  $('#grid').innerHTML = Array(+$('#k').value).fill('<div class="sk"></div>').join('');

  const fd = new FormData();
  fd.append('file', file);
  fd.append('text', $('#cond').value.trim());
  fd.append('k', $('#k').value);
  fd.append('gender', gender);
  fd.append('part', $('#part').value);
  fd.append('alpha', $('#alpha').value / 100);
  fd.append('pool', $('#pool').value);

  try {
    const r = await fetch('/api/search', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'Search request failed');
    show(j);
  } catch (e) {
    $('#err').style.display = 'block';
    $('#err').textContent = 'Error: ' + e.message;
    $('#grid').innerHTML = '<div class="empty"><div class="empty-icon">⚠️</div><div class="empty-title">Search Failed</div><div class="empty-desc">' + e.message + '</div></div>';
  }
  upd();
};

function show(j) {
  const q = $('#qbar');
  q.style.display = 'flex';
  q.innerHTML = '<img src="' + $('#preview').src + '">' +
    '<span class="qbar-plus">+</span>' +
    '<span class="qbar-badge">' + $('#cond').value.trim().replace(/</g, '&lt;') + '</span>' +
    '<span class="qbar-meta">' + j.results.length + ' results (' + (j.part || 'any') + ') in ' + j.ms + ' ms</span>';

  $('#grid').innerHTML = j.results.map((r, i) =>
    '<figure>' +
      '<div class="card-badge">#' + (i + 1) + '</div>' +
      '<div class="card-tag ' + r.gender + '">' + r.gender + (r.est ? ' ~' : '') + '</div>' +
      '<img src="' + r.url + '" loading="lazy" alt="Garment match">' +
      '<figcaption>' +
        '<div class="score-bar">' +
          '<label>Look</label>' +
          '<div class="score-track"><i class="score-fill" style="width:' + Math.min(100, Math.max(0, Math.round(r.img_score * 100))) + '%"></i></div>' +
          '<b class="score-num">' + r.img_score.toFixed(2) + '</b>' +
        '</div>' +
        '<div class="score-bar condition">' +
          '<label title="Matches your text better than this % of the whole gallery">Condition</label>' +
          '<div class="score-track"><i class="score-fill" style="width:' + r.cond_pct + '%"></i></div>' +
          '<b class="score-num">' + r.cond_pct + '%</b>' +
        '</div>' +
        '<small class="item-filename">' + r.name + '</small>' +
      '</figcaption>' +
    '</figure>'
  ).join('');

  document.querySelectorAll('#grid figure img').forEach(im => im.onclick = () => {
    $('#big').src = im.src;
    $('#modal').style.display = 'grid';
  });
}

$('#modal').onclick = () => $('#modal').style.display = 'none';
</script>
</body>
</html>'''

with open("cir_server.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the HTML block
start_marker = 'HTML = r"""'
end_marker = '"""\n\n\nif __name__ == "__main__":'

start_idx = code.find(start_marker)
end_idx = code.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print(f"Error finding markers: start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

new_code = code[:start_idx] + 'HTML = r"""' + HTML_CONTENT + '"""\n\n\n' + code[end_idx + len('"""\n\n\n'):]

with open("cir_server.py", "w", encoding="utf-8") as f:
    f.write(new_code)

print("Successfully replaced HTML in cir_server.py!")"""


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_model", default=CFG["img_model"])
    ap.add_argument("--txt_model", default=CFG["txt_model"])
    ap.add_argument("--gallery", nargs="+", default=CFG["gallery"], help="folders or json files with a 'shop' list")
    ap.add_argument("--limit", type=int, default=CFG["limit"], help="max images sampled per folder source")
    ap.add_argument("--host", default="0.0.0.0", help="host to bind the server (e.g. 0.0.0.0 or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--build", action="store_true", help="only build the gallery index and exit")
    a = ap.parse_args()
    CFG.update(img_model=a.img_model, txt_model=a.txt_model, gallery=a.gallery, limit=a.limit)
    if a.build:
        get(CFG["img_model"]); get(CFG["txt_model"]); build_index()
    else:
        uvicorn.run(app, host=a.host, port=a.port)