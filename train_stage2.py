import json, random, argparse
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

DEV = "cuda"

def _t(o):
    return o if hasattr(o, "shape") else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])

NORM = T.Normalize([0.5] * 3, [0.5] * 3)
shop_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])

def street_tf(erase):
    ops = [T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), NORM]
    if erase:                                   # occlusion augmentation (extension)
        ops.append(T.RandomErasing(p=0.5, scale=(0.05, 0.25)))
    return T.Compose(ops)

class TripletDS(Dataset):
    def __init__(self, path, erase):
        self.items = [x for x in json.load(open(path)) if len(x["user"]) >= 2 and x["shop"]]
        self.tf = street_tf(erase)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        q, h = random.sample(it["user"], 2)            # anchor + hard positive (street)
        p = random.choice(it["shop"])                  # positive (shop)
        ld = lambda f: Image.open(f).convert("RGB")
        return self.tf(ld(q)), self.tf(ld(h)), shop_tf(ld(p))

def nt_xent(za, zb, tau):
    """Anchor i's positive is zb[i]; every other sample in the batch is a negative."""
    za, zb = F.normalize(za.float(), dim=-1), F.normalize(zb.float(), dim=-1)
    N = za.size(0)
    z = torch.cat([za, zb]); sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="triplets_train.json")
    ap.add_argument("--ckpt", default="ckpt_stage1")
    ap.add_argument("--out", default="ckpt_stage2")      # keep 'stage' or 'siglip' in the name
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wp", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--erase", action="store_true")
    ap.add_argument("--train_last", type=int, default=0)  # train only last N vision blocks (0 = all)
    a = ap.parse_args()

    model = AutoModel.from_pretrained(a.ckpt).to(DEV)
    proc = AutoProcessor.from_pretrained(a.ckpt)
    for p in model.text_model.parameters(): p.requires_grad = False       # freeze text tower
    try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e: print("no gradient checkpointing:", e)
    if a.train_last > 0:
        vm = model.vision_model
        for p in vm.parameters(): p.requires_grad = False
        for blk in vm.encoder.layers[-a.train_last:]:
            for p in blk.parameters(): p.requires_grad = True
        for name in ("post_layernorm", "head"):
            if hasattr(vm, name):
                for p in getattr(vm, name).parameters(): p.requires_grad = True
        print("trainable params (M):", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    model.train()

    dl = DataLoader(TripletDS(a.data, a.erase), batch_size=a.bs, shuffle=True, drop_last=True, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(a.epochs):
        run = 0.0
        for step, (q, h, p) in enumerate(tqdm(dl, desc=f"epoch {ep+1}/{a.epochs}")):
            x = torch.cat([q, h, p]).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                f = _t(model.get_image_features(pixel_values=x))
            za, zh, zp = f.chunk(3)
            loss = nt_xent(za, zh, a.tau) + a.wp * nt_xent(za, zp, a.tau)    # Eq. 6
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
            if step % 25 == 0: print(f"  step {step} loss {loss.item():.4f}")
        sched.step()
        print(f"epoch {ep+1} mean loss {run/len(dl):.4f}")
        model.save_pretrained(a.out); proc.save_pretrained(a.out)
    print("saved to", a.out)