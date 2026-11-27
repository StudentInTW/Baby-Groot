"""
mouth_patch_train.py（2026/11/23 起；v6 口鼻小塊模型的第 3 步，在 .venv-vlm 跑：要 PyTorch、torchvision、safetensors）
吃 mouth_patch_build.py 存的口鼻小塊（YuNet 找到的臉、依 5 個特徵點轉正，中間 96 見方是鼻子到下巴），
判斷「口鼻露出（綠的幀）」對「沒露出（黃、紅的幀裡被找到的假臉）」。要取代的是 v4 的口鼻膚色門檻：
主流程只在 YuNet 找到臉時才問它，所以它不用管臉轉開的情況。

骨幹 ResNet18，預訓練權重 timm/resnet18.a1_in1k（Hugging Face，Apache-2.0；model.safetensors 放
model/pretrained/resnet18.a1_in1k/，鍵名跟 torchvision 的 resnet18 一樣），輸入 96，12 個 epoch。
訓練資料只有自己拍的娃娃（hailo/prep_calib.py SPLITS 的 calib 那一半）；真實嬰兒、SyRIP、網路影像都不能進來。
取樣：每一批資料（百貨公司、台南家裡、寶寶1、11/19、寶寶2）每類一樣重，批裡每支影片／每個照片批再平分
  （照影片平分的話，21 支的寶寶2 會佔掉快四成）；露出的原圖取樣兩倍，因為其中 25% 會被貼布改成沒露出，貼完兩類差不多一樣多。
增強：
  幾何：從存下來的 1.6 倍範圍隨機平移（±8%）、縮放（0.85–1.15）、旋轉 ±10°、左右翻
  畫質：亮度、對比、飽和度、色相、灰階（0.3）、高斯模糊（0.3）、銳化（0.2）、先縮小再放大（0.4，模擬監視器上口鼻只有 20–60 像素）
  貼布遮擋（預設開，--no-paste 關）：綠的小塊，25% 機率貼一塊布蓋住鼻子和嘴巴（標籤改成沒露出），
      15% 機率只貼在下巴或最外側、沒碰到口鼻（標籤不變）。布料是 calib 頭部特寫下緣的包巾、衣服（只用訓練那一半）。
只用 --train 指定的組訓練；其他組（hold、hold_1123、field）只評估。「組:影片1,影片2」只取那幾支。
輸出 patch/pred_<tag>.csv（每張臉 p_visible，trained＝這張有沒有拿來訓練）、model_patch_<tag>.pt。

v6 的模型是這三顆（種子 0、1、2；筆電 RTX 3050 各約 4 分鐘），再用 mouth_patch_export.py 合成一個 ONNX：
    .venv-vlm/Scripts/python -m experiments.mouth_patch_train --train calib calib_1123 --seed 0 --tag p_base_s0
    .venv-vlm/Scripts/python -m experiments.mouth_patch_train --train calib calib_1123 --seed 1 --tag p_base_s1
    .venv-vlm/Scripts/python -m experiments.mouth_patch_train --train calib calib_1123 --seed 2 --tag p_base_s2
第二版（2026/11/26）：新分法的 calib（mouth_patch_crops --suffix _r2）＋床位裡 0.3–0.8 分的臉（mouth_patch_build --low）。
低分的臉的標籤：黃、紅的幀＝沒露出；綠的幀只在「主流程的臉 0 張、低分的臉剛好 1 張」時當露出，其他不訓練
（綠的幀裡多出來的低分臉可能是耳朵、手上的假臉）。--sets 只讀這幾組（不讀評估組，也不替它們預測）：
    .venv-vlm/Scripts/python -m experiments.mouth_patch_train --train calib_r2 calib_1123_r2 --sets calib_r2 calib_1123_r2 --seed 0 --tag p_r2_s0
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parent.parent
A3 = ROOT / "outputs" / "airway3"
P = A3 / "patch"
TIMM_R18 = ROOT / "model" / "pretrained" / "resnet18.a1_in1k" / "model.safetensors"
SETS = ("calib", "calib_1123", "hold", "hold_1123", "field")
TRAINABLE = ("calib", "calib_1123", "calib_r2", "calib_1123_r2")   # 只有 calib 那一半能訓練；其他只評估
CROPS_OF = {"calib": "crops_calib.csv", "calib_1123": "crops_calib_1123.csv",    # 貼布的布料從哪個頭部特寫清單取（只取訓練組）
            "calib_r2": "crops_calib_r2.csv", "calib_1123_r2": "crops_calib_1123.csv"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def source(s, g):
    """取樣平衡用的資料批次。"""
    if s in ("calib_1123", "calib_1123_r2"):
        return "寶寶2"
    if g.startswith(("photo_batch2", "photo_batch3")) or g in ("IMG_8115", "IMG_8150", "IMG_8429", "IMG_8432"):
        return "百貨公司"
    if g.startswith("photo_batch4") or g.startswith("IMG_87"):
        return "台南家裡"
    if g.startswith("寶寶1"):
        return "寶寶1"
    return g
S, CTXS = 96, 154     # 中間口鼻那塊、存檔的範圍（96×1.6）


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_rows(sets):
    rows = []
    for s in sets:
        if not (P / s / "faces.csv").exists():
            print(f"[略過] {s} 還沒有 faces.csv", flush=True)
            continue
        samp = {r["name"]: r for r in read_csv(P / s / "samples.csv")}
        for f in read_csv(P / s / "faces.csv"):
            r = samp[f["name"]]
            y = 1 if r["label"] == "ok" else 0
            if r["staged"] == "warn":          # 不確定的不訓練
                y = -1
            if f.get("low") == "1" and y == 1 and not (r["faces"] == "0" and r.get("n_low") == "1"):
                y = -1                          # 綠的幀裡低分的臉：只有主流程的臉 0 張、低分的臉剛好 1 張才當露出
            rows.append({**f, "set": s, "group": r["group"], "label": r["label"], "staged": r["staged"], "y": y,
                         "path": str(P / s / "p" / f["patch"])})
    return rows


def cloth_bank(files=("crops_calib.csv", "crops_calib_1123.csv"), n=600, seed=0):
    """calib 頭部特寫的下緣（下巴以下，多半是包巾、衣服）切成方塊當布料。只用訓練那一半（files＝訓練組的頭部特寫清單）。"""
    rng = random.Random(seed)
    rows = [r for f in files for r in read_csv(A3 / f) if r["crop"]]
    bank = []
    for r in rng.sample(rows, min(n, len(rows))):
        img = cv2.imdecode(np.fromfile(str(A3 / "crops_old" / r["crop"]), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        y0 = int(h * rng.uniform(0.74, 0.8)); x0 = int(w * rng.uniform(0.15, 0.35))
        tile = img[y0:h, x0:x0 + int(w * 0.5)]
        if tile.size and tile.std() > 4:
            bank.append(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    return bank


def paste(img, bank, rng, cover):
    """img：CTXS 見方 RGB uint8。cover=True：布蓋住鼻子（中間 0.27）和嘴巴（0.65）；False：只蓋下巴（>0.82）或最外側（<0.15）。"""
    h, w = img.shape[:2]
    o = (CTXS - S) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx - o) / S, (yy - o) / S          # 中間那塊的比例座標
    kind = rng.choice(("below", "side", "diag"))
    wav = 0.03 * np.sin(u * rng.uniform(6, 14) + rng.uniform(0, 6)) + 0.02 * np.sin(v * rng.uniform(6, 14))
    if kind == "below":                        # 被子從下面拉上來
        edge = rng.uniform(-0.1, 0.2) if cover else rng.uniform(0.84, 1.0)
        mask = v + wav > edge
    elif kind == "side":                       # 布從旁邊蓋過來
        edge = rng.uniform(0.62, 1.05) if cover else rng.uniform(-0.2, 0.12)
        side = u if rng.random() < 0.5 else 1 - u
        mask = side + wav < edge
    else:                                      # 斜斜蓋過臉
        ang = rng.uniform(0, 2 * np.pi); nx, ny = np.cos(ang), np.sin(ang)
        d = (u - 0.5) * nx + (v - 0.46) * ny   # 鼻嘴中點附近
        thr = rng.uniform(0.25, 0.45) if cover else rng.uniform(-0.75, -0.5)
        mask = d + wav < thr
        pts = [(0.5, 0.27), (0.5, 0.65)]
        under = [(px - 0.5) * nx + (py - 0.46) * ny < thr for px, py in pts]
        if cover and not all(under) or (not cover and any(under)):
            return None
    if cover and not (mask[int(o + 0.27 * S), int(o + 0.5 * S)] and mask[int(o + 0.65 * S), int(o + 0.5 * S)]):
        return None
    if not cover and (mask[int(o + 0.27 * S), int(o + 0.5 * S)] or mask[int(o + 0.65 * S), int(o + 0.5 * S)]):
        return None
    tex = bank[rng.randrange(len(bank))]
    tex = cv2.resize(tex, (w, h)) if rng.random() < 0.5 else cv2.resize(cv2.flip(tex, 1), (w, h))
    tex = np.clip(tex.astype(np.float32) * rng.uniform(0.8, 1.15) + rng.uniform(-15, 15), 0, 255)
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), rng.uniform(0.8, 2.5))[..., None]
    shadow = 1 - 0.25 * np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6)[..., None] - m, 0, 1)
    out = img.astype(np.float32) * shadow * (1 - m) + tex * m
    return np.clip(out, 0, 255).astype(np.uint8)


def lowres(x):
    """模擬監視器：先縮到 20–60 像素再放大回來。x：C×H×W float"""
    h = x.shape[-1]
    s = random.randint(20, 60)
    small = nn.functional.interpolate(x[None], size=(s, s), mode="area")
    return nn.functional.interpolate(small, size=(h, h), mode="bilinear", align_corners=False)[0]


def tfs():
    train = T.Compose([
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.85, 1.15), fill=114),
        T.CenterCrop(S),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
        T.RandomGrayscale(p=0.3),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        T.RandomAdjustSharpness(2.0, p=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.RandomApply([T.Lambda(lowres)], p=0.4),
        T.Normalize(MEAN, STD),
    ])
    ev = T.Compose([T.CenterCrop(S), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    return train, ev


def _groups_by(rows):
    """(批次, 類) → 那一類出現在哪幾組（影片／照片批）。"""
    out = {}
    for r in rows:
        out.setdefault((source(r["set"], r["group"]), r["y"]), set()).add((r["set"], r["group"]))
    return out


class Patches(Dataset):
    def __init__(self, rows, tf, bank=None, p_cover=0.0, p_safe=0.0):
        self.rows, self.tf, self.bank, self.pc, self.ps = rows, tf, bank, p_cover, p_safe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        y = r["y"]
        if self.bank and y == 1:
            pc = self.pc
            q = random.random()
            if q < pc + self.ps:
                cover = q < pc
                for _ in range(6):
                    out = paste(img, self.bank, random, cover)
                    if out is not None:
                        img, y = out, (0 if cover else 1)
                        break
        return self.tf(torch.from_numpy(img).permute(2, 0, 1)), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True,
                    help="拿來訓練的組：calib、calib_1123；「組:影片1,影片2」只取那幾支")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-paste", action="store_true")
    ap.add_argument("--predict", action="store_true", help="不訓練：讀 model_patch_<tag>.pt，重新預測全部的組（後來才建好的組）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", nargs="+", default=list(SETS), help="讀哪幾組的小塊（訓練＋預測）；預設全部")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = {}
    for item in args.train:
        s, _, g = item.partition(":")
        if s not in TRAINABLE:
            raise SystemExit(f"{s} 只能評估、不能訓練")
        spec[s] = set(g.split(",")) if g else None
    rows = load_rows(args.sets)
    train_rows = [r for r in rows if r["set"] in spec and (spec[r["set"]] is None or r["group"] in spec[r["set"]])
                  and r["y"] in (0, 1)]
    for s, gs in spec.items():
        missing = (gs or set()) - {r["group"] for r in train_rows if r["set"] == s}
        if missing:
            raise SystemExit(f"{s} 裡沒有 {missing}")
    if args.predict:                   # --train 要跟訓練時一樣（trained 欄才對）
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(P / f"model_patch_{args.tag}.pt", map_location=dev))
        model.to(dev)
        return predict(model, rows, train_rows, args.tag, dev, time.perf_counter())
    bank = None if args.no_paste else cloth_bank(sorted({CROPS_OF[s] for s in spec}), seed=args.seed)
    by_src = Counter((source(r["set"], r["group"]), r["y"]) for r in train_rows)
    print(f"[{args.tag}] 訓練 {len(train_rows)} 張臉 {dict(Counter(r['y'] for r in train_rows))}（1＝露出）"
          f"｜{dict(sorted(by_src.items()))}｜布料 {len(bank) if bank else 0}｜{dev}", flush=True)
    train_tf, _ = tfs()
    count = Counter((r["set"], r["group"], r["y"]) for r in train_rows)
    n_groups = Counter({k: len(v) for k, v in _groups_by(train_rows).items()})
    weights = [(2.0 if r["y"] == 1 else 1.0) / (n_groups[(source(r["set"], r["group"]), r["y"])] * count[(r["set"], r["group"], r["y"])])
               for r in train_rows]
    loader = DataLoader(Patches(train_rows, train_tf, bank, 0.25, 0.15), batch_size=64, num_workers=0,
                        sampler=WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True))
    from safetensors.torch import load_file
    model = torchvision.models.resnet18()
    model.load_state_dict(load_file(str(TIMM_R18)))
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(dev)
    head = list(model.fc.parameters()); body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train(); tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev).float()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        if ep in (0, args.epochs - 1):
            print(f"  epoch {ep + 1}/{args.epochs}：loss {tot / n:.3f}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    torch.save(model.state_dict(), P / f"model_patch_{args.tag}.pt")
    predict(model, rows, train_rows, args.tag, dev, t0)


def predict(model, rows, train_rows, tag, dev, t0):
    _, ev_tf = tfs()
    model.eval()
    out, trained = [], {id(r) for r in train_rows}
    with torch.no_grad():
        for k in range(0, len(rows), 256):
            batch = rows[k:k + 256]
            x = torch.stack([Patches([r], ev_tf)[0][0] for r in batch]).to(dev)
            for r, p in zip(batch, torch.sigmoid(model(x).squeeze(1)).cpu().tolist()):
                out.append({"set": r["set"], "group": r["group"], "name": r["name"], "k": r["k"],
                            "trained": int(id(r) in trained), "p": round(p, 4)})
    with open(P / f"pred_{tag}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"→ patch/pred_{tag}.csv（{len(out)} 張臉，組 {sorted({r['set'] for r in out})}，{time.perf_counter() - t0:.0f} 秒）", flush=True)


if __name__ == "__main__":
    sys.exit(main())
