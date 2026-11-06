"""
airway_train.py
口鼻分類器（實驗，2026/11/06 起）：頭部特寫 → 「口鼻看得到」的機率。calib 那一半訓練、hold 那一半評估
（outputs/airway/crops.csv，先在主環境跑 experiments.airway_crops）。

模型：ResNet18，從 torchvision 的 ImageNet 預訓練權重開始（torchvision 的程式是 BSD-3；權重沒有另外標授權，
torchvision 提醒預訓練權重可能受訓練資料的條款限制，ImageNet 只允許非商用——能不能用在作品裡還沒決定），
換掉最後一層、輸出 1 個 logit。選它是因為只用 ReLU：
Hailo 的 DFC 量化最穩（RTMO 的 SiLU 做不了 equalization 吃過虧），Hailo Model Zoo 也有 10H 版。輸入 160×160
（頭部特寫裡臉約 50 px）。
增強：隨機裁切縮放、±15° 旋轉、左右翻轉、亮度對比、30% 灰階（模擬 IR）、模糊、變暗。取樣時每一組（影片、照片批次）
的每一類各佔一樣多，不讓長影片或某一類主導。**門檻固定 0.5，不在 hold 上調。**

流程：先在 calib 裡做「留一組」交叉驗證（每次拿掉一支影片或一批照片，看沒看過的那組準不準），
再用整個 calib 訓練最後的模型，在 hold 上評估一次。也在 VLM 實驗用過的 IMG_8429 頭部特寫上算一次，跟 VLM 直接比。

環境：.venv-vlm（有 PyTorch；主環境刻意不裝）。第一次會下載 torchvision 的 ResNet18 權重（約 45 MB）。
    .venv-vlm/Scripts/python -m experiments.airway_train
    .venv-vlm/Scripts/python -m experiments.airway_train --no-cv        # 跳過交叉驗證
輸出：outputs/airway/model.pt、hold_pred.csv（每張的機率）、終端機摘要；--onnx 另存 model.onnx（要裝 onnx）
"""

import argparse
import csv
import random
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
AIR = ROOT / "outputs" / "airway"
VLM = ROOT / "outputs" / "vlm_spike"
SIZE = 160
TH = 0.5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def dim(x):
    return x * random.uniform(0.3, 0.7)


TRAIN_TF = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15, fill=114),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
    T.RandomGrayscale(p=0.3),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
    T.ToDtype(torch.float32, scale=True),
    T.RandomApply([T.Lambda(dim)], p=0.2),
    T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize(SIZE, antialias=True), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf      # items：[(圖檔路徑, 目標 0/1), ...]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, target = self.items[i]
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return self.tf(torch.from_numpy(img).permute(2, 0, 1)), torch.tensor(float(target))


def items_of(rows):
    return [(AIR / "crops" / r["crop"], int(r["target"])) for r in rows]


def make_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def train(rows, epochs, seed, device, quiet=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    count = Counter((r["group"], r["target"]) for r in rows)
    weights = [1.0 / count[(r["group"], r["target"])] for r in rows]
    loader = DataLoader(Crops(items_of(rows), TRAIN_TF), batch_size=32, num_workers=0,
                        sampler=WeightedRandomSampler(weights, num_samples=len(rows), replacement=True))
    model = make_model().to(device)
    head = list(model.fc.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        if not quiet and (ep == 0 or ep == epochs - 1 or (ep + 1) % 4 == 0):
            print(f"    epoch {ep + 1}/{epochs}：loss {total / n:.3f}", flush=True)
    return model


@torch.no_grad()
def predict(model, items, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(Crops(items, EVAL_TF), batch_size=64, num_workers=0):
        probs += torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().tolist()
    return probs


def auc(probs, targets):
    """P(看得到的那張分數比看不到的高)；只有一類時回傳 None。"""
    pos = [p for p, t in zip(probs, targets) if t == 1]
    neg = [p for p, t in zip(probs, targets) if t == 0]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def report(title, rows, probs):
    vis = [p for r, p in zip(rows, probs) if r["target"] == "1"]
    hid = [p for r, p in zip(rows, probs) if r["target"] == "0"]
    a = auc(probs, [int(r["target"]) for r in rows])
    ok_v = sum(p >= TH for p in vis)
    ok_h = sum(p < TH for p in hid)
    print(f"  {title:<16} 看得到判對 {ok_v}/{len(vis)}｜看不到判對 {ok_h}/{len(hid)}"
          + (f"｜AUC {a:.3f}" if a is not None else ""))


def second_check(title, rows, probs, p_key=None):
    """偵測器（v3＋床位）判看得到的幀：漏掉的危險分類器說看不到幾幀、安全幀多叫幾次（跟 VLM 實驗同樣的定義）。"""
    shown = [(r, p) for r, p in zip(rows, probs) if r["v3_slot"] == "visible"]
    missed = [p for r, p in shown if r["label"] == "alert"]
    fine = [p for r, p in shown if r["label"] == "ok" and r["staged"] != "prone"]
    print(f"  {title:<16} 偵測器漏掉的危險補到 {sum(p < TH for p in missed)}/{len(missed)}"
          f"｜安全畫面多出的提醒 {sum(p < TH for p in fine)}/{len(fine)}")


def laptop_frames():
    """筆電 VLM（448、床頭範圍）那次的每幀結果。那次還跑了只能內部評估的影片，檔名後面會接它的名稱，
    所以用找的、不寫死：IMG_8429_IMG_8432 開頭、不是 336／chip／head／wholeframe 版本的那個。"""
    tags = ("_336", "_chip", "_head", "_wholeframe")
    found = [p for p in VLM.glob("frames_IMG_8429_IMG_8432*.csv") if not p.stem.endswith(tags)]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def vlm_compare(model, device):
    """VLM 實驗用過的 IMG_8429 頭部特寫（hold 那一半），同一批幀上跟 VLM 直接比。"""
    frames = VLM / "frames_IMG_8429_IMG_8432_336_head.csv"
    src = VLM / "head_crops" / "head_crops.csv"
    if not frames.exists() or not src.exists():
        return
    src = {r["crop"]: r["src"] for r in read_csv(src)}
    rows = [r for r in read_csv(frames) if r["video"] == "IMG_8429" and src.get(r["crop"][:-4]) == "head"]
    probs = predict(model, [(VLM / "head_crops" / (r["crop"][:-4] + ".png"), 0) for r in rows], device)
    print(f"\n== 跟 VLM 比：IMG_8429 的 VLM 取樣幀（有頭部特寫的 {len(rows)} 幀）")
    second_check("口鼻分類器", rows, probs)
    for name, path, th in (("VLM 筆電 448 床頭", laptop_frames(), 0.4),
                           ("VLM 晶片 頭部", VLM / "frames_IMG_8429_IMG_8432_chip_head.csv", 0.5)):
        if path is not None and path.exists():
            ref = {(r["video"], r["doll"], r["t"]): float(r["p_mouth"]) for r in read_csv(path)}
            ps = [ref[(r["video"], r["doll"], r["t"])] for r in rows]
            # 門檻不同：second_check 用 TH，這裡把 VLM 的分數平移成「同一個門檻」再比
            second_check(f"{name}（{th}）", rows, [p - th + TH for p in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cv", action="store_true", help="跳過 calib 裡的留一組交叉驗證")
    ap.add_argument("--onnx", action="store_true", help="另存 model.onnx（之後轉 HEF；要裝 onnx）")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in read_csv(AIR / "crops.csv") if r["crop"]]
    calib = [r for r in rows if r["half"] == "calib"]
    hold = [r for r in rows if r["half"] == "hold"]
    print(f"有頭部特寫的：calib {len(calib)}（看得到 {sum(r['target'] == '1' for r in calib)}）、"
          f"hold {len(hold)}（看得到 {sum(r['target'] == '1' for r in hold)}）｜{device}", flush=True)

    t0 = time.perf_counter()
    if not args.no_cv:
        print("\n== calib 裡留一組交叉驗證（拿掉的那組沒看過）")
        for g in sorted({r["group"] for r in calib}):
            model = train([r for r in calib if r["group"] != g], args.epochs, args.seed, device, quiet=True)
            test = [r for r in calib if r["group"] == g]
            report(g, test, predict(model, items_of(test), device))
        print(f"  （{time.perf_counter() - t0:.0f} 秒）")

    print(f"\n== 用整個 calib 訓練（{args.epochs} 個 epoch）", flush=True)
    model = train(calib, args.epochs, args.seed, device)
    torch.save(model.state_dict(), AIR / "model.pt")
    probs = predict(model, items_of(hold), device)
    with open(AIR / "hold_pred.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "group", "label", "staged", "note", "v3_slot", "p_visible"])
        writer.writeheader()
        for r, p in zip(hold, probs):
            writer.writerow({k: r[k] for k in ("name", "group", "label", "staged", "note", "v3_slot")}
                            | {"p_visible": round(p, 4)})

    print(f"\n== hold（只能評估的那一半；門檻 {TH}）")
    for g in sorted({r["group"] for r in hold}):
        idx = [i for i, r in enumerate(hold) if r["group"] == g]
        report(g, [hold[i] for i in idx], [probs[i] for i in idx])
    report("合計", hold, probs)
    side = [i for i, r in enumerate(hold) if r["label"] == "ok" and "側" in r["note"]]
    if side:
        print(f"  {'其中側臉':<16} 看得到判對 {sum(probs[i] >= TH for i in side)}/{len(side)}")
    vids = [i for i, r in enumerate(hold) if r["kind"] == "video"]
    print("\n== 當第二道檢查（hold 的影片幀；偵測器判看得到的那些）")
    second_check("口鼻分類器", [hold[i] for i in vids], [probs[i] for i in vids])
    n_none = Counter((r["group"], r["label"]) for r in read_csv(AIR / "crops.csv") if r["half"] == "hold" and not r["crop"])
    if n_none:
        print(f"  找不到人、沒有頭部特寫的幀（交給偵測器）：{dict(n_none)}")
    vlm_compare(model, device)

    if args.onnx:
        torch.onnx.export(model.cpu().eval(), torch.zeros(1, 3, SIZE, SIZE), str(AIR / "model.onnx"),
                          input_names=["image"], output_names=["logit"], opset_version=17, dynamo=False)
        print(f"→ {AIR / 'model.onnx'}")
    print(f"\n→ {AIR / 'model.pt'}、{AIR / 'hold_pred.csv'}（共 {time.perf_counter() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
