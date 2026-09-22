"""
spike_posture.py —— 實驗（結論：此路不通，保留當紀錄，別重做）

問題：只用關鍵點的幾何特徵，能不能分嬰兒睡姿（尤其安全關鍵的「仰 vs 趴」）？
資料：SyRIP_Posture（train600 訓練、validate100 報告）。預設用 GT 關鍵點（隔離「幾何是否可分」，
不受偵測器誤差干擾）。分兩組特徵：
  full  —— 含臉部（眼距、鼻相對肩、可見臉點數）
  body  —— 只用身體（肩/髖/四肢），不看臉——近似俯視鏡頭看不到正面臉時能用的訊號
分類器：numpy softmax 邏輯回歸（環境沒有 sklearn）。

結論（2026/9/22，validate100）：坐 91%、站 80% 分得好，但**仰睡召回只有 7–21%**（幾乎全被判成趴）——
仰躺和趴睡的骨架投影幾乎一樣，差別在外觀（看到臉 vs 後腦勺/背），不在幾何。只用身體更慘。
→ 別做「用關鍵點分仰/趴」的姿態分類器；改強化「臉」這一關（sideface_spike.py，三級 ok/warn/alert）。
SyRIP 沒有「側躺」類別，側臉問題不能用它驗。

用法：uv run python spike_posture.py [--rtmo]  （--rtmo 改用 RTMO 偵測到的關鍵點做端到端）
SyRIP 只能內部評估：數字不進簡報/影片/GitHub。
"""
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
ZIP = ROOT / "dataset" / "SyRIP_Posture.zip"
CLASSES = ["Supine", "Prone", "Sitting", "Standing"]

# COCO 17
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(17)


def load(split):
    z = zipfile.ZipFile(ZIP)
    ann = next(n for n in z.namelist()
               if n.startswith(f"SyRIP_Posture/annotations/{split}/") and n.endswith(".json"))
    d = json.loads(z.read(ann))
    images = {im["id"]: im for im in d["images"]}
    out = []
    for a in d["annotations"]:
        im = images[a["image_id"]]
        k = np.array(a["keypoints"], float).reshape(17, 3)
        out.append({"file": im["file_name"], "posture": im["posture"], "kpts": k,
                    "path": f"SyRIP_Posture/images/{split}/{im['file_name']}"})
    return z, out


def pt(k, i):
    if k[i, 2] > 0 and not (k[i, 0] == 0 and k[i, 1] == 0):
        return k[i, :2]
    return None


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) if a is not None and b is not None else np.nan


def mid(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def features(k, mode):
    """從 17x3 關鍵點算特徵。mode: 'full' 或 'body'。缺的點 → NaN（之後用訓練均值補）。"""
    vis = [pt(k, i) for i in range(17)]
    pts = np.array([p for p in vis if p is not None])
    if len(pts) < 3:
        return np.full(13 if mode == "full" else 9, np.nan)
    bw = pts[:, 0].max() - pts[:, 0].min()
    bh = pts[:, 1].max() - pts[:, 1].min()
    scale = float(np.hypot(bw, bh)) + 1e-6
    ls, rs, lh, rh = vis[LSH], vis[RSH], vis[LHIP], vis[RHIP]
    sh_mid, hip_mid = mid(ls, rs), mid(lh, rh)
    center = pts.mean(axis=0)

    sh_w = dist(ls, rs) / scale
    hip_w = dist(lh, rh) / scale
    torso = dist(sh_mid, hip_mid) / scale
    sw_hw = (dist(ls, rs) + 1e-6) / (dist(lh, rh) + 1e-6)
    aspect = bw / (bh + 1e-6)
    limbs = [vis[i] for i in (LEL, REL, LWR, RWR, LKN, RKN, LAN, RAN)]
    spread = np.nanmean([dist(center, p) / scale for p in limbs])
    leg_v = np.nanmean([abs(vis[a][1] - vis[h][1]) / scale
                        for a, h in ((LAN, LHIP), (RAN, RHIP)) if vis[a] is not None and vis[h] is not None]) \
        if any(vis[a] is not None for a in (LAN, RAN)) else np.nan
    n_vis = len(pts) / 17.0

    feats = [sh_w, hip_w, torso, sw_hw, aspect, spread, leg_v, n_vis, (bh / (bw + 1e-6))]
    if mode == "full":
        eye = dist(vis[LEYE], vis[REYE]) / scale
        ear = dist(vis[LEAR], vis[REAR]) / scale
        n_face = sum(vis[i] is not None for i in (NOSE, LEYE, REYE, LEAR, REAR)) / 5.0
        if vis[NOSE] is not None and sh_mid is not None and hip_mid is not None:
            axis = np.array(sh_mid) - np.array(hip_mid)
            axis = axis / (np.linalg.norm(axis) + 1e-6)
            nose_proj = float(np.dot(np.array(vis[NOSE]) - np.array(sh_mid), axis)) / scale
        else:
            nose_proj = np.nan
        feats += [eye, ear, n_face, nose_proj]
    return np.array(feats, float)


def build(samples, mode):
    X = np.array([features(s["kpts"], mode) for s in samples])
    y = np.array([CLASSES.index(s["posture"]) for s in samples])
    return X, y


def softmax_train(X, y, n_cls, epochs=4000, lr=0.5, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.01, (d, n_cls))
    b = np.zeros(n_cls)
    Y = np.eye(n_cls)[y]
    for _ in range(epochs):
        z = X @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (X.T @ g + l2 * W)
        b -= lr * g.sum(axis=0)
    return W, b


def confusion(y, yp, n_cls):
    m = np.zeros((n_cls, n_cls), int)
    for a, p in zip(y, yp):
        m[a, p] += 1
    return m


def run(mode, tr, va):
    Xtr, ytr = build(tr, mode)
    Xva, yva = build(va, mode)
    mean = np.nanmean(Xtr, axis=0)
    std = np.where(np.isnan(Xtr), mean, Xtr).std(axis=0) + 1e-6

    def norm(X):
        return (np.where(np.isnan(X), mean, X) - mean) / std

    W, b = softmax_train(norm(Xtr), ytr, len(CLASSES))
    yp = (norm(Xva) @ W + b).argmax(axis=1)
    acc = (yp == yva).mean()
    print(f"\n===== 特徵組：{mode}（{Xtr.shape[1]} 維）=====")
    print(f"validate100 整體準確率：{acc:.1%}")
    cm = confusion(yva, yp, len(CLASSES))
    print("混淆矩陣（列＝真實，欄＝預測）：")
    print("            " + "".join(f"{c[:4]:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        row = "".join(f"{cm[i, j]:>9}" for j in range(len(CLASSES)))
        tot = cm[i].sum()
        print(f"{c:>10}  {row}   召回 {cm[i,i]/tot if tot else 0:.0%}（{cm[i,i]}/{tot}）")
    sp = [(a, p) for a, p in zip(yva, yp) if a in (0, 1)]
    both = [(a, p) for a, p in sp if p in (0, 1)]
    if both:
        n_sup = sum(a == 0 for a, _ in sp)
        n_pro = sum(a == 1 for a, _ in sp)
        print(f"仰 vs 趴（安全關鍵）：仰召回 {sum(a==0 and p==0 for a,p in sp)}/{n_sup}、"
              f"趴召回 {sum(a==1 and p==1 for a,p in sp)}/{n_pro}"
              f"；兩類都預測在仰/趴時二分準確率 {sum(a==p for a,p in both)/len(both):.0%}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtmo", action="store_true", help="改用 RTMO 偵測到的關鍵點（端到端）")
    args = ap.parse_args()
    _, tr = load("train600")
    _, va = load("validate100")
    print(f"train {len(tr)}（{dict(Counter(s['posture'] for s in tr))}）")
    print(f"validate {len(va)}（{dict(Counter(s['posture'] for s in va))}）")
    if args.rtmo:
        import cv2
        from infant_face_v3 import load_rtmo
        rp = load_rtmo()
        for split, samples in (("train600", tr), ("validate100", va)):
            z = zipfile.ZipFile(ZIP)
            for s in samples:
                img = cv2.imdecode(np.frombuffer(z.read(s["path"]), np.uint8), cv2.IMREAD_COLOR)
                dets = rp(img)
                if dets:
                    gt = s["kpts"][:5, :2][s["kpts"][:5, 2] > 0]
                    gt = gt.mean(axis=0) if len(gt) else np.array([img.shape[1] / 2, img.shape[0] / 2])
                    s["kpts"] = min(dets, key=lambda d: np.linalg.norm(d[1][:5, :2].mean(axis=0) - gt))[1]
                else:
                    s["kpts"] = np.zeros((17, 3))
            print(f"  RTMO 關鍵點取完：{split}")
    for mode in ("full", "body"):
        run(mode, tr, va)


if __name__ == "__main__":
    main()
