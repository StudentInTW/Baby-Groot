"""
hailo/prep_calib.py
準備 YuNet、RTMO 量化用的校正圖、和量化前後比對用的圖，存成 hailo/calib/*.npy（不進 repo）。

在 repo 根目錄、專案環境執行（要 RTMO 找頭部，跟主流程一樣）：
  uv run python -m hailo.prep_calib --coco-dir hailo/calib/coco
  uv run python -m hailo.prep_calib --model rtmo --coco-dir hailo/calib/coco     # RTMO 的，檔名前面加 rtmo_
  --coco-dir 裡要有 annotations_trainval2017.zip（http://images.cocodataset.org/annotations/annotations_trainval2017.zip）；
  有 val2017.zip 就從裡面讀圖，沒有就只下載挑到的那幾張到 val2017/（不用下載整包 5000 張）。

做法：
- 娃娃的圖不自己裁切，而是把主流程原封不動跑一次（crib_detect.detect、infant_face_v4.detect_faces_v4），
  把每次送進 YuNet 的圖記下來：整張圖、每個人的頭部、床位範圍。這樣校正圖跟晶片上實際會收到的一模一樣。
  每張圖跑三種條件：原圖、灰階、暗光 ×0.3（evaluation/degrade.py，跟評估一樣）。
- HEF 的輸入固定 640×640：長寬都不超過 640 的圖放左上角、右下補 0（yunet_decode.pad_to，OpenCV 內部也這樣補）；
  超過 640 的（整張圖的第二個尺度，長邊 1280）先不收，怎麼接還沒決定（hailo/結果.md「接上主流程前要處理」）。
- 娃娃資料依影片和拍攝批次分兩半：calib 給校正，hold 留著比對量化前後。照片相鄰幾張幾乎一樣，
  所以整批分，不逐張隨機分。網路影片和真實嬰兒的圖不收。
- COCO 2017 val 只挑允許商用的授權（CC BY、No known copyright、美國政府作品），檔名和授權記在 coco_meta.csv。
- RTMO（--model rtmo）：主流程每一幀只跑一次 RTMO、吃整張圖，所以直接用 rtmlib 自己的前處理
  （RTMO.preprocess：等比例縮放到 640×640、可以放大、放左上角、其餘補 114），不用跑整個主流程。
  娃娃的分法、條件、COCO 挑的圖（同一個 seed）都跟 YuNet 一樣。

輸出（hailo/calib/）：coco.npy、coco_hold.npy、doll_calib.npy、doll_hold.npy（RTMO 的前面加 rtmo_），
都是 (N, 640, 640, 3) 的 BGR uint8；每個檔案旁邊有 *_meta.csv，記每張圖的來源。
"""

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from evaluation import video_labels
from evaluation.degrade import degrade, tag
from evaluation.eval_doll import IMG_DIR, LABELS
from hailo.yunet_decode import SIZE, pad_to
from pipeline import crib_detect, infant_face_v4, infant_face_v5
from pipeline.face_common import FACE_MODEL, FACE_SCORE_TH, ROOT
from pipeline.infant_face_v3 import load_rtmo, make_persons

OUT = ROOT / "hailo" / "calib"
SPLITS = {
    # 每床鏡頭的兩支（8429、8432）各放一邊；8429 比較長、有空床，留給比對。
    # 11/08 台南那批（一隻照護娃娃、每床直拍：影片 IMG_8735–8744、照片第 4 批）整批放 calib；
    # IMG_8743 每一幀都有手臂，沒有標註、不收。rtmo_D1.hef、yunet_A.hef 的校正圖是加這批之前做的。
    "calib": {"videos": ("IMG_8115", "IMG_8150", "IMG_8432",
                         "IMG_8735", "IMG_8739", "IMG_8740", "IMG_8741", "IMG_8742", "IMG_8744"),
              "batches": ("2", "3", "4")},
    # 嬰兒翻身：AI 生成的灰階夜視俯拍，唯一像 IR 的畫面；授權（生成工具的條款）確認前只當比對用
    "hold": {"videos": ("IMG_8116", "IMG_8184", "IMG_8429", "嬰兒翻身"), "batches": ("1",)},
}
CONDITIONS = ((False, 0.0), (True, 0.0), (False, 0.3))   # 原圖、灰階、暗光 ×0.3
COCO_LICENSES = (4, 7, 8)  # COCO 的授權代號：4 CC BY 2.0、7 No known copyright restrictions、8 United States Government Work


class Recorder:
    """包住 OpenCV 的 FaceDetectorYN：每次 detect 先把送進 YuNet 的圖記下來，再照常偵測。"""

    def __init__(self, det):
        self.det, self.calls = det, []

    def __getattr__(self, name):
        return getattr(self.det, name)

    def detect(self, img):
        caller = sys._getframe(1).f_code.co_name   # detect_faces_lm／head_faces_lm／slot_faces
        kind = "head" if "head" in caller else "slot" if "slot" in caller else "whole"
        self.calls.append((kind, img.copy()))
        return self.det.detect(img)


class Reservoir:
    """從不知道總數的一串圖裡，平均隨機留 n 張（固定 seed，可重現）。"""

    def __init__(self, n, seed):
        self.n, self.rng, self.items, self.seen = n, random.Random(seed), [], 0

    def add(self, img, meta):
        self.seen += 1
        if len(self.items) < self.n:
            self.items.append((img, meta))
        else:
            j = self.rng.randrange(self.seen)
            if j < self.n:
                self.items[j] = (img, meta)


def photo_batches():
    """{照片: 拍攝批次}。"""
    with open(LABELS, encoding="utf-8-sig") as fh:
        return {r["file"]: r["batch"] for r in csv.DictReader(fh)}


def sources(split, every):
    """這一半的每張照片、每支影片每 every 秒一幀：產生 (來源名稱, 圖, 床位, 方向, 雜訊 seed)。照片沒有床位。"""
    batches = SPLITS[split]["batches"]
    for f, b in sorted(photo_batches().items()):
        if b in batches:
            img = cv2.imread(str(IMG_DIR / f))
            if img is not None:
                yield f, img, None, None, 0          # eval_doll 的 degrade 用預設 seed 0
    ann = video_labels.load()
    for name in SPLITS[split]["videos"]:
        slots = {d: s["rect"] for d, s in ann[name]["slots"].items()}
        rots = {d: s["rot"] for d, s in ann[name]["slots"].items()}
        cap = cv2.VideoCapture(str(video_labels.find_video(name)))
        step = max(1, round(cap.get(cv2.CAP_PROP_FPS) * every))
        i = 0
        while cap.grab():
            if i % step == 0:
                _, frame = cap.retrieve()
                yield f"{name}@{i}", frame, slots, rots, i   # eval_video 的雜訊 seed＝幀號
            i += 1
        cap.release()


def rtmo_model():
    """跟 infant_face_v3.load_rtmo 同一個模型、同樣的設定；回傳 rtmlib 的 RTMO 物件（要用它的 preprocess）。"""
    from rtmlib import Body
    return Body(pose="rtmo", mode="balanced", backend="onnxruntime", device="cpu").pose_model


def yunet_input(img):
    """COCO 給 YuNet：跟主流程整張圖的第一個尺度一樣，長邊縮到 640、只縮不放大，再補邊到 640×640。"""
    h, w = img.shape[:2]
    s = min(1.0, SIZE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)))
    return pad_to(img), img.shape[1], img.shape[0]


def doll_split(split, n, every, seed, model="yunet"):
    keep, big, t0 = Reservoir(n, seed), Counter(), time.perf_counter()
    if model == "rtmo":
        rtmo = rtmo_model()
        for k, (name, img, _, _, noise_seed) in enumerate(sources(split, every), 1):
            for gray, dim in CONDITIONS:
                x = degrade(img, gray, dim, seed=noise_seed)
                keep.add(rtmo.preprocess(x)[0], {"source": name, "cond": tag(gray, dim) or "orig", "kind": "whole",
                                                 "w": x.shape[1], "h": x.shape[0]})
        print(f"{split}：RTMO 呼叫 {keep.seen} 次（{time.perf_counter() - t0:.0f}s），留 {len(keep.items)} 張")
        return keep.items
    run_pose = load_rtmo()
    rec = Recorder(cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH))
    det_v5 = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), infant_face_v5.FACE_TH_V5)
    for k, (name, img, slots, rots, noise_seed) in enumerate(sources(split, every), 1):
        for gray, dim in CONDITIONS:
            x = degrade(img, gray, dim, seed=noise_seed)
            rec.calls = []
            if slots is None:
                infant_face_v4.detect_faces_v4(rec, x, make_persons(run_pose(x)))
            else:
                crib_detect.detect(run_pose, rec, det_v5, x, slots, rots)
            for kind, inp in rec.calls:
                h, w = inp.shape[:2]
                if max(h, w) > SIZE:
                    big[kind] += 1
                    continue
                keep.add(pad_to(inp), {"source": name, "cond": tag(gray, dim) or "orig", "kind": kind, "w": w, "h": h})
        if k % 20 == 0:
            print(f"  {split}：{k} 張來源、{keep.seen} 次 YuNet 呼叫（{time.perf_counter() - t0:.0f}s）", flush=True)
    print(f"{split}：YuNet 呼叫 {keep.seen + sum(big.values())} 次，超過 640 沒收 {dict(big)}，留 {len(keep.items)} 張")
    return keep.items


def coco(coco_dir, n, seed, to_input=yunet_input):
    with zipfile.ZipFile(coco_dir / "annotations_trainval2017.zip") as z:
        info = json.loads(z.read("annotations/instances_val2017.json"))
    names = {lic["id"]: lic["name"] for lic in info["licenses"]}
    count = Counter(im["license"] for im in info["images"])
    print("COCO val2017 授權：" + "、".join(f"{names[k]} {count[k]}" for k in sorted(count)))
    ok = sorted((im for im in info["images"] if im["license"] in COCO_LICENSES), key=lambda im: im["id"])
    if len(ok) < n:
        raise SystemExit(f"可商用授權的圖只有 {len(ok)} 張，不到 {n}")
    picked = random.Random(seed).sample(ok, len(ok))   # 全部打亂：前 n 張給校正，剩下的留著比對
    folder = coco_dir / "val2017"
    if not (coco_dir / "val2017.zip").exists():   # 沒有整包就只下載挑到的圖（來源同一個網站）
        fetch(folder, [im["coco_url"] for im in picked])
    items = []
    z = zipfile.ZipFile(coco_dir / "val2017.zip") if (coco_dir / "val2017.zip").exists() else None
    for im in picked:
        raw = z.read(f"val2017/{im['file_name']}") if z else (folder / im["file_name"]).read_bytes()
        inp, w, h = to_input(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
        items.append((inp, {"source": im["file_name"], "license": names[im["license"]],
                            "flickr_url": im.get("flickr_url", ""), "w": w, "h": h}))
    print(f"COCO：可商用 {len(ok)} 張，隨機挑 {n} 張校正、{len(ok) - n} 張比對（seed {seed}）")
    return items[:n], items[n:]


def fetch(folder, urls, workers=8):
    """把還沒下載的圖抓下來（同時 workers 個連線）。"""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import urlopen
    folder.mkdir(parents=True, exist_ok=True)
    todo = [u for u in urls if not (folder / u.rsplit("/", 1)[1]).exists()]

    def get(url):
        path = folder / url.rsplit("/", 1)[1]
        with urlopen(url, timeout=60) as r:
            data = r.read()
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").replace(path)

    print(f"COCO：下載 {len(todo)} 張到 {folder}")
    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(get, todo), 1):
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


def save(name, items):
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}.npy", np.stack([img for img, _ in items]))
    with open(OUT / f"{name}_meta.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(items[0][1]))
        wr.writeheader()
        wr.writerows(meta for _, meta in items)
    kinds = Counter((m.get("kind", "whole"), m.get("cond", "orig")) for _, m in items)
    print(f"存好 {OUT / name}.npy：{len(items)} 張；種類 {dict(sorted(kinds.items()))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yunet", choices=("yunet", "rtmo"))
    ap.add_argument("--coco-dir", type=Path, help="放 annotations_trainval2017.zip 的資料夾；不給就不做 COCO")
    ap.add_argument("--coco-n", type=int, default=768,
                    help="COCO 校正張數（val2017 可商用的只有 862 張；剩下的存成 coco_hold 比對用）")
    ap.add_argument("--calib-n", type=int, default=512, help="娃娃校正圖張數")
    ap.add_argument("--hold-n", type=int, default=384, help="娃娃比對圖張數")
    ap.add_argument("--every", type=float, default=3.0, help="影片每幾秒取一幀")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-doll", action="store_true")
    args = ap.parse_args()

    pre = "rtmo_" if args.model == "rtmo" else ""
    if args.coco_dir:
        if args.model == "rtmo":
            rtmo = rtmo_model()
            to_input = lambda img: (rtmo.preprocess(img)[0], img.shape[1], img.shape[0])   # noqa: E731
        else:
            to_input = yunet_input
        calib, hold = coco(args.coco_dir, args.coco_n, args.seed, to_input)
        save(pre + "coco", calib)
        if hold:
            save(pre + "coco_hold", hold)
    if not args.skip_doll:
        save(pre + "doll_calib", doll_split("calib", args.calib_n, args.every, args.seed, args.model))
        save(pre + "doll_hold", doll_split("hold", args.hold_n, args.every, args.seed + 1, args.model))


if __name__ == "__main__":
    main()
