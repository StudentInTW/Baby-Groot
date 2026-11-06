"""
hailo/compare_yunet_chip.py
UGen300 上的 YuNet（hailo/yunet_hef.py）vs OpenCV FaceDetectorYN：同一張圖、同樣的尺寸，逐臉比分數和框。

送三種主流程實際會送的圖：整張圖長邊 640、長邊 1280（晶片版切成 640×640 小塊）、長邊 640 那張中心的 320×320
（像頭部裁切）。門檻 0.5 下找臉、IoU > 0.5 配對；最後一欄是門檻 0.7（主流程用的）以上的臉數差，所有圖加總。

  uv run python -m hailo.compare_yunet_chip model/yunet_A.hef            # 娃娃照片平均挑 40 張
  uv run python -m hailo.compare_yunet_chip model/yunet_A.hef --n 100
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from hailo.yunet_hef import YuNetHef
from pipeline.face_common import FACE_MODEL, FACE_SCALES, FACE_SCORE_TH

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "dataset" / "infant_jpg_without_video"   # 娃娃照片（eval_doll.py 用的那批）
TH = 0.5


def run(det, img):
    det.setInputSize((img.shape[1], img.shape[0]))
    _, f = det.detect(img)
    return np.zeros((0, 15), np.float32) if f is None else f


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def inputs_for(img):
    h, w = img.shape[:2]
    out = {}
    for side in FACE_SCALES:
        s = min(1.0, side / max(h, w))
        out[f"長邊 {side}"] = cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img
    small = out[f"長邊 {FACE_SCALES[0]}"]
    cy, cx = small.shape[0] // 2, small.shape[1] // 2
    out["中心 320×320"] = np.ascontiguousarray(small[max(0, cy - 160):cy + 160, max(0, cx - 160):cx + 160])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hef", type=Path)
    ap.add_argument("--images", default=str(IMG_DIR / "**" / "*.jpg"), help="圖片的 glob（預設：娃娃照片）")
    ap.add_argument("--n", type=int, default=40, help="平均挑幾張（預設 40）")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        raise SystemExit(f"找不到圖：{args.images}")
    paths = paths[::max(1, len(paths) // args.n)][:args.n]
    cv = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), TH)
    chip = YuNetHef(args.hef, TH)
    stats = {}
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)   # 中文路徑也讀得到
        for k, x in inputs_for(img).items():
            a, b = run(cv, x), run(chip, x)
            st = stats.setdefault(k, {"cv": 0, "chip": 0, "matched": 0, "sd": [], "bd": [], "flip": 0})
            st["cv"] += len(a)
            st["chip"] += len(b)
            used = set()
            for fa in a:
                best, bj = 0.5, None
                for j, fb in enumerate(b):
                    if j not in used and (v := iou(fa, fb)) > best:
                        best, bj = v, j
                if bj is not None:
                    used.add(bj)
                    st["matched"] += 1
                    st["sd"].append(abs(float(fa[14] - b[bj][14])))
                    st["bd"].append(float(np.abs(fa[:4] - b[bj][:4]).max()))
            st["flip"] += abs(int((a[:, 14] >= FACE_SCORE_TH).sum()) - int((b[:, 14] >= FACE_SCORE_TH).sum()))
    print(f"{args.hef.name} vs OpenCV：{len(paths)} 張，門檻 {TH} 找臉、IoU > 0.5 配對")
    for k, st in stats.items():
        sd, bd = np.array(st["sd"] or [0]), np.array(st["bd"] or [0])
        print(f"  {k:12s} 臉數 OpenCV {st['cv']}／晶片 {st['chip']}、配到 {st['matched']}｜"
              f"分數差 中位 {np.median(sd):.3f} 最大 {sd.max():.3f}｜框差 px 中位 {np.median(bd):.1f} 最大 {bd.max():.1f}｜"
              f"門檻 {FACE_SCORE_TH} 以上臉數差 {st['flip']}")


if __name__ == "__main__":
    main()
