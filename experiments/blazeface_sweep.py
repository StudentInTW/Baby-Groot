"""
experiments/blazeface_sweep.py
縮小實驗的對照：同一組縮小畫面，換 MediaPipe BlazeFace（full-range／short-range）在床位範圍找臉，
跟 scale_sweep.py 的 YuNet（v3＋床位）比，看小臉時能不能補回「臉露出卻判看不到」的誤報、又會不會把被蓋住的臉也當成看得到。

MediaPipe 跟專案的 OpenCV 版本衝突，所以用獨立環境跑（不動專案環境）：
  uv run --no-project --python 3.12 --with mediapipe==0.10.21 python -m experiments.blazeface_sweep \
      --video outputs/field/<日期>/<name>.mp4 --slot x0,y0,x1,y1 --scales ... --every 5 \
      --yunet-csv outputs/scale_sweep/<scale_sweep 的輸出>.csv
參數跟 scale_sweep.py 一樣（畫面用同一個 make_canvas 產生，標註一樣直接讀 .csv 的按鍵）。MediaPipe 套件（含模型）是 Apache-2.0。
輸出：outputs/scale_sweep/<name>_blazeface_<時間>.csv／.txt。
"""

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from experiments.scale_sweep import NAMES, OUT_DIR, WANT, iter_frames, load_labels, make_canvas

THRESHOLDS = (0.5, 0.3)  # 0.5 是 MediaPipe 預設


def best_face(detector, crop):
    """回傳 (最高分數, 那張臉的寬度 px)；沒找到就是 (0, 0)。"""
    res = detector.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    best = (0.0, 0)
    for d in res.detections or []:
        score = float(d.score[0])
        if score > best[0]:
            best = (score, round(d.location_data.relative_bounding_box.width * crop.shape[1]))
    return best


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--scales", default="1,0.7,0.5,0.4,0.3,0.2,0.15")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--jpeg", type=int, default=80)
    ap.add_argument("--yunet-csv", help="scale_sweep.py 的輸出，用來並排比較")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    video = Path(args.video)
    labels = load_labels(args.csv or video.with_suffix(".csv"))
    slot = tuple(float(v) for v in args.slot.split(","))
    scales = sorted((float(s) for s in args.scales.split(",")), reverse=True)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(args.rot)
    fd = mp.solutions.face_detection
    full = fd.FaceDetection(model_selection=1, min_detection_confidence=0.05)
    short = fd.FaceDetection(model_selection=0, min_detection_confidence=0.05)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{video.stem}_blazeface_{datetime.now():%m%d_%H%M%S}"
    fields = ["frame", "t", "key", "scale", "full_score", "full_px", "short_score", "short_px"]
    rows, n, t0 = [], 0, time.time()
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t, key, frame in iter_frames(video, labels, args.every):
            for s in scales:
                canvas, map_slot = make_canvas(frame, s, args.jpeg)
                x0, y0, x1, y1 = map_slot(slot)
                h, wd = canvas.shape[:2]
                crop = canvas[int(y0 * h):int(y1 * h), int(x0 * wd):int(x1 * wd)]
                if rot is not None:
                    crop = cv2.rotate(crop, rot)
                fs, fp = best_face(full, crop)
                ss, sp = best_face(short, crop)
                row = {"frame": i, "t": t, "key": key, "scale": s, "full_score": round(fs, 3), "full_px": fp,
                       "short_score": round(ss, 3), "short_px": sp}
                w.writerow(row)
                rows.append(row)
            n += 1
            if n % 50 == 0:
                print(f"{n} 幀，{time.time() - t0:.0f} 秒", flush=True)
            if args.limit and n >= args.limit:
                break

    yunet = {}
    if args.yunet_csv:
        for r in csv.DictReader(open(args.yunet_csv, encoding="utf-8-sig")):
            yunet[(int(r["frame"]), float(r["scale"]))] = r

    keys = []
    for r in rows:
        if r["key"] not in keys:
            keys.append(r["key"])
    lines = [f"{video.name}  床位 {slot}  每 {args.every} 幀取 1 幀、共 {n} 幀  JPEG {args.jpeg}",
             "每格是「判對」的比例：看得到的擺法＝判成看得到；看不到的擺法＝判成看不到（判成看得到就是漏報）。",
             "YuNet＝現在的 v3＋床位；BF 全距／BF 近距＝BlazeFace full-range／short-range（分數門檻）；"
             "YuNet或BF＝兩個有一個找到臉就算看得到（給「誤報少多少、漏報多多少」看）"]
    for s in scales:
        rs = [r for r in rows if r["scale"] == s]
        yx = [yunet.get((r["frame"], s)) for r in rs]
        px = [int(y["face_px"]) for y in yx if y and int(y["face_px"])]
        lines.append(f"\n縮放 {s:g}（YuNet 量到的臉寬中位數 {statistics.median(px) if px else 0:.0f} px）")
        hdr = ["擺法", "幀數", "YuNet"] + [f"BF全距≥{th}" for th in THRESHOLDS] + [f"BF近距≥{THRESHOLDS[0]}",
                                                                                f"YuNet或BF全距≥{THRESHOLDS[0]}"]
        lines.append(" | ".join(hdr))
        for key in keys:
            want = WANT[key]
            g = [(r, yunet.get((r["frame"], s))) for r in rs if r["key"] == key]
            if not g or want is None:
                continue

            def ok(visible):
                return (want == "visible") == visible
            ys = [y for _, y in g if y]
            cells = [f"{sum(ok(y['v3_slot'] == 'visible') for y in ys) / len(ys):.0%}" if ys else "—"]
            for th in THRESHOLDS:
                cells.append(f"{sum(ok(r['full_score'] >= th) for r, _ in g) / len(g):.0%}")
            cells.append(f"{sum(ok(r['short_score'] >= THRESHOLDS[0]) for r, _ in g) / len(g):.0%}")
            if ys:
                cells.append(f"{sum(ok(y is not None and (y['v3_slot'] == 'visible' or r['full_score'] >= THRESHOLDS[0])) for r, y in g) / len(g):.0%}")
            else:
                cells.append("—")
            lines.append(" | ".join([f"{key} {NAMES[key]}", str(len(g))] + cells))
    text = "\n".join(lines)
    Path(f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"→ {stem}.csv / .txt")


if __name__ == "__main__":
    main()
