"""
experiments/scale_sweep.py
縮小實驗：臉要多大才判得對？

把一支每床鏡頭的錄影（rtsp_check --record 存的 .mp4，加上同名 .csv 的按鍵標註）每一幀縮小，
貼到 1920×1080 灰底畫面的右下角，模擬「同一張床在比較遠、比較廣的鏡頭裡」，再跑跟現場一樣的床位判讀
（crib_detect：RTMO＋YuNet，v3＋床位是基準）。看臉縮到幾像素時，每種擺法開始判錯。

這是模擬：真的遠距離鏡頭還有鏡頭變形、拍攝角度、對焦、串流壓縮（這裡用 JPEG 近似），結果只能當大概的門檻，
而且比真實樂觀。錄影來自場域，只能內部評估：檔名用參數給，不寫進程式。

標註直接讀 .csv 的按鍵（擺法和應該的判定用 rtsp_check 的 LABELS；按 0 和沒按的幀不算）。
現場按錯的地方先改好 .csv 再跑（原始按鍵另存 <name>_raw.csv），不在這裡用參數補。

用法（在 repo 根目錄）：
  uv run python -m experiments.scale_sweep --video outputs/field/<日期>/<name>.mp4 --slot x0,y0,x1,y1 \
      --scales 1,0.7,0.5,0.4,0.3,0.2,0.15 --every 5
輸出：outputs/scale_sweep/<name>_<時間>.csv（每幀×每個縮放一列）、.txt（每個縮放×擺法的結果表）。
make_canvas()、load_labels()、iter_frames() 只用 cv2／numpy，BlazeFace 的比較程式（另一個環境）也用這幾個。
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from experiments.rtsp_check import LABELS  # noqa: E402  跟現場標註同一份按鍵表

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "scale_sweep"
CANVAS_W, CANVAS_H = 1920, 1080
GRAY = 128
NAMES = {key: zh for key, (_, zh, _) in LABELS.items()}
WANT = {key: want for key, (_, _, want) in LABELS.items()}  # visible／hidden；空床是 None


def load_labels(csv_path):
    """回傳每一幀的 (t, key)；key 是 None 代表不算（沒按鍵或按 0）。rtsp_check 的錄影每一幀對應 CSV 的一列。"""
    out = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        key = r["key"]
        out.append((float(r["t"]), key if key in LABELS and key != "0" else None))
    return out


def make_canvas(frame, scale, jpeg=80):
    """把整幀縮成 scale 倍，貼在 1920×1080 灰底的右下角；回傳 (畫面, 換算床位座標的函式)。
    scale=1 時畫面就是原圖（只做同樣的 JPEG 壓縮，讓各縮放可以比較）。"""
    h, w = frame.shape[:2]
    if scale >= 1:
        canvas, (ox, oy, sw, sh) = frame.copy(), (0, 0, w, h)
        cw, ch = w, h
    else:
        sw, sh = round(w * scale), round(h * scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        cw, ch = CANVAS_W, CANVAS_H
        canvas = np.full((ch, cw, 3), GRAY, np.uint8)
        ox, oy = cw - sw, ch - sh
        canvas[oy:oy + sh, ox:ox + sw] = small
    if jpeg:
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        canvas = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def map_slot(slot):
        x0, y0, x1, y1 = slot
        return ((ox + x0 * sw) / cw, (oy + y0 * sh) / ch, (ox + x1 * sw) / cw, (oy + y1 * sh) / ch)

    return canvas, map_slot


def iter_frames(video, labels, every):
    """依序讀錄影，只回傳有標註、而且是第 every 幀的那些：(幀號, t, key, 畫面)。"""
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(labels):
            break
        t, key = labels[i]
        if key and i % every == 0:
            yield i, t, key, frame
        i += 1
    cap.release()


def longest_wrong_run(rows):
    """依時間排序，連續判錯最長幾秒（取樣間隔內的近似值）。"""
    best, start = 0.0, None
    for r in sorted(rows, key=lambda r: r["t"]):
        if not r["ok"]:
            start = r["t"] if start is None else start
            best = max(best, r["t"] - start)
        else:
            start = None
    return best


def summarize(rows, scales):
    keys = []
    for r in rows:
        if r["key"] not in keys:
            keys.append(r["key"])
    base_px = [r["face_px"] for r in rows if r["scale"] == max(scales) and r["face_px"]]
    lines = []
    for s in scales:
        rs = [r for r in rows if r["scale"] == s]
        px = [r["face_px"] for r in rs if r["face_px"]]
        nominal = statistics.median(base_px) * s if base_px else 0
        lines.append(f"\n縮放 {s:g}：臉寬約 {nominal:.0f} px（量到的中位數 {statistics.median(px) if px else 0:.0f}，"
                     f"{len(px)}/{len(rs)} 幀有找到臉）")
        lines.append("擺法 | 幀數 | 判對 | 床位裡找到人 | 連續判錯最長（秒）")
        for key in keys:
            g = [r for r in rs if r["key"] == key]
            if not g:
                continue
            if WANT[key] is None:
                right = f"床位裡沒人 {sum(r['persons_in_slot'] == 0 for r in g) / len(g):.0%}"
                run = "—"
            else:
                right = f"{sum(r['ok'] for r in g) / len(g):.0%}"
                run = f"{longest_wrong_run(g):.0f}"
            lines.append(f"{key} {NAMES[key]} | {len(g)} | {right} | "
                         f"{sum(r['persons_in_slot'] > 0 for r in g) / len(g):.0%} | {run}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", help="預設跟 --video 同名的 .csv")
    ap.add_argument("--slot", required=True, help="原畫面的床位 x0,y0,x1,y1（0–1）")
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--scales", default="1,0.7,0.5,0.4,0.3,0.2,0.15")
    ap.add_argument("--every", type=int, default=5, help="每幾幀取一幀（錄影每秒 5 幀，5 就是每秒 1 幀）")
    ap.add_argument("--jpeg", type=int, default=80, help="模擬串流壓縮的 JPEG 品質（0 不壓）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前幾幀（試跑用）")
    args = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from pipeline import crib_detect, infant_face_v3, infant_face_v5
    from pipeline.face_common import FACE_SCORE_TH, create_face_detector
    infant_face_v3.DET_TH = 0.15  # 跟 eval_video.py、rtsp_check.py 一樣
    run_pose = infant_face_v3.load_rtmo()
    det = create_face_detector(FACE_SCORE_TH)
    det_v5 = create_face_detector(infant_face_v5.FACE_TH_V5)

    video = Path(args.video)
    labels = load_labels(args.csv or video.with_suffix(".csv"))
    slot = tuple(float(v) for v in args.slot.split(","))
    scales = sorted((float(s) for s in args.scales.split(",")), reverse=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{video.stem}_{datetime.now():%m%d_%H%M%S}"
    fields = ["frame", "t", "key", "scale", "v3_slot", "v4_slot", "v5_slot", "ok", "persons", "persons_in_slot",
              "faces_in_slot", "face_px", "slot_face_scores", "ms"]
    rows, n, t0 = [], 0, time.time()
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t, key, frame in iter_frames(video, labels, args.every):
            for s in scales:
                canvas, map_slot = make_canvas(frame, s, args.jpeg)
                sl = map_slot(slot)
                tt = time.time()
                res, persons, faces = crib_detect.detect(run_pose, det, det_v5, canvas, {"S": sl}, {"S": args.rot})
                ms = (time.time() - tt) * 1000
                h, wd = canvas.shape[:2]
                inside = [f_ for f_ in faces if sl[0] * wd <= (f_[0] + f_[2]) / 2 <= sl[2] * wd
                          and sl[1] * h <= (f_[1] + f_[3]) / 2 <= sl[3] * h]
                r = res["S"]
                want = WANT[key]
                row = {"frame": i, "t": t, "key": key, "scale": s, "v3_slot": r["v3_slot"],
                       "v4_slot": r["v4_slot"], "v5_slot": r["v5_slot"], "ok": want is None or r["v3_slot"] == want,
                       "persons": len(persons), "persons_in_slot": r["persons"], "faces_in_slot": len(inside),
                       "face_px": round(max((f_[2] - f_[0] for f_ in inside), default=0)),
                       "slot_face_scores": r["slot_face_scores"], "ms": round(ms)}
                w.writerow(row)
                rows.append(row)
            n += 1
            if n % 50 == 0:
                print(f"{n} 幀，{time.time() - t0:.0f} 秒", flush=True)
            if args.limit and n >= args.limit:
                break
    head = (f"{video.name}  床位 {slot}  rot {args.rot}  每 {args.every} 幀取 1 幀、共 {n} 幀  JPEG {args.jpeg}  "
            f"RTMO {'UGen300' if os.environ.get('FG_RTMO_HEF') else 'CPU'}\n判定：v3＋床位（基準）")
    text = head + "\n" + summarize(rows, scales)
    Path(f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"→ {stem}.csv / .txt")


if __name__ == "__main__":
    main()
