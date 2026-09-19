"""
eval_syrip.py
用 SyRIP_Posture（700 張真實嬰兒圖，含睡姿標籤）評估管線，並比較三個姿態模型。

  - 偵測率：GT 嬰兒框有沒有被找到（IoU ≥ 0.5）
  - 誤報率：GT 標註「臉看得到」的嬰兒，系統卻判「臉看不見 → ALERT」的比例（只算有找到的嬰兒）
  - 另外列出 29 張 GT「臉看不到」的圖有幾張判 ALERT（張數少、情境雜，只當參考）

人臉判斷（--face）：
  v2 —— 整張圖找臉，再配給頭部最近的人（infant_face_spike.py）
  v3 —— v2 再加上「每個人的頭部區域轉正後找臉」和「合併重複框」（infant_face_v3.py）
姿態模型（--models）：
  yolov8  —— Ultralytics YOLOv8n-pose（AGPL）
  rtmpose —— rtmlib：YOLOX-m 找人 + RTMPose-m 估姿態（Apache-2.0）
  rtmo    —— rtmlib：RTMO-m 一次找出所有人和姿態（Apache-2.0）
三個模型的找人門檻統一 0.3，比較才公平（v2 的 YOLOv8 平常用預設 0.25）。

SyRIP 的真實圖來自 Google Images／YouTube，只能內部評估：
outputs/syrip_eval/ 裡的東西不可放進簡報、影片或 GitHub，簡報只能引用數字。

用法：
    uv run python eval_syrip.py                       # 人臉判斷 v2、三個模型
    uv run python eval_syrip.py --face v3             # 人臉判斷 v3（輸出檔名加 _v3）
    uv run python eval_syrip.py --limit 10            # 先跑 10 張試試
    uv run python eval_syrip.py --models yolov8 rtmo  # 只跑指定模型
"""

import argparse
import csv
import json
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from infant_face_spike import (  # v2 的人臉偵測、配對規則與門檻
    FACE_MODEL, FACE_SCORE_TH, POSE_MODEL, assign_faces, detect_faces, make_person,
)
from infant_face_v3 import detect_faces_v3, load_rtmo, make_persons

ROOT = Path(__file__).parent
SYRIP_ZIP = ROOT / "dataset" / "SyRIP_Posture.zip"
OUT_DIR = ROOT / "outputs" / "syrip_eval"
DET_TH = 0.3          # YOLOv8、YOLOX 的找人門檻（RTMO 用 infant_face_v3.DET_TH，同為 0.3）
IOU_TH = 0.5          # 預測框和 GT 框 IoU ≥ 這個值，才算找到這個嬰兒
MAX_SHEET = 12        # 每個模型最多存幾張誤報縮圖，供人工複查


def load_syrip():
    """回傳 (zip, 樣本清單)。每個樣本：檔名、zip 內路徑、睡姿、GT 框(xyxy)、GT 臉看不看得到、GT 頭中心。"""
    z = zipfile.ZipFile(SYRIP_ZIP)
    samples = []
    for split in ("train600", "validate100"):
        ann_path = next(n for n in z.namelist()
                        if n.startswith(f"SyRIP_Posture/annotations/{split}/") and n.endswith(".json"))
        data = json.loads(z.read(ann_path))
        images = {im["id"]: im for im in data["images"]}
        for a in data["annotations"]:
            im = images[a["image_id"]]
            k = np.array(a["keypoints"], dtype=float).reshape(17, 3)
            x, y, w, h = a["bbox"]
            samples.append({
                "file": im["file_name"],
                "path": f"SyRIP_Posture/images/{split}/{im['file_name']}",
                "posture": im["posture"],
                "box": np.array([x, y, x + w, y + h]),
                "face_visible": int((k[:3, 2] == 2).sum()) >= 2,  # 鼻子、雙眼至少兩個標為看得到
                "head": k[:5, :2].mean(axis=0),
            })
    return z, samples


def load_models(names):
    """回傳 {名稱: 找人函式}；找人函式輸入 BGR 圖，輸出 [(box_xyxy, kpts(17,3)), ...]。"""
    models = {}
    if "yolov8" in names:
        from ultralytics import YOLO
        yolo = YOLO(str(POSE_MODEL))

        def run_yolo(img):
            r = yolo(img, conf=DET_TH, verbose=False)[0]
            if r.keypoints is None:
                return []
            return list(zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()))
        models["yolov8"] = run_yolo

    if "rtmpose" in names:
        from rtmlib import Body
        body = Body(mode="balanced", backend="onnxruntime", device="cpu")
        body.det_model.score_thr = DET_TH

        def run_rtmpose(img):
            boxes = body.det_model(img)
            if len(boxes) == 0:  # 不能把空的框丟給 RTMPose：它會把整張圖當成一個人
                return []
            kpts, scores = body.pose_model(img, bboxes=boxes)
            return [(b, np.concatenate([k, s[:, None]], axis=1)) for b, k, s in zip(boxes, kpts, scores)]
        models["rtmpose"] = run_rtmpose

    if "rtmo" in names:
        models["rtmo"] = load_rtmo()
    return models


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def thumbnail(img, sample, person, faces):
    """誤報複查用縮圖：藍＝GT 嬰兒框、紅＝系統找到的嬰兒框和頭部配對範圍、黃＝偵測到的臉。"""
    vis = img.copy()
    t = max(2, round(max(img.shape[:2]) / 400))
    for (x1, y1, x2, y2), color in ((sample["box"], (255, 0, 0)), (person["box"], (0, 0, 255))):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
    cv2.circle(vis, tuple(int(v) for v in person["head"]), int(person["head_r"]), (0, 0, 255), t)
    for x1, y1, x2, y2, _ in faces:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), t)
    h, w = vis.shape[:2]
    s = min(400 / w, 280 / h)
    small = cv2.resize(vis, (round(w * s), round(h * s)))
    thumb = np.full((300, 400, 3), 255, np.uint8)
    thumb[20:20 + small.shape[0], :small.shape[1]] = small
    cv2.putText(thumb, f"{sample['posture']} {sample['file']}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    return thumb


def save_sheet(tiles, path, cols=4):
    if not tiles:
        return
    tiles = tiles + [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def report(rows, secs, n_images, face):
    print(f"\n== SyRIP_Posture：{n_images} 張真實嬰兒圖｜人臉判斷 {face}"
          f"（人臉偵測平均 {1000 * secs['face'] / n_images:.0f} ms／張）")
    for m, rs in rows.items():
        found = [r for r in rs if r["found"]]
        fv = [r for r in found if r["gt_face_visible"]]
        fh = [r for r in found if not r["gt_face_visible"]]
        fa = sum(r["alert"] for r in fv)
        no_face = sum(r["alert"] and r["faces_in_image"] == 0 for r in fv)
        herr = np.median([r["head_err"] for r in found]) if found else float("nan")
        print(f"\n[{m}] 偵測率 {len(found) / len(rs):.1%}（{len(found)}/{len(rs)}）"
              f"｜誤報率 {fa / max(1, len(fv)):.1%}（{fa}/{len(fv)}，其中整張圖沒找到任何臉 {no_face}）"
              f"｜臉看不到→ALERT {sum(r['alert'] for r in fh)}/{len(fh)}"
              f"｜頭部誤差中位數 {herr:.3f}（÷GT 框對角線）"
              f"｜額外偵測 {np.mean([r['extra_persons'] for r in rs]):.2f} 人／張"
              f"｜姿態模型 {1000 * secs[m] / len(rs):.0f} ms／張")
        by = defaultdict(lambda: [0, 0])
        for r in fv:
            by[r["posture"]][0] += r["alert"]
            by[r["posture"]][1] += 1
        print("    誤報依睡姿：" + "、".join(f"{p} {a}/{t}（{a / t:.0%}）" for p, (a, t) in sorted(by.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov8", "rtmpose", "rtmo"])
    ap.add_argument("--face", choices=["v2", "v3"], default="v2")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張（0＝全部）")
    args = ap.parse_args()

    z, samples = load_syrip()
    if args.limit:
        samples = samples[:args.limit]
    models = load_models(args.models)
    face_det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = {m: [] for m in models}
    sheets = {m: [] for m in models}
    secs = defaultdict(float)
    for n, s in enumerate(samples, 1):
        img = cv2.imdecode(np.frombuffer(z.read(s["path"]), np.uint8), cv2.IMREAD_COLOR)
        if args.face == "v2":  # v2 的臉跟姿態模型無關，每張圖找一次就好
            t = time.perf_counter()
            faces = detect_faces(face_det, img)
            secs["face"] += time.perf_counter() - t
        for m, run in models.items():
            t = time.perf_counter()
            dets = run(img)
            secs[m] += time.perf_counter() - t
            if args.face == "v3":  # v3 要依每個人的頭部區域找臉，所以每個模型各找一次
                persons = make_persons(dets)
                t = time.perf_counter()
                faces = detect_faces_v3(face_det, img, persons)
                secs["face"] += (time.perf_counter() - t) / len(models)
            else:
                persons = [p for p in (make_person(b, k) for b, k in dets) if p is not None]
            assign_faces(persons, faces)
            ious = [iou(p["box"], s["box"]) for p in persons]
            best = int(np.argmax(ious)) if ious else -1
            p = persons[best] if best >= 0 and ious[best] >= IOU_TH else None
            alert = p is not None and not p["faces"]
            rows[m].append({
                "file": s["file"], "posture": s["posture"], "gt_face_visible": s["face_visible"],
                "found": p is not None, "iou": round(ious[best], 3) if ious else 0.0,
                "faces_in_image": len(faces), "faces_assigned": len(p["faces"]) if p else 0,
                "alert": alert, "extra_persons": len(persons) - int(p is not None),
                "head_err": round(float(np.linalg.norm(p["head"] - s["head"])
                                        / np.hypot(*(s["box"][2:] - s["box"][:2]))), 3) if p else "",
            })
            if alert and s["face_visible"] and len(sheets[m]) < MAX_SHEET:
                sheets[m].append(thumbnail(img, s, p, faces))
        if n % 50 == 0:
            print(f"  {n}/{len(samples)}", flush=True)

    suffix = "" if args.face == "v2" else f"_{args.face}"
    for m, rs in rows.items():
        with open(OUT_DIR / f"{m}{suffix}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rs[0]))
            writer.writeheader()
            writer.writerows(rs)
        save_sheet(sheets[m], OUT_DIR / f"{m}{suffix}_false_alarms.jpg")
    report(rows, secs, len(samples), args.face)


if __name__ == "__main__":
    main()
