"""
mouth_patch_crops.py（2026/11/22 起；v6 口鼻小塊模型的資料第 1 步）
挑訓練（和評估）要用的幀，存每個樣本的頭部特寫。特寫本身不進模型：
  - crops_<half>.csv 決定 mouth_patch_build.py 要跑哪些幀（影片第幾秒、照片哪一隻娃娃、標籤）；
  - crops_old/ 的頭部特寫下緣（包巾、衣服）是 mouth_patch_train.py「貼布」增強的布料（只用 calib 那一半）。
跟 experiments/airway_crops.py 同一套取樣：照片配到的娃娃、影片乾淨片段每秒 --fps 幀（片段頭尾各留 0.25 秒）；
分兩半照 hailo/prep_calib.py 的 SPLITS。calib 另外加 11/19 的側睡角度序列（dataset/20261119.MOV，
標註在 labels/train_20261119.csv：只拿來訓練，不在 doll_video_labels.csv 裡，不算進 eval_video 的合計）。
特寫存兩種大小：crops_old（邊長＝人框長邊 × 0.6，跟主流程給 YuNet 的頭部特寫一樣）、crops_tight（× 0.4，對照用）。

v6 的模型（model/mouth_patch_ens3.onnx）用的是這兩次的輸出：
    uv run python -m experiments.mouth_patch_crops --half calib --exclude-prefix 寶寶2                 # 11/22，找人門檻 0.3
    uv run python -m experiments.mouth_patch_crops --half calib --prefix 寶寶2 --suffix _1123 --det-th 0.15   # 11/23 那批
評估那一半（只給實驗看，正式評估用 eval_video／eval_doll／eval_syrip）：--half hold、--half field。
輸出：outputs/airway3/crops_<half><suffix>.csv、crops_old/、crops_tight/（outputs/ 不進 repo）
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "airway3"          # 口鼻小塊模型的中間檔都放這裡
LABELS_1119 = ROOT / "labels" / "train_20261119.csv"
VIDEO_1119 = ROOT / "dataset" / "20261119.MOV"
TARGET = {"ok": 1, "alert": 0}
FRACS = {"old": 1.0, "tight": 0.4 / 0.6}


def crop(img, person, size, frac):
    from pipeline.infant_face_v3 import up_angle
    angle = up_angle(person)
    cx, cy = (float(v) for v in person["head"])
    s = size / max(2 * person["head_r"] * frac, 1.0)
    M = cv2.getRotationMatrix2D((cx, cy), 0.0 if angle is None else angle, s)
    M[:, 2] += (size / 2 - cx, size / 2 - cy)
    return cv2.warpAffine(img, M, (size, size), borderValue=(114, 114, 114))


def main():
    from evaluation import video_labels
    from evaluation.eval_doll import IMG_DIR, load_labels, match
    from experiments.airway_crops import nearest, slot_person, v3_slots
    from hailo.prep_calib import SPLITS
    from pipeline.infant_face_v3 import load_rtmo, make_persons
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", required=True, choices=list(SPLITS))
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--prefix", default="", help="只做名稱這樣開頭的影片（不做照片和 20261119），例如 寶寶2")
    ap.add_argument("--exclude-prefix", default="", help="不做名稱這樣開頭的影片")
    ap.add_argument("--suffix", default="", help="輸出 crops_<half><suffix>.csv")
    ap.add_argument("--det-th", type=float, default=0,
                    help="RTMO 找人門檻（預設不改＝infant_face_v3 的 0.3；主流程 eval_video、live_view 用 0.15）")
    args = ap.parse_args()
    if args.det_th:
        import pipeline.infant_face_v3 as v3
        v3.DET_TH = args.det_th
    half, spec = args.half, SPLITS[args.half]
    for k in FRACS:
        (OUT / f"crops_{k}").mkdir(parents=True, exist_ok=True)
    run_pose = load_rtmo()
    rows, t0 = [], time.perf_counter()

    def save(name, img, person, base):
        if person is not None:
            for k, frac in FRACS.items():
                cv2.imencode(".png", crop(img, person, 224, frac))[1].tofile(str(OUT / f"crops_{k}" / f"{name}.png"))
        rows.append({**base, "name": name, "crop": "" if person is None else f"{name}.png",
                     "head_r": "" if person is None else round(float(person["head_r"]), 1)})

    photos = {} if args.prefix else {f: ds for f, ds in load_labels().items() if ds[0]["batch"] in spec["batches"]}
    for f, dolls in photos.items():
        img = cv2.imdecode(np.fromfile(str(IMG_DIR / f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        persons = make_persons(run_pose(img))
        d2p = match(persons, dolls, img.shape[1], img.shape[0])
        for j, d in enumerate(dolls):
            if d["label"] not in TARGET:
                continue
            base = {"half": half, "kind": "photo", "group": f"photo_batch{d['batch']}", "source": f, "doll": d["doll"], "t": "",
                    "label": d["label"], "staged": d["staged"], "note": d["note"], "ori": "", "tri": "", "v3_slot": ""}
            save(f"photo_{Path(f).stem}_{d['doll']}", img, persons[d2p[j]] if j in d2p else None, base)
    print(f"{half} 照片 {len(photos)} 張（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    ann = video_labels.load()
    videos = [(v, ann[v]["segments"], ann[v]["slots"], video_labels.find_video(v)) for v in spec["videos"]
              if v in ann and not (ann[v]["local"] and half != "field") and v != "嬰兒翻身"]
    if args.exclude_prefix:
        videos = [v for v in videos if not v[0].startswith(args.exclude_prefix)]
    if args.prefix:
        videos = [v for v in videos if v[0].startswith(args.prefix)]
    elif half == "calib":   # 11/19 側睡角度序列：只拿來訓練，標註不在 doll_video_labels.csv
        segs = list(csv.DictReader(open(LABELS_1119, encoding="utf-8-sig")))
        for s in segs:
            s["start"], s["end"] = float(s["start"]), float(s["end"])
        videos.append(("20261119", segs, {"S": {"rect": (0.0, 0.0, 1.0, 1.0)}}, VIDEO_1119))
    for video, segs, slots, path in videos:
        slots3, cap, n = v3_slots(video), cv2.VideoCapture(str(path)), 0
        for seg in segs:
            if seg["label"] not in TARGET and seg.get("tri") != "skip":
                continue
            for t in np.arange(seg["start"] + 0.25, seg["end"] - 0.25 + 1e-6, 1 / args.fps):
                t = round(float(t), 2)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                persons = make_persons(run_pose(frame))
                base = {"half": half, "kind": "video", "group": video, "source": video, "doll": seg["doll"], "t": t,
                        "label": seg["label"], "staged": seg["staged"], "note": seg["note"], "ori": seg.get("ori", ""),
                        "tri": seg.get("tri", ""), "v3_slot": nearest(slots3.get(seg["doll"], []), t)}
                save(f"{video}_{seg['doll']}_{t:07.2f}", frame, slot_person(frame, persons, slots[seg["doll"]]["rect"]), base)
                n += 1
        print(f"{half} {video}：{n} 幀（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    keys = ["name", "half", "kind", "group", "source", "doll", "t", "label", "staged", "tri", "ori", "note", "v3_slot", "head_r", "crop"]
    with open(OUT / f"crops_{half}{args.suffix}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"→ crops_{half}{args.suffix}.csv：{sum(bool(r['crop']) for r in rows)}/{len(rows)} 有頭部特寫", flush=True)


if __name__ == "__main__":
    main()
