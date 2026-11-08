"""
airway_crops.py
口鼻分類器（實驗，2026/11/06 起）的資料：每個人的頭部特寫 → 「口鼻看得到／看不到」。

為什麼：頭轉開時 YuNet 會把耳朵當成臉（漏掉危險）；晶片上的 VLM 又把側臉全部說成看不到
（hailo/結果.md「VLM 複判也放上晶片」）。改成一個只回答這一題的小分類器：吃主流程本來就算好的頭部特寫，
每次幾毫秒，可以每幀跑，也可以用我們自己的娃娃資料訓練。

頭部特寫跟主流程給 YuNet 的那張一樣（infant_face_v3.head_faces）：以 RTMO 的頭為中心、邊長＝人框長邊 × 0.6、
依身體方向轉正，這裡存成 --size × --size。

資料（都是自己拍的娃娃；分兩半的規則跟 hailo/prep_calib.py 的 SPLITS 一樣，依影片和拍攝批次分）：
  calib（可以訓練）：影片 IMG_8115、IMG_8150、IMG_8432；照片第 2、3 批；
                     11/08 台南那批：影片 IMG_8735、8739、8740、8741、8742、8744，照片第 4 批
  hold（只能評估）：影片 IMG_8116、IMG_8184、IMG_8429；照片第 1 批
  AI 生成的嬰兒翻身.mp4 不用；SyRIP、網路影片不能拿來訓練。
標籤：
  照片：labels/doll_labels.csv 每隻娃娃的 label（ok＝看得到、alert＝看不到；partial 不用），
        用 eval_doll.match 的規則把偵測到的人配給娃娃。
  影片：labels/doll_video_labels.csv 的乾淨片段（沒有手），每秒取 --fps 幀，片段頭尾各留 0.25 秒；
        empty（空床）不用。趴睡但口鼻看得到（staged prone）算看得到。
        另外記主流程（v3＋床位）那一幀的判定（outputs/video_eval/<影片>_frames.csv，要先跑過 eval_video），
        評估時才算得出「偵測器漏掉的危險補到幾幀、安全畫面多叫幾次」。

用法：
    uv run python -m experiments.airway_crops                 # 預設 224×224、影片每秒 2 幀
輸出：outputs/airway/crops/<名稱>.png、outputs/airway/crops.csv（找不到人的也記一列，crop 欄空白）
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "airway"
EVAL_DIR = ROOT / "outputs" / "video_eval"
TARGET = {"ok": 1, "alert": 0}      # 1＝口鼻看得到


def crop_head(img, person, size):
    """頭部特寫：跟 infant_face_v3.head_faces 同一個幾何（頭中心、邊長＝配對直徑、轉正），輸出 size×size。"""
    from pipeline.infant_face_v3 import up_angle
    angle = up_angle(person)
    cx, cy = (float(v) for v in person["head"])
    s = size / max(2 * person["head_r"], 1.0)
    M = cv2.getRotationMatrix2D((cx, cy), 0.0 if angle is None else angle, s)
    M[:, 2] += (size / 2 - cx, size / 2 - cy)
    return cv2.warpAffine(img, M, (size, size), borderValue=(114, 114, 114))


def slot_person(img, persons, rect):
    """頭中心落在床位範圍（比例座標）裡、框最大的那個人；沒有就回傳 None。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect
    inside = [p for p in persons if x0 * w <= p["head"][0] <= x1 * w and y0 * h <= p["head"][1] <= y1 * h]
    if not inside:
        return None
    return max(inside, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))


def v3_slots(video):
    """{娃娃: [(t, v3_slot), ...]}，讀 eval_video 存的每幀判定。"""
    path = EVAL_DIR / f"{video}_frames.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["doll"], []).append((float(r["t"]), r["v3_slot"]))
    return out


def nearest(seq, t):
    return min(seq, key=lambda x: abs(x[0] - t))[1] if seq else ""


def main():
    from evaluation import video_labels
    from evaluation.eval_doll import IMG_DIR, load_labels, match
    from hailo.prep_calib import SPLITS
    from pipeline.infant_face_v3 import load_rtmo, make_persons, rtmo_tag

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224, help="頭部特寫存成幾乘幾（預設 224；訓練時再縮）")
    ap.add_argument("--fps", type=float, default=2, help="影片每秒取幾幀（預設 2）")
    args = ap.parse_args()

    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    run_pose = load_rtmo()
    print(f"RTMO：{rtmo_tag() or '原本的 ONNX（CPU）'}", flush=True)
    rows, t0 = [], time.perf_counter()

    def save(name, img, person, base):
        crop = "" if person is None else f"{name}.png"
        if person is not None:
            cv2.imencode(".png", crop_head(img, person, args.size))[1].tofile(str(crops_dir / crop))
        rows.append({**base, "name": name, "crop": crop})

    annotations = video_labels.load()
    for half, spec in SPLITS.items():
        # 照片：每隻娃娃一列
        photos = {f: ds for f, ds in load_labels().items() if ds[0]["batch"] in spec["batches"]}
        for f, dolls in photos.items():
            img = cv2.imdecode(np.fromfile(str(IMG_DIR / f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[略過] 讀不到 {f}")
                continue
            persons = make_persons(run_pose(img))
            doll2person = match(persons, dolls, img.shape[1], img.shape[0])
            for j, d in enumerate(dolls):
                if d["label"] not in TARGET:
                    continue
                base = {"half": half, "kind": "photo", "group": f"photo_batch{d['batch']}", "source": f,
                        "doll": d["doll"], "t": "", "label": d["label"], "staged": d["staged"], "note": d["note"],
                        "target": TARGET[d["label"]], "v3_slot": "", "persons": len(persons)}
                person = persons[doll2person[j]] if j in doll2person else None
                save(f"photo_{Path(f).stem}_{d['doll']}", img, person, base)
        print(f"{half} 照片：{len(photos)} 張（{time.perf_counter() - t0:.0f} 秒）", flush=True)

        # 影片：乾淨片段每秒 --fps 幀
        for video in spec["videos"]:
            if video not in annotations or annotations[video]["local"] or video == "嬰兒翻身":
                continue
            ann, slots3 = annotations[video], v3_slots(video)
            cap = cv2.VideoCapture(str(video_labels.find_video(video)))
            n = 0
            for seg in ann["segments"]:
                if seg["label"] not in TARGET:
                    continue
                slot = ann["slots"][seg["doll"]]
                for t in np.arange(seg["start"] + 0.25, seg["end"] - 0.25 + 1e-6, 1 / args.fps):
                    t = round(float(t), 2)
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    persons = make_persons(run_pose(frame))
                    base = {"half": half, "kind": "video", "group": video, "source": video, "doll": seg["doll"],
                            "t": t, "label": seg["label"], "staged": seg["staged"], "note": seg["note"],
                            "target": TARGET[seg["label"]], "v3_slot": nearest(slots3.get(seg["doll"], []), t),
                            "persons": len(persons)}
                    save(f"{video}_{seg['doll']}_{t:07.2f}", frame, slot_person(frame, persons, slot["rect"]), base)
                    n += 1
            print(f"{half} {video}：{n} 幀（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    keys = ["name", "half", "kind", "group", "source", "doll", "t", "label", "staged", "note", "target",
            "v3_slot", "persons", "crop"]
    with open(OUT_DIR / "crops.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("\n== 每一組：有頭部特寫／全部（看得到、看不到）")
    for half in SPLITS:
        for g in sorted({r["group"] for r in rows if r["half"] == half}):
            rs = [r for r in rows if r["half"] == half and r["group"] == g]
            got = [r for r in rs if r["crop"]]
            vis = sum(r["target"] == 1 for r in got)
            print(f"  {half:<5} {g:<14} {len(got)}/{len(rs)}（看得到 {vis}、看不到 {len(got) - vis}）")
    print(f"→ {OUT_DIR / 'crops.csv'}")


if __name__ == "__main__":
    main()
