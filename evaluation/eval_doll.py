"""
eval_doll.py
用娃娃擺拍照片評估 v3 管線。**計分單位是「每隻娃娃」**，不是「每張照片」——
一張照片裡兩隻娃娃都危險、系統只抓到一隻時，用整張圖算會被判對，漏掉的那隻看不出來。

真實標註：`labels/doll_labels.csv`（每隻娃娃一列，人工標）
  x, y    娃娃軀幹中心；身體被被子蓋住時改標頭（佔整張圖寬高的比例）。拿來把偵測到的人配回是哪隻娃娃
  label   口鼻看不看得到：ok＝看得到／partial＝只看得到一部分／alert＝看不到
          （依「側躺但看不到口鼻一律算警示」的標準；partial 不計入誤報率，另外列）
  staged  拍攝意圖：safe 安全／warn 注意／danger 危險（資料夾和檔名，2026/9/21 Harry 逐隻確認過）。
          「注意」可能口鼻看得到（趴睡側臉）也可能看不到（這個角度看不到）；跟 label 矛盾的會另外列出來
  pose    supine 仰／side 側／prone 趴／covered 身體被被子蓋住
  batch   拍攝批次：1 站著手持拍／2 口鼻遮蓋補拍／3 近似天花板的俯拍（影片截圖）
床外（展示架上）的娃娃、頭幾乎在畫面外的娃娃不標，系統若偵測到會算成額外偵測。

配對規則：娃娃的標註點落在哪個人框裡就算配到那個人（一對一，取框心最近的）。
配不到人＝這隻娃娃沒被偵測到；沒配到娃娃的人框＝額外偵測（賣場背景的真人、人偶、抱枕）。

指標（都以娃娃為單位）：
  偵測率     有幾隻娃娃被姿態模型找到
  警示召回率 口鼻看不到的娃娃有幾隻發出警示。分兩種：
             「系統層級」把沒偵測到的娃娃也算漏（產品實際表現）
             「偵測到的」只看有偵測到的（跟 SyRIP 的算法一致，可互相比較）
  誤報率     口鼻看得到的娃娃被判成警示的比例
另外印出「以整張照片為單位」的數字，用來對照兩種算法差多少；
以及「床位模式」的模擬：假設安裝時已框好每個床位（用標註位置代替），
床位裡找不到人、或找到人但看不到臉，都算警示——被子蓋住身體時姿態模型找不到人，靠這個補。

用法：
    uv run python -m evaluation.eval_doll                     # 原圖
    uv run python -m evaluation.eval_doll --dim 0.3           # 模擬關燈（亮度降到 30%，見 degrade.py）
    uv run python -m evaluation.eval_doll --gray              # 模擬 IR 夜視（灰階）；可以跟 --dim 一起用
    uv run python -m evaluation.eval_doll --max-side 1280     # 模擬解析度較低的天花板攝影機
    uv run python -m evaluation.eval_doll --save-vis          # 每張都存標註圖（預設只存判錯的）
    uv run python -m evaluation.eval_doll --face v4 --det-th 0.15  # v4（v3＋口鼻檢查）＋建議的找人門檻
輸出：outputs/doll_eval/dolls<suffix>.csv、判錯的標註圖 outputs/doll_eval/vis<suffix>/
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from pipeline import face_common
from pipeline import infant_face_v3
from pipeline import infant_face_v4
from pipeline.face_common import (
    FACE_MODEL, FACE_SCORE_TH, GREEN, MIN_BODY_KPTS, RED, YELLOW, assign_faces, put_label,
)
from evaluation.degrade import degrade, describe
from pipeline.infant_face_v3 import detect_faces_v3, load_rtmo, make_persons

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "dataset" / "infant_jpg_without_video"
LABELS = ROOT / "labels" / "doll_labels.csv"
OUT_DIR = ROOT / "outputs" / "doll_eval"


def load_labels():
    """回傳 {檔案: [娃娃, ...]}，照 CSV 的順序。"""
    with open(LABELS, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    by_file = defaultdict(list)
    for r in rows:
        r["x"], r["y"] = float(r["x"]), float(r["y"])
        by_file[r["file"]].append(r)
    return by_file


def match(persons, dolls, w, h):
    """把偵測到的人配給娃娃：標註點落在人框內就是候選，一對一取框心最近的。
    回傳 {娃娃索引: 人索引}。"""
    pairs = []
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["box"]
        cx, cy, diag = (x1 + x2) / 2, (y1 + y2) / 2, max(np.hypot(x2 - x1, y2 - y1), 1.0)
        for j, d in enumerate(dolls):
            gx, gy = d["x"] * w, d["y"] * h
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                pairs.append((np.hypot(gx - cx, gy - cy) / diag, i, j))
    taken_p, taken_d = set(), {}
    for _, i, j in sorted(pairs):
        if i not in taken_p and j not in taken_d:
            taken_p.add(i)
            taken_d[j] = i
    return taken_d


def draw(img, persons, faces, dolls, doll2person):
    """紅／綠＝系統判定，藍點＝人工標註的娃娃位置（括號是真實標註）。"""
    vis = img.copy()
    scale = max(0.6, img.shape[1] / 1600)
    thick = max(2, round(scale * 2))
    covered = [f for p in persons for f in p.get("covered", [])]  # v4：口鼻檢查拿掉的臉畫橘色
    for face in faces:
        x1, y1, x2, y2, score = face
        is_covered = any(face is c for c in covered)
        color = infant_face_v4.ORANGE if is_covered else YELLOW
        text = f"mouth covered {score:.2f}" if is_covered else f"face {score:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        put_label(vis, text, x1, y2 + 25 * scale, color, scale * 0.7, max(1, thick - 1))
    person2doll = {i: j for j, i in doll2person.items()}
    for i, p in enumerate(persons):
        alert = not p["faces"]
        x1, y1, x2, y2 = p["box"]
        color = RED if alert else GREEN
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        tag = f"#{i + 1}" if i in person2doll else f"#{i + 1} extra"
        put_label(vis, f"{tag} {'ALERT' if alert else 'OK'}", x1, y1 - 8, color, scale, thick)
    h, w = img.shape[:2]
    for j, d in enumerate(dolls):
        gx, gy = int(d["x"] * w), int(d["y"] * h)
        cv2.circle(vis, (gx, gy), round(6 * scale), (255, 120, 0), -1)
        put_label(vis, f"doll{j + 1} {d['label']}", gx + 10 * scale, gy, (255, 120, 0), scale * 0.8, thick)
    return vis


def report(rows, secs, n_images, title):
    """rows：每隻娃娃一列。"""
    def rate(hit, tot):
        return f"{hit}/{tot}（{hit / tot:.1%}）" if tot else f"{hit}/0（—）"

    def block(name, rs):
        found = [r for r in rs if r["found"]]
        by = lambda lab: [r for r in rs if r["label"] == lab]
        alert_gt, ok_gt, part_gt = by("alert"), by("ok"), by("partial")
        f_alert = [r for r in alert_gt if r["found"]]
        f_ok = [r for r in ok_gt if r["found"]]
        f_part = [r for r in part_gt if r["found"]]
        print(f"\n[{name}] 娃娃 {len(rs)} 隻（口鼻看不到 {len(alert_gt)}、看得到 {len(ok_gt)}、部分 {len(part_gt)}）")
        print(f"    偵測率　　　　　　{rate(len(found), len(rs))}")
        print(f"    警示召回（系統層級）{rate(sum(r['alert'] for r in alert_gt), len(alert_gt))}"
              f"　←沒偵測到的娃娃算漏")
        print(f"    警示召回（偵測到的）{rate(sum(r['alert'] for r in f_alert), len(f_alert))}")
        print(f"    誤報率　　　　　　{rate(sum(r['alert'] for r in f_ok), len(f_ok))}")
        if part_gt:
            print(f"    部分可見→警示　　 {rate(sum(r['alert'] for r in f_part), len(f_part))}（不計入上面兩項）")

    print(f"\n{'=' * 78}\n== {title}｜{n_images} 張照片、{len(rows)} 隻娃娃"
          f"（姿態 {1000 * secs['pose'] / n_images:.0f} ms／張、人臉 {1000 * secs['face'] / n_images:.0f} ms／張）")
    block("全部", rows)
    single = [r for r in rows if r["dolls_in_image"] == 1]
    multi = [r for r in rows if r["dolls_in_image"] > 1]
    block("單娃娃照片", single)
    block("雙娃娃照片", multi)

    print("\n-- 依情境資料夾（口鼻看不到→警示｜口鼻看得到→誤報）")
    for folder in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == folder]
        a = [r for r in rs if r["label"] == "alert"]
        o = [r for r in rs if r["label"] == "ok"]
        print(f"    {folder:<16} 娃娃 {len(rs):>3}　偵測 {rate(sum(r['found'] for r in rs), len(rs)):<16}"
              f"　警示 {rate(sum(r['alert'] for r in a), len(a)):<16}　誤報 {rate(sum(r['alert'] for r in o), len(o))}")

    for title, key in (("依拍攝批次（1 站著拍／2 口鼻遮蓋補拍／3 俯拍，影片截圖）", "batch"), ("依姿勢", "pose"),
                       ("依拍攝意圖（danger 危險／warn 注意／safe 安全）", "staged")):
        print(f"\n-- {title}")
        for v in sorted({r[key] for r in rows}):
            rs = [r for r in rows if r[key] == v]
            a = [r for r in rs if r["label"] == "alert"]
            o = [r for r in rs if r["label"] == "ok"]
            print(f"    {v:<10} 娃娃 {len(rs):>3}　偵測 {rate(sum(r['found'] for r in rs), len(rs)):<16}"
                  f"　警示 {rate(sum(r['alert'] for r in a), len(a)):<16}　誤報 {rate(sum(r['alert'] for r in o), len(o))}")

    # 對照：如果用「整張照片有沒有警示」來算（使用者擔心會高估的算法）
    imgs = defaultdict(list)
    for r in rows:
        imgs[r["file"]].append(r)
    danger_imgs = [rs for rs in imgs.values() if any(r["label"] == "alert" for r in rs)]
    img_hit = sum(any(r["alert"] for r in rs) for rs in danger_imgs)
    dolls_hit = sum(r["alert"] for rs in danger_imgs for r in rs if r["label"] == "alert")
    dolls_tot = sum(r["label"] == "alert" for rs in danger_imgs for r in rs)
    two = [rs for rs in danger_imgs if sum(r["label"] == "alert" for r in rs) > 1]
    part_only = sum(0 < sum(r["alert"] for r in rs if r["label"] == "alert")
                    < sum(r["label"] == "alert" for r in rs) for rs in two)
    print(f"\n-- 計分單位差多少：有危險娃娃的照片 {len(danger_imgs)} 張"
          f"\n    以「整張照片有沒有警示」算：{rate(img_hit, len(danger_imgs))}"
          f"\n    以「每隻娃娃」算：　　　　 {rate(dolls_hit, dolls_tot)}"
          f"\n    兩隻都危險的照片 {len(two)} 張，其中 {part_only} 張只抓到一隻"
          f"（用整張圖算會被判對，漏掉的那隻看不出來）")

    slot = lambda r: not r["found"] or r["alert"]  # 床位裡找不到人也警示
    print("\n-- 床位模式（假設安裝時框好床位；床位裡找不到人、或找到人但看不到臉，都警示）"
          "\n    誤報率的分母是全部口鼻看得到的娃娃，沒偵測到的也算誤報")
    for name, rs in [("全部", rows)] + [(f"第 {b} 批", [r for r in rows if r["batch"] == b])
                                         for b in sorted({r["batch"] for r in rows})]:
        a = [r for r in rs if r["label"] == "alert"]
        o = [r for r in rs if r["label"] == "ok"]
        print(f"    {name:<6} 警示召回 {rate(sum(slot(r) for r in a), len(a)):<16}"
              f"　誤報率 {rate(sum(slot(r) for r in o), len(o))}")

    per_img = {r["file"]: (r["extra_persons"], r["extra_alert"]) for r in rows}
    n_extra = sum(v[0] for v in per_img.values())
    n_extra_alert = sum(v[1] for v in per_img.values())
    print(f"\n-- 額外偵測（配不到娃娃的人框，多半是賣場背景的真人／人偶／抱枕）：每張 "
          f"{n_extra / n_images:.2f} 個（{sum(v[0] > 0 for v in per_img.values())} 張有）"
          f"\n    其中 {n_extra_alert} 個沒配到臉、會跳出警示（每張 {n_extra_alert / n_images:.2f} 個）"
          f"——上面的誤報率只算標註過的娃娃，不含這些；實機要靠「只監測床位範圍」濾掉")

    lost = [r for r in rows if not r["found"]]
    if lost:
        filtered = sum(r["raw_det"] for r in lost)
        print(f"\n-- 沒偵測到的娃娃 {len(lost)} 隻：姿態模型有框到、被「可見身體關鍵點 < "
              f"{MIN_BODY_KPTS} 個」的雜訊過濾器濾掉 {filtered} 隻；模型根本沒輸出 {len(lost) - filtered} 隻")

    miss = [r for r in rows if r["label"] == "alert" and not r["alert"]]
    if miss:
        print(f"\n-- 漏掉的危險娃娃 {len(miss)} 隻（口鼻看不到卻沒警示）")
        for r in miss:
            why = ("沒偵測到這隻娃娃（" + ("被雜訊過濾器濾掉" if r["raw_det"] else "模型沒輸出") + "）"
                   if not r["found"] else f"配到假臉 {r['face_scores']}"
                   + (f"（口鼻檢查 {r['mouth_skin']} 沒擋掉）" if r["mouth_skin"] else ""))
            print(f"    {r['file']} #{r['doll']}　{why}　{r['note']}")
    fa = [r for r in rows if r["label"] == "ok" and r["alert"]]
    if fa:
        print(f"\n-- 誤報 {len(fa)} 隻（口鼻看得到卻警示）")
        for r in fa:
            why = f"口鼻檢查誤擋（{r['covered_skin']}）" if r["covered_skin"] else "沒找到臉"
            print(f"    {r['file']} #{r['doll']}　{r['pose']}　{why}　{r['note']}")

    # 「注意」可能口鼻看得到（趴睡側臉）也可能看不到（這個角度看不到），兩種都不算矛盾
    diff = [r for r in rows if (r["staged"] == "safe" and r["label"] == "alert")
            or (r["staged"] == "danger" and r["label"] == "ok")]
    if diff:
        print(f"\n-- 人工標註和拍攝意圖矛盾 {len(diff)} 隻（安全卻看不到口鼻／危險卻看得到口鼻，請 Harry 確認）")
        for r in diff:
            print(f"    {r['file']} #{r['doll']}　標註 {r['label']}／拍攝意圖 {r['staged']}　{r['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=float, default=0, help="模擬關燈：亮度乘上這個倍率（0＝不處理）")
    ap.add_argument("--gray", action="store_true", help="模擬 IR 夜視：轉成灰階（見 degrade.py）")
    ap.add_argument("--max-side", type=int, default=0, help="長邊縮到這個像素（0＝原圖）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張")
    ap.add_argument("--save-vis", action="store_true", help="每張都存標註圖（預設只存判錯的）")
    ap.add_argument("--face", choices=["v3", "v4"], default="v3",
                    help="v4＝v3 再加口鼻檢查（infant_face_v4.py），輸出檔名加 _v4")
    # 以下三個是調門檻用的，只改這支評估程式跑的值，不動 infant_face_v3.py 的預設
    ap.add_argument("--det-th", type=float, default=0, help=f"RTMO 找人門檻（預設 {infant_face_v3.DET_TH}）")
    ap.add_argument("--face-th", type=float, default=0, help=f"整張圖的人臉門檻（預設 {FACE_SCORE_TH}）")
    ap.add_argument("--head-face-th", type=float, default=0,
                    help=f"頭部裁切的人臉門檻（預設 {infant_face_v3.HEAD_FACE_TH}）")
    args = ap.parse_args()

    if args.det_th:
        infant_face_v3.DET_TH = args.det_th
    if args.face_th:  # 兩個模組各自 import 過這個常數，兩邊都要改
        face_common.FACE_SCORE_TH = infant_face_v3.FACE_SCORE_TH = args.face_th
    if args.head_face_th:
        infant_face_v3.HEAD_FACE_TH = args.head_face_th

    by_file = load_labels()
    files = sorted(by_file)[:args.limit or None]
    knobs = [("dim", args.dim), ("px", args.max_side), ("det", args.det_th),
             ("face", args.face_th), ("hface", args.head_face_th)]
    suffix = ("_v4" if args.face == "v4" else "") + ("_gray" if args.gray else "") \
        + "".join(f"_{k}{v:g}" for k, v in knobs if v)
    vis_dir = OUT_DIR / f"vis{suffix}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    run_pose = load_rtmo()
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), args.face_th or FACE_SCORE_TH)
    rows, secs = [], defaultdict(float)

    for n, f in enumerate(files, 1):
        img = cv2.imread(str(IMG_DIR / f))
        if img is None:
            print(f"[略過] 讀不到 {f}")
            continue
        if args.max_side:
            s = min(1.0, args.max_side / max(img.shape[:2]))
            if s < 1:
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
        img = degrade(img, args.gray, args.dim)
        h, w = img.shape[:2]

        t = time.perf_counter()
        dets = run_pose(img)
        persons = make_persons(dets)
        secs["pose"] += time.perf_counter() - t
        t = time.perf_counter()
        if args.face == "v4":
            faces = infant_face_v4.detect_faces_v4(detector, img, persons)
            assign_faces(persons, faces)
            infant_face_v4.drop_covered_faces(img, persons)
        else:
            faces = detect_faces_v3(detector, img, persons)
            assign_faces(persons, faces)
        secs["face"] += time.perf_counter() - t

        dolls = by_file[f]
        doll2person = match(persons, dolls, w, h)
        wrong = False
        for j, d in enumerate(dolls):
            p = persons[doll2person[j]] if j in doll2person else None
            alert = p is not None and not p["faces"]
            wrong |= (d["label"] == "alert") != alert
            gx, gy = d["x"] * w, d["y"] * h  # 姿態模型原本有沒有輸出這隻娃娃（過濾之前）
            raw = any(b[0] <= gx <= b[2] and b[1] <= gy <= b[3] for b, _ in dets)
            rows.append({
                "raw_det": raw,
                "file": f, "folder": f.split("/")[0], "doll": d["doll"], "label": d["label"],
                "staged": d["staged"], "pose": d["pose"], "batch": d["batch"], "dolls_in_image": len(dolls),
                "found": p is not None, "alert": alert,
                "faces_assigned": len(p["faces"]) if p else 0,
                "face_scores": " ".join(f"{x[4]:.2f}" for x in p["faces"]) if p else "",
                # v4：留下的臉和被口鼻檢查拿掉的臉，各自「下半臉像皮膚的比例」
                "mouth_skin": " ".join(f"{x.skin:.2f}" for x in p["faces"] if getattr(x, "skin", None) is not None)
                if p else "",
                "covered_skin": " ".join(f"{x.skin:.2f}" for x in p.get("covered", [])) if p else "",
                "faces_in_image": len(faces), "persons_in_image": len(persons),
                "extra_persons": len(persons) - len(doll2person),
                "extra_alert": sum(not q["faces"] for i, q in enumerate(persons)
                                   if i not in doll2person.values()),
                "note": d["note"],
            })
        if args.save_vis or wrong:
            out = draw(img, persons, faces, dolls, doll2person)
            s = min(1.0, 1400 / max(out.shape[:2]))  # 存小一點，4284×5712 的原圖太大
            cv2.imwrite(str(vis_dir / f"{f.replace('/', '__')}"),
                        cv2.resize(out, (round(out.shape[1] * s), round(out.shape[0] * s))) if s < 1 else out,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        if n % 10 == 0:
            print(f"  {n}/{len(files)}", flush=True)

    with open(OUT_DIR / f"dolls{suffix}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    title = ("娃娃擺拍｜v4（v3＋口鼻檢查）" if args.face == "v4" else "娃娃擺拍｜v3（RTMO＋YuNet）") \
        + (f"｜{describe(args.gray, args.dim)}" if args.gray or args.dim else "") \
        + (f"｜長邊 {args.max_side}px" if args.max_side else "｜原圖 4284×5712") \
        + (f"｜找人門檻 {args.det_th:g}" if args.det_th else "") \
        + (f"｜整張圖人臉門檻 {args.face_th:g}" if args.face_th else "") \
        + (f"｜頭部人臉門檻 {args.head_face_th:g}" if args.head_face_th else "")
    report(rows, secs, len(files), title)
    print(f"\n明細：outputs/doll_eval/dolls{suffix}.csv｜標註圖：outputs/doll_eval/vis{suffix}/")


if __name__ == "__main__":
    main()
