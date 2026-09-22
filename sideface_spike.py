"""
sideface_spike.py —— 實驗：把「臉」這一關從兩級（看得到/看不到）改成三級的可行性測試

背景（2026/9/22）：6 分鐘正常睡姿影片 IMG_8184 顯示，只問「看不看得到正面臉」會把「側睡（側臉、口鼻其實看得到）」
誤判成危險。SyRIP 姿態 spike（spike_posture.py）證明「用關鍵點幾何分仰/趴」行不通（仰睡召回 7–21%），
差別在外觀不在骨架 → 修法不是做姿態模型，而是**強化臉這一關**：
    正面清楚 → ok｜側臉但口鼻區清楚 → warn（需關注）｜完全沒臉/口鼻被埋 → alert

這支 spike 用我們自己標好的娃娃影片（有側睡，SyRIP 沒有「側躺」類別）驗兩件事：
  (a) 把 YuNet 門檻降到能抓到側臉後，YuNet 的 5 個特徵點（雙眼、鼻、兩嘴角）幾何能不能分
      「正臉 / 側臉 / 假臉（卡通臉、後腦勺）」——這決定三級分法可不可行。
  (b) RTMO 的肩寬壓縮能不能判「側躺」——備用訊號。

GT 類別（由 doll_video_labels.csv 的 label＋note 推出）：
  frontal 正臉（ok、note 沒有「側」）｜profile 側臉（ok、note 有「側/頭轉」）｜buried 臉埋住（label=alert）

用法：uv run python sideface_spike.py            # 全部影片
      uv run python sideface_spike.py IMG_8184   # 只跑一支
      uv run python sideface_spike.py --fps 2 --low 0.3
輸出：終端統計；不寫檔（純分析）。娃娃是自己拍的，數字可內部參考。
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np

import infant_face_v3
from face_common import FACE_MODEL
from infant_face_v3 import load_rtmo, make_persons

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "dataset" / "doll_video"
SLOTS = ROOT / "doll_video_slots.csv"
LABELS = ROOT / "doll_video_labels.csv"
SLOT_INPUT = 640
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def read_csv(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_ann():
    slots, rot, segs = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for r in read_csv(SLOTS):
        slots[r["video"]][r["doll"]] = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
        rot[r["video"]][r["doll"]] = int(r["rot"])
    for r in read_csv(LABELS):
        r["start"], r["end"] = float(r["start"]), float(r["end"])
        segs[r["video"]].append(r)
    return slots, rot, segs


def category(seg):
    if seg["label"] == "alert":
        return "buried"
    note = seg["note"]
    if any(w in note for w in ("側", "頭轉")):
        return "profile"
    return "frontal"


def slot_face_landmarks(det, frame, rect, rot):
    """在床位範圍依方向轉正後找臉，回傳轉正座標下的 [(score, box, lm(5,2)), ...]（門檻已由 det 設定）。
    轉正座標＝臉大致直立，方便算幾何。"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    s = min(1.0, SLOT_INPUT / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    img = cv2.rotate(crop, ROTATE[rot]) if rot else crop
    det.setInputSize((img.shape[1], img.shape[0]))
    _, found = det.detect(img)
    out = []
    for f in [] if found is None else found:
        box = f[:4].astype(float)
        lm = f[4:14].reshape(5, 2).astype(float)   # 轉正影像座標（右眼,左眼,鼻,右嘴角,左嘴角）
        out.append((float(f[-1]), box, lm))
    return out


def face_geom(box, lm):
    """從 5 特徵點算「正臉度/側臉度」與結構合理性。座標為臉大致直立的影像座標。"""
    re, le, nose, rm, lm_ = lm
    eye = np.hypot(*(le - re)) + 1e-6
    eye_mid = (le + re) / 2
    mouth_mid = (rm + lm_) / 2
    ax = (le - re) / eye                       # 沿雙眼方向（水平軸）
    perp = np.array([-ax[1], ax[0]])           # 垂直軸
    # 側臉指標：鼻沿水平軸偏離雙眼中點多少（正臉≈0，側臉偏向露出的一側）
    nose_off = abs(float(np.dot(nose - eye_mid, ax)) / eye)
    # 雙眼到鼻的水平距離不對稱（正臉≈0，側臉一大一小）
    d_re = abs(float(np.dot(nose - re, ax)) / eye)
    d_le = abs(float(np.dot(nose - le, ax)) / eye)
    eye_asym = abs(d_re - d_le)
    # 結構合理性（真臉：眼在上、鼻在中、嘴在下；沿垂直軸）
    eye_nose_v = float(np.dot(nose - eye_mid, perp)) / eye        # 應 > 0（鼻在眼下方）
    nose_mouth_v = float(np.dot(mouth_mid - nose, perp)) / eye    # 應 > 0（嘴在鼻下方）
    mouth_w = np.hypot(*(lm_ - rm)) / eye
    return dict(nose_off=nose_off, eye_asym=eye_asym, eye_nose_v=eye_nose_v,
                nose_mouth_v=nose_mouth_v, mouth_w=mouth_w)


def in_slot(rect, x, y, w, h):
    return rect[0] * w <= x <= rect[2] * w and rect[1] * h <= y <= rect[3] * h


def shoulder_ratio(persons, rect, w, h):
    """床位裡的人：肩寬 / 軀幹長（側躺→肩前後疊、投影肩寬小）。回傳 None 若量不到。"""
    best = None
    for p in persons:
        hx, hy = p["head"]
        if in_slot(rect, hx, hy, w, h):
            best = p
            break
    if best is None:
        return None
    k = best.get("kpts")
    if k is None:
        return None
    ls, rs, lh, rh = k[5], k[6], k[11], k[12]
    if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
        return None
    sh_w = np.hypot(ls[0] - rs[0], ls[1] - rs[1])
    sh_mid = (ls[:2] + rs[:2]) / 2
    hip_mid = (lh[:2] + rh[:2]) / 2
    torso = np.hypot(*(sh_mid - hip_mid)) + 1e-6
    return float(sh_w / torso)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--fps", type=float, default=2, help="每秒取幾幀（預設 2）")
    ap.add_argument("--low", type=float, default=0.3, help="YuNet 低門檻，降到能抓側臉（預設 0.3）")
    ap.add_argument("--shoulder", action="store_true", help="也跑 RTMO 量肩寬（較慢）")
    args = ap.parse_args()

    infant_face_v3.DET_TH = 0.15
    slots, rots, segs = load_ann()
    names = args.videos or sorted(segs)
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), args.low)
    run_pose = load_rtmo() if args.shoulder else None

    feats = defaultdict(list)      # category -> list of geom dicts (最佳臉)
    found_cnt = defaultdict(lambda: [0, 0])   # category -> [找到臉, 總取樣]
    shoulders = defaultdict(list)

    for name in names:
        mov = VIDEO_DIR / f"{name}.MOV"
        if not mov.exists():
            continue
        cap = cv2.VideoCapture(str(mov))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for seg in segs[name]:
            doll = seg["doll"]
            if doll not in slots[name]:
                continue
            rect, rot = slots[name][doll], rots[name][doll]
            cat = category(seg)
            t = seg["start"]
            while t <= seg["end"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                cands = slot_face_landmarks(det, frame, rect, rot)
                found_cnt[cat][1] += 1
                if cands:
                    found_cnt[cat][0] += 1
                    score, box, lm = max(cands, key=lambda c: c[0])
                    g = face_geom(box, lm)
                    g["score"] = score
                    feats[cat].append(g)
                if run_pose is not None:
                    persons = make_persons(run_pose(frame))
                    sr = shoulder_ratio(persons, rect, w, h)
                    if sr is not None:
                        shoulders[cat].append(sr)
                t += 1.0 / args.fps
        cap.release()
        print(f"  掃完 {name}", flush=True)

    def summ(vals):
        a = np.array(vals)
        return f"n={len(a):3d} 中位 {np.median(a):5.2f} 平均 {a.mean():5.2f}±{a.std():.2f}" if len(a) else "n=0"

    print("\n===== (a) YuNet 低門檻（{:.2f}）＋特徵點幾何 =====".format(args.low))
    print("類別        找到臉率            score            nose_off(側臉↑)     eye_asym(側臉↑)   eye→nose_v(真臉>0)  nose→mouth_v(真臉>0)")
    for cat in ("frontal", "profile", "buried"):
        fc = found_cnt[cat]
        rate = f"{fc[0]}/{fc[1]}（{fc[0]/max(fc[1],1):.0%}）"
        fs = feats[cat]
        if fs:
            def col(key):
                a = np.array([g[key] for g in fs])
                return f"{np.median(a):.2f}/{a.mean():.2f}±{a.std():.2f}"
            print(f"{cat:10s} {rate:16s}  {col('score'):18s} {col('nose_off'):18s} {col('eye_asym'):16s} {col('eye_nose_v'):18s} {col('nose_mouth_v')}")
        else:
            print(f"{cat:10s} {rate:16s}  （沒找到臉）")
    print("   （中位/平均±標準差。側臉度 nose_off、eye_asym 應該 profile > frontal；結構 eye→nose_v、nose→mouth_v 真臉應 >0，假臉/後腦勺會亂）")

    # 簡單分界看可分性：frontal vs profile 用 eye_asym
    fa = [g["eye_asym"] for g in feats["frontal"]]
    pa = [g["eye_asym"] for g in feats["profile"]]
    if fa and pa:
        best_t, best_acc = 0, 0
        for t in np.linspace(0, 1, 101):
            acc = (np.mean(np.array(fa) < t) + np.mean(np.array(pa) >= t)) / 2
            if acc > best_acc:
                best_acc, best_t = acc, t
        print(f"\n   frontal vs profile 只用 eye_asym 的最佳單一分界：門檻 {best_t:.2f} → 平衡準確率 {best_acc:.0%}")

    if run_pose is not None:
        print("\n===== (b) RTMO 肩寬/軀幹（側躺→小） =====")
        for cat in ("frontal", "profile", "buried"):
            print(f"{cat:10s} {summ(shoulders[cat])}")


if __name__ == "__main__":
    main()
