"""
mouth_patch_build.py（2026/11/23 起；v6 口鼻小塊模型的資料第 2 步，先跑 mouth_patch_crops.py）
照主流程（床位模式、找人門檻 0.15，跟 eval_video 一樣）重跑 crops_<組>.csv 的每個取樣幀，
床位裡每張被 YuNet 找到的臉（v3＋床位算的那些，含 v4 判蓋住的）用 5 個特徵點切口鼻小塊
（pipeline/infant_face_v6.mouth_patch；多留 1.6 倍範圍給訓練時隨機平移、縮放，中間 96×96 就是主流程切的那塊）。
同時記這一幀的 v3＋床位、v4＋床位（跟 crib_detect.detect 一樣）。照片沒有床位：用 eval_doll 的規則配到那隻娃娃的人，取他身上的臉。

組：calib、calib_1123（訓練）；hold、hold_1123、field（只評估；每兩列取一列＝每秒 1 幀）。
標籤由 mouth_patch_train.py 決定：綠（label ok）的幀裡的臉＝口鼻露出；黃、紅的幀裡被找到的臉＝沒露出
（多半是 YuNet 把只露眼睛、側臉朝下的頭當成臉）。

    uv run python -m experiments.mouth_patch_build calib
    uv run python -m experiments.mouth_patch_build calib_1123
    uv run python -m experiments.mouth_patch_build hold 50     # 試跑：只做前 50 幀，輸出到 patch/hold_try/
--low（2026/11/26，第二版）：另外切床位裡 0.3–0.8 分的臉（v5 那顆低門檻偵測器找到、主流程 0.8 門檻丟掉的；faces.csv 的 low 欄＝1，
samples.csv 的 n_low＝這一幀幾張）。床位找臉的門檻降下來時，這些臉（多半是側睡朝下、趴睡時長在臉頰、耳朵、手上的假臉）
才會交給口鼻模型，第一版沒訓練過。--out 換輸出資料夾名稱：
    uv run python -m experiments.mouth_patch_build calib_r2 --low
    uv run python -m experiments.mouth_patch_build calib_1123 --low --out calib_1123_r2
輸出：outputs/airway3/patch/<組>/p/*.png（154×154 小塊）、samples.csv（每幀一列）、faces.csv（每張臉一列）
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A3 = ROOT / "outputs" / "airway3"
SETS = ("calib", "hold", "field", "calib_1123", "hold_1123")
EVAL_SETS = ("hold", "field", "hold_1123")
CTX = 1.6     # 存檔的範圍＝口鼻小塊 × 1.6


class Reader:
    """依時間讀幀：往前不遠就一直 grab，往回或跳很遠才 seek（長影片 seek 很慢）。"""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.pos = -1     # 上一個讀到的幀號

    def frame_at(self, idx):
        if idx <= self.pos or idx - self.pos > self.fps * 3:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx - 1
        while self.pos < idx - 1:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, fr = self.cap.read()
        self.pos += 1
        return fr if ok else None

    def at(self, t):
        return self.frame_at(int(round(t * self.fps)))


class Done(Exception):
    pass


def main(argv=None):
    import pipeline.infant_face_v3 as v3mod
    v3mod.DET_TH = 0.15
    from pipeline import crib_detect, infant_face_v4
    from pipeline.face_common import FACE_SCORE_TH, assign_faces, create_face_detector
    from pipeline.infant_face_v3 import load_rtmo, make_persons
    from pipeline.infant_face_v5 import FACE_TH_V5
    from pipeline.infant_face_v6 import PATCH, mouth_patch

    argv = list(sys.argv[1:] if argv is None else argv)
    low = "--low" in argv
    argv = [a for a in argv if a != "--low"]
    out_name = None
    if "--out" in argv:
        i = argv.index("--out")
        out_name, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    if not argv or not (A3 / f"crops_{argv[0]}.csv").exists():
        raise SystemExit(f"用法：python -m experiments.mouth_patch_build <組> [試跑幀數] [--low] [--out 資料夾]；"
                         f"組＝outputs/airway3/crops_<組>.csv（mouth_patch_crops.py 的輸出），例如 {'、'.join(SETS)}")
    which = argv[0]
    limit = int(argv[1]) if len(argv) > 1 else 0
    out = A3 / "patch" / ((out_name or which) + ("_try" if limit else ""))
    (out / "p").mkdir(parents=True, exist_ok=True)
    run_pose, det = load_rtmo(), create_face_detector(FACE_SCORE_TH)
    det_low = create_face_detector(FACE_TH_V5) if low else None
    samples, faces_rows, t0 = [], [], time.perf_counter()

    def process(base, frame, rect=None, rot=0, person_pick=None):
        """跑一幀，存床位裡（或指定的人身上）每張臉的口鼻小塊。rect=None 時用 person_pick(persons) 選人（照片）。"""
        h, w = frame.shape[:2]
        persons = make_persons(run_pose(frame))
        faces = infant_face_v4.detect_faces_v4(det, frame, persons)
        assign_faces(persons, faces)
        v3f = [list(p["faces"]) for p in persons]
        infant_face_v4.drop_covered_faces(frame, persons)
        if rect is not None:
            x0, y0, x1, y1 = rect
            ins = lambda f: x0 * w <= (f[0] + f[2]) / 2 <= x1 * w and y0 * h <= (f[1] + f[3]) / 2 <= y1 * h  # noqa: E731
            found = crib_detect.slot_faces(det, frame, rect, rot)
            for f in found:
                f.skin = infant_face_v4.mouth_skin_ratio(frame, f.lm)
            fs = [f for pf in v3f for f in pf if ins(f)] + list(found)
            n_person = sum(x0 * w <= p["head"][0] <= x1 * w and y0 * h <= p["head"][1] <= y1 * h for p in persons)
            lows = [] if det_low is None else [f for f in crib_detect.slot_faces(det_low, frame, rect, rot, FACE_TH_V5)
                                               if f[4] < crib_detect.SLOT_FACE_TH]
        else:
            i = person_pick(persons)
            fs = [] if i is None else v3f[i]
            n_person = int(i is not None)
            lows = []
        samples.append({**base, "persons": n_person, "faces": len(fs), "v3": bool(fs),
                        "v4": any(f.skin >= infant_face_v4.MOUTH_SKIN_MIN for f in fs),
                        **({"n_low": len(lows)} if low else {})})
        for f in lows:
            f.skin = infant_face_v4.mouth_skin_ratio(frame, f.lm)
        for k, f in enumerate(fs + lows):
            lm = np.asarray(f.lm, np.float64).reshape(5, 2)
            patch = mouth_patch(frame, lm, PATCH, CTX)
            if patch is None:
                continue
            fn = f"{base['name']}_f{k}.png"
            cv2.imencode(".png", patch)[1].tofile(str(out / "p" / fn))
            m = float(np.linalg.norm((lm[3] + lm[4]) / 2 - (lm[0] + lm[1]) / 2))
            faces_rows.append({"name": base["name"], "k": k, "score": round(float(f[4]), 3), "skin": round(float(f.skin), 3),
                               **({"low": int(k >= len(fs))} if low else {}),
                               "m": round(m, 1), "patch": fn,
                               "eye": round(float(np.linalg.norm(lm[1] - lm[0])), 1),      # 雙眼距離（v4 小臉門檻看這個）
                               "fw": round(float(f[2] - f[0]), 1), "fh": round(float(f[3] - f[1]), 1),
                               "lm": " ".join(f"{v:.1f}" for v in lm.ravel())})
        if len(samples) % 50 == 0:
            print(f"  {which}：{len(samples)} 幀、{len(faces_rows)} 張臉（{time.perf_counter() - t0:.0f} 秒）", flush=True)
        if limit and len(samples) >= limit:
            raise Done

    try:
        build(which, process)
    except Done:
        pass
    for fn, rows in (("samples.csv", samples), ("faces.csv", faces_rows)):
        if rows:
            with open(out / fn, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"→ patch/{out.name}：{len(samples)} 幀、{len(faces_rows)} 張臉（{time.perf_counter() - t0:.0f} 秒）", flush=True)


def build(which, process):
    from evaluation import video_labels
    from evaluation.eval_doll import IMG_DIR, load_labels, match
    t0 = time.perf_counter()
    from experiments.mouth_patch_crops import TRAIN_ONLY
    rows = list(csv.DictReader(open(A3 / f"crops_{which}.csv", encoding="utf-8-sig")))
    if which in EVAL_SETS:   # 評估組每秒 1 幀
        keep, by = [], defaultdict(list)
        for r in rows:
            by[(r["group"], r["doll"], r["label"], r["staged"], r["note"])].append(r)
        for rs in by.values():
            keep += rs[::2]
        rows = keep
    ann = video_labels.load()
    dolls_of = load_labels()
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["source"] if r["kind"] == "photo" else r["group"]].append(r)
    for src, rs in by_video.items():
        base_of = lambda r: {k: r[k] for k in ("name", "half", "group", "doll", "t", "label", "staged", "note")}  # noqa: E731
        if rs[0]["kind"] == "photo":
            img = cv2.imdecode(np.fromfile(str(IMG_DIR / src), np.uint8), cv2.IMREAD_COLOR)
            dolls = dolls_of[src]
            for r in rs:
                j = next(i for i, d in enumerate(dolls) if d["doll"] == r["doll"])

                def pick(persons, j=j):
                    return match(persons, dolls, img.shape[1], img.shape[0]).get(j)
                process(base_of(r), img, person_pick=pick)
            continue
        path = video_labels.find_video(src)
        if src in TRAIN_ONLY:                     # 只訓練的影片：床位寫在 TRAIN_ONLY
            _, rect, rot = TRAIN_ONLY[src]
        rd = Reader(path)
        for r in sorted(rs, key=lambda r: float(r["t"])):
            if src not in TRAIN_ONLY:
                s = ann[src]["slots"][r["doll"]]
                rect, rot = s["rect"], s["rot"]
            fr = rd.at(float(r["t"]))
            if fr is not None:
                process(base_of(r), fr, rect, rot)
        print(f"{which} {src}：{len(rs)} 列（{time.perf_counter() - t0:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
