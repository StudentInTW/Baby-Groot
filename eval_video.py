"""
eval_video.py
用娃娃影片評估「每個床位」的判定，跟 Harry 的時間軸比對。照片只能看單張判對沒，影片才量得到：
  - 危險片段有沒有發出警示（危險持續到片段結束時，系統是不是在響）
  - 安全片段會不會亂叫（誤報）

標註（repo 根目錄）：
  doll_video_slots.csv   每支影片、每隻娃娃一個床位方框（畫面比例座標），和床位方向 rot
                         （把床位轉正要轉幾度；安裝時設定頭朝哪邊）。頭在框裡的人就算這個床位。
  doll_video_labels.csv  每段狀態的開始、結束秒數。只列手不在畫面裡的乾淨片段；
                         手在擺娃娃的過渡期不計分（label 意義同 doll_labels.csv：ok 口鼻看得到／alert 看不到）。

比較三種判定：
  v3        床位裡偵測到人、那個人找不到臉 → 看不到口鼻；床位裡沒偵測到人 → 不警示（v3 本來的行為）
  v3＋床位  床位模式（docs/娃娃測試結果.md §4）：除了透過人找臉，也直接在床位範圍依床位方向轉正後找臉
            ——身體被被子蓋住、姿態模型找不到人時，臉露出來還是找得到。床位裡找不到臉就算看不到口鼻。
  v4＋床位  再加口鼻檢查（infant_face_v4.py，實驗版）
警示規則：每個床位各自一個狀態機，連續 --on 秒看不到 → 警示；連續 --off 秒看得到 → 解除；
看得到要持續 --blip 秒以上才會把「看不到」的計時歸零（防閃爍）。

用法：
    uv run python eval_video.py                  # 跑 doll_video_labels.csv 裡所有影片
    uv run python eval_video.py IMG_8150         # 只跑一支
    uv run python eval_video.py --reuse          # 不重跑偵測，用上次存的每幀結果重算（改 --on/--off 時用）
輸出：outputs/video_eval/<影片>_frames.csv（每幀判定）、<影片>_timeline.png（時間軸對照圖）、
      <影片>_preview.mp4（標註影片，縮小版）
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

import infant_face_v3
import infant_face_v4
import infant_face_v5
from face_common import FACE_MODEL, FACE_SCORE_TH, GREEN, RED, YELLOW, assign_faces, put_label
from infant_face_v3 import load_rtmo, make_persons

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "dataset" / "doll_video"
LABELS = ROOT / "doll_video_labels.csv"
SLOTS = ROOT / "doll_video_slots.csv"
OUT_DIR = ROOT / "outputs" / "video_eval"
MODES = ("v3", "v3_slot", "v4_slot", "v5_slot")
MODE_NAMES = {"v3": "v3", "v3_slot": "v3+slot", "v4_slot": "v4+slot", "v5_slot": "v5+slot"}
PREVIEW_W = 540     # 標註影片的寬度
SLOT_INPUT = 640    # 床位範圍縮到長邊這麼多像素再找臉
SLOT_FACE_TH = 0.8  # 床位範圍找到的臉要多高分才算（跟 v3 頭部裁切一樣：裁切後比較容易把抱枕圖案看成臉）
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_annotations():
    """回傳 {影片: {"slots": {娃娃: (x0, y0, x1, y1)}, "segments": [段落, ...]}}。"""
    videos = defaultdict(lambda: {"slots": {}, "rot": {}, "segments": []})
    for r in read_csv(SLOTS):
        videos[r["video"]]["slots"][r["doll"]] = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
        videos[r["video"]]["rot"][r["doll"]] = int(r["rot"])
    for r in read_csv(LABELS):
        r["start"], r["end"] = float(r["start"]), float(r["end"])
        videos[r["video"]]["segments"].append(r)
    return videos


def gt_at(segments, doll, t):
    """這一刻這隻娃娃的標註；落在過渡期（不計分）就回傳 None。"""
    return next((s for s in segments if s["doll"] == doll and s["start"] <= t <= s["end"]), None)


def slot_faces(detector, frame, rect, rot, min_score=SLOT_FACE_TH):
    """在床位範圍裡找臉：依床位方向轉正後找，框和特徵點換回原圖座標。min_score 可降低以抓側臉（v5）。"""
    h, w = frame.shape[:2]
    x0, y0 = int(rect[0] * w), int(rect[1] * h)
    crop = frame[y0:int(rect[3] * h), x0:int(rect[2] * w)]
    s = min(1.0, SLOT_INPUT / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    img = cv2.rotate(crop, ROTATE[rot]) if rot else crop
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, found = detector.detect(img)
    faces = []
    for f in [] if found is None else found:
        if f[-1] < min_score:
            continue
        pts = np.vstack([f[:2], f[:2] + f[2:4], f[4:14].reshape(5, 2)]).astype(np.float64)  # 框的兩角＋5 個特徵點
        u, v = pts[:, 0].copy(), pts[:, 1].copy()
        if rot == 90:        # 轉正後的座標 (u, v) → 轉正前的裁切座標 (x, y)
            pts[:, 0], pts[:, 1] = v, ch - u
        elif rot == 180:
            pts[:, 0], pts[:, 1] = cw - u, ch - v
        elif rot == 270:
            pts[:, 0], pts[:, 1] = cw - v, u
        pts = pts / s + (x0, y0)
        (bx0, by0), (bx1, by1) = pts[:2].min(axis=0), pts[:2].max(axis=0)
        faces.append(infant_face_v4.Face((bx0, by0, bx1, by1, float(f[-1])), pts[2:]))
    return faces


def detect(run_pose, detector, det_v5, frame, slots, rots):
    """一幀：回傳每個床位在四種判定下的狀態，和畫圖用的偵測結果。
    狀態：visible 看得到口鼻／hidden 看不到／none 床位裡沒偵測到人（只有 v3 會出現）。
    v5 額外輸出三級 tier（ok 正臉／warn 側臉／alert）；binary 狀態把 ok、warn 都當 visible（不硬警示）。"""
    h, w = frame.shape[:2]
    persons = make_persons(run_pose(frame))
    faces = infant_face_v4.detect_faces_v4(detector, frame, persons)  # 跟 v3 找到的臉逐一相同，多帶特徵點
    assign_faces(persons, faces)
    v3_faces = [list(p["faces"]) for p in persons]
    infant_face_v4.drop_covered_faces(frame, persons)
    v4_faces = [list(p["faces"]) for p in persons]

    result, extra = {}, []
    for doll, (x0, y0, x1, y1) in slots.items():
        def in_slot(x, y):
            return x0 * w <= x <= x1 * w and y0 * h <= y <= y1 * h

        def face_in_slot(f):
            return in_slot((f[0] + f[2]) / 2, (f[1] + f[3]) / 2)

        inside = [i for i, p in enumerate(persons) if in_slot(*p["head"])]
        found = slot_faces(detector, frame, (x0, y0, x1, y1), rots[doll])
        for f in found:
            f.skin = infant_face_v4.mouth_skin_ratio(frame, f.lm)
        extra += found
        # v3：照原本的做法，床位裡的人有臉就算看得到（臉在哪都算）
        any_v3 = any(v3_faces[i] for i in inside)
        # 床位模式：臉要落在這個床位裡才算（依位置，不透過人框——兩隻被框成一個人時才不會互相拿走對方的臉）
        slot_v3 = any(face_in_slot(f) for fs in v3_faces for f in fs) or bool(found)
        slot_v4 = any(face_in_slot(f) for fs in v4_faces for f in fs) \
            or any(f.skin >= infant_face_v4.MOUTH_SKIN_MIN for f in found)
        # v5：混合三級——backbone 用 v4_slot（保住召回），visible 再分正臉 ok／側臉 warn；
        #     backbone 判 alert 時，只有低門檻找到「乾淨側臉」才把 alert 降成 warn（found5）
        backbone5 = [f for fs in v4_faces for f in fs if face_in_slot(f)] + list(found)
        found5 = slot_faces(det_v5, frame, (x0, y0, x1, y1), rots[doll], min_score=infant_face_v5.FACE_TH_V5)
        v5_tier = infant_face_v5.tier_hybrid(slot_v4, backbone5, found5, frame)
        result[doll] = {
            "persons": len(inside),
            "v3": "visible" if any_v3 else ("hidden" if inside else "none"),
            "v3_slot": "visible" if slot_v3 else "hidden",
            "v4_slot": "visible" if slot_v4 else "hidden",
            "v5_slot": "visible" if v5_tier in ("ok", "warn") else "hidden",
            "v5_tier": v5_tier,
            "face_scores": " ".join(f"{f[4]:.2f}" for i in inside for f in persons[i]["faces"] + persons[i]["covered"]),
            "slot_face_scores": " ".join(f"{f[4]:.2f}" for f in found),
            "mouth_skin": " ".join(f"{f.skin:.2f}" for i in inside for f in persons[i]["faces"] + persons[i]["covered"]
                                   ) + ("｜" + " ".join(f"{f.skin:.2f}" for f in found) if found else ""),
        }
    return result, persons, faces + extra


class AlertState:
    """一個床位的警示狀態機：連續 on 秒看不到口鼻 → 警示；連續 off 秒看得到 → 解除。
    防閃爍（context.md §8 第 5 項）：看得到要持續 blip 秒以上，才會把「看不到」的計時歸零
    ——偶爾冒出一幀假臉，不會讓計時重來。"""

    def __init__(self, on, off, blip):
        self.on, self.off, self.blip = on, off, blip
        self.alert, self.since_hidden, self.since_visible = False, None, None

    def update(self, t, hidden):
        if hidden:
            self.since_visible = None
            self.since_hidden = t if self.since_hidden is None else self.since_hidden
        else:
            self.since_visible = t if self.since_visible is None else self.since_visible
            if t - self.since_visible >= self.blip - 1e-6:
                self.since_hidden = None
            if t - self.since_visible >= self.off - 1e-6:
                self.alert = False
        if self.since_hidden is not None and t - self.since_hidden >= self.on - 1e-6:
            self.alert = True
        return self.alert


def alert_track(times, hidden, on, off, blip):
    """回傳每一幀是否在警示中。"""
    state = AlertState(on, off, blip)
    return [state.update(t, hid) for t, hid in zip(times, hidden)]


def evaluate(rows, segments, on, off, blip):
    """rows：每幀每個床位一列。回傳每個段落、每種判定的結果。"""
    by_doll = defaultdict(list)
    for r in rows:
        by_doll[r["doll"]].append(r)
    alerts = {}
    for doll, rs in by_doll.items():
        rs.sort(key=lambda r: r["t"])
        times = [r["t"] for r in rs]
        for m in MODES:
            alerts[doll, m] = (times, alert_track(times, [r[m] == "hidden" for r in rs], on, off, blip),
                               [r[m] == "hidden" for r in rs])
    results = []
    for s in segments:
        times = alerts[s["doll"], MODES[0]][0]
        if not any(s["start"] <= t <= s["end"] for t in times):
            continue  # 這段沒有跑到（--limit-sec 試跑）
        res = {"seg": s}
        for m in MODES:
            times, alert, hidden = alerts[s["doll"], m]
            idx = [i for i, t in enumerate(times) if s["start"] <= t <= s["end"]]
            frac_hidden = sum(hidden[i] for i in idx) / max(len(idx), 1)
            if s["label"] == "alert":  # 危險一直持續到這段結束，那一刻還在警示才算抓到
                # （過渡期手擋住畫面時可能已經響了；只看開頭有響，會把「手擋住」誤算成抓到）
                res[m] = {"hidden": frac_hidden, "caught": alert[idx[-1]],
                          "coverage": sum(alert[i] for i in idx) / len(idx)}
            else:  # 口鼻看得到：扣掉開頭 off 秒（前一段的警示還在解除中），之後還在警示就算誤報
                idx_late = [i for i in idx if times[i] >= s["start"] + off]
                res[m] = {"hidden": frac_hidden, "false_alert": any(alert[i] for i in idx_late) if idx_late else None}
        results.append(res)
    return results, alerts


def report(name, results, on, off, blip):
    def pct(a, b):
        return f"{a}/{b}（{a / b:.0%}）" if b else "—"

    print(f"\n{'=' * 78}\n== {name}｜警示規則：連續 {on:g} 秒看不到才警示、連續 {off:g} 秒看得到才解除、看得到不到 {blip:g} 秒不算數")
    print(f"\n{'時間':<11}{'娃娃':<3}{'標註':<7}" + "".join(f"{MODE_NAMES[m]:>18}" for m in MODES) + "  說明")
    for r in results:
        s = r["seg"]
        cells = []
        for m in MODES:
            x = r[m]
            if s["label"] == "alert":
                txt = "抓到" if x["caught"] else "漏掉"
            else:
                txt = "—" if x["false_alert"] is None else ("誤報" if x["false_alert"] else "正常")
            cells.append(f"{txt}（{x['hidden']:.0%}）")
        span = f"{int(s['start'] // 60)}:{s['start'] % 60:04.1f}-{int(s['end'] // 60)}:{s['end'] % 60:04.1f}"
        print(f"{span:<11}{s['doll']:<3}{s['label']:<7}" + "".join(f"{c:>16}" for c in cells) + f"  {s['note']}")
    print("    （括號＝這段裡每幀判成看不到口鼻的比例；危險片段要到結束那一刻還在警示才算抓到）")
    summarize(results)


def summarize(results, title=None):
    if title:
        print(f"\n{'=' * 78}\n== {title}")
    danger = [r for r in results if r["seg"]["label"] == "alert"]
    safe = [r for r in results if r["seg"]["label"] == "ok"]
    for m in MODES:
        caught = [r for r in danger if r[m]["caught"]]
        cover = sum(r[m]["coverage"] for r in danger) / max(len(danger), 1)
        scored = [r for r in safe if r[m]["false_alert"] is not None]
        fa = sum(r[m]["false_alert"] for r in scored)
        print(f"    {MODE_NAMES[m]:<8} 危險片段抓到 {len(caught)}/{len(danger)}（{len(caught) / max(len(danger), 1):.0%}，"
              f"危險時間裡有 {cover:.0%} 在警示）｜安全片段誤報 {fa}/{len(scored)}")


def draw_timeline(name, rows, alerts, segments, duration, out_path):
    """時間軸對照圖：每隻娃娃一組——標註、三種判定（細條＝每幀判定，粗條＝警示中）。"""
    W, left, row_h, gap = 1800, 110, 26, 14
    dolls = sorted({r["doll"] for r in rows})
    n_rows = 1 + len(MODES)
    H = 60 + len(dolls) * (n_rows * row_h + gap + 24) + 40
    img = np.full((H, W, 3), 255, np.uint8)
    xs = lambda t: int(left + (W - left - 20) * t / duration)
    for t in range(0, int(duration) + 1, 10):  # 時間軸
        x = xs(t)
        cv2.line(img, (x, 40), (x, H - 30), (225, 225, 225), 1)
        cv2.putText(img, f"{t // 60}:{t % 60:02d}", (x - 16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, f"{name}  (green = mouth/nose visible, red = not visible / alert, gray = no person, "
                     f"blank GT = hands in frame, not scored)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y = 50
    colors = {"ok": (80, 180, 80), "alert": (60, 60, 220), "partial": (0, 170, 255)}
    for doll in dolls:
        cv2.putText(img, f"{doll} doll", (10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += 24
        cv2.putText(img, "GT", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for s in segments:
            if s["doll"] == doll:
                cv2.rectangle(img, (xs(s["start"]), y + 3), (xs(s["end"]), y + row_h - 3), colors[s["label"]], -1)
        y += row_h
        rs = sorted((r for r in rows if r["doll"] == doll), key=lambda r: r["t"])
        for m in MODES:
            cv2.putText(img, MODE_NAMES[m], (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            times, alert, _ = alerts[doll, m]
            for i, r in enumerate(rs):  # 上半：每幀判定
                x0, x1 = xs(r["t"]), max(xs(r["t"]) + 1, xs(r["t"] + 0.2))
                c = {"visible": (80, 180, 80), "hidden": (60, 60, 220), "none": (170, 170, 170)}[r[m]]
                cv2.rectangle(img, (x0, y + 3), (x1, y + row_h // 2), c, -1)
                if alert[i]:  # 下半：警示中
                    cv2.rectangle(img, (x0, y + row_h // 2 + 1), (x1, y + row_h - 3), (30, 30, 170), -1)
            y += row_h
        y += gap
    cv2.imwrite(str(out_path), img)


def draw_preview(frame, t, slots, persons, faces, alert_now, segments, mode):
    """標註影片的一幀：床位框（紅＝警示中，依 --preview 選的判定）、人框、臉（黃＝找到的臉，橘＝口鼻檢查判成被蓋住）。"""
    h, w = frame.shape[:2]
    vis = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 3)
    for f in faces:
        is_covered = getattr(f, "skin", None) is not None and f.skin < infant_face_v4.MOUTH_SKIN_MIN
        c = infant_face_v4.ORANGE if is_covered else YELLOW
        cv2.rectangle(vis, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), c, 5)
    for doll, (x0, y0, x1, y1) in slots.items():
        alarm = alert_now[doll]
        color = RED if alarm else GREEN
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h)), color, 8)
        seg = gt_at(segments, doll, t)
        gt = "GT: -" if seg is None else f"GT: {'visible' if seg['label'] == 'ok' else 'NOT visible'}"
        put_label(vis, f"{doll} {'ALERT' if alarm else 'OK'}", x0 * w + 10, y0 * h + 60, color, 1.8, 5)
        put_label(vis, gt, x0 * w + 10, y1 * h - 20, (90, 90, 90), 1.1, 3)
    put_label(vis, f"{int(t // 60)}:{t % 60:04.1f}", 20, 80, (0, 0, 0), 2.2, 5)
    put_label(vis, f"slot color: {MODE_NAMES[mode]} | orange face = mouth/nose covered (v4 check)",
              20, h - 30, (60, 60, 60), 1.0, 2)
    s = PREVIEW_W / w
    return cv2.resize(vis, (PREVIEW_W, round(h * s)), interpolation=cv2.INTER_AREA)


def run_video(name, ann, args, run_pose, detector, det_v5):
    """逐幀偵測（每秒 --fps 幀），存每幀結果和標註影片。回傳 rows、影片長度。"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{name}.MOV"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / args.fps))
    slots, rots, segments = ann["slots"], ann["rot"], ann["segments"]
    writer, rows, i, t0 = None, [], 0, time.perf_counter()
    live = {d: AlertState(args.on, args.off, args.blip) for d in slots}  # 標註影片的床位框顏色（--preview 選的判定）
    while True:
        if not cap.grab() or (args.limit_sec and i / src_fps > args.limit_sec):
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            t = i / src_fps
            states, persons, faces = detect(run_pose, detector, det_v5, frame, slots, rots)
            alert_now = {}
            for doll, st in states.items():
                seg = gt_at(segments, doll, t)
                rows.append({"t": round(t, 3), "doll": doll, "gt": "" if seg is None else seg["label"],
                             "persons": st["persons"], **{m: st[m] for m in MODES},
                             "v5_tier": st["v5_tier"],
                             "face_scores": st["face_scores"], "slot_face_scores": st["slot_face_scores"],
                             "mouth_skin": st["mouth_skin"]})
                alert_now[doll] = live[doll].update(t, st[args.preview] == "hidden")
            pv = draw_preview(frame, t, slots, persons, faces, alert_now, segments, args.preview)
            if writer is None:
                writer = cv2.VideoWriter(str(OUT_DIR / f"{name}_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (pv.shape[1], pv.shape[0]))
            writer.write(pv)
            if len(rows) // len(slots) % 50 == 0:
                print(f"  {name} {t:5.1f}/{n / src_fps:.1f}s（{time.perf_counter() - t0:.0f}s）", flush=True)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    with open(OUT_DIR / f"{name}_frames.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows, n / src_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="影片名稱（不含副檔名）；不給就跑標註檔裡全部")
    ap.add_argument("--fps", type=float, default=5, help="每秒取幾幀跑偵測（預設 5）")
    ap.add_argument("--on", type=float, default=2, help="連續幾秒看不到口鼻才警示（預設 2）")
    ap.add_argument("--off", type=float, default=2, help="連續幾秒看得到口鼻才解除警示（預設 2）")
    ap.add_argument("--blip", type=float, default=1, help="看得到要持續幾秒才把看不到的計時歸零（預設 1；0＝不防閃爍）")
    ap.add_argument("--det-th", type=float, default=0.15, help="RTMO 找人門檻（預設 0.15，見 docs/娃娃測試結果.md §7）")
    ap.add_argument("--reuse", action="store_true", help="不重跑偵測，讀上次存的 <影片>_frames.csv 重算")
    ap.add_argument("--limit-sec", type=float, default=0, help="只跑前幾秒（試跑用；會覆蓋上次的輸出）")
    ap.add_argument("--preview", choices=MODES[1:], default="v3_slot",
                    help="標註影片的床位框顏色依哪一種判定（預設 v3_slot；v4 還沒通過真實嬰兒的誤報測試）")
    args = ap.parse_args()

    infant_face_v3.DET_TH = args.det_th
    videos = load_annotations()
    names = args.videos or sorted(videos)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_pose = detector = det_v5 = None
    all_results = []
    for name in names:
        ann = videos[name]
        if args.reuse:
            rows = read_csv(OUT_DIR / f"{name}_frames.csv")
            for r in rows:
                r["t"] = float(r["t"])
            duration = max(r["t"] for r in rows) + 1
        else:
            if run_pose is None:
                run_pose = load_rtmo()
                detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
                det_v5 = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), infant_face_v5.FACE_TH_V5)
            rows, duration = run_video(name, ann, args, run_pose, detector, det_v5)
        results, alerts = evaluate(rows, ann["segments"], args.on, args.off, args.blip)
        report(name, results, args.on, args.off, args.blip)
        draw_timeline(name, rows, alerts, ann["segments"], duration, OUT_DIR / f"{name}_timeline.png")
        all_results += results
    if len(names) > 1:
        summarize(all_results, f"合計 {len(names)} 支影片")
    print(f"\n輸出：outputs/video_eval/（每幀判定 *_frames.csv、時間軸對照 *_timeline.png、標註影片 *_preview.mp4）")


if __name__ == "__main__":
    main()
