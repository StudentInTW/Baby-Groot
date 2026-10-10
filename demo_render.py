"""
demo_render.py
示範畫面（給 3 分鐘影片用）：左邊是每床鏡頭畫面（娃娃重演），右邊是這張床的燈號和判斷依據。

燈號（跟簡報第 10 頁一致）：
  綠  口鼻看得到
  黃  口鼻看不到、3 秒倒數中；或偵測器說看得到、VLM 說看不到（請看一眼）——只在護理站畫面，不推播
  紅  連續 3 秒看不到 → 推播給護理師
  藍  照護中（大人在床邊），暫停計時
  灰  空床
資料來源：eval_video.py 的每幀判定（v3＋床位）、crib_context.py 的照護中／空床狀態機、
vlm_spike.py --every 的 VLM 時間軸（有的話；連續 VLM_HOLD 次 P(看得到口鼻) < VLM_TH 才亮「請看一眼」）。
臉框是重跑偵測畫的（只跑輸出的那一段）。

用法：
    uv run python demo_render.py IMG_8432 --start 30 --end 70   # 單床：左邊鏡頭、右邊燈號與判斷依據
    uv run python demo_render.py --grid                          # 護理站畫面示意：8 格（GRID），每格循環播一段娃娃影片
    uv run python demo_render.py --split                         # 開場雙畫面：左家長手機、右護理站（預設 IMG_8432 86–100 秒）
輸出：outputs/demo/<影片>_<開始>-<結束>.mp4、outputs/demo/nurse_station.mp4、outputs/demo/split_open.mp4（1920×1080）
護理站畫面是示意：8 張床都是同兩支娃娃影片的不同片段，燈號是系統對那一段的真實判定。
開場雙畫面的燈號也是系統真實輸出；家長手機畫面和護理師手機推播是介面原型（畫面角落標 PROTOTYPE_TAG）。
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import crib_context
import eval_video
import infant_face_v5
import video_labels
from face_common import FACE_MODEL, FACE_SCORE_TH
from infant_face_v3 import load_rtmo
from video_labels import read_csv

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs" / "demo"
VLM_TH = 0.4     # P(看得到口鼻) 低於這個算「VLM 說看不到」（vlm_spike.py：漏掉的危險幀 13/13、安全幀多 9/38）
VLM_HOLD = 2     # 連續幾次（每秒一次）才亮「請看一眼」
W, H = 1920, 1080
PRODUCT = "FirstGlance"  # 產品名（暫定），改這裡就好
PROTOTYPE_TAG = "UI prototype · doll re-enactment"
CLOCK0 = 14 * 3600 + 32 * 60  # 畫面上的時鐘：主角 07 號片段（GRID）開頭＝14:32:00，各段影片的時鐘對得上


def _first_existing(*paths):
    return next((p for p in paths if Path(p).exists()), paths[-1])


# Windows 用 Arial；其他系統（雲端開發、Mac）找不到就用 DejaVu Sans
FONT = _first_existing("C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = _first_existing("C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
STATES = {  # 顯示狀態 → (顏色 RGB, 標題, 說明)
    "green": ((46, 125, 50), "GREEN", "Nose and mouth visible"),
    "count": ((214, 158, 0), "YELLOW", "Nose and mouth hidden: 3-second timer running"),
    "look": ((214, 158, 0), "YELLOW", "Detector sees a face, the VLM doesn't: please take a look"),
    "red": ((217, 45, 32), "RED", "Hidden for 3 s: silent alert sent to the nurse"),
    "care": ((21, 101, 192), "BLUE", "Adult at the crib: timer paused"),
    "empty": ((110, 120, 130), "GRAY", "Empty crib: matches its empty photo, no alert"),
}


GRID = [  # (床號, 影片, 開始秒, 結束秒)：每一格播哪一段，播完停在最後一幀（睡著的娃娃本來就不太動）。
    # 片段都從燈號已經穩定的時刻開始（10/10 用 timeline_states 挑過）；07 是主角（綠 → 藍 → 黃 → 紅），
    # 04 是空床（灰），08 是偵測器把耳朵當成臉、VLM 不同意（黃「請看一眼」）
    ("01", "IMG_8429", 9, 53), ("02", "IMG_8432", 8, 27), ("03", "IMG_8429", 72, 103), ("04", "IMG_8432", 181, 193),
    ("05", "IMG_8429", 175.5, 188), ("06", "IMG_8432", 65, 81.5), ("07", "IMG_8432", 78, 110), ("08", "IMG_8432", 41, 58),
]
EVENT_TEXT = {"red": "nose and mouth hidden 3 s: alert sent", "look": "VLM disagrees: please take a look",
              "care": "adult at the crib", "empty": "baby out of the crib"}


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def clock_text(t_src, seconds=True):
    """來源影片 t_src 秒 → 畫面上的時鐘（07 號片段開頭＝14:32:00）。"""
    c = CLOCK0 + int(t_src - GRID[6][2])
    return f"{c // 3600:02d}:{c // 60 % 60:02d}" + (f":{c % 60:02d}" if seconds else "")


def display_state(ctx, state, hidden, vlm_hidden=False):
    """CribState 的狀態 → 顯示燈號（STATES 的 key）。live_view.py 也用（沒有 VLM）。"""
    if state == "alert":
        return "red"
    if state in ("care", "empty"):
        return state
    if hidden and ctx.alarm.since_hidden is not None:
        return "count"
    if not hidden and vlm_hidden:
        return "look"
    return "green"


def timeline_states(name, slot, end):
    """跑 crib_context 的狀態機到 end 秒，回傳每個取樣時刻的顯示狀態與訊號。"""
    frames = {float(r["t"]): r for r in read_csv(eval_video.OUT_DIR / f"{name}_frames.csv")}
    vlm_path = ROOT / "outputs" / "vlm_spike" / f"timeline_{name}.csv"
    vlm = [(float(r["t"]), float(r["p_mouth"])) for r in read_csv(vlm_path)] if vlm_path.exists() else []
    ctx, rows = crib_context.CribState(), []
    for t, motion, diff in crib_context.features(name, slot):
        if t > end + 0.5:
            break
        r = frames.get(t)
        if r is None:
            continue
        hidden = r["v3_slot"] == "hidden"
        state = ctx.update(t, hidden, motion, diff)
        checks = [p for tv, p in vlm if tv <= t][-VLM_HOLD:]
        vlm_hidden = len(checks) == VLM_HOLD and all(p < VLM_TH for p in checks)
        since = ctx.alarm.since_hidden
        show = display_state(ctx, state, hidden, vlm_hidden)
        rows.append({"t": t, "show": show, "hidden": hidden, "count": t - since if since is not None else 0.0,
                     "adult": ctx.last_motion is not None and t - ctx.last_motion < crib_context.CARE_HOLD,
                     "matches": diff is not None and diff < crib_context.EMPTY_TH,
                     "vlm": checks[-1] if checks else None})
    return rows


def face_boxes(name, ann, start, end):
    """重跑偵測（只跑輸出的那一段），回傳 {取樣時刻: [臉框, ...]}。"""
    run_pose = load_rtmo()
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    det_v5 = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), infant_face_v5.FACE_TH_V5)
    slots = {d: s["rect"] for d, s in ann["slots"].items()}
    rots = {d: s["rot"] for d, s in ann["slots"].items()}
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / crib_context.FPS))
    boxes, i = {}, 0
    while cap.grab():
        t = round(i / fps, 3)
        if t > end:
            break
        if t >= start - 0.3 and i % step == 0:
            _, frame = cap.retrieve()
            _, _, faces = eval_video.detect(run_pose, det, det_v5, frame, slots, rots)
            boxes[t] = [tuple(int(v) for v in f[:4]) for f in faces]
        i += 1
    return boxes


def latest(seq, t):
    """seq 依時間排序的 (時刻, 值)；回傳 t 之前最後一個值。"""
    best = None
    for tv, v in seq:
        if tv > t:
            break
        best = v
    return best


def draw_panel(canvas, row, rows, start, end, t, crib_label):
    color, title, desc = STATES[row["show"]]
    d = ImageDraw.Draw(canvas)
    x0 = 760
    d.text((x0, 60), crib_label, font=font(30), fill=(90, 100, 110))
    d.rounded_rectangle((x0, 110, 1860, 290), radius=28, fill=color)
    d.text((x0 + 40, 128), title, font=font(84, True), fill=(255, 255, 255))
    d.text((x0 + 40, 232), desc, font=font(32), fill=(255, 255, 255))
    y = 330
    if row["show"] == "count":  # 3 秒倒數條
        frac = min(row["count"] / 3.0, 1.0)
        d.rounded_rectangle((x0, y, 1860, y + 36), radius=18, fill=(235, 228, 205))
        d.rounded_rectangle((x0, y, x0 + int((1860 - x0) * frac), y + 36), radius=18, fill=color)
        d.text((x0, y + 46), f"{row['count']:.1f} s of 3 s", font=font(28), fill=(90, 100, 110))
    y = 450
    d.text((x0, y), "WHAT THE BOX SEES", font=font(26, True), fill=(16, 74, 133))
    signals = [
        ("Face detector, every frame", "nose and mouth NOT visible" if row["hidden"] else "nose and mouth visible",
         (217, 45, 32) if row["hidden"] else (46, 125, 50)),
        ("VLM second look, every second",
         "—" if row["vlm"] is None else f"P(nose and mouth visible) = {row['vlm']:.2f}",
         (110, 120, 130) if row["vlm"] is None else ((214, 158, 0) if row["vlm"] < VLM_TH else (46, 125, 50))),
        ("Adult at the crib (edge motion)", "yes: timer paused" if row["adult"] else "no",
         (21, 101, 192) if row["adult"] else (110, 120, 130)),
        ("Matches this crib's empty photo", "yes" if row["matches"] else "no",
         (110, 120, 130)),
    ]
    for i, (label, value, c) in enumerate(signals):
        yy = y + 50 + i * 82
        d.ellipse((x0, yy + 10, x0 + 22, yy + 32), fill=c)
        d.text((x0 + 40, yy), label, font=font(26), fill=(90, 100, 110))
        d.text((x0 + 40, yy + 32), value, font=font(30, True), fill=(30, 30, 30))
    # 時間軸：這一段每個取樣時刻的燈號，加上目前位置
    ty, tx0, tx1 = 900, x0, 1860
    span = [r for r in rows if start <= r["t"] <= end]
    for r in span:
        xa = tx0 + int((tx1 - tx0) * (r["t"] - start) / (end - start))
        xb = tx0 + int((tx1 - tx0) * (r["t"] + 0.2 - start) / (end - start))
        d.rectangle((xa, ty, max(xa + 1, xb), ty + 34), fill=STATES[r["show"]][0])
    cx = tx0 + int((tx1 - tx0) * (t - start) / (end - start))
    d.rectangle((cx - 2, ty - 10, cx + 2, ty + 44), fill=(20, 20, 20))
    d.text((tx0, ty + 48), f"{int(start // 60)}:{start % 60:04.1f}", font=font(22), fill=(90, 100, 110))
    d.text((tx1 - 70, ty + 48), f"{int(end // 60)}:{end % 60:04.1f}", font=font(22), fill=(90, 100, 110))
    d.text((x0, 975), "White box: a face the detector found", font=font(24), fill=(120, 130, 140))
    d.text((x0, 1010), "Doll re-enactment · laptop prototype · not real nursery footage",
           font=font(24), fill=(120, 130, 140))


def render_grid(fps=30):
    """護理站畫面示意：左邊 4×2 格床位畫面（各自的燈號外框），右邊是事件列表。
    一開始就是黃（VLM）或灰的床列為事件；其他開頭狀態不算事件，之後每次換燈號才記。"""
    anns = video_labels.load()
    rows = {}
    for video in sorted({g[1] for g in GRID}):
        slot = next(iter(anns[video]["slots"].values()))
        rows[video] = [(r["t"], r) for r in timeline_states(video, slot, max(g[3] for g in GRID if g[1] == video))]
    caps = [cv2.VideoCapture(str(video_labels.find_video(v))) for _, v, _, _ in GRID]
    for cap, (_, _, start, _) in zip(caps, GRID):
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    duration = GRID[6][3] - GRID[6][2]
    out = OUT_DIR / "nurse_station.mp4"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    prev, events, held = {}, [], {}
    tw, th = 360, 540
    for n in range(int(duration * fps)):
        t_rel = n / fps
        canvas = np.full((H, W, 3), (251, 249, 247), np.uint8)
        shows = []
        for k, (cap, (label, video, start, end)) in enumerate(zip(caps, GRID)):
            t_src = min(start + t_rel, end)
            ok, frame = (True, held[label]) if start + t_rel > end and label in held else cap.read()
            if ok:
                held[label] = frame
            row = latest(rows[video], t_src) or rows[video][0][1]
            shows.append(row["show"])
            color = STATES[row["show"]][0][::-1]
            x, y = (k % 4) * tw, (k // 4) * th
            if ok:
                img = cv2.resize(frame, (round(frame.shape[1] * (th - 20) / frame.shape[0]), th - 20),
                                 interpolation=cv2.INTER_AREA)
                ox = x + (tw - img.shape[1]) // 2
                canvas[y + 10:y + 10 + img.shape[0], ox:ox + img.shape[1]] = img
                cv2.rectangle(canvas, (ox, y + 10), (ox + img.shape[1] - 1, y + th - 11), color, 10)
            first = label not in prev
            if row["show"] != prev.get(label) and row["show"] in EVENT_TEXT and (not first or row["show"] in ("look", "empty")):
                events.insert(0, (t_rel, label, row["show"]))
            prev[label] = row["show"]
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for k, (label, *_rest) in enumerate(GRID):
            x, y = (k % 4) * tw + 40, (k // 4) * th + 26
            color, title, _ = STATES[shows[k]]
            d.rounded_rectangle((x, y, x + 150, y + 44), radius=14, fill=color)
            d.text((x + 12, y + 6), f"{label}  {title}", font=font(26, True), fill=(255, 255, 255))
        x0 = 1460
        d.text((x0, 30), "Nurse station", font=font(40, True), fill=(16, 74, 133))
        clock = CLOCK0 + int(t_rel)
        d.text((x0, 84), f"{clock // 3600:02d}:{clock // 60 % 60:02d}:{clock % 60:02d} · 8 cribs", font=font(26),
               fill=(90, 100, 110))
        counts = {c: sum(STATES[s][1] == c for s in shows) for c in ("GREEN", "YELLOW", "RED", "BLUE", "GRAY")}
        for i, (c, v) in enumerate(counts.items()):
            col = next(STATES[s][0] for s in STATES if STATES[s][1] == c)
            d.rounded_rectangle((x0 + i * 88, 134, x0 + i * 88 + 78, 196), radius=12, fill=col)
            d.text((x0 + i * 88 + 28, 144), str(v), font=font(34, True), fill=(255, 255, 255))
        d.text((x0, 226), "EVENTS", font=font(24, True), fill=(16, 74, 133))
        for i, (te, label, show) in enumerate(events[:7]):
            yy = 266 + i * 96
            clock_e = CLOCK0 + int(te)
            d.ellipse((x0, yy + 8, x0 + 20, yy + 28), fill=STATES[show][0])
            d.text((x0 + 32, yy), f"Crib {label} · {STATES[show][1]}", font=font(26, True), fill=(30, 30, 30))
            d.text((x0 + 32, yy + 34), f"{clock_e // 3600:02d}:{clock_e // 60 % 60:02d}:{clock_e % 60:02d} · "
                   f"{EVENT_TEXT[show]}", font=font(19), fill=(90, 100, 110))
        d.text((x0, 1000), "Mock-up: all 8 cribs are the same doll clips;", font=font(19), fill=(120, 130, 140))
        d.text((x0, 1026), "each light is our system's real output.", font=font(19), fill=(120, 130, 140))
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"→ {out}（{duration:.0f} 秒）")


def fit(frame, w, h):
    """把畫面等比例縮到放得進 w×h（不裁切）。"""
    s = min(w / frame.shape[1], h / frame.shape[0])
    return cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)


def wrap(d, text, fnt, width):
    """依像素寬度斷行。"""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and d.textlength(trial, font=fnt) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def draw_push(img, x1, y1, text, t_since, w=380, h=176):
    """護理師手機推播小卡：從右邊滑進來，前 0.8 秒左右震動。只有文字，沒有影像。"""
    slide = min(t_since / 0.35, 1.0)
    shake = int(9 * np.sin(t_since * 60) * max(0.0, 1 - t_since / 0.8)) if t_since > 0.35 else 0
    x0 = x1 - w + int((1 - slide) ** 2 * (w + 60)) + shake
    y0 = y1 - h
    d = ImageDraw.Draw(img)
    d.text((x0 + 4, y0 - 34), "Nurse's phone", font=font(22), fill=(110, 120, 130))
    d.rounded_rectangle((x0 + 4, y0 + 6, x0 + w + 4, y1 + 6), radius=26, fill=(205, 205, 205))  # 影子
    d.rounded_rectangle((x0, y0, x0 + w, y1), radius=26, fill=(255, 255, 255), outline=(225, 225, 225), width=2)
    d.rounded_rectangle((x0 + 22, y0 + 24, x0 + 74, y0 + 76), radius=12, fill=STATES["red"][0])
    d.text((x0 + 38, y0 + 26), "!", font=font(40, True), fill=(255, 255, 255))
    d.text((x0 + 92, y0 + 24), f"{PRODUCT.upper()} · now", font=font(22), fill=(110, 120, 130))
    for i, line in enumerate(wrap(d, text, font(27, True), w - 112)[:3]):
        d.text((x0 + 92, y0 + 56 + i * 34), line, font=font(27, True), fill=(30, 30, 30))


def render_split(video, start, end, crib="07", fps_out=None):
    """開場雙畫面：左邊家長手機的直播（原始畫面，沒有任何 AI 標記），右邊同一段的護理站單床卡片（燈號是系統真實輸出）。
    紅燈一亮，右下角跳出護理師手機推播；最後 2 秒疊字幕。"""
    ann = video_labels.load()[video]
    slot = next(iter(ann["slots"].values()))
    rows = timeline_states(video, slot, end)
    samples = [(r["t"], r) for r in rows]
    red_at = next((r["t"] for r in rows if start <= r["t"] <= end and r["show"] == "red"), None)
    cap = cv2.VideoCapture(str(video_labels.find_video(video)))
    fps = fps_out or cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "split_open.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    bg, ink, muted = (251, 249, 247), (30, 30, 30), (110, 120, 130)
    # 手機（左半）：機身、螢幕、狀態列、直播列、影片區
    px0, py0, px1, py1 = 480 - 250, 90, 480 + 250, 1050
    sx0, sy0, sx1, sy1 = px0 + 18, py0 + 18, px1 - 18, py1 - 18
    vy0, vy1 = sy0 + 120, sy1 - 40
    # 護理站卡片（右半）：影片框、燈號面板
    cx0, cy0, cx1, cy1 = 1000, 150, 1470, 1000
    n = 0
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, frame = cap.read()
        if not ok or t > end:
            break
        row = latest(samples, t) or samples[0][1]
        color, title, desc = STATES[row["show"]]
        canvas = np.full((H, W, 3), bg[::-1], np.uint8)
        canvas[:, :960] = (36, 32, 30)
        cam = fit(frame, sx1 - sx0, vy1 - vy0)
        oy, ox = vy0 + (vy1 - vy0 - cam.shape[0]) // 2, sx0 + (sx1 - sx0 - cam.shape[1]) // 2
        phone_cam = (ox, oy, cam)
        card = fit(frame, cx1 - cx0, cy1 - cy0)
        kx, ky = cx0 + (cx1 - cx0 - card.shape[1]) // 2, cy0 + (cy1 - cy0 - card.shape[0]) // 2
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.text((60, 30), "Parent's phone · live stream", font=font(30, True), fill=(235, 235, 235))
        d.rounded_rectangle((px0, py0, px1, py1), radius=64, fill=(12, 12, 14), outline=(90, 90, 96), width=3)
        d.rounded_rectangle((sx0, sy0, sx1, sy1), radius=48, fill=(0, 0, 0))
        d.rounded_rectangle((480 - 60, sy0 + 12, 480 + 60, sy0 + 40), radius=14, fill=(12, 12, 14))  # 瀏海
        d.text((sx0 + 36, sy0 + 12), clock_text(t, seconds=False), font=font(24, True), fill=(255, 255, 255))
        d.rounded_rectangle((sx1 - 76, sy0 + 16, sx1 - 36, sy0 + 34), radius=5, outline=(255, 255, 255), width=2)
        d.rectangle((sx1 - 72, sy0 + 20, sx1 - 46, sy0 + 30), fill=(255, 255, 255))
        d.rounded_rectangle((sx0 + 24, sy0 + 66, sx0 + 112, sy0 + 102), radius=8, fill=(217, 45, 32))
        d.text((sx0 + 38, sy0 + 71), "LIVE", font=font(24, True), fill=(255, 255, 255))
        d.text((sx0 + 128, sy0 + 70), f"Live · Crib {crib}", font=font(26, True), fill=(255, 255, 255))
        d.text((sx1 - 24 - d.textlength(clock_text(t), font=font(22)), sy0 + 74), clock_text(t), font=font(22),
               fill=(190, 190, 190))
        d.rounded_rectangle((480 - 70, sy1 - 22, 480 + 70, sy1 - 14), radius=4, fill=(200, 200, 200))
        img.paste(Image.fromarray(cv2.cvtColor(phone_cam[2], cv2.COLOR_BGR2RGB)), (phone_cam[0], phone_cam[1]))
        # 護理站卡片
        d.text((1000, 30), f"{PRODUCT} · nurse station", font=font(30, True), fill=(16, 74, 133))
        d.text((1000, 76), f"{clock_text(t)} · Crib {crib}", font=font(26), fill=muted)
        img.paste(Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB)), (kx, ky))
        d.rectangle((kx - 6, ky - 6, kx + card.shape[1] + 5, ky + card.shape[0] + 5), outline=color, width=14)
        x0 = 1510
        d.text((x0, 150), f"Crib {crib}", font=font(30), fill=muted)
        d.rounded_rectangle((x0, 196, 1880, 300), radius=24, fill=color)
        d.text((x0 + 28, 208), title, font=font(64, True), fill=(255, 255, 255))
        for i, line in enumerate(wrap(d, desc, font(28), 1880 - x0)):
            d.text((x0, 322 + i * 38), line, font=font(28), fill=ink)
        if row["show"] == "count":
            frac = min(row["count"] / 3.0, 1.0)
            d.rounded_rectangle((x0, 460, 1880, 492), radius=16, fill=(235, 228, 205))
            d.rounded_rectangle((x0, 460, x0 + int((1880 - x0) * frac), 492), radius=16, fill=color)
            d.text((x0, 502), f"{row['count']:.1f} s of 3 s", font=font(26), fill=muted)
        if red_at is not None and t >= red_at:
            draw_push(img, 1890, 840, f"Crib {crib} · nose and mouth hidden 3 s", t - red_at)
        d.text((W - 20 - d.textlength(PROTOTYPE_TAG, font=font(20)), H - 34), PROTOTYPE_TAG, font=font(20),
               fill=muted)
        if t >= end - 2:  # 最後 2 秒：字幕淡入
            a = min((t - (end - 2)) / 0.4, 1.0)
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.rectangle((0, H - 200, W, H - 70), fill=(0, 0, 0, int(190 * a)))
            msg = "Same crib, same second. The nurse knows first."
            ld.text(((W - ld.textlength(msg, font=font(54, True))) / 2, H - 168), msg, font=font(54, True),
                    fill=(255, 255, 255, int(255 * a)))
            img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    if red_at is None:
        print(f"[注意] {video} {start:g}–{end:g} 秒沒有紅燈，不會出現推播小卡")
    print(f"→ {out}（{n} 幀，{n / fps:.1f} 秒；紅燈在 {red_at} 秒）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="影片名稱（--grid 時不用給）")
    ap.add_argument("--start", type=float, help="開始秒數")
    ap.add_argument("--end", type=float, help="結束秒數")
    ap.add_argument("--crib", default="Crib 07 · per-crib camera", help="右上角的床位名稱")
    ap.add_argument("--grid", action="store_true", help="改畫護理站畫面示意（8 格）")
    ap.add_argument("--split", action="store_true",
                    help="改畫開場雙畫面（左家長手機、右護理站）；影片預設 IMG_8432、86–100 秒")
    args = ap.parse_args()
    if args.grid:
        return render_grid()
    if args.split:
        return render_split(args.video or "IMG_8432", 86 if args.start is None else args.start,
                            100 if args.end is None else args.end)

    ann = video_labels.load()[args.video]
    slot = next(iter(ann["slots"].values()))
    rows = timeline_states(args.video, slot, args.end)
    samples = [(r["t"], r) for r in rows]
    boxes = sorted(face_boxes(args.video, ann, args.start, args.end).items())

    cap = cv2.VideoCapture(str(video_labels.find_video(args.video)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.video}_{args.start:g}-{args.end:g}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, frame = cap.read()
        if not ok or t > args.end:
            break
        row = latest(samples, t) or samples[0][1]
        color = STATES[row["show"]][0]
        for (x0, y0, x1, y1) in latest(boxes, t) or []:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 6)
        s = H / frame.shape[0]
        cam = cv2.resize(frame, (round(frame.shape[1] * s), H), interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W, 3), (251, 249, 247), np.uint8)
        cx = 60
        canvas[:, cx:cx + cam.shape[1]] = cam
        cv2.rectangle(canvas, (cx + 8, 8), (cx + cam.shape[1] - 9, H - 9), color[::-1], 16)
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw_panel(img, row, rows, args.start, args.end, t, args.crib)
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    print(f"→ {out}（{n} 幀，{n / fps:.1f} 秒）")


if __name__ == "__main__":
    main()
