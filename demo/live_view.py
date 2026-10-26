"""
live_view.py
筆電即時判讀（給影片「真的在跑」那一段錄螢幕用）：讀一支每床鏡頭的娃娃影片，按影片原速播放，同時即時跑
v3＋床位判讀（每秒 FPS 幀，crib_detect.detect()）和床位情境狀態機（crib_state.CribState：照護中暫停計時、空床灰燈），
用 OpenCV 視窗顯示：左邊鏡頭畫面＋臉框＋燈號外框，右邊燈號與判斷依據，右上角是每幀處理毫秒數、目前判讀 FPS、在什麼硬體上跑。

不是事先算好的：不讀 outputs/ 的任何東西，每一幀都是當下才算（床位特徵也是逐幀算，見 crib_state.EdgeFeatures）。
處理不過來時會跳過畫面、維持原速（跟真的裝置一樣），右上角的 FPS 就會掉到 5 以下。
空床參考照（床位檔的 empty_ref）在開始前先讀一張，等於安裝時拍的空床照。
VLM 不即時跑（筆電 GPU 在另一個虛擬環境），畫面上不顯示 VLM，所以沒有「請看一眼」的黃燈。

用法：
    uv run python -m demo.live_view IMG_8432 --start 80                 # 開視窗；q／Esc 結束、空白鍵暫停
    uv run python -m demo.live_view IMG_8432 --start 80 --end 112 --save outputs/demo/live_IMG_8432.mp4   # 同時把視窗畫面存一份
錄螢幕：Windows 按 Win+Alt+R（Xbox Game Bar）或用 OBS。
"""

import argparse
import os
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np
from PIL import Image, ImageDraw

from demo import demo_render
from demo.demo_render import PRODUCT, STATES, font, wrap
from evaluation import crib_context, video_labels
from pipeline import crib_detect, crib_state, infant_face_v3, infant_face_v5
from pipeline.face_common import FACE_MODEL, FACE_SCORE_TH

FPS = crib_context.FPS  # 每秒判讀幾幀（跟 eval_video.py、crib_context.py 一樣）
TITLE = f"{PRODUCT} · live"


def draw_panel(size, cam_box, show, row, metrics, args):
    """右邊面板＋右上角效能數字（PIL；只在每次判讀後重畫，每秒 FPS 次）。cam_box 是鏡頭畫面的位置，留白給每幀貼影像。"""
    w, h = size
    color, title, desc = STATES[show]
    img = Image.new("RGB", (w, h), (251, 249, 247))
    d = ImageDraw.Draw(img)
    cx0, cy0, cx1, cy1 = cam_box
    d.rectangle((cx0 - 14, cy0 - 14, cx1 + 13, cy1 + 13), fill=color)
    k = w / 1600
    x0, x1 = cx1 + int(60 * k), w - int(40 * k)
    f = lambda size, bold=False: font(max(10, int(size * k)), bold)  # noqa: E731
    # 右上角：效能
    d.rounded_rectangle((x0, int(30 * k), x1, int(200 * k)), radius=int(16 * k), fill=(30, 32, 36))
    d.ellipse((x0 + int(22 * k), int(52 * k), x0 + int(40 * k), int(70 * k)), fill=(217, 45, 32))
    d.text((x0 + int(52 * k), int(46 * k)), "LIVE · computed now, not pre-rendered", font=f(24, True), fill=(255, 255, 255))
    ms = "—" if metrics["ms"] is None else f"{metrics['ms']:.0f} ms"
    rate = "—" if metrics["fps"] is None else f"{metrics['fps']:.1f} fps"
    d.text((x0 + int(22 * k), int(88 * k)), f"Per-frame processing  {ms}", font=f(28, True), fill=(255, 255, 255))
    d.text((x0 + int(22 * k), int(124 * k)), f"Analysis rate  {rate}  (target {args.fps:g})", font=f(24), fill=(210, 210, 210))
    d.text((x0 + int(22 * k), int(158 * k)), f"Running on: {args.device}", font=f(24), fill=(210, 210, 210))
    # 燈號
    y = int(240 * k)
    d.text((x0, y), f"{args.crib} · {args.video} {metrics['clock']}", font=f(26), fill=(90, 100, 110))
    d.rounded_rectangle((x0, y + int(44 * k), x1, y + int(160 * k)), radius=int(24 * k), fill=color)
    d.text((x0 + int(32 * k), y + int(56 * k)), title, font=f(76, True), fill=(255, 255, 255))
    y += int(180 * k)
    for i, line in enumerate(wrap(d, desc, f(30), x1 - x0)):
        d.text((x0, y + i * int(40 * k)), line, font=f(30), fill=(30, 30, 30))
    y += int(84 * k)
    if show == "count":  # 3 秒倒數條
        frac = min(row["count"] / 3.0, 1.0)
        d.rounded_rectangle((x0, y, x1, y + int(30 * k)), radius=int(15 * k), fill=(235, 228, 205))
        d.rounded_rectangle((x0, y, x0 + int((x1 - x0) * frac), y + int(30 * k)), radius=int(15 * k), fill=color)
        d.text((x0, y + int(38 * k)), f"{row['count']:.1f} s of 3 s", font=f(24), fill=(90, 100, 110))
    y += int(96 * k)
    d.text((x0, y), "WHAT THE BOX SEES", font=f(24, True), fill=(16, 74, 133))
    signals = [
        ("Face detector (pose + face, crib area)", "nose and mouth NOT visible" if row["hidden"] else "nose and mouth visible",
         STATES["red"][0] if row["hidden"] else STATES["green"][0]),
        ("Adult at the crib (edge motion)", "yes: timer paused" if row["adult"] else "no",
         STATES["care"][0] if row["adult"] else STATES["empty"][0]),
        ("Matches this crib's empty photo", "yes" if row["matches"] else "no", STATES["empty"][0]),
    ]
    for i, (label, value, c) in enumerate(signals):
        yy = y + int((44 + i * 72) * k)
        d.ellipse((x0, yy + int(8 * k), x0 + int(20 * k), yy + int(28 * k)), fill=c)
        d.text((x0 + int(36 * k), yy), label, font=f(24), fill=(90, 100, 110))
        d.text((x0 + int(36 * k), yy + int(30 * k)), value, font=f(28, True), fill=(30, 30, 30))
    d.text((x0, h - int(44 * k)), "White box: a face the detector found · doll re-enactment · laptop prototype",
           font=f(22), fill=(120, 130, 140))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Analyzer(threading.Thread):
    """背景判讀：主迴圈照影片原速播放，每 1/fps 秒交一幀進來；上一幀還沒判讀完就不收
    （判讀跟不上時 FPS 會掉，跟真的裝置一樣）。畫面因此不會因為判讀而卡頓，燈號比畫面晚一次判讀的時間。
    onnxruntime、OpenCV 運算時會放開 GIL，所以用執行緒就好。"""

    def __init__(self, args, doll, slots, rots, run_pose, det, det_v5, feat, ctx, size, cam_box, scale):
        super().__init__(daemon=True)
        self.args, self.doll, self.slots, self.rots = args, doll, slots, rots
        self.run_pose, self.det, self.det_v5, self.feat, self.ctx = run_pose, det, det_v5, feat, ctx
        self.size, self.cam_box, self.scale = size, cam_box, scale
        self.cond = threading.Condition()
        self.inbox, self.busy, self.stopped = None, False, False
        self.panel, self.boxes, self.count = None, [], 0
        self.done = deque(maxlen=int(args.fps * 3) + 1)  # 最近幾次判讀完成的時間，算判讀 FPS
        self.ms_hist = deque(maxlen=max(1, int(args.fps)))  # 最近一秒的處理毫秒數，取平均顯示

    def submit(self, t, frame):
        with self.cond:
            if self.busy or self.inbox is not None:
                return False
            self.inbox = (t, frame.copy())
            self.cond.notify()
            return True

    def latest(self):
        with self.cond:
            return self.panel, self.boxes

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify()

    def metrics(self, t):
        fps = (len(self.done) - 1) / (self.done[-1] - self.done[0]) if len(self.done) > 2 else None
        ms = sum(self.ms_hist) / len(self.ms_hist) if self.ms_hist else None
        return {"ms": ms, "fps": fps, "clock": f"{int(t // 60)}:{t % 60:04.1f}"}

    def run(self):
        while True:
            with self.cond:
                while self.inbox is None and not self.stopped:
                    self.cond.wait()
                if self.stopped:
                    return
                (t, frame), self.inbox, self.busy = self.inbox, None, True
            tic = time.perf_counter()
            result, _, faces = crib_detect.detect(self.run_pose, self.det, self.det_v5, frame, self.slots, self.rots)
            hidden = result[self.doll]["v3_slot"] == "hidden"
            motion, diff = self.feat.update(frame)
            state = self.ctx.update(t, hidden, motion, diff)
            show = demo_render.display_state(self.ctx, state, hidden)
            self.ms_hist.append((time.perf_counter() - tic) * 1000)
            self.done.append(time.perf_counter())
            self.count += 1
            since = self.ctx.alarm.since_hidden
            row = {"hidden": hidden, "count": t - since if since is not None else 0.0,
                   "adult": self.ctx.last_motion is not None and t - self.ctx.last_motion < crib_state.CARE_HOLD,
                   "matches": diff is not None and diff < crib_state.EMPTY_TH}
            panel = draw_panel(self.size, self.cam_box, show, row, self.metrics(t), self.args)
            boxes = [tuple(int(v * self.scale) for v in f[:4]) for f in faces]
            with self.cond:
                self.panel, self.boxes, self.busy = panel, boxes, False


def main():
    ap = argparse.ArgumentParser(description="筆電即時判讀（錄螢幕用）")
    ap.add_argument("video", help="影片名稱（每床鏡頭、床位檔裡只有一個床位）")
    ap.add_argument("--start", type=float, default=0, help="從第幾秒開始")
    ap.add_argument("--end", type=float, help="播到第幾秒（預設播完）")
    ap.add_argument("--fps", type=float, default=FPS, help=f"每秒判讀幾幀（預設 {FPS}，跟評估一樣）")
    ap.add_argument("--det-th", type=float, default=0.15, help="RTMO 找人門檻（預設 0.15，跟 eval_video.py 一樣）")
    ap.add_argument("--device", default="laptop CPU (Intel i7-11800H)", help="右上角「Running on」的文字")
    ap.add_argument("--crib", default="Crib 07", help="床位名稱")
    ap.add_argument("--width", type=int, default=1600, help="視窗寬度（高度＝寬度×9/16）")
    ap.add_argument("--save", help="同時把視窗畫面存成影片（.mp4）")
    ap.add_argument("--no-window", action="store_true", help="不開視窗（搭配 --save 用）")
    args = ap.parse_args()

    ann = video_labels.load()[args.video]
    if len(ann["slots"]) != 1:
        raise SystemExit(f"{args.video} 一個畫面有 {len(ann['slots'])} 個床位；這支只處理每床鏡頭（一個畫面一張床）")
    doll, slot = next(iter(ann["slots"].items()))
    print("載入模型（RTMO、YuNet）…")
    infant_face_v3.DET_TH = args.det_th
    run_pose = infant_face_v3.load_rtmo()
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    det_v5 = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), infant_face_v5.FACE_TH_V5)
    slots, rots = {doll: slot["rect"]}, {doll: slot["rot"]}

    cap = cv2.VideoCapture(str(video_labels.find_video(args.video)))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    feat = crib_state.EdgeFeatures(slot["rect"], crib_state.read_empty_ref(cap, slot))  # 安裝時拍的空床照
    ctx = crib_state.CribState()
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    W = args.width // 2 * 2
    H = W * 9 // 16 // 2 * 2
    margin = int(40 * W / 1600)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min((H - 2 * margin) / fh, (W * 0.55) / fw)
    cw, ch = round(fw * s), round(fh * s)
    cam_box = (margin, (H - ch) // 2, margin + cw, (H - ch) // 2 + ch)

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"avc1"), src_fps, (W, H))
        if not writer.isOpened():
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
    if not args.no_window:
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, W, H)

    analyzer = Analyzer(args, doll, slots, rots, run_pose, det, det_v5, feat, ctx, (W, H), cam_box, s)
    analyzer.start()
    interval = 1 / args.fps - 0.5 / src_fps   # 距離上次送去判讀超過這麼久，就把這一幀送去
    last_sent, wall0, frames, skipped = -1e9, None, 0, 0
    print("開始（q／Esc 結束、空白鍵暫停）")
    while True:
        t = cap.get(cv2.CAP_PROP_POS_FRAMES) / src_fps
        ok, frame = cap.read()
        if not ok or (args.end is not None and t > args.end):
            break
        if t - last_sent >= interval and analyzer.submit(t, frame):
            last_sent = t
        if wall0 is None:  # 第一幀：等模型暖機、判讀完才開始計時
            while analyzer.panel is None:
                time.sleep(0.01)
            wall0 = time.perf_counter() - (t - args.start)
        panel, boxes = analyzer.latest()
        canvas = panel.copy()
        cam = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
        for (x0, y0, x1, y1) in boxes:
            cv2.rectangle(cam, (x0, y0), (x1, y1), (255, 255, 255), max(2, W // 400))
        canvas[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2]] = cam
        if writer is not None:
            writer.write(canvas)
        frames += 1

        # 維持影片原速：快了就等，慢了就跳過幾幀
        ahead = (t - args.start) - (time.perf_counter() - wall0) + 1 / src_fps
        key = -1
        if args.no_window:
            if ahead > 0:
                time.sleep(ahead)
        else:
            cv2.imshow(TITLE, canvas)
            key = cv2.waitKey(max(1, int(ahead * 1000))) & 0xFF
        for _ in range(max(0, int(-ahead * src_fps))):
            cap.grab()
            skipped += 1
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused_at = time.perf_counter()
            while (key := cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q"), 27):
                pass
            if key != ord(" "):
                break
            wall0 += time.perf_counter() - paused_at
    analyzer.stop()
    cap.release()
    if writer is not None:
        writer.release()
        print(f"→ {args.save}")
    cv2.destroyAllWindows()
    m = analyzer.metrics(0)
    print(f"顯示 {frames} 幀、為了維持原速跳過 {skipped} 幀；判讀 {analyzer.count} 次，最後判讀 FPS "
          f"{m['fps'] or 0:.1f}、每幀處理平均 {m['ms'] or 0:.0f} ms")


if __name__ == "__main__":
    main()
