"""
live_view.py
筆電即時判讀（給影片「真的在跑」那一段錄螢幕用）：讀一支每床鏡頭的娃娃影片，按影片原速播放，同時即時跑
主流程 v7＋床位判讀（每秒 FPS 幀，crib_detect.detect()；--mode 可換回 v6_slot、v3_slot 等對照）和床位情境狀態機（crib_state.CribState：照護中暫停計時、空床灰燈），
用 OpenCV 視窗顯示：左邊鏡頭畫面＋臉框＋燈號外框，右邊燈號與判斷依據，右上角是每幀處理毫秒數、目前判讀 FPS、在什麼硬體上跑。

不是事先算好的：不讀 outputs/ 的任何東西，每一幀都是當下才算（床位特徵也是逐幀算，見 crib_state.EdgeFeatures）。
處理不過來時會跳過畫面、維持原速（跟真的裝置一樣），右上角的 FPS 就會掉到 5 以下。
空床參考照（床位檔的 empty_ref）在開始前先讀一張，等於安裝時拍的空床照。
VLM 不即時跑（筆電 GPU 在另一個虛擬環境），畫面上不顯示 VLM，所以沒有「請看一眼」的黃燈。

用法：
    uv run python -m demo.live_view IMG_8432 --start 80                 # 開視窗；q／Esc 結束、空白鍵暫停
    uv run python -m demo.live_view IMG_8432 --start 80 --end 112 --save outputs/demo/live_IMG_8432.mp4   # 同時把視窗畫面存一份
    uv run python -m demo.live_view IMG_8432 --start 80 --fault error@90          # 測離線狀態（紫燈）：第 90 秒起判讀出錯 5 秒
    uv run python -m demo.live_view IMG_8432 --start 80 --chip                    # 決賽展演的設定：RTMO、口鼻模型、RetinaFace 在 UGen300（CHIP_FINALS）
    uv run python -m demo.live_view IMG_8432 --start 80 --chip --fault hang@90 kill@105   # 測晶片卡住、當掉時的紫燈
錄螢幕：Windows 按 Win+Alt+R（Xbox Game Bar）或用 OBS。
離線（紫燈）：判讀出錯、或超過 2 秒沒判讀完（晶片斷線重開、卡住）就不顯示最後的燈號，見 crib_state.Liveness。
"""

import argparse
import os
import re
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
from pipeline import crib_detect, crib_state, infant_face_v3, infant_face_v5, infant_face_v6, infant_face_v7
from pipeline.face_common import FACE_SCORE_TH, create_face_detector

FPS = crib_context.FPS  # 每秒判讀幾幀（跟 eval_video.py、crib_context.py 一樣）
# 決賽展演的晶片設定（2026/11/25 定，加 --chip 就套用）：RTMO（D1）和口鼻小塊模型（F，全部 16 位元）在 UGen300，YuNet 留在 CPU；
# 11/27 起 v7 的 RetinaFace（640×480）也放晶片（CPU 上一次約 73 ms、晶片約 16 ms）。
# 晶片放子程序、一次送一張（長時間測試最穩的接法，見 hailo/結果.md「裝置斷線」）。開發、重訓時不加：口鼻模型在 CPU 跑，重訓不用重編 HEF。
CHIP_FINALS = {"FG_CHIP_PROCESS": "1", "FG_RTMO_HEF": "model/rtmo_D1.hef", "FG_MOUTH_HEF": "model/mouth_F.hef",
               "FG_RETINA_HEF": "model/retinaface_v1_640x480.hef"}
TITLE = f"{PRODUCT} · live"


def ascii_only(text):
    """畫面字型（Arial）沒有中文：錯誤訊息裡的中文（例如晶片子程序的「晶片呼叫失敗：」）拿掉，只留英文部分。"""
    if not text:
        return text
    text = re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7f]+", " ", text))
    return re.sub(r":(\s*:)+", ":", text).strip(" :")


def fit_lines(d, text, size, width, bold=False, max_lines=2, min_ratio=0.75, wrap_first=False):
    """把 text 放進 width 像素寬（11/28：橫的鏡頭畫面讓右邊面板變窄，「Running on …」和最下面的說明被切掉）：
    先把字從 size 縮到 size×min_ratio，還放不下就斷成最多 max_lines 行（行數不夠再繼續縮，最小 10 像素）；
    wrap_first=True 時先用原本的字級斷行（高度夠、寧可多一行也不要字太小時用）。
    只能一行、縮到最小還放不下時，尾巴改成「…」。回傳 (字型, [行])。"""
    def wrapped(s):
        f = font(s, bold)
        lines = wrap(d, text, f, width)
        ok = len(lines) <= max_lines and all(d.textlength(line, font=f) <= width for line in lines)
        return f, lines, ok

    low = max(10, int(size * min_ratio))
    if wrap_first and max_lines > 1:
        for s in range(int(size), low - 1, -1):
            f, lines, ok = wrapped(s)
            if ok:
                return f, lines
    for s in range(int(size), low - 1, -1):
        f = font(s, bold)
        if d.textlength(text, font=f) <= width:
            return f, [text]
    if max_lines > 1:
        for s in range(low, 9, -1):
            f, lines, ok = wrapped(s)
            if ok:
                return f, lines
        return f, lines[:max_lines]
    f = font(low, bold)
    cut = text
    while cut and d.textlength(cut + "…", font=f) > width:
        cut = cut[:-1]
    return f, [cut + "…" if cut != text else text]


def draw_fit(d, xy, text, size, width, fill, bold=False, max_lines=2, up=False, gap=1.2, wrap_first=False):
    """畫一段放得進 width 的字（fit_lines）。up=True 時 xy 是最後一行的位置，多出來的行往上長（給貼底的說明用）。
    回傳用掉的高度。"""
    f, lines = fit_lines(d, text, size, width, bold, max_lines, wrap_first=wrap_first)
    lh = int(f.size * gap)
    x, y = xy
    if up:
        y -= lh * (len(lines) - 1)
    for i, line in enumerate(lines):
        d.text((x, y + i * lh), line, font=f, fill=fill)
    return lh * len(lines)


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
    inner = x1 - x0 - int(44 * k)   # 深色框裡可以寫字的寬度（左右各留 22）
    draw_fit(d, (x0 + int(52 * k), int(46 * k)), "LIVE · computed now, not pre-rendered", 24 * k, inner - int(30 * k),
             (255, 255, 255), bold=True, max_lines=1)
    ms = "—" if metrics["ms"] is None else f"{metrics['ms']:.0f} ms"
    rate = "—" if metrics["fps"] is None else f"{metrics['fps']:.1f} fps"
    draw_fit(d, (x0 + int(22 * k), int(88 * k)), f"Per-frame processing  {ms}", 28 * k, inner, (255, 255, 255),
             bold=True, max_lines=1)
    draw_fit(d, (x0 + int(22 * k), int(124 * k)), f"Analysis rate  {rate}  (target {args.fps:g})", 24 * k, inner,
             (210, 210, 210), max_lines=1)
    # 「Running on」最長（晶片設定時三個模型名）：放不下就縮字、再不行斷兩行（深色框底在 200，兩行剛好放得下）
    draw_fit(d, (x0 + int(22 * k), int(156 * k)), f"Running on: {args.device}", 24 * k, inner, (210, 210, 210),
             max_lines=2, gap=1.05)
    # 燈號
    y = int(240 * k)
    draw_fit(d, (x0, y), f"{args.crib} · {args.video} {metrics['clock']}", 26 * k, x1 - x0, (90, 100, 110), max_lines=1)
    d.rounded_rectangle((x0, y + int(44 * k), x1, y + int(160 * k)), radius=int(24 * k), fill=color)
    d.text((x0 + int(32 * k), y + int(56 * k)), title, font=f(76, True), fill=(255, 255, 255))
    y += int(180 * k)
    for i, line in enumerate(wrap(d, desc, f(30), x1 - x0)):
        d.text((x0, y + i * int(40 * k)), line, font=f(30), fill=(30, 30, 30))
    y += int(84 * k)
    if show == "offline":  # 離線：不顯示判斷依據（都是舊的），改顯示多久沒判讀、為什麼
        y += int(96 * k)
        d.text((x0, y), "WHY", font=f(24, True), fill=(16, 74, 133))
        reason = ascii_only(row["error"]) or "no answer from the analysis (chip busy or reconnecting)"
        signals = [("Last completed analysis", f"{row['silent']:.0f} s ago", color),
                   ("Reason", reason[:60], color)]
        for i, (label, value, c) in enumerate(signals):
            yy = y + int((44 + i * 72) * k)
            d.ellipse((x0, yy + int(8 * k), x0 + int(20 * k), yy + int(28 * k)), fill=c)
            draw_fit(d, (x0 + int(36 * k), yy), label, 24 * k, x1 - x0 - int(36 * k), (90, 100, 110), max_lines=1)
            draw_fit(d, (x0 + int(36 * k), yy + int(30 * k)), value, 28 * k, x1 - x0 - int(36 * k), (30, 30, 30),
                     bold=True, max_lines=1)
        draw_fit(d, (x0, h - int(44 * k)), "Lights resume by themselves after the next completed analysis · laptop prototype",
                 22 * k, x1 - x0, (120, 130, 140), up=True, wrap_first=True)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
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
        draw_fit(d, (x0 + int(36 * k), yy), label, 24 * k, x1 - x0 - int(36 * k), (90, 100, 110), max_lines=1)
        draw_fit(d, (x0 + int(36 * k), yy + int(30 * k)), value, 28 * k, x1 - x0 - int(36 * k), (30, 30, 30),
                 bold=True, max_lines=1)
    draw_fit(d, (x0, h - int(44 * k)), "White box: a face the detector found · doll re-enactment · laptop prototype",
             22 * k, x1 - x0, (120, 130, 140), up=True)   # 先縮字：斷兩行會壓到上面的判斷依據
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def parse_faults(specs):
    """--fault 種類@秒 → [(秒, 種類)]，照時間排好。"""
    out = []
    for s in specs or []:
        kind, _, sec = s.partition("@")
        if kind not in FAULTS or not sec:
            raise SystemExit(f"--fault {s}：要寫成 種類@秒，種類是 {'／'.join(FAULTS)}")
        out.append((float(sec), kind))
    return sorted(out)


FAULTS = ("error", "kill", "hang")
ERROR_S = 5   # --fault error：判讀連續出錯幾秒


class Analyzer(threading.Thread):
    """背景判讀：主迴圈照影片原速播放，每 1/fps 秒交一幀進來；上一幀還沒判讀完就不收
    （判讀跟不上時 FPS 會掉，跟真的裝置一樣）。畫面因此不會因為判讀而卡頓，燈號比畫面晚一次判讀的時間。
    onnxruntime、OpenCV 運算時會放開 GIL，所以用執行緒就好。
    離線（crib_state.Liveness）：判讀出錯、或超過 2 秒沒有判讀完（晶片斷線重開、卡住），latest() 就給紫色的離線畫面，
    不給最後一次的燈號；下一次判讀成功自己恢復。"""

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
        self.live = crib_state.Liveness()
        self.errors = self.lapses = self.resumes = 0
        self.last_t = None   # 上一次判讀成功的那一幀的影片時間
        self.faults, self.error_until = parse_faults(args.fault), -1.0
        self._off_key, self._off_panel, self._shown_off, self._off_since = None, None, False, None

    def inject_fault(self, t):
        """--fault 測試用：到了影片第幾秒製造一次故障。error：接下來 ERROR_S 秒判讀直接出錯（不用晶片）；
        kill：砍掉晶片子程序（等於晶片當掉，FG_CHIP_PROCESS=1 才有）；hang：暫停晶片子程序（等於卡住，主程序等 chip_process.CALL_TIMEOUT_S
        沒回應就砍掉重開：11/28 晚起 2 秒、一次好幾張口鼻小塊時每張多 PIPE_PER_IMAGE_S，舊版 12 秒）。"""
        while self.faults and t >= self.faults[0][0]:
            at, kind = self.faults.pop(0)
            print(f"[故障測試] 影片 {t:.1f} 秒：{kind}", flush=True)
            if kind == "error":
                self.error_until = at + ERROR_S
                continue
            import hailo.chip_process as cp
            if cp._proc is None:
                print("[故障測試] 沒有晶片子程序（要設 FG_CHIP_PROCESS=1、FG_RTMO_HEF），略過", flush=True)
            elif kind == "kill":
                cp._proc.kill()
            else:
                import psutil
                psutil.Process(cp._proc.pid).suspend()
        if t < self.error_until:
            raise RuntimeError("simulated analysis error (--fault error)")

    def submit(self, t, frame):
        with self.cond:
            if self.busy or self.inbox is not None:
                return False
            self.inbox = (t, frame.copy())
            self.cond.notify()
            return True

    def latest(self, t):
        """目前該顯示的面板和臉框。離線時給離線面板（每秒重畫一次，顯示多久沒判讀），臉框不畫（都是舊的）。"""
        off = self.live.offline()
        if off != self._shown_off:
            self._shown_off = off
            if off:
                self.lapses += 1
                self._off_since = time.monotonic()
                print(f"[離線] 影片 {t:.1f} 秒：{self.live.silent_for():.1f} 秒沒有判讀"
                      + (f"（{self.live.error[:80]}）" if self.live.error else ""), flush=True)
            else:
                print(f"[恢復] 影片 {t:.1f} 秒：離線 {time.monotonic() - self._off_since:.1f} 秒", flush=True)
        if not off:
            with self.cond:
                return self.panel, self.boxes
        key = (int(self.live.silent_for()), self.live.error)
        if key != self._off_key:
            row = {"silent": self.live.silent_for(), "error": self.live.error}
            self._off_key = key
            self._off_panel = draw_panel(self.size, self.cam_box, "offline", row, self.metrics(t), self.args)
        return self._off_panel, []

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
            try:
                self.inject_fault(t)
                result, _, faces = crib_detect.detect(self.run_pose, self.det, self.det_v5, frame, self.slots, self.rots,
                                                      mouth=self.args.mouth, retina=self.args.retina)
            except Exception as e:   # 晶片重開、重送之後還是失敗等：這床離線，等下一幀再試；狀態機不動
                self.live.fail(e)
                self.errors += 1
                if self.errors <= 3 or self.errors % 50 == 0:
                    print(f"[判讀出錯 {self.errors}] 影片 {t:.1f} 秒：{type(e).__name__}: {str(e)[:100]}", flush=True)
                with self.cond:
                    self.busy = False
                continue
            hidden = result[self.doll][self.args.mode] == "hidden"
            if self.last_t is not None and t - self.last_t > crib_state.OFFLINE_AFTER:
                # 這一幀跟上一次判讀的幀隔太久（中間離線）：用畫面時間判斷，因為卡住時那一幀是卡住前送進來的，
                # 恢復後的「下一幀」才是隔了十幾秒的那一幀
                self.feat.reset()
                self.ctx.resume(self.last_t)
                self.resumes += 1
            self.last_t = t
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
            self.live.ok()   # 新面板放好之後才算恢復，不然會閃一下斷線前的舊燈號


def main():
    ap = argparse.ArgumentParser(description="筆電即時判讀（錄螢幕用）")
    ap.add_argument("video", help="影片名稱（每床鏡頭、床位檔裡只有一個床位）")
    ap.add_argument("--start", type=float, default=0, help="從第幾秒開始")
    ap.add_argument("--end", type=float, help="播到第幾秒（預設播完）")
    ap.add_argument("--fps", type=float, default=FPS, help=f"每秒判讀幾幀（預設 {FPS}，跟評估一樣）")
    ap.add_argument("--det-th", type=float, default=0.15, help="RTMO 找人門檻（預設 0.15，跟 eval_video.py 一樣）")
    ap.add_argument("--chip", action="store_true", help="用決賽展演的晶片設定（CHIP_FINALS：RTMO、口鼻模型、RetinaFace 在 UGen300）")
    ap.add_argument("--device", help="右上角「Running on」的文字（預設照環境變數寫哪些模型在 UGen300）")
    ap.add_argument("--crib", default="Crib 07", help="床位名稱")
    ap.add_argument("--width", type=int, default=1600, help="視窗寬度（高度＝寬度×9/16）")
    ap.add_argument("--save", help="同時把視窗畫面存成影片（.mp4）")
    ap.add_argument("--no-window", action="store_true", help="不開視窗（搭配 --save 用）")
    ap.add_argument("--mode", default=crib_detect.MAIN_MODE, choices=crib_detect.MODES_V7[1:],
                    help=f"燈號依哪一種判定（預設主流程 {crib_detect.MAIN_MODE}，要有口鼻小塊模型和 RetinaFace 模型檔；"
                         "v6_slot 不用 RetinaFace；都沒有的話用 v3_slot）")
    ap.add_argument("--fault", nargs="*", metavar="種類@秒",
                    help="測試離線狀態：影片第幾秒製造故障，可以給好幾個。error＝判讀連續出錯 5 秒（不用晶片）；"
                         "kill＝砍掉晶片子程序；hang＝暫停晶片子程序（後兩個要 FG_CHIP_PROCESS=1）")
    args = ap.parse_args()
    if args.chip:   # 要在載入模型之前設；殘留的 YuNet HEF、合批設定拿掉，免得混進來
        for k in ("FG_YUNET_HEF", "FG_CHIP_BATCH"):
            os.environ.pop(k, None)
        os.environ.update(CHIP_FINALS)
    if args.device is None:
        on_chip = [n for n, env in (("RTMO", "FG_RTMO_HEF"), ("YuNet", "FG_YUNET_HEF"), ("mouth/nose", "FG_MOUTH_HEF"),
                                    ("RetinaFace", "FG_RETINA_HEF")) if os.environ.get(env)]
        args.device = f"ASUS UGen300 ({', '.join(on_chip)}) + laptop CPU" if on_chip else "laptop CPU (Intel i7-11800H)"
    print(f"Running on: {args.device}" + ("（晶片在子程序）" if os.environ.get("FG_CHIP_PROCESS") == "1" else ""))

    ann = video_labels.load()[args.video]
    if len(ann["slots"]) != 1:
        raise SystemExit(f"{args.video} 一個畫面有 {len(ann['slots'])} 個床位；這支只處理每床鏡頭（一個畫面一張床）")
    doll, slot = next(iter(ann["slots"].items()))
    print("載入模型（RTMO、YuNet" + ("、口鼻小塊模型" if args.mode in ("v6_slot", "v7_slot") else "")
          + ("、RetinaFace" if args.mode == "v7_slot" else "") + "）…")
    infant_face_v3.DET_TH = args.det_th
    run_pose = infant_face_v3.load_rtmo()
    det = create_face_detector(FACE_SCORE_TH)
    det_v5 = create_face_detector(infant_face_v5.FACE_TH_V5)
    args.mouth = infant_face_v6.load_mouth_model(required=True) if args.mode in ("v6_slot", "v7_slot") else None
    args.retina = infant_face_v7.load_retina(required=True) if args.mode == "v7_slot" else None
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
        if wall0 is None:  # 第一幀：等模型暖機、判讀完才開始計時（第一幀就出錯的話直接顯示離線）
            while analyzer.panel is None and analyzer.live.error is None:
                time.sleep(0.01)
            wall0 = time.perf_counter() - (t - args.start)
        panel, boxes = analyzer.latest(t)
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
          f"{m['fps'] or 0:.1f}、每幀處理平均 {m['ms'] or 0:.0f} ms；離線 {analyzer.lapses} 次、判讀出錯 {analyzer.errors} 次、"
          f"斷線後重接狀態機 {analyzer.resumes} 次")


if __name__ == "__main__":
    main()
