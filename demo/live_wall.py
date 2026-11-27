"""
live_wall.py
決賽展演（12/19）的護理站牆面：8 支預錄的每床娃娃影片當 8 支每床鏡頭，同時即時判讀（2026/11/28 寫）。

- 每床一條執行緒解碼，照真實速度播放（308 床的錄影比真實快 1.16–2.13 倍，播放時放慢回真實速度，SPEEDUP）；
  另一條執行緒即時跑主流程 v7＋床位判讀（crib_detect.detect，跟 live_view.py、評估同一段程式）和床位狀態機
  （crib_state.CribState：3 秒警示、照護中暫停計時、空床灰燈；crib_state.Liveness：超過 2 秒沒判讀完或出錯就紫燈）。
  判讀時拿那一床「當下最新的一幀」，判讀跟不上時每床每秒判讀的幀數會掉（8 床在晶片上約每床 1.5 幀），畫面照原速播、不跟著卡。
- 牆面 2×4 格：每格是那床的鏡頭畫面（裁成床位附近）、燈號外框、左上角床號和燈號、黃燈的 3 秒倒數條、白框是找到的臉、
  右下角是這床目前每秒判讀幾幀。右邊是各燈號床數和事件列表；上面是時鐘、合計判讀速度、在什麼硬體上跑。
  燈號規則跟 live_view.py 一樣（demo_render.display_state）：綠、黃（倒數中）、紅、藍（照護中）、灰（空床）、紫（離線）。
- 紅燈推播（demo/ntfy_push.py）：某床變紅燈時推播到手機（標題「FirstGlance · Crib 03」、內容「Nose and mouth hidden for 3 s.
  Check now.」），只有文字、不送畫面；同一床一次紅燈只推一次（解除後再紅才再推）。topic 放環境變數 FG_NTFY_TOPIC，沒設就不推。
  背景送、逾時短，送不出去只記在事件列表，判讀照常。離線（紫燈）不推播（Harry 11/28 定）：--push-offline 才推，預設不推。
- 不是事先算好的：不讀 outputs/ 的任何東西。空床參考照（床位檔的 empty_ref）開始前先讀一張，等於安裝時拍的空床照。
- 播放清單（--playlist，預設 demo/wall_playlist.json）：每床一串片段 {video, start, end}，可以接不同支影片
  （例如 11/23 那批一支只有一個姿勢）。片段之間的剪接點不拿前後兩幀比「床邊有沒有動」（不是真的有人來），
  換了床位框或空床照就重新開始算。所有床一起播完一輪（最長那床）就全部從頭再播（--once 只播一輪）；先播完的床停在最後一幀、繼續判讀。

用法：
    uv run python -m demo.live_wall                                   # CPU（開發用）、播放清單裡的 8 床
    uv run python -m demo.live_wall --cribs 01 02                     # 只跑其中幾床（CPU 測試）
    uv run python -m demo.live_wall --clip bed308_v4_122418:50:100    # 不用播放清單，直接指定片段（影片:開始:結束，可給好幾個）
    uv run python -m demo.live_wall --chip                            # 決賽展演的設定（live_view.CHIP_FINALS：RTMO、口鼻、RetinaFace 在 UGen300）
    $env:FG_NTFY_TOPIC="<topic>"; uv run python -m demo.live_wall --chip --fullscreen    # 展演：紅燈推播到手機
    uv run python -m demo.live_wall --chip --no-window --no-push --duration 36000         # 整晚長時間測試（不開視窗、不推播、10 小時）
    uv run python -m demo.live_wall --chip --fault hang@60 kill@120 error@180:03          # 測紫燈：晶片卡住、當掉、03 床判讀出錯 5 秒
按鍵：q／Esc 結束、空白鍵暫停、r 從頭重播（燈號、事件、推播紀錄都重來；自動播完一輪時事件列表不清）、f 全螢幕切換、
      o 現場示範紫燈（--chip 時暫停晶片子程序＝晶片卡住，約 13.5 秒後自己恢復；CPU 時所有床判讀出錯 5 秒）。
紀錄：每分鐘印一行、存一次 outputs/demo_wall/wall_<開始時間>.json（每床判讀次數、每秒幾幀、最長多久沒判讀、離線次數、推播）；
      同名的 _minutes.csv 每分鐘一列（這一分鐘每床判讀幾次、最長多久沒判讀、離線和晶片重開累計、畫面幀數、CPU %），
      _events.jsonl 每個事件一行（紅、藍、灰、離線、恢復，含日期時間）。
"""

import argparse
import datetime
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np
from PIL import Image, ImageDraw

from demo import demo_render, ntfy_push
from demo.demo_render import PRODUCT, STATES, font
from demo.live_view import CHIP_FINALS, FPS, ascii_only, draw_fit, fit_lines
from evaluation import video_labels
from pipeline import crib_detect, crib_state, infant_face_v3, infant_face_v5, infant_face_v6, infant_face_v7
from pipeline.face_common import FACE_SCORE_TH, create_face_detector

ROOT = Path(__file__).resolve().parent.parent
PLAYLIST = ROOT / "demo" / "wall_playlist.json"
LOG_DIR = ROOT / "outputs" / "demo_wall"
TITLE = f"{PRODUCT} · nurse station"
DISPLAY_FPS = 30   # 牆面每秒重畫幾次（影片照各自的原速播，這裡只是畫面更新）
COLS, ROWS = 4, 2
# 錄影比真實快幾倍（播放時放慢回真實速度）。308 床那 3 支是 rtsp_check --record 每判讀一次才存一幀、檔案卻標每秒 5 幀，
# 照它每列的真實時間算（outputs/field/20261117/*.csv，見 docs/影片測試結果.md §3l）；另一間月子中心那支現場判讀時每秒 5 幀，比真實快 1.8%。
SPEEDUP = {"bed308_main_121857": 2.135, "bed308_v4_122418": 1.527, "bed308_v5_123331": 1.161, "pp_main_215118": 1.018}
EVENT_TEXT = {"red": "nose and mouth hidden 3 s", "care": "adult at the crib", "empty": "baby out of the crib",
              "offline": "no analysis for 2 s: check in person", "online": "analysis resumed"}
COUNT_ORDER = ("GREEN", "YELLOW", "RED", "BLUE", "GRAY", "OFFLINE")
FAULTS = ("error", "kill", "hang")
ERROR_S = 5        # --fault error：判讀連續出錯幾秒（跟 live_view 一樣）
OFFLINE_GROUP_S = 1.0   # --push-offline：這麼多秒內一起離線的床合成一則推播（晶片斷線時 8 床會同時離線）
BG, INK, MUTED, BRAND, LINE = (251, 249, 247), (30, 30, 30), (110, 120, 130), (16, 74, 133), (228, 225, 220)


def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── 播放清單 ──

class Segment:
    """一床的一段片段：影片 video 的 start–end 秒（影片時間），照真實速度播（speedup：錄影比真實快幾倍）。"""

    _refs = {}   # 影片檔 → 空床照（同一支影片的好幾段共用一張）
    _warned = set()

    def __init__(self, ann, video, start=0.0, end=None, speedup=None, view=None, first=False, file=None):
        """first：讀這一段的第一幀留著（每床第一段才要，開始前暖機用；4K 一張 25 MB，不要每段都留）。
        file：改播這個檔（例如 demo/wall_clips.py 把 4K 轉成的 1080 版）；床位標註照 video 那支（畫面比例座標，
        換解析度不影響）。檔案還不在就播原檔、印一次提醒。"""
        if video not in ann:
            raise SystemExit(f"播放清單的 {video} 沒有床位標註（labels/doll_video_slots.csv、local_video_slots.csv）")
        if len(ann[video]["slots"]) != 1:
            raise SystemExit(f"{video} 一個畫面有 {len(ann[video]['slots'])} 個床位；牆面只放每床鏡頭（一個畫面一張床）")
        self.video, self.path, self.converted = video, video_labels.find_video(video), False
        if file:
            alt = Path(file) if Path(file).is_absolute() else ROOT / file
            if alt.exists():
                self.path, self.converted = alt, True
            elif video not in Segment._warned:
                Segment._warned.add(video)
                print(f"[注意] {file} 還沒轉好，{video} 先播原檔（python -m demo.wall_clips 轉檔）")
        self.doll, slot = next(iter(ann[video]["slots"].items()))
        self.rect, self.rot = slot["rect"], slot["rot"]
        cap = cv2.VideoCapture(str(self.path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.start, self.end = float(start), min(float(end), length) if end is not None else length
        if self.end <= self.start:
            raise SystemExit(f"{video}：片段 {self.start:g}–{self.end:g} 秒是空的（影片長 {length:.1f} 秒）")
        key = str(self.path)   # 依檔案存：空床照要跟播的那支同解析度，床位特徵的小圖才對得上
        if key not in Segment._refs:
            Segment._refs[key] = crib_state.read_empty_ref(cap, slot)   # 安裝時拍的空床照（沒有就 None）
        self.ref, self.ref_at = Segment._refs[key], slot["empty_ref"]
        self.first = None
        if first:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start * 1000)
            ok, self.first = cap.read()
            if not ok:
                raise SystemExit(f"{video} 讀不到第 {self.start:g} 秒")
        cap.release()
        self.speed = 1.0 / (speedup or SPEEDUP.get(video, 1.0))   # 真實 1 秒播幾秒影片
        self.real = (self.end - self.start) / self.speed          # 這段播多久（真實秒數）
        self.view = view   # 牆面上顯示的範圍（畫面比例）；None＝床位框四周多留一點

    def feat_key(self):
        """床位框和空床照一樣的兩段，床位特徵可以接著算（只重設移動量）。"""
        return (self.rect, str(self.path) if self.ref is not None else None, self.ref_at)

    def geometry(self, tw, th):
        """牆面一格 tw×th 要顯示原圖的哪一塊、縮多少、貼在格子的哪裡：床位框（或 view）四周多留 6%，
        再往左右或上下補到格子的長寬比（補不到就留黑邊）。回傳 (x0, y0, x1, y1, 縮放, ox, oy, 寬, 高)。"""
        fw, fh = self.size
        r = self.view or self.rect
        x0, y0, x1, y1 = r[0] * fw, r[1] * fh, r[2] * fw, r[3] * fh
        mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(fw, x1 + mx), min(fh, y1 + my)
        aspect = tw / th
        w, h = x1 - x0, y1 - y0
        if w / h < aspect:   # 太窄：左右補
            want = min(fw, h * aspect)
            cx = min(max((x0 + x1) / 2, want / 2), fw - want / 2)
            x0, x1 = cx - want / 2, cx + want / 2
        else:                # 太扁：上下補
            want = min(fh, w / aspect)
            cy = min(max((y0 + y1) / 2, want / 2), fh - want / 2)
            y0, y1 = cy - want / 2, cy + want / 2
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        s = min(tw / (x1 - x0), th / (y1 - y0))
        w, h = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
        return x0, y0, x1, y1, s, (tw - w) // 2, (th - h) // 2, w, h


def parse_clip(spec):
    """--clip 影片[:開始[:結束]] → dict。"""
    parts = spec.split(":")
    clip = {"video": parts[0]}
    if len(parts) > 1 and parts[1]:
        clip["start"] = float(parts[1])
    if len(parts) > 2 and parts[2]:
        clip["end"] = float(parts[2])
    return clip


def load_cribs(args):
    """回傳 [(床號, [Segment, ...], 說明), ...]。"""
    ann = video_labels.load()
    if args.clip:
        spec = [{"crib": f"{i + 1:02d}", "clips": [parse_clip(c)]} for i, c in enumerate(args.clip)]
        speedup, files = {}, {}
    else:
        cfg = json.loads(Path(args.playlist).read_text(encoding="utf-8"))
        spec, speedup, files = cfg["cribs"], cfg.get("speedup", {}), cfg.get("files", {})
    if args.original:
        files = {}
    out = []
    for c in spec:
        if args.cribs and c["crib"] not in args.cribs:
            continue
        segs = [Segment(ann, k["video"], k.get("start", 0.0), k.get("end"), k.get("speedup", speedup.get(k["video"])),
                        k.get("view"), first=i == 0, file=files.get(k["video"])) for i, k in enumerate(c["clips"])]
        out.append((c["crib"], segs, c.get("note", "")))
    if not out:
        raise SystemExit("沒有床可以播（--cribs 的床號不在播放清單裡？）")
    if len(out) > COLS * ROWS:
        raise SystemExit(f"牆面最多 {COLS * ROWS} 床，播放清單有 {len(out)} 床")
    return out


# ── 時鐘 ──

class ShowClock:
    """展演的時鐘：牆上時鐘扣掉暫停的時間（Liveness 也用這個，暫停時不會變紫燈）。play_time() 是這一輪播了幾秒。"""

    def __init__(self):
        self.t0, self.paused_at, self.loop_t0 = time.monotonic(), None, 0.0

    def now(self):
        return (self.paused_at if self.paused_at is not None else time.monotonic()) - self.t0

    @property
    def paused(self):
        return self.paused_at is not None

    def pause(self):
        if self.paused_at is None:
            self.paused_at = time.monotonic()

    def resume(self):
        if self.paused_at is not None:
            self.t0 += time.monotonic() - self.paused_at
            self.paused_at = None

    def play_time(self):
        return self.now() - self.loop_t0

    def new_loop(self):
        self.loop_t0 = self.now()


# ── 每床：解碼 ──

class Feed(threading.Thread):
    """一床的鏡頭：照真實速度解碼，隨時留著最新一幀（給判讀）和縮好的格子畫面（給牆面）。
    落後時跳過幾幀（只解碼不轉換），跟真的鏡頭一樣不會越拖越慢。播完（或影片比片段先結束）就停在最後一幀，
    每 HOLD_S 秒把同一幀再交出去一次，判讀繼續跑（真的鏡頭拍到不動的床也是一直有畫面）。"""

    HOLD_S = 0.2

    def __init__(self, crib, segs, clock, tile):
        super().__init__(daemon=True, name=f"feed-{crib}")
        self.crib, self.segs, self.clock, self.tile = crib, segs, clock, tile
        self.offsets = list(np.cumsum([0.0] + [s.real for s in segs[:-1]]))
        self.duration = sum(s.real for s in segs)
        self.geoms = [s.geometry(*tile) for s in segs]
        self.lock = threading.Lock()
        self.epoch, self.seq, self.latest, self.thumb, self.thumb_seg = 0, 0, None, None, 0
        self.stopped = False
        self.decoded = self.skipped = 0
        self.caps = {}

    def restart(self):
        """從頭播（主執行緒呼叫）：作廢目前的幀，讀到一半的那一幀也不收。"""
        with self.lock:
            self.epoch += 1
            self.latest = None

    def snapshot(self):
        with self.lock:
            return self.latest

    def view(self):
        with self.lock:
            return self.thumb, self.thumb_seg

    def locate(self, pt):
        """這一輪的第 pt 秒（真實時間）→ (第幾段, 影片時間)；播完了回傳 (None, None)。"""
        for i, (off, seg) in enumerate(zip(self.offsets, self.segs)):
            if pt < off + seg.real:
                return i, seg.start + (pt - off) * seg.speed
        return None, None

    def cap(self, seg):
        key = str(seg.path)
        if key not in self.caps:
            self.caps[key] = cv2.VideoCapture(key)
        return self.caps[key]

    def make_thumb(self, frame, i):
        x0, y0, x1, y1, _, ox, oy, w, h = self.geoms[i]
        tw, th = self.tile
        thumb = np.zeros((th, tw, 3), np.uint8)
        thumb[oy:oy + h, ox:ox + w] = cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
        return thumb

    def publish(self, epoch, pt, i, frame, thumb):
        with self.lock:
            if epoch != self.epoch:   # 讀這一幀的時候按了重播：不收
                return
            self.seq += 1
            self.latest = {"seq": self.seq, "epoch": epoch, "pt": pt, "seg": i, "frame": frame}
            if thumb is not None:
                self.thumb, self.thumb_seg = thumb, i

    def run(self):
        epoch, cur, cap, done, held_at = None, None, None, None, -1e9
        while not self.stopped:
            if self.epoch != epoch:
                epoch, cur, done = self.epoch, None, None
            if self.clock.paused:
                time.sleep(0.02)
                continue
            pt = self.clock.play_time()
            i, src_t = self.locate(pt)
            if i is None or i == done:   # 播完、或影片比片段先結束：停在最後一幀，繼續交給判讀
                last = self.snapshot()
                if last is not None and pt - held_at >= self.HOLD_S:
                    held_at = pt
                    self.publish(epoch, pt, last["seg"], last["frame"], None)
                time.sleep(0.02)
                continue
            seg = self.segs[i]
            if i != cur:   # 換段（或從頭播）：跳到這一段的開頭
                cap, cur = self.cap(seg), i
                cap.set(cv2.CAP_PROP_POS_MSEC, seg.start * 1000)
            t_next = cap.get(cv2.CAP_PROP_POS_FRAMES) / seg.fps   # 下一幀的影片時間
            if t_next > src_t + 0.5 / seg.fps:   # 還沒到它的時間
                time.sleep(min(0.02, (t_next - src_t) / seg.speed))
                continue
            while t_next + 1 / seg.fps <= src_t:   # 落後超過一幀：跳過
                if not cap.grab():
                    break
                t_next += 1 / seg.fps
                self.skipped += 1
            ok, frame = cap.read()
            if not ok or t_next >= seg.end:
                done = i
                continue
            self.decoded += 1
            fpt = self.offsets[i] + (t_next - seg.start) / seg.speed   # 這一幀在這一輪的真實時間
            self.publish(epoch, fpt, i, frame, self.make_thumb(frame, i))
            held_at = fpt
        for c in self.caps.values():
            c.release()


# ── 每床：判讀 ──

class Analyzer(threading.Thread):
    """一床的即時判讀：拿 Feed 當下最新的一幀，跑 crib_detect.detect（跟 live_view、評估一樣），再更新床位狀態機。
    每床自己一個 YuNet（OpenCV 的偵測器不能多條執行緒共用）；RTMO、口鼻模型、RetinaFace 共用（晶片版透過
    hailo/chip_process 排隊、一次一張）。燈號變了通知 Wall（事件列表、推播）。"""

    def __init__(self, crib, feed, models, args, wall):
        super().__init__(daemon=True, name=f"analyze-{crib}")
        self.crib, self.feed, self.args, self.wall = crib, feed, args, wall
        self.run_pose, self.mouth, self.retina = models
        self.det = create_face_detector(FACE_SCORE_TH)
        self.det_v5 = create_face_detector(infant_face_v5.FACE_TH_V5)
        self.lock = threading.Lock()
        self.stopped = False
        self.count = self.errors = self.resumes = self.alerts = self.lapses = 0
        self.max_gap, self.last_done, self.offline_s = 0.0, None, 0.0
        self.max_gap_min = 0.0   # 這一分鐘裡最長多久沒判讀（每分鐘的紀錄讀完歸零）
        self.ms_hist = deque(maxlen=10)
        self.done = deque(maxlen=12)   # 最近幾次判讀完成的時間（展演時鐘），算這床每秒判讀幾幀
        self.offline, self.offline_since = False, None
        self.reset()

    def reset(self):
        """從頭播：狀態機、床位特徵、心跳全部重來（主執行緒呼叫，要在 feed.restart() 之後）。"""
        with self.lock:
            self.epoch = self.feed.epoch
            self.ctx, self.feat, self.feat_key, self.seg = crib_state.CribState(), None, None, None
            self.live = crib_state.Liveness(clock=self.wall.clock.now)
            self.last_seq, self.last_pt = None, None
            self.show, self.in_alert, self.count_s = None, False, 0.0
            self.boxes, self.box_seg = [], None

    def warmup(self, frame, seg):
        """開始前先判讀一次（模型、晶片第一次跑比較慢），結果不用。"""
        crib_detect.detect(self.run_pose, self.det, self.det_v5, frame, {seg.doll: seg.rect}, {seg.doll: seg.rot},
                           mouth=self.mouth, retina=self.retina)

    def rate(self):
        d = self.done
        return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 2 and d[-1] > d[0] else None

    def run(self):
        interval = 1 / self.args.fps
        while not self.stopped:
            snap = self.feed.snapshot()
            with self.lock:
                epoch, last_seq, last_pt = self.epoch, self.last_seq, self.last_pt
            if (snap is None or snap["epoch"] != epoch or snap["seq"] == last_seq
                    or (last_pt is not None and snap["pt"] - last_pt < interval - 0.02)):
                time.sleep(0.005)
                continue
            self.last_seq = snap["seq"]
            seg, frame, pt = self.feed.segs[snap["seg"]], snap["frame"], snap["pt"]
            tic = time.perf_counter()
            try:
                self.wall.inject_error(self.crib)
                result, _, faces = crib_detect.detect(self.run_pose, self.det, self.det_v5, frame, {seg.doll: seg.rect},
                                                      {seg.doll: seg.rot}, mouth=self.mouth, retina=self.retina)
            except Exception as e:   # 晶片重開後還是失敗等：這床離線，等下一幀再試；狀態機不動
                self.live.fail(e)
                self.errors += 1
                if self.errors <= 3 or self.errors % 50 == 0:
                    print(f"[判讀出錯 {self.errors}] Crib {self.crib}：{type(e).__name__}: {str(e)[:100]}", flush=True)
                time.sleep(interval)   # 一直出錯時不要每一幀都重試
                continue
            ms = (time.perf_counter() - tic) * 1000
            with self.lock:
                if epoch != self.epoch:   # 判讀途中按了重播：這一幀作廢
                    continue
                hidden = result[seg.doll][self.args.mode] == "hidden"
                if self.feat is None or snap["seg"] != self.seg:
                    # 開始或剪接點：剪接前後兩幀不拿來比「床邊有沒有動」；換了床位框、空床照就重新開始算
                    if self.feat is None or seg.feat_key() != self.feat_key:
                        self.feat, self.feat_key = crib_state.EdgeFeatures(seg.rect, seg.ref), seg.feat_key()
                    else:
                        self.feat.reset()
                    self.seg = snap["seg"]
                elif self.last_pt is not None and pt - self.last_pt > crib_state.OFFLINE_AFTER:
                    # 跟上一次判讀的幀隔太久（中間離線）：同 live_view，斷線前的移動紀錄作廢、離線那段當作看不到
                    self.feat.reset()
                    self.ctx.resume(self.last_pt)
                    self.resumes += 1
                self.last_pt = pt
                motion, diff = self.feat.update(frame)
                state = self.ctx.update(pt, hidden, motion, diff)
                show = demo_render.display_state(self.ctx, state, hidden)
                since = self.ctx.alarm.since_hidden
                self.count_s = pt - since if since is not None else 0.0
                self.boxes, self.box_seg = dedupe_boxes([tuple(f[:4]) for f in faces]), snap["seg"]
                prev, self.show = self.show, show
                new_alert = state == "alert" and not self.in_alert
                self.in_alert = state == "alert"
                self.alerts += new_alert
                self.ms_hist.append(ms)
                t = self.wall.clock.now()
                if self.last_done is not None:
                    self.max_gap = max(self.max_gap, t - self.last_done)
                    self.max_gap_min = max(self.max_gap_min, t - self.last_done)
                self.last_done = t
                self.done.append(t)
                self.count += 1
            self.live.ok()   # 新燈號放好之後才算恢復
            if show != prev:
                self.wall.on_show(self, prev, show, new_alert)


# ── 牆面 ──

def dedupe_boxes(boxes, iou=0.5):
    """同一張臉常被找到好幾次（頭部裁切、床位範圍、RetinaFace），畫面上只畫一個框：重疊超過 iou 的只留第一個。"""
    keep = []
    for b in boxes:
        for k in keep:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            union = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if union > 0 and inter / union > iou:
                break
        else:
            keep.append(b)
    return keep


def pill(text, bg, size, fg=(255, 255, 255), bold=True, pad=(0.6, 0.28), radius=0.45):
    """圓角標籤（PIL 畫好轉成 BGR＋透明度，貼到 OpenCV 畫面上）。"""
    f = font(max(10, int(size)), bold)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=f)
    px, py = int(size * pad[0]), int(size * pad[1])
    w, h = r - l + 2 * px, int(size * 1.0) + 2 * py
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(h * radius), fill=tuple(bg) + (255,) if len(bg) == 3 else tuple(bg))
    d.text((px - l, (h - (b - t)) // 2 - t), text, font=f, fill=fg)
    a = np.asarray(img)
    return cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR), a[:, :, 3:4].astype(np.float32) / 255


def blit(canvas, patch, x, y):
    img, alpha = patch
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    canvas[y0:y1, x0:x1] = (img[y0 - y:y1 - y, x0 - x:x1 - x] * a + region * (1 - a)).astype(np.uint8)


def dashed_rect(img, p0, p1, color, thick, dash):
    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thick)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thick)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thick)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thick)


class Wall:
    """牆面：畫面版面、事件列表、推播、故障測試、統計。主執行緒用；Analyzer 透過 on_show 通知燈號變化。"""

    def __init__(self, args, size, n):
        self.args, (self.W, self.H) = args, size
        self.k = k = self.W / 1920
        self.clock = ShowClock()
        self.lock = threading.Lock()
        self.events, self.version = deque(maxlen=60), 0
        self.notifier = None
        self.top, self.side = int(76 * k), int(400 * k)
        pad = self.pad = int(14 * k)
        gw, gh = self.W - self.side - 2 * pad, self.H - self.top - 2 * pad
        self.tw, self.th = (gw - (COLS - 1) * pad) // COLS, (gh - (ROWS - 1) * pad) // ROWS
        self.tiles = [(pad + (i % COLS) * (self.tw + pad), self.top + pad + (i // COLS) * (self.th + pad))
                      for i in range(COLS * ROWS)]
        self.n = n
        self.cribs = []   # [(Feed, Analyzer)]
        self._pills, self._base, self._base_key = {}, None, None
        self.faults = parse_faults(args.fault)
        self.error_until = {}   # 床號或 "*" → 展演時鐘幾秒前判讀都出錯
        self.pending_off, self.pending_off_at = [], None
        self.frames = 0
        self.event_log = None   # main() 設：每個事件另外附加一行到這個 .jsonl（長時間測試事後查哪一分鐘斷線）

    # 事件、推播
    def add_event(self, kind, crib, text, push=None):
        ev = {"t": now_str(), "kind": kind, "crib": crib, "text": text, "push": push}
        with self.lock:
            self.events.appendleft(ev)
            self.version += 1
            if self.event_log is not None:
                with open(self.event_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="milliseconds"),
                                         "show_s": round(self.clock.now(), 2), "kind": kind, "crib": crib,
                                         "text": text}, ensure_ascii=False) + "\n")
        print(f"[{ev['t']}] " + (f"Crib {crib} · " if crib else "") + f"{kind}：{text}", flush=True)
        return ev

    def on_show(self, an, prev, show, new_alert):
        """Analyzer 的執行緒呼叫：燈號變了。紅、藍、灰記成事件（開頭的灰也記：開機時就是空床）；新的紅燈推播。"""
        if show not in EVENT_TEXT or (prev is None and show != "empty"):
            return
        ev = self.add_event(show, an.crib, EVENT_TEXT[show])
        if show == "red" and new_alert and self.notifier is not None and self.notifier.enabled:
            if self.notifier.send(ntfy_push.crib_title(an.crib), ntfy_push.ALERT_TEXT, event=ev, red_at=time.monotonic()):
                with self.lock:
                    ev["push"] = "sending"
                    self.version += 1

    def on_push_result(self, item, ok, dt, info):
        ev = item.get("event")
        if ev is None:
            return
        with self.lock:
            ev["push"] = f"phone alert sent ({dt:.1f} s)" if ok else f"phone alert failed ({info[:40]})"
            ev["push_ok"] = ok
            self.version += 1

    def poll_offline(self):
        """主迴圈每一幀呼叫：每床的心跳 → 離線（紫燈）的開始和結束記成事件；--push-offline 時合成一則推播。"""
        for _, an in self.cribs:
            off = an.live.offline()
            if off == an.offline:
                continue
            an.offline = off
            if off:
                an.lapses += 1
                an.offline_since = self.clock.now()
                if an.live.error:   # 判讀出錯是馬上離線，不是等 2 秒
                    text = f"analysis failing: check in person ({ascii_only(an.live.error)[:50]})"
                else:
                    text = EVENT_TEXT["offline"]
                self.add_event("offline", an.crib, text)
                if self.args.push_offline and self.notifier is not None and self.notifier.enabled:
                    self.pending_off.append(an.crib)
                    self.pending_off_at = self.pending_off_at or time.monotonic()
            else:
                dur = self.clock.now() - an.offline_since
                an.offline_s += dur
                self.add_event("online", an.crib, f"{EVENT_TEXT['online']} after {dur:.1f} s")
        if self.pending_off and time.monotonic() - self.pending_off_at >= OFFLINE_GROUP_S:
            cribs, self.pending_off, self.pending_off_at = self.pending_off, [], None
            if len(cribs) == 1:
                title, where = ntfy_push.crib_title(cribs[0]), "this crib"
            else:
                title, where = f"{PRODUCT} · {len(cribs)} cribs", "the cribs"
            self.notifier.send(title, f"Not being analysed right now. Check {where} in person.")

    # 故障測試
    def inject_error(self, crib):
        now = self.clock.now()
        if now < self.error_until.get(crib, -1) or now < self.error_until.get("*", -1):
            raise RuntimeError("simulated analysis error (--fault error)")

    def run_faults(self):
        now = self.clock.now()
        while self.faults and now >= self.faults[0][0]:
            at, kind, crib = self.faults.pop(0)
            print(f"[故障測試] 展演第 {now:.1f} 秒：{kind}" + (f"（Crib {crib}）" if crib else ""), flush=True)
            if kind == "error":
                self.error_until[crib or "*"] = at + ERROR_S
                continue
            import hailo.chip_process as cp
            if cp._proc is None:
                print("[故障測試] 沒有晶片子程序（要 --chip 或設 FG_CHIP_PROCESS=1、FG_RTMO_HEF），略過", flush=True)
            elif kind == "kill":
                cp._proc.kill()
            else:
                import psutil
                psutil.Process(cp._proc.pid).suspend()

    def restart(self, reason, clear=False):
        """從頭播：所有床的片段、狀態機、推播紀錄（哪床已經推過）重來。clear＝連事件列表也清掉（按 r 時；
        自動播完一輪不清，事件列表像真的護理站一樣一直往下記）。"""
        self.clock.new_loop()
        for feed, an in self.cribs:
            feed.restart()
            an.reset()
        if clear:
            with self.lock:
                self.events.clear()
                self.version += 1
        print(f"[{now_str()}] 從頭播（{reason}）", flush=True)

    def simulate_offline(self):
        """o 鍵：現場示範紫燈。晶片在子程序時把它暫停（等於晶片卡住：所有床約 13.5 秒沒判讀，主程序逾時後砍掉重開）；
        沒有晶片子程序時改成所有床判讀出錯 ERROR_S 秒。"""
        import hailo.chip_process as cp
        if cp._proc is not None:
            self.faults.insert(0, (self.clock.now(), "hang", None))
        else:
            self.faults.insert(0, (self.clock.now(), "error", None))

    # 統計
    def cribs_text(self):
        return f"{self.n} crib" + ("s" if self.n != 1 else "")

    def total_rate(self):
        rates = [an.rate() for _, an in self.cribs]
        rates = [r for r in rates if r is not None]
        return sum(rates) if rates else None

    def counts(self):
        out = dict.fromkeys(COUNT_ORDER, 0)
        for _, an in self.cribs:
            show = "offline" if an.offline else an.show
            if show is not None:
                out[STATES[show][1]] += 1
        return out

    # 畫面
    def pill(self, key, text, bg, size, **kw):
        if key not in self._pills:
            self._pills[key] = pill(text, bg, size, **kw)
        return self._pills[key]

    def draw_base(self, counts, rate):
        """上方列＋右邊欄（PIL）：每秒重畫一次，事件或燈號統計變了也重畫。"""
        k, W, H = self.k, self.W, self.H
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        # 上方列
        d.rectangle((0, 0, W, self.top), fill=(255, 255, 255))
        d.line((0, self.top, W, self.top), fill=LINE, width=max(1, int(k)))
        s = int(15 * k)
        for i, c in enumerate(("green", "count", "red", "care")):
            x, y = int(24 * k) + (i % 2) * (s + 2), int(21 * k) + (i // 2) * (s + 2)
            d.rounded_rectangle((x, y, x + s, y + s), radius=int(4 * k), fill=STATES[c][0])
        x = int(24 * k) + 2 * s + int(16 * k)
        d.text((x, int(18 * k)), PRODUCT, font=font(int(30 * k), True), fill=BRAND)
        x += d.textlength(PRODUCT, font=font(int(30 * k), True)) + int(16 * k)
        d.text((x, int(24 * k)), "Nurse station", font=font(int(24 * k)), fill=MUTED)
        x += d.textlength("Nurse station", font=font(int(24 * k))) + int(22 * k)
        live = "LIVE · computed now, not pre-rendered"
        lw = d.textlength(live, font=font(int(17 * k), True))
        d.rounded_rectangle((x, int(20 * k), x + lw + int(44 * k), int(56 * k)), radius=int(10 * k), fill=(30, 32, 36))
        d.ellipse((x + int(12 * k), int(31 * k), x + int(26 * k), int(45 * k)), fill=STATES["red"][0])
        d.text((x + int(34 * k), int(27 * k)), live, font=font(int(17 * k), True), fill=(255, 255, 255))
        left = x + lw + int(70 * k)
        clock = now_str()
        cf = font(int(30 * k), True)
        cw = d.textlength(clock, font=cf)
        d.text((W - int(24 * k) - cw, int(20 * k)), clock, font=cf, fill=INK)
        right = W - int(24 * k) - cw - int(24 * k)
        per = rate / self.n if rate else None
        line1 = (f"{self.cribs_text()} · analysis {rate:.1f} frames/s in total · {per:.1f} per crib" if rate
                 else f"{self.cribs_text()} · starting analysis…")
        line2 = f"Running on: {self.args.device}"
        for i, (text, bold) in enumerate(((line1, True), (line2, False))):
            f, lines = fit_lines(d, text, 18 * k, right - left, bold, max_lines=1, min_ratio=0.7)
            d.text((right - d.textlength(lines[0], font=f), int((14 + i * 26) * k)), lines[0], font=f,
                   fill=INK if bold else MUTED)
        # 右邊欄
        x0, x1 = W - self.side, W
        d.rectangle((x0, self.top, x1, H), fill=(255, 255, 255))
        d.line((x0, self.top, x0, H), fill=LINE, width=max(1, int(k)))
        x0, x1 = x0 + int(24 * k), x1 - int(24 * k)
        y = self.top + int(20 * k)
        d.text((x0, y), "Nurse station", font=font(int(30 * k), True), fill=BRAND)
        y += int(44 * k)
        y += draw_fit(d, (x0, y), f"{self.cribs_text()} · AI runs on the box, no video stored", 16 * k, x1 - x0, MUTED)
        y += int(12 * k)
        gap = int(6 * k)
        bw = (x1 - x0 - gap * (len(COUNT_ORDER) - 1)) / len(COUNT_ORDER)
        colors = {STATES[s][1]: STATES[s][0] for s in ("green", "count", "red", "care", "empty", "offline")}
        for i, title in enumerate(COUNT_ORDER):
            bx = int(x0 + i * (bw + gap))
            d.rounded_rectangle((bx, y, int(bx + bw), y + int(62 * k)), radius=int(10 * k), fill=colors[title])
            num = str(counts[title])
            nf = font(int(26 * k), True)
            d.text((bx + (bw - d.textlength(num, font=nf)) / 2, y + int(6 * k)), num, font=nf, fill=(255, 255, 255))
            lf = fit_lines(d, title, 10 * k, bw - 4, True, max_lines=1, min_ratio=0.8)[0]
            d.text((bx + (bw - d.textlength(title, font=lf)) / 2, y + int(40 * k)), title, font=lf, fill=(255, 255, 255))
        y += int(86 * k)
        d.text((x0, y), "EVENTS", font=font(int(15 * k), True), fill=BRAND)
        y += int(28 * k)
        foot = int(H - 112 * k)
        with self.lock:
            events = list(self.events)
        for ev in events:
            lines = 2 + bool(ev["push"])
            if y + int((lines * 22 + 16) * k) > foot:
                break
            color = STATES["green"][0] if ev["kind"] == "online" else STATES[{"red": "red", "care": "care", "empty": "empty",
                                                                            "offline": "offline"}[ev["kind"]]][0]
            d.ellipse((x0, y + int(5 * k), x0 + int(14 * k), y + int(19 * k)), fill=color)
            tx = x0 + int(26 * k)
            title = f"Crib {ev['crib']} · " + ("BACK ONLINE" if ev["kind"] == "online" else STATES[ev["kind"]][1])
            d.text((tx, y), title, font=font(int(17 * k), True), fill=INK)
            draw_fit(d, (tx, y + int(23 * k)), f"{ev['t']} · {ev['text']}", 14 * k, x1 - tx, MUTED, max_lines=1)
            if ev["push"]:
                c = STATES["green"][0] if ev.get("push_ok") else (STATES["red"][0] if "failed" in ev["push"] else MUTED)
                draw_fit(d, (tx, y + int(43 * k)), ev["push"], 14 * k, x1 - tx, c, bold=True, max_lines=1)
            y += int((lines * 21 + 14) * k)
            d.line((x0, y - int(7 * k), x1, y - int(7 * k)), fill=LINE, width=1)
        push = self.notifier.describe() if self.notifier is not None else "off (--no-push)"
        fy = foot + int(10 * k)
        fy += draw_fit(d, (x0, fy), f"Phone alerts: {push}", 14 * k, x1 - x0, INK, bold=True, max_lines=1)
        draw_fit(d, (x0, fy + int(4 * k)), "Doll re-enactment · recorded per-crib camera clips at real speed · "
                 "every light is computed live", 14 * k, x1 - x0, MUTED, max_lines=3, wrap_first=True)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def render(self):
        counts, rate = self.counts(), self.total_rate()
        key = (int(time.time()), self.version, tuple(counts.values()))
        if key != self._base_key:
            self._base_key, self._base = key, self.draw_base(counts, rate)
        canvas = self._base.copy()
        k, tw, th = self.k, self.tw, self.th
        bw = max(4, int(8 * k))
        for i, (x, y) in enumerate(self.tiles):
            if i >= len(self.cribs):   # 沒有用到的格子（開發時只跑幾床）
                cv2.rectangle(canvas, (x, y), (x + tw - 1, y + th - 1), (225, 222, 218), -1)
                continue
            feed, an = self.cribs[i]
            thumb, tseg = feed.view()
            tile = canvas[y:y + th, x:x + tw]
            if thumb is not None:
                tile[:] = thumb
            else:
                tile[:] = (17, 17, 17)
            show = "offline" if an.offline else an.show
            if show is not None and show != "offline" and an.box_seg == tseg:   # 臉框（最近一次判讀的；離線時不畫）
                gx0, gy0, _, _, s, ox, oy, _, _ = feed.geoms[tseg]
                for bx0, by0, bx1, by1 in an.boxes:
                    p0 = (int(x + ox + (bx0 - gx0) * s), int(y + oy + (by0 - gy0) * s))
                    p1 = (int(x + ox + (bx1 - gx0) * s), int(y + oy + (by1 - gy0) * s))
                    cv2.rectangle(canvas, p0, p1, (255, 255, 255), max(1, int(2 * k)))
            if show is None:
                color = (110, 110, 110)
            else:
                color = STATES[show][0][::-1]
                if show == "red" and int(time.monotonic() * 2) % 2:   # 紅燈外框閃
                    color = (128, 138, 255)
            p0, p1 = (x + bw // 2, y + bw // 2), (x + tw - 1 - bw // 2, y + th - 1 - bw // 2)
            if show == "offline":   # 離線：虛線框（燈號不明，畫面照常）
                cv2.rectangle(canvas, (x, y), (x + tw - 1, y + th - 1), (40, 40, 40), bw)
                dashed_rect(canvas, p0, p1, color, bw, int(22 * k))
            else:
                cv2.rectangle(canvas, p0, p1, color, bw)
            label = f"{an.crib}  {STATES[show][1]}" if show is not None else f"{an.crib}  …"
            bg = STATES[show][0] if show is not None else (110, 110, 110)
            blit(canvas, self.pill(("tag", label, bg), label, bg, 20 * k), x + int(16 * k), y + int(16 * k))
            if show == "count":   # 3 秒倒數條
                bx0, bx1, by = x + int(18 * k), x + tw - int(18 * k), y + th - int(30 * k)
                cv2.rectangle(canvas, (bx0, by), (bx1, by + int(10 * k)), (235, 235, 235), -1)
                cv2.rectangle(canvas, (bx0, by), (bx0 + int((bx1 - bx0) * min(an.count_s / 3, 1)), by + int(10 * k)),
                              color, -1)
            if show == "offline":
                silent = an.live.silent_for()
                msg = f"OFFLINE · no analysis {silent:.0f} s"
                p = self.pill(("off", msg), msg, STATES["offline"][0], 18 * k)
                blit(canvas, p, x + (tw - p[0].shape[1]) // 2, y + (th - p[0].shape[0]) // 2)
            r = an.rate()
            if r is not None:
                msg = f"{r:.1f} frames/s"
                p = self.pill(("rate", msg), msg, (20, 20, 20, 170), 13 * k, bold=False)
                blit(canvas, p, x + tw - p[0].shape[1] - int(14 * k), y + th - p[0].shape[0] - int(44 * k))
        if all(an.offline for _, an in self.cribs):   # 全部離線（晶片斷線、卡住）：牆面上方一條
            silent = min(an.live.silent_for() for _, an in self.cribs)
            msg = f"OFFLINE · all {len(self.cribs)} cribs · no analysis for {silent:.0f} s · check the cribs in person"
            p = self.pill(("banner", msg), msg, STATES["offline"][0], 22 * k)
            blit(canvas, p, (self.W - self.side - p[0].shape[1]) // 2, self.top + int(26 * k))
        if len(self._pills) > 400:   # 秒數一直變的標籤：別讓快取無限長大
            self._pills.clear()
        return canvas

    def summary(self, started, wall_s):
        import hailo.chip_process as cp
        n = self.notifier
        return {
            "started": started, "seconds": round(wall_s, 1), "show_seconds": round(self.clock.now(), 1),
            "device": self.args.device, "mode": self.args.mode, "fps_target": self.args.fps,
            "display_fps": round(self.frames / wall_s, 1) if wall_s else None,
            "chip_restarts": cp.restarts,
            "push": {"enabled": bool(n and n.enabled), "sent": n.sent if n else 0, "failed": n.failed if n else 0,
                     "dropped": n.dropped if n else 0,
                     "median_s": round(float(np.median(n.latencies)), 2) if n and n.latencies else None},
            "cribs": [{"crib": an.crib, "clips": [f"{s.video} {s.start:g}-{s.end:g}" for s in feed.segs],
                       "analyses": an.count, "per_s": round(an.count / self.clock.now(), 2) if self.clock.now() else None,
                       "ms_recent": round(float(np.mean(an.ms_hist)), 0) if an.ms_hist else None,
                       "max_gap_s": round(an.max_gap, 2), "offline_lapses": an.lapses,
                       "offline_s": round(an.offline_s + (self.clock.now() - an.offline_since if an.offline else 0), 1),
                       "errors": an.errors, "resumes": an.resumes, "red_alerts": an.alerts,
                       "decoded": feed.decoded, "skipped": feed.skipped} for feed, an in self.cribs],
        }


class MinuteLog:
    """每分鐘一列的時間序列（長時間測試用，跟主 session 的 soak 測速對照）：這一分鐘每床判讀幾次、每床最長多久沒判讀、
    到目前為止的離線次數、晶片重開次數、這一分鐘畫面重畫幾幀、CPU 使用率。"""

    def __init__(self, path, wall):
        import psutil
        self.path, self.wall, self.psutil = path, wall, psutil
        self.prev_counts, self.prev_frames = None, 0
        psutil.cpu_percent(None)   # 第一次呼叫只是起算點
        cribs = [an.crib for _, an in wall.cribs]
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(",".join(["time", "show_min"] + [f"n_{c}" for c in cribs] + [f"gap_{c}" for c in cribs]
                              + ["offline_lapses", "chip_restarts", "display_frames", "cpu_pct"]) + "\n")

    def write(self):
        import hailo.chip_process as cp
        w = self.wall
        counts = [an.count for _, an in w.cribs]
        prev = self.prev_counts or [0] * len(counts)
        gaps = []
        for _, an in w.cribs:
            gaps.append(round(an.max_gap_min, 2))
            an.max_gap_min = 0.0
        row = ([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{w.clock.now() / 60:.1f}"]
               + [str(c - p) for c, p in zip(counts, prev)] + [str(g) for g in gaps]
               + [str(sum(an.lapses for _, an in w.cribs)), str(cp.restarts), str(w.frames - self.prev_frames),
                  f"{self.psutil.cpu_percent(None):.0f}"])
        self.prev_counts, self.prev_frames = counts, w.frames
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            fh.write(",".join(row) + "\n")


def parse_faults(specs):
    """--fault 種類@秒[:床號] → [(秒, 種類, 床號或 None)]，照時間排好。秒是展演開始後幾秒（扣掉暫停）。"""
    out = []
    for s in specs or []:
        kind, _, rest = s.partition("@")
        sec, _, crib = rest.partition(":")
        if kind not in FAULTS or not sec:
            raise SystemExit(f"--fault {s}：要寫成 種類@秒（error 可以加 :床號），種類是 {'／'.join(FAULTS)}")
        out.append((float(sec), kind, crib or None))
    return sorted(out, key=lambda f: f[0])


def main():
    ap = argparse.ArgumentParser(description="決賽展演：8 床護理站牆面即時判讀＋紅燈推播")
    ap.add_argument("--playlist", default=str(PLAYLIST), help="播放清單（預設 demo/wall_playlist.json）")
    ap.add_argument("--clip", nargs="*", metavar="影片:開始:結束", help="不用播放清單，直接給每床一段（給幾個就幾床）")
    ap.add_argument("--cribs", nargs="*", help="只跑播放清單裡這幾床（床號，例如 01 02）")
    ap.add_argument("--original", action="store_true", help="不用播放清單 files 指的轉檔版，播原檔（對照用）")
    ap.add_argument("--fps", type=float, default=FPS, help=f"每床每秒最多判讀幾幀（預設 {FPS}；判讀跟不上時會更少）")
    ap.add_argument("--det-th", type=float, default=0.15, help="RTMO 找人門檻（預設 0.15，跟 eval_video.py 一樣）")
    ap.add_argument("--chip", action="store_true", help="決賽展演的晶片設定（live_view.CHIP_FINALS）")
    ap.add_argument("--device", help="上方「Running on」的文字（預設照環境變數寫哪些模型在 UGen300）")
    ap.add_argument("--mode", default=crib_detect.MAIN_MODE, choices=crib_detect.MODES_V7[1:],
                    help=f"燈號依哪一種判定（預設主流程 {crib_detect.MAIN_MODE}）")
    ap.add_argument("--width", type=int, default=1920, help="畫面寬度（高度＝寬度×9/16）")
    ap.add_argument("--fullscreen", action="store_true", help="一開始就全螢幕（f 鍵切換）")
    ap.add_argument("--no-window", action="store_true", help="不開視窗（長時間測試用）")
    ap.add_argument("--save", help="同時把牆面存成影片（.mp4；會多吃 CPU，備用影片才開）")
    ap.add_argument("--snapshot", help="結束時把最後一個畫面存成圖（.png；檢查版面用）")
    ap.add_argument("--once", action="store_true", help="只播一輪（預設播完全部從頭再播）")
    ap.add_argument("--duration", type=float, help="跑幾秒就結束（預設不限）")
    ap.add_argument("--log-every", type=float, default=60, help="每幾秒印一行、存一次紀錄（預設 60）")
    ap.add_argument("--no-push", action="store_true", help="不推播（就算設了 FG_NTFY_TOPIC）")
    ap.add_argument("--push-offline", action="store_true", help="離線（紫燈）也推播（預設不推；同時離線的床合成一則）")
    ap.add_argument("--fault", nargs="*", metavar="種類@秒[:床號]",
                    help="測紫燈：展演第幾秒製造故障。error＝判讀出錯 5 秒（加 :床號 只有那床）；"
                         "kill＝砍掉晶片子程序；hang＝暫停晶片子程序（後兩個要 --chip）")
    args = ap.parse_args()
    if args.chip:   # 要在載入模型之前設；殘留的 YuNet HEF、合批設定拿掉（跟 live_view 一樣）
        for k in ("FG_YUNET_HEF", "FG_CHIP_BATCH"):
            os.environ.pop(k, None)
        os.environ.update(CHIP_FINALS)
    if args.device is None:
        on_chip = [n for n, env in (("RTMO", "FG_RTMO_HEF"), ("YuNet", "FG_YUNET_HEF"), ("mouth/nose", "FG_MOUTH_HEF"),
                                    ("RetinaFace", "FG_RETINA_HEF")) if os.environ.get(env)]
        args.device = f"ASUS UGen300 ({', '.join(on_chip)}) + laptop CPU" if on_chip else "laptop CPU (Intel i7-11800H)"
    print(f"Running on: {args.device}" + ("（晶片在子程序、一次一張）" if os.environ.get("FG_CHIP_PROCESS") == "1" else ""))

    cribs = load_cribs(args)
    show_len = max(sum(s.real for s in segs) for _, segs, _ in cribs)
    for crib, segs, note in cribs:
        parts = " → ".join(f"{s.video}{'（轉檔版）' if s.converted else ''} {s.start:g}–{s.end:g} 秒"
                           + (f"（放慢 {1 / s.speed:.2f} 倍）" if s.speed < 0.999 else "")
                           for s in segs)
        print(f"  Crib {crib}：{parts}，共 {sum(s.real for s in segs):.0f} 秒" + (f"｜{note}" if note else ""))
    print(f"一輪 {show_len:.0f} 秒" + ("（播完停止）" if args.once else "（播完從頭再播）"))

    print("載入模型（RTMO、YuNet" + ("、口鼻小塊模型" if args.mode in ("v6_slot", "v7_slot") else "")
          + ("、RetinaFace" if args.mode == "v7_slot" else "") + "）…")
    infant_face_v3.DET_TH = args.det_th
    run_pose = infant_face_v3.load_rtmo()
    mouth = infant_face_v6.load_mouth_model(required=True) if args.mode in ("v6_slot", "v7_slot") else None
    retina = infant_face_v7.load_retina(required=True) if args.mode == "v7_slot" else None

    W = args.width // 2 * 2
    H = W * 9 // 16 // 2 * 2
    wall = Wall(args, (W, H), len(cribs))
    for crib, segs, _ in cribs:
        feed = Feed(crib, segs, wall.clock, (wall.tw, wall.th))
        wall.cribs.append((feed, Analyzer(crib, feed, (run_pose, mouth, retina), args, wall)))
    print("暖機（每床先判讀一次）…")
    for (_, an), (_, segs, _) in zip(wall.cribs, cribs):
        an.warmup(segs[0].first, segs[0])
    if not args.no_push:
        wall.notifier = ntfy_push.Notifier(on_result=wall.on_push_result)
    print(f"推播：{wall.notifier.describe() if wall.notifier else 'off (--no-push)'}"
          + ("；離線也推" if args.push_offline else ""))

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"avc1"), DISPLAY_FPS, (W, H))
        if not writer.isOpened():
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), DISPLAY_FPS, (W, H))
    full = args.fullscreen
    if not args.no_window:
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, W, H)
        if full:
            cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    started = datetime.datetime.now()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"wall_{started:%Y%m%d_%H%M%S}.json"
    wall.event_log = log_path.with_name(log_path.stem + "_events.jsonl")
    minutes = MinuteLog(log_path.with_name(log_path.stem + "_minutes.csv"), wall)
    wall.clock = ShowClock()   # 暖機、開視窗的時間不算
    for feed, an in wall.cribs:
        feed.clock = wall.clock
        an.reset()
    for feed, an in wall.cribs:
        feed.start()
        an.start()
    print("開始（q／Esc 結束、空白鍵暫停、r 從頭重播、f 全螢幕、o 示範紫燈）")
    wall0, next_log, frame_dt = time.monotonic(), args.log_every, 1 / DISPLAY_FPS
    next_frame, canvas = time.monotonic(), None
    try:
        while True:
            if wall.clock.play_time() >= show_len and not wall.clock.paused:
                if args.once:
                    break
                wall.restart("播完一輪")
            if args.duration and wall.clock.now() >= args.duration:
                break
            wall.poll_offline()
            wall.run_faults()
            canvas = wall.render()
            wall.frames += 1
            if writer is not None:
                writer.write(canvas)
            next_frame += frame_dt
            wait = next_frame - time.monotonic()
            if wait < -0.5:   # 畫面落後太多（例如暫停、拖動視窗）：不追，從現在重新算
                next_frame, wait = time.monotonic(), 0
            key = -1
            if args.no_window:
                time.sleep(max(0.0, wait))
            else:
                cv2.imshow(TITLE, canvas)
                key = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if wall.clock.paused:
                    wall.clock.resume()
                    print(f"[{now_str()}] 繼續", flush=True)
                else:
                    wall.clock.pause()
                    print(f"[{now_str()}] 暫停（空白鍵繼續）", flush=True)
            if key == ord("r"):
                wall.clock.resume()
                wall.restart("按了 r", clear=True)
            if key == ord("o"):
                wall.simulate_offline()
            if key == ord("f") and not args.no_window:
                full = not full
                cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if wall.clock.now() >= next_log:   # 每分鐘一行＋存一次紀錄（長時間測試中途當掉也有資料）
                next_log += args.log_every
                s = wall.summary(f"{started:%Y-%m-%d %H:%M:%S}", time.monotonic() - wall0)
                log_path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
                minutes.write()
                per = " ".join(f"{c['crib']}:{c['per_s']}" for c in s["cribs"])
                print(f"[{now_str()}] 第 {wall.clock.now() / 60:.0f} 分：每床每秒判讀 {per}；最長沒判讀 "
                      f"{max(c['max_gap_s'] for c in s['cribs']):.1f} 秒；離線 {sum(c['offline_lapses'] for c in s['cribs'])} 次；"
                      f"晶片重開 {s['chip_restarts']} 次；畫面 {s['display_fps']} 幀/秒；推播 {s['push']['sent']}／失敗 {s['push']['failed']}",
                      flush=True)
    except KeyboardInterrupt:
        print("（Ctrl+C）")
    for feed, an in wall.cribs:
        feed.stopped = an.stopped = True
    if writer is not None:
        writer.release()
        print(f"→ {args.save}")
    cv2.destroyAllWindows()
    if args.snapshot and canvas is not None:   # cv2.imwrite 在 Windows 不吃中文路徑，改用 imencode
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Path(args.snapshot).write_bytes(cv2.imencode(".png", canvas)[1].tobytes())
        print(f"→ {args.snapshot}")
    s = wall.summary(f"{started:%Y-%m-%d %H:%M:%S}", time.monotonic() - wall0)
    log_path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"跑了 {s['seconds']:.0f} 秒（展演時鐘 {s['show_seconds']:.0f} 秒）；畫面 {s['display_fps']} 幀/秒；晶片重開 {s['chip_restarts']} 次；"
          f"推播送出 {s['push']['sent']}、失敗 {s['push']['failed']}" + (f"、中位數 {s['push']['median_s']} 秒" if s['push']['median_s'] else ""))
    for c in s["cribs"]:
        print(f"  Crib {c['crib']}：判讀 {c['analyses']} 次（每秒 {c['per_s']}，最近每次 {c['ms_recent']} ms），最長 {c['max_gap_s']} 秒沒判讀；"
              f"離線 {c['offline_lapses']} 次共 {c['offline_s']} 秒、出錯 {c['errors']} 次；紅燈 {c['red_alerts']} 次；"
              f"解碼 {c['decoded']} 幀、跳過 {c['skipped']} 幀")
    print(f"→ {log_path}")


if __name__ == "__main__":
    main()
