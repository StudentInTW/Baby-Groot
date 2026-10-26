"""
pipeline/crib_state.py
每張床的狀態機：crib_detect.py 每幀判「口鼻看不看得到」，這裡決定燈號。

AlertState：連續 on 秒看不到口鼻 → 警示；連續 off 秒看得到 → 解除（預設 3／2 秒）。

CribState（2026/10/10，原型）：每床鏡頭再加照護中暫停計時、空床不響。
每床鏡頭只拍一張床，床位模式「找不到臉就算看不到」→ 寶寶被抱去餵奶、拍嗝後，空床一定響（補拍影片 2/2）。
不讓護理師每次抱走都手動標記（太增加負擔），改成：
  1. 大人在床邊 → 照護中：不累積「看不到」的時間。判法：床位邊緣一圈有明顯移動（大人一定從畫面邊緣伸手進來），
     連續 3 幀有 2 幀以上就算，動作停了 CARE_HOLD 秒才算離開。先扣掉整個畫面的平移（鏡頭晃動）再比：
     手機拍的 IMG_8429 會晃，不扣的話沒有手的片段有 17% 幀被當成有人在動，扣掉後 5%（剩下的是有人在調整手機）。
     已經在響的警示不會因為有人在動而解除（移動也可能是寶寶在掙扎）；要嘛臉重新看得到，要嘛判成空床。
  2. 大人離開後 ARM 秒內，臉不見、而且畫面跟這張床的空床參考畫面相符（色差 < EMPTY_TH）連續 CONFIRM 秒 → 空床（灰燈）。
     不能只看「抓不到嬰兒」：空床上 RTMO 會在皺床單上找到人，被蓋住的寶寶又常常抓不到人，兩邊都會錯。
  3. 沒有大人來過就不可能變空床（寶寶不會自己離開）：看到臉之後就關掉空床判定，直到下一次有大人來。
     開機時還沒看過臉，允許直接判空床。
  不確定就走一般 3 秒計時——寧可多叫一次。

EdgeFeatures 每幀算兩個便宜的特徵（都在縮成長邊 160 像素的小圖上算，比找臉便宜得多）：
  床位邊緣一圈的移動量（跟前一幀比）、跟空床參考畫面的色差。空床參考畫面是安裝時拍一張。
評估見 evaluation/crib_context.py。
"""

import math
import os
from collections import deque

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

SMALL = 160          # 特徵小圖的長邊
BAND = 0.12          # 床位邊緣一圈的寬度（床位寬、高的比例）
PIXEL_TH = 25        # 灰階差超過這麼多算「這個像素在動」
MOTION_TH = 0.02     # 邊緣一圈有這麼多比例的像素在動，這一幀算「邊緣有動作」
CARE_HOLD = 1.5      # 動作停了幾秒才算大人離開
ARM = 10             # 大人離開後幾秒內允許判空床
EMPTY_TH = 14        # 跟空床參考的平均色差低於這個算「相符」（IMG_8432：空床 3.9–9.4，有娃娃的片段最低 19.4）
CONFIRM = 2          # 相符要持續幾秒才判空床（比 3 秒警示短，才不會先閃紅燈）


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


class CribState:
    """一張床的狀態：ok（綠）／alert（紅）／care（照護中，暫停計時）／empty（空床，灰）。"""

    def __init__(self, on=3, off=2, blip=1):
        self.alarm = AlertState(on, off, blip)
        self.recent = deque(maxlen=3)
        self.last_motion = None
        self.arm_deadline = math.inf   # 開機時還沒看過臉：允許判空床
        self.match_since = None
        self.empty = False

    def update(self, t, hidden, motion, ref_diff):
        self.recent.append(motion >= MOTION_TH)
        if sum(self.recent) >= 2:
            self.last_motion = t
        if self.last_motion is not None and t - self.last_motion < CARE_HOLD:
            self.empty, self.match_since = False, None
            self.arm_deadline = t + ARM
            if hidden:  # 暫停：照護中看不到臉不算時間；已經在響的不解除
                self.alarm.since_hidden = self.alarm.since_visible = None
            else:
                self.alarm.update(t, False)
            return "alert" if self.alarm.alert else "care"

        matches = ref_diff is not None and ref_diff < EMPTY_TH
        if self.empty:
            if hidden and matches:
                return "empty"
            self.empty = False  # 看到臉、或床上多了東西 → 回到一般監看
        if hidden and matches and (t <= self.arm_deadline or self.match_since is not None):
            self.match_since = t if self.match_since is None else self.match_since
            if t - self.match_since >= CONFIRM - 1e-6:
                self.empty, self.match_since = True, None
                self.alarm = AlertState(self.alarm.on, self.alarm.off, self.alarm.blip)
                return "empty"
        else:
            self.match_since = None
        alert = self.alarm.update(t, hidden)
        if not hidden:
            self.arm_deadline = -math.inf  # 寶寶在床上：要等下一次有大人來，才可能變空床
        return "alert" if alert else "ok"


def small(frame, rect):
    h, w = frame.shape[:2]
    crop = frame[int(rect[1] * h):int(rect[3] * h), int(rect[0] * w):int(rect[2] * w)]
    s = SMALL / max(crop.shape[:2])
    crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(crop, (5, 5), 0)


def edge_motion(prev, cur, band):
    """床位邊緣一圈「在動」的像素比例。先用 phase correlation 估整張小圖的平移（鏡頭晃動）、把前一幀對齊再相減；
    對齊後邊界外推出來的那幾排像素不算。"""
    (dx, dy), _ = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32))
    h, w = cur.shape
    aligned = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    mx, my = int(math.ceil(abs(dx))) + 1, int(math.ceil(abs(dy))) + 1
    valid = np.zeros_like(band)
    valid[my:h - my, mx:w - mx] = True
    return float((cv2.absdiff(cur, aligned) > PIXEL_TH)[band & valid].mean())


def border_mask(shape):
    h, w = shape
    bh, bw = max(1, int(h * BAND)), max(1, int(w * BAND))
    band = np.ones((h, w), bool)
    band[bh:h - bh, bw:w - bw] = False
    return band


class EdgeFeatures:
    """逐幀算床位特徵（live_view.py 即時用；crib_context.features() 批次也用這個，兩邊算法一樣）。
    每次 update 給一幀（要是每秒固定幀數取樣的那幾幀），回傳 (邊緣移動比例, 跟空床參考的色差或 None)。"""

    def __init__(self, rect, ref_frame=None):
        self.rect = rect
        self.ref = small(ref_frame, rect).astype(np.float32) if ref_frame is not None else None
        self.prev, self.band = None, None

    def update(self, frame):
        img = small(frame, self.rect)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.band = border_mask(gray.shape) if self.band is None else self.band
        motion = edge_motion(self.prev, gray, self.band) if self.prev is not None else 0.0
        diff = float(np.abs(img.astype(np.float32) - self.ref).mean()) if self.ref is not None else None
        self.prev = gray
        return motion, diff


def read_empty_ref(cap, slot):
    """讀床位的空床參考畫面（安裝時拍的空床照；沒有就回傳 None），讀完把影片倒回開頭。"""
    if slot["empty_ref"] is None:
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, slot["empty_ref"] * 1000)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame if ok else None
