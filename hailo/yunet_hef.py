"""
hailo/yunet_hef.py
YuNet 在 UGen300 上跑：OpenCV FaceDetectorYN 的替身，介面一樣（setInputSize、detect、setScoreThreshold…），
主流程不用改。主流程設環境變數 FG_YUNET_HEF（face_common.create_face_detector）就會改用這個。

HEF 的輸入固定 640×640，主流程送進 YuNet 的圖有四種大小（hailo/結果.md「接上主流程前要處理」）：
- 長寬都不超過 640（整張圖長邊 640、頭部 320×320、床位範圍）：放左上角、右邊和下面補 0（yunet_decode.pad_to），
  解碼時只留中心在原圖範圍裡的格子（valid_wh），跟 OpenCV 一樣。
- 比 640 大（整張圖長邊 1280 那個尺度）：切成互相重疊的 640×640 小塊一次送進晶片，每塊各自解碼，
  只留中心落在這一塊「負責範圍」裡的臉（重疊區從中間分給兩塊），座標換回原圖，最後整體再做一次 NMS。
  重疊至少 OVERLAP px，比這個小的臉一定完整落在負責它的那一塊；更大的臉長邊 640 那個尺度會找到。
同一張圖連續送兩次（det 和 det_v5 在同一個床位範圍用不同門檻各找一次）時，第二次直接用上一次的晶片輸出。
"""

import math
import threading
from pathlib import Path

import numpy as np

from hailo.hailo_infer import Hef
from hailo.yunet_decode import NMS_TH, SIZE, STRIDES, TOP_K, decode, nms, pad_to

OVERLAP = 160
# HEF 的 1 通道輸出（sigmoid）對應原本 ONNX（model/face_detection_yunet_2023mar.onnx）的哪個輸出：
# 從 ONNX 的輸出往回追到產生它的節點（11/06 查過）；HailoRT 用 get_original_names_from_vstream_name 回報這些名稱
SIGMOID_ROLE = {"Sigmoid_85": "cls_8", "Sigmoid_93": "cls_16", "Sigmoid_101": "cls_32",
                "Sigmoid_109": "obj_8", "Sigmoid_117": "obj_16", "Sigmoid_125": "obj_32"}
_RUNNERS = {}


class _Runner:
    """一顆 YuNet HEF（同一個路徑共用一份）：把輸出對應到 yunet_decode 用的名稱，記住上一張圖的結果。"""

    def __init__(self, path):
        self.chip = Hef(path)
        self.roles = []
        for name, shape in self.chip.outputs:
            ch, n = shape[-1], int(np.prod(shape[:-1]))
            stride = {(SIZE // s) ** 2: s for s in STRIDES}.get(n)
            if ch == 1:     # cls、obj 形狀一樣，只能靠原本的節點名稱分
                role = next((SIGMOID_ROLE[o] for o in self.chip.original_names[name] if o in SIGMOID_ROLE), None)
            else:           # bbox（4 通道）、kps（10 通道）用形狀就分得出來
                kind = {4: "bbox", 10: "kps"}.get(ch)
                role = f"{kind}_{stride}" if kind else None
            if role is None or stride is None or not role.endswith(f"_{stride}"):
                raise ValueError(f"{path}：認不出輸出 {name} {shape}（原本的節點 {self.chip.original_names[name]}）")
            self.roles.append(role)
        if len(set(self.roles)) != 12:
            raise ValueError(f"{path}：輸出對應不完整 {self.roles}")
        self._last = threading.local()   # 每條執行緒記自己的上一張（多床同時處理時不會互相拿錯）

    def __call__(self, batch):
        """batch：640×640 BGR uint8 的 list → 每張一個 {名稱: 陣列}。"""
        last = getattr(self._last, "pair", None)
        if len(batch) == 1 and last is not None and np.array_equal(batch[0], last[0]):
            return [last[1]]
        outs = self.chip(np.stack(batch))
        res = [{role: o[k] for role, o in zip(self.roles, outs)} for k in range(len(batch))]
        if len(batch) == 1:
            self._last.pair = (batch[0], res[0])
        return res


def spans(n, size=SIZE, overlap=OVERLAP):
    """一個方向切成幾段：[(起點, 長度, 負責範圍起點, 負責範圍終點)]。不超過 size 就一段。"""
    if n <= size:
        return [(0, n, 0, n)]
    k = math.ceil((n - overlap) / (size - overlap))
    starts = [round(i * (n - size) / (k - 1)) for i in range(k)]
    bounds = [0] + [(starts[i] + starts[i - 1] + size) // 2 for i in range(1, k)] + [n]
    return [(starts[i], size, bounds[i], bounds[i + 1]) for i in range(k)]


class YuNetHef:
    """cv2.FaceDetectorYN 的替身。detect 回傳 (1, faces)，faces 是 (N, 15) float32 或 None（沒找到臉），跟 OpenCV 一樣。"""

    def __init__(self, hef_path, score_th=0.9, nms_th=NMS_TH, top_k=TOP_K, input_wh=(320, 320)):
        key = str(Path(hef_path).resolve())
        if key not in _RUNNERS:
            _RUNNERS[key] = _Runner(key)
        self.runner = _RUNNERS[key]
        self.score_th, self.nms_th, self.top_k = float(score_th), float(nms_th), int(top_k)
        self.input_wh = tuple(input_wh)

    # OpenCV 的介面。輸入尺寸只是記著：晶片版每次照圖的實際大小處理
    def setInputSize(self, wh):
        self.input_wh = (int(wh[0]), int(wh[1]))

    def getInputSize(self):
        return self.input_wh

    def setScoreThreshold(self, th):
        self.score_th = float(th)

    def getScoreThreshold(self):
        return self.score_th

    def setNMSThreshold(self, th):
        self.nms_th = float(th)

    def getNMSThreshold(self):
        return self.nms_th

    def setTopK(self, k):
        self.top_k = int(k)

    def getTopK(self):
        return self.top_k

    def detect(self, img):
        h, w = img.shape[:2]
        tiles = [(x, y, tw, th, (cx0, cx1, cy0, cy1))
                 for y, th, cy0, cy1 in spans(h) for x, tw, cx0, cx1 in spans(w)]
        raws = self.runner([pad_to(img[y:y + th, x:x + tw]) for x, y, tw, th, _ in tiles])
        found = []
        for (x, y, tw, th, (cx0, cx1, cy0, cy1)), raw in zip(tiles, raws):
            f = decode(raw, self.score_th, valid_wh=(tw, th), nms_th=self.nms_th, top_k=self.top_k)
            if len(tiles) > 1 and len(f):
                cx, cy = f[:, 0] + f[:, 2] / 2 + x, f[:, 1] + f[:, 3] / 2 + y
                f = f[(cx >= cx0) & (cx < cx1) & (cy >= cy0) & (cy < cy1)]
            if len(f):
                f = f.copy()
                f[:, [0, 4, 6, 8, 10, 12]] += x
                f[:, [1, 5, 7, 9, 11, 13]] += y
                found.append(f)
        if not found:
            return 1, None
        faces = np.concatenate(found)
        if len(tiles) > 1:
            faces = faces[nms(faces[:, :4], faces[:, 14], self.nms_th, self.top_k)]
        return 1, faces
