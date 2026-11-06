"""
hailo/yunet_decode.py
YuNet（2023mar）的主機端解碼：只用 numpy，ONNX（onnxruntime）和 Hailo（模擬器、晶片）的輸出都用這支。

YuNet 的網路只輸出 12 個特徵層（cls／obj／bbox／kps × stride 8／16／32），框、特徵點、NMS 都在主機上算。
現在主流程由 OpenCV 的 FaceDetectorYN 代勞；HEF 的輸入尺寸固定，接上晶片後要自己解碼。
算法照 OpenCV 4.x 的 modules/objdetect/src/face_detect.cpp。

輸入圖的規定（跟 OpenCV 一樣）：BGR、0–255、不做正規化；圖小於 640×640 就在右邊和下面補 0（pad_to）。
"""

import numpy as np

SIZE = 640                      # ONNX／HEF 的輸入固定 640×640
STRIDES = (8, 16, 32)
NMS_TH = 0.3                    # 跟 FaceDetectorYN 的預設一樣
TOP_K = 5000
OUTPUT_NAMES = [f"{k}_{s}" for k in ("cls", "obj", "bbox", "kps") for s in STRIDES]
CHANNELS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}


def pad_to(img, size=SIZE):
    """把長寬都不超過 size 的圖放左上角、右邊和下面補 0，變成 size×size（OpenCV 內部也是這樣補）。"""
    h, w = img.shape[:2]
    if h > size or w > size:
        raise ValueError(f"圖 {w}×{h} 比 {size}×{size} 大，要先縮小或切塊")
    out = np.zeros((size, size, 3), np.uint8)
    out[:h, :w] = img
    return out


def to_rows(name, arr, size=SIZE):
    """把一個輸出層整理成 (格數, 通道數)，格子順序是 row-major（r * cols + c）。
    ONNX 的輸出是 (1, 格數, 通道)；Hailo 的是 (1, H, W, C) 或 (H, W, C)，兩種都收。"""
    kind, stride = name.split("_")
    n = (size // int(stride)) ** 2
    a = np.asarray(arr, np.float32).reshape(-1, CHANNELS[kind])
    if a.shape[0] != n:
        raise ValueError(f"{name}：預期 {n} 格，拿到 {a.shape[0]}")
    return a


def decode(outputs, score_th, size=SIZE, valid_wh=None, nms_th=NMS_TH, top_k=TOP_K):
    """outputs：{名稱: 陣列}，12 個 YuNet 輸出（cls、obj 要已經過 sigmoid）。
    回傳 (N, 15) 的 float32：x, y, w, h, 5 個特徵點 (x, y), score——跟 FaceDetectorYN.detect 的格式一樣。
    valid_wh：原圖（補邊前）的 (寬, 高)。OpenCV 只在「補到 32 倍數」的範圍內找臉，補到 640 的那一大片
    沒有對應；給這個參數就只保留中心落在原圖範圍內的格子，結果才跟 OpenCV 一樣。"""
    faces = []
    for s in STRIDES:
        cols = size // s
        cls = np.clip(to_rows(f"cls_{s}", outputs[f"cls_{s}"], size)[:, 0], 0, 1)
        obj = np.clip(to_rows(f"obj_{s}", outputs[f"obj_{s}"], size)[:, 0], 0, 1)
        bbox = to_rows(f"bbox_{s}", outputs[f"bbox_{s}"], size)
        kps = to_rows(f"kps_{s}", outputs[f"kps_{s}"], size)
        idx = np.arange(cols * cols)
        r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        if valid_wh is not None:
            pw, ph = (-(-v // 32) * 32 for v in valid_wh)  # OpenCV 把輸入補到 32 的倍數
            keep = (c < pw // s) & (r < ph // s)
        else:
            keep = np.ones(len(idx), bool)
        score = np.sqrt(cls * obj)
        keep &= score >= score_th
        if not keep.any():
            continue
        r, c, score, bbox, kps = r[keep], c[keep], score[keep], bbox[keep], kps[keep]
        cx, cy = (c + bbox[:, 0]) * s, (r + bbox[:, 1]) * s
        w, h = np.exp(bbox[:, 2]) * s, np.exp(bbox[:, 3]) * s
        lm = np.empty((len(r), 10), np.float32)
        lm[:, 0::2] = (kps[:, 0::2] + c[:, None]) * s
        lm[:, 1::2] = (kps[:, 1::2] + r[:, None]) * s
        faces.append(np.column_stack([cx - w / 2, cy - h / 2, w, h, lm, score]).astype(np.float32))
    if not faces:
        return np.zeros((0, 15), np.float32)
    faces = np.concatenate(faces)
    return faces[nms(faces[:, :4], faces[:, 14], nms_th, top_k)]


def nms(xywh, scores, th, top_k):
    """貪婪 NMS（跟 cv2.dnn.NMSBoxes 一樣：分數由高到低，IoU 大於 th 的丟掉）。回傳保留的索引。"""
    order = np.argsort(-scores, kind="stable")[:top_k]
    x1, y1 = xywh[:, 0], xywh[:, 1]
    x2, y2 = x1 + xywh[:, 2], y1 + xywh[:, 3]
    area = xywh[:, 2] * xywh[:, 3]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= th]
    return np.array(keep, int)


def onnx_session(model_path):
    """onnxruntime 的 YuNet；回傳 run(img_640) → {名稱: 陣列}。img_640 是 640×640 的 BGR uint8。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def run(img):
        x = img.astype(np.float32).transpose(2, 0, 1)[None]
        return dict(zip(OUTPUT_NAMES, sess.run(OUTPUT_NAMES, {"input": x})))
    return run
