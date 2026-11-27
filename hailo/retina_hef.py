"""
hailo/retina_hef.py
v7 的 RetinaFace 在 UGen300 上跑：infant_face_v7.RetinaFace 的替身（faces(img, th) 介面一樣），主流程不用改。
主流程設環境變數 FG_RETINA_HEF（infant_face_v7.load_retina）就會改用這個。

HEF 是 hailo/compile_hef.py --model retina 編的（ONNX 由 hailo/retina_port.py 從 Hailo 給的 TFLite 搬過來）：
- 輸入：640×480（高×寬）的 RGB uint8，床位裁切放左上角、其餘補 0。model script 照 Hailo Model Zoo 的
  retinaface_mobilenet_v1.alls：晶片上先 bgr_to_rgb 換回 BGR、再減平均 (104, 117, 123)，所以主機送 RGB
  （11/27 在 calib 300 幀比過：送 RGB 跟 CPU 版判定一致 97.3%，送 BGR 也是 97.3%；Hailo 原本 736×1280 那顆送 RGB 98.3%、BGR 96.7%）。
- 輸出：三層各 3 個卷積（框 8、分類 4、特徵點 20 通道）；softmax、anchor 解碼、NMS 在主機做，跟 ONNX 版同一段程式。
FG_CHIP_PROCESS=1 時晶片在另一個程序（hailo/chip_process.py）。

  uv run python -m hailo.retina_hef model/retinaface_v1_640x480.hef   # 晶片跟 ONNX 比（要接 UGen300）
"""

import argparse
import time
from pathlib import Path

import numpy as np

from hailo.hailo_infer import open_hef
from pipeline.infant_face_v7 import RETINA_MODEL, RETINA_TH, RetinaFace


class RetinaHef(RetinaFace):
    def __init__(self, path):   # 不呼叫 RetinaFace.__init__：不用 onnxruntime
        self.path = Path(path)
        self.chip = open_hef(self.path)
        self.setup(*self.chip.input_shape[:2])
        self.by_shape = {shape: k for k, (_, shape) in enumerate(self.chip.outputs)}

    def raw(self, x):
        outs = self.chip(x[:, :, ::-1])        # 送 RGB（見檔頭）
        return {shape: outs[k][0] for shape, k in self.by_shape.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hef", type=Path)
    ap.add_argument("--data", default="retina_calib.npy", help="prep_calib.py --model retina 存的 .npy（hailo/calib/，RGB）")
    args = ap.parse_args()
    from hailo.prep_calib import OUT as CALIB_DIR
    imgs = [np.ascontiguousarray(i[:, :, ::-1]) for i in np.load(CALIB_DIR / args.data)]   # RGB → BGR
    chip, ref = RetinaHef(args.hef), RetinaFace(RETINA_MODEL)
    same, diffs, ms = 0, [], []
    for img in imgs:
        t = time.perf_counter()
        a = chip.faces(img, 0.3)
        ms.append((time.perf_counter() - t) * 1000)
        b = ref.faces(img, 0.3)
        top = lambda fs: max((f[1] for f in fs), default=0.0)   # noqa: E731
        diffs.append(abs(top(a) - top(b)))
        same += (top(a) >= RETINA_TH) == (top(b) >= RETINA_TH)
    d = np.array(diffs)
    print(f"{len(imgs)} 張：最高分差 中位數 {np.median(d):.3f}、95% {np.percentile(d, 95):.3f}；"
          f"「有一張分數 ≥ {RETINA_TH} 的臉」跟 ONNX 一樣 {same}/{len(imgs)}｜晶片版每張（含縮放、解碼）中位數 {np.median(ms):.1f} ms")


if __name__ == "__main__":
    main()
