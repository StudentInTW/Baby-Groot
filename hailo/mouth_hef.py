"""
hailo/mouth_hef.py
v6 的口鼻小塊模型在 UGen300 上跑：infant_face_v6.MouthModel 的替身（probs(img, faces) 介面一樣），主流程不用改。
主流程設環境變數 FG_MOUTH_HEF（infant_face_v6.load_mouth_model）就會改用這個。

HEF 是 hailo/compile_hef.py --model mouth 編的：
- 輸入：96×96 的 BGR uint8 小塊（infant_face_v6.mouth_patch 切的那塊）；正規化在晶片上（model script 的 normalization）。
- 輸出：三顆 ResNet18 的 logit。sigmoid 和三顆平均在主機做，跟 ONNX 版（model/mouth_patch_ens3.onnx）同一個定義。
一幀裡要判的小塊（床位裡所有的臉）一次送進晶片。FG_CHIP_PROCESS=1 時晶片在另一個程序（hailo/chip_process.py）。

  uv run python -m hailo.mouth_hef model/mouth_A.hef --data mouth_hold.npy   # 晶片跟 ONNX 比（要接 UGen300）
"""

import argparse
import time
from pathlib import Path

import numpy as np

from hailo.hailo_infer import open_hef
from pipeline.infant_face_v6 import MouthModel


class MouthHef(MouthModel):
    def __init__(self, path):   # 不呼叫 MouthModel.__init__：不用 onnxruntime
        self.path = Path(path)
        self.chip = open_hef(self.path)

    def predict(self, patches):
        logits = np.asarray(self.chip(patches)[0], np.float64).reshape(len(patches), -1)
        return (1 / (1 + np.exp(-logits))).mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hef", type=Path)
    ap.add_argument("--data", default="mouth_hold.npy", help="prep_calib.py --model mouth 存的 .npy（hailo/calib/）")
    args = ap.parse_args()
    import csv
    from collections import defaultdict
    from hailo.prep_calib import OUT as CALIB_DIR
    from pipeline.infant_face_v6 import MOUTH_MODEL, MOUTH_P_MIN
    data = np.load(CALIB_DIR / args.data)
    groups = defaultdict(list)   # prep_calib 存的 *_meta.csv：每張小塊來自哪一組（hold、hold_1123、field）
    meta = (CALIB_DIR / args.data).with_name(Path(args.data).stem + "_meta.csv")
    if meta.exists():
        with open(meta, encoding="utf-8-sig") as fh:
            for i, m in enumerate(csv.DictReader(fh)):
                groups[m["set"]].append(i)
    chip = MouthHef(args.hef)
    ref = MouthModel(MOUTH_MODEL)
    t0 = time.perf_counter()
    p_chip = np.concatenate([chip.predict(data[i:i + 16]) for i in range(0, len(data), 16)])
    dt = time.perf_counter() - t0
    p_ref = np.concatenate([ref.predict(data[i:i + 256]) for i in range(0, len(data), 256)])

    def describe(idx):
        d = np.abs(p_chip[idx] - p_ref[idx])
        flip = (p_chip[idx] >= MOUTH_P_MIN) != (p_ref[idx] >= MOUTH_P_MIN)
        lost = flip & (p_ref[idx] >= MOUTH_P_MIN)
        return (f"{len(d)} 張：機率差 中位 {np.median(d):.4f}、95% {np.percentile(d, 95):.4f}、最大 {d.max():.4f}｜"
                f"翻轉 {int(flip.sum())}（看得到→看不到 {int(lost.sum())}、看不到→看得到 {int((flip & ~lost).sum())}）")

    print(f"全部 {describe(np.arange(len(data)))}｜晶片每張 {dt / len(data) * 1000:.2f} ms（一次 16 張）")
    for g, idx in sorted(groups.items()):
        print(f"  {g}：{describe(np.array(idx))}")


if __name__ == "__main__":
    main()
