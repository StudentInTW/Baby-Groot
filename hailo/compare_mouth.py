"""
hailo/compare_mouth.py
量化後的口鼻小塊模型（DFC 模擬器）跟主流程的 ONNX（model/mouth_patch_ens3.onnx）比：同一批 96×96 小塊，
三顆平均的「口鼻露出」機率差多少、以 0.5（infant_face_v6.MOUTH_P_MIN）為界翻轉幾張。在 WSL 的 DFC 虛擬環境執行。

  . ~/hailo_dfc/bin/activate && cd ~/work
  python "/mnt/c/Users/user/ASUS hackathon/hailo/compare_mouth.py" --har mouth/mouth_A_quantized.har --data mouth_hold.npy

--data 是 prep_calib.py --model mouth 存的 .npy（hailo/calib/），旁邊的 *_meta.csv 用來分組（hold、hold_1123、field）。
fp_optimized＝DFC 的浮點模擬，含 model script 加的正規化（確認 parse 和正規化沒錯，應該跟 ONNX 幾乎一樣；
native 不含 model script 的修改，吃原始像素會錯，所以不用）；quantized＝8 位元模擬（晶片上會看到的）。
每張小塊的結果存成 <out>/compare_mouth_<har>_<資料>.csv。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ONNX = REPO / "model" / "mouth_patch_ens3.onnx"
CALIB_DIR = REPO / "hailo" / "calib"
P_MIN = 0.5


def sigmoid_mean(logits):
    logits = np.asarray(logits, np.float64).reshape(len(logits), -1)   # 三顆的 logit
    return (1 / (1 + np.exp(-logits))).mean(1)


def emulate(har, data, context):
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=str(har))
    ctx_type = {"quantized": InferenceContext.SDK_QUANTIZED, "fp_optimized": InferenceContext.SDK_FP_OPTIMIZED}[context]
    with runner.infer_context(ctx_type) as ctx:
        out = runner.infer(ctx, data.astype(np.float32), batch_size=64)
    return sigmoid_mean(out)


def onnx_probs(data):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"bgr": data[i:i + 256]})[0] for i in range(0, len(data), 256)])


def describe(p, q):
    d = np.abs(p - q)
    flip = (p >= P_MIN) != (q >= P_MIN)
    return (f"機率差 中位 {np.median(d):.4f}、95% {np.percentile(d, 95):.4f}、最大 {d.max():.4f}｜"
            f"翻轉 {int(flip.sum())}/{len(p)}（看得到→看不到 {int((flip & (p >= P_MIN)).sum())}、"
            f"看不到→看得到 {int((flip & (p < P_MIN)).sum())}）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", nargs="+", required=True, type=Path)
    ap.add_argument("--data", default="mouth_hold.npy")
    ap.add_argument("--out", type=Path, default=Path.home() / "work" / "mouth")
    args = ap.parse_args()

    data = np.load(CALIB_DIR / args.data)
    with open((CALIB_DIR / args.data).with_name(Path(args.data).stem + "_meta.csv"), encoding="utf-8-sig") as fh:
        meta = list(csv.DictReader(fh))
    p_onnx = onnx_probs(data)
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["set"]].append(i)
    args.out.mkdir(parents=True, exist_ok=True)
    for har in args.har:
        res = {c: emulate(har, data, c) for c in ("fp_optimized", "quantized")}
        print(f"\n== {har.name}（{len(data)} 張，{args.data}）")
        for c, p in res.items():
            print(f"  {c:<9} 全部：{describe(p_onnx, p)}")
        for g, idx in sorted(groups.items()):
            print(f"  quantized {g}：{describe(p_onnx[idx], res['quantized'][idx])}")
        out = args.out / f"compare_mouth_{har.stem}_{Path(args.data).stem}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["patch", "set", "label", "p_onnx", "p_fp", "p_quant"])
            for i, m in enumerate(meta):
                wr.writerow([m["patch"], m["set"], m["label"], f"{p_onnx[i]:.5f}", f"{res['fp_optimized'][i]:.5f}",
                             f"{res['quantized'][i]:.5f}"])
        print(f"  → {out}")


if __name__ == "__main__":
    main()
