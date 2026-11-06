"""
hailo/compare_yunet.py
量化後的 YuNet（DFC 模擬器）跟 ONNX 比：同一批 640×640 的圖，找到的臉、分數、框差多少。在 WSL 的 DFC 虛擬環境執行。

  . ~/hailo_dfc/bin/activate && cd ~/work
  python "/mnt/c/Users/user/ASUS hackathon/hailo/compare_yunet.py" --har yunet/yunet_A_quantized.har yunet/yunet_B_quantized.har \
      --data doll_hold.npy doll_calib.npy:384

--data 是 prep_calib.py 存的 .npy（預設在 hailo/calib/），旁邊的 *_meta.csv 用來分條件（原圖／灰階／暗光）和種類
（整張圖／頭部／床位）。門檻照主流程：整張圖 0.7（FACE_SCORE_TH）、頭部和床位 0.8（HEAD_FACE_TH、SLOT_FACE_TH）。

比的東西：
- 臉數不一樣的圖有幾張（有臉在門檻上下翻轉）——這會直接改變燈號，最重要
- 配對到的臉：分數差、框的四個邊差幾 px（640×640 座標）
- 原始輸出的 SNR（dB）：12 個輸出層各自的 10·log10(Σonnx² / Σ(onnx−量化)²)
每張圖的結果存成 <out>/compare_<模型>_<資料>.csv。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from yunet_decode import OUTPUT_NAMES, decode, onnx_session

REPO = Path(__file__).resolve().parent.parent
ONNX = REPO / "model" / "face_detection_yunet_2023mar.onnx"
CALIB_DIR = REPO / "hailo" / "calib"
TH = {"whole": 0.7, "head": 0.8, "slot": 0.8}
LOW = 0.5   # 先用低門檻解碼，才看得到「差一點過門檻」的臉


def load(spec):
    name, _, n = spec.partition(":")
    path = CALIB_DIR / name
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr[:int(n)] if n else arr)
    meta_path = path.with_name(path.stem + "_meta.csv")
    with open(meta_path, encoding="utf-8-sig") as fh:
        meta = list(csv.DictReader(fh))[:len(arr)]
    return arr, meta


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def match(ref, got):
    """ONNX 的臉跟量化後的臉一對一配對（IoU > 0.5，分數高的先配）。回傳 [(ref_i, got_j)]。"""
    pairs, used = [], set()
    for i in np.argsort(-ref[:, 14]):
        best, bj = 0.5, None
        for j in range(len(got)):
            if j not in used and (v := iou(ref[i], got[j])) > best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj))
    return pairs


def quantized_outputs(har, data, context):
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=str(har))
    ctx_type = {"quantized": InferenceContext.SDK_QUANTIZED, "native": InferenceContext.SDK_NATIVE,
                "fp_optimized": InferenceContext.SDK_FP_OPTIMIZED}[context]
    with runner.infer_context(ctx_type) as ctx:
        outs = runner.infer(ctx, data.astype(np.float32), batch_size=8)
    return dict(zip(OUTPUT_NAMES, outs))   # 輸出順序跟 ONNX 一樣（parse 時 DFC 依 ONNX 的輸出順序對應）


def compare(har, spec, args, onnx_run):
    data, meta = load(spec)
    q = quantized_outputs(har, data, args.context)
    rows, snr_num, snr_den = [], defaultdict(float), defaultdict(float)
    for k in range(len(data)):
        ref_raw = onnx_run(data[k])
        got_raw = {n: q[n][k] for n in OUTPUT_NAMES}
        for n in OUTPUT_NAMES:
            r = ref_raw[n].reshape(-1)
            g = np.asarray(got_raw[n], np.float32).reshape(-1)
            snr_num[n] += float((r ** 2).sum())
            snr_den[n] += float(((r - g) ** 2).sum())
        m = meta[k]
        kind = m.get("kind", "whole")
        th = TH[kind]
        ref, got = decode(ref_raw, LOW), decode(got_raw, LOW)
        n_ref, n_got = int((ref[:, 14] >= th).sum()), int((got[:, 14] >= th).sum())
        pairs = match(ref, got)
        sd = [abs(float(ref[i, 14] - got[j, 14])) for i, j in pairs if max(ref[i, 14], got[j, 14]) >= th]
        bd = [float(np.abs(np.r_[ref[i, :2], ref[i, :2] + ref[i, 2:4]] - np.r_[got[j, :2], got[j, :2] + got[j, 2:4]]).max())
              for i, j in pairs if max(ref[i, 14], got[j, 14]) >= th]
        rows.append({"i": k, "source": m.get("source", ""), "cond": m.get("cond", "orig"), "kind": kind,
                     "faces_onnx": n_ref, "faces_hailo": n_got,
                     "max_score_diff": round(max(sd), 4) if sd else "", "max_box_diff_px": round(max(bd), 2) if bd else "",
                     "scores_onnx": " ".join(f"{s:.3f}" for s in ref[:, 14] if s >= th - 0.1),
                     "scores_hailo": " ".join(f"{s:.3f}" for s in got[:, 14] if s >= th - 0.1)})
    snr = {n: 10 * np.log10(snr_num[n] / max(snr_den[n], 1e-12)) for n in OUTPUT_NAMES}
    out = args.out / f"compare_{Path(har).stem.replace('_quantized', '')}_{Path(spec.split(':')[0]).stem}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows, snr, out


def summarize(rows):
    faces = sum(r["faces_onnx"] for r in rows)
    flips = sum(r["faces_onnx"] != r["faces_hailo"] for r in rows)
    sd = [r["max_score_diff"] for r in rows if r["max_score_diff"] != ""]
    bd = [r["max_box_diff_px"] for r in rows if r["max_box_diff_px"] != ""]
    pct = lambda v, p: f"{np.percentile(v, p):.3f}" if v else "-"
    return (f"{len(rows):4d} 張｜ONNX 臉 {faces:4d}｜臉數不同 {flips:3d} 張｜"
            f"分數差 中位 {pct(sd, 50)}、95% {pct(sd, 95)}、最大 {pct(sd, 100)}｜"
            f"框差 px 中位 {pct(bd, 50)}、95% {pct(bd, 95)}、最大 {pct(bd, 100)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", nargs="+", required=True, type=Path)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--context", default="quantized", choices=("quantized", "native", "fp_optimized"),
                    help="native＝parse 後的浮點模擬（用來確認輸出對應、解碼沒寫錯，應該跟 ONNX 幾乎一樣）")
    ap.add_argument("--out", type=Path, default=Path.home() / "work" / "yunet")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_run = onnx_session(ONNX)
    for har in args.har:
        for spec in args.data:
            rows, snr, out = compare(har, spec, args, onnx_run)
            print(f"\n=== {har.name} × {spec}（{args.context}）→ {out.name}")
            print("  全部        " + summarize(rows))
            for key in sorted({(r["cond"], r["kind"]) for r in rows}):
                print(f"  {key[0]:6s} {key[1]:5s} " + summarize([r for r in rows if (r["cond"], r["kind"]) == key]))
            print("  SNR dB：" + "、".join(f"{n} {v:.1f}" for n, v in snr.items()))


if __name__ == "__main__":
    main()
