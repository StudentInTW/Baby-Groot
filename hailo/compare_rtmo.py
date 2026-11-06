"""
hailo/compare_rtmo.py
量化後的 RTMO-m（DFC 模擬器跑前段＋主機跑後段）跟原本的 ONNX 比。在 WSL 的 DFC 虛擬環境執行。

  . ~/hailo_dfc/bin/activate && cd ~/work
  python "/mnt/c/Users/user/ASUS hackathon/hailo/compare_rtmo.py" --har rtmo/rtmo_A_quantized.har \
      --data rtmo_doll_hold.npy rtmo_coco_hold.npy

晶片上實跑（UGen300 插著、pyHailoRT 裝在專案環境，見 hailo_infer.py）：在 Windows 用 --hef 代替 --har
  uv run --with onnx python hailo/compare_rtmo.py --hef model/rtmo_D6.hef --data rtmo_doll_hold.npy rtmo_coco_hold.npy \
      --out outputs/hailo/onchip

--data 是 prep_calib.py --model rtmo 存的 .npy（預設在 hailo/calib/），旁邊的 *_meta.csv 用來分條件（原圖／灰階／暗光）。
ONNX 的標準答案用「原本的 ONNX 多輸出切點張量」那一份（rtmo_post.with_cut_outputs）：dets、keypoints 跟原本的 ONNX
一樣（rtmo_post.py --check 驗過，最大差 0.002 px），順便拿到前段 8 個輸出算 SNR。

後處理照主流程（infant_face_v3.load_rtmo）：後段的框先過分數門檻 0.3（DET_TH）、再做 NMS（IoU 0.45，rtmlib 預設）。
比的東西（座標都是 640×640 輸入上的 px）：
- 人數不一樣的圖有幾張——主流程靠它判斷有沒有人、床上有沒有寶寶
- 配對到的人（框 IoU > 0.5）：分數差、框的四個邊最大差
- 關鍵點：兩邊都 ≥ 0.3（KPT_TH）的點位置差幾 px；頭部 5 點（鼻、眼、耳）另外算——主流程用它們找頭部範圍；
  可見度翻轉（一邊 ≥ 0.3、另一邊沒有）的點有幾個
- 前段 8 個輸出的 SNR（dB）：10·log10(Σonnx² / Σ(onnx−量化)²)
每張圖的結果存成 <out>/compare_<模型>_<資料>.csv（晶片上跑的是 compare_<模型>_chip_<資料>.csv）。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtmo_post import ONNX, Post, cut_tensors, load, to_nchw, with_cut_outputs

DET_TH, NMS_THR, KPT_TH = 0.3, 0.45, 0.3   # infant_face_v3.DET_TH、rtmlib RTMO 的 nms_thr、face_common.KPT_TH
ROLES = ("cls16", "box16", "vis16", "feat16", "cls32", "box32", "vis32", "feat32")   # 依 rtmo_post.CUT_NODES 的順序


def nms(boxes, scores, thr):
    """跟 rtmlib 的 nms 一樣（tools/object_detection/post_processings.py，Apache-2.0）。"""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= thr)[0] + 1]
    return keep


def people(dets, kpts):
    """後段輸出 → 主流程看到的人：(框 xyxy, 分數, 關鍵點 (17,3))，分數高的在前。"""
    boxes, scores = dets[0, :, :4], dets[0, :, 4]
    idx = np.where(scores > DET_TH)[0]
    keep = idx[nms(boxes[idx], scores[idx], NMS_THR)] if len(idx) else idx
    return boxes[keep], scores[keep], kpts[0, keep]


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def match(ref_boxes, got_boxes):
    pairs, used = [], set()
    for i in range(len(ref_boxes)):           # 已經依分數排好
        best, bj = 0.5, None
        for j in range(len(got_boxes)):
            if j not in used and (v := iou(ref_boxes[i], got_boxes[j])) > best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((i, bj))
    return pairs


def quantized_outputs(har, data, context, batch, chunk):
    """模擬器跑前段。每次只餵 chunk 張：一次把 384 張（640×640 float 約 1.9 GB）交給模擬器，
    筆電 4 GB 的顯示記憶體連 batch 1 都不夠（2026/11/05，量化模擬；浮點模擬 32 張沒事）。"""
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=str(har))
    ctx_type = {"quantized": InferenceContext.SDK_QUANTIZED, "native": InferenceContext.SDK_NATIVE,
                "fp_optimized": InferenceContext.SDK_FP_OPTIMIZED}[context]
    parts = []
    with runner.infer_context(ctx_type) as ctx:
        for s in range(0, len(data), chunk):
            outs = runner.infer(ctx, data[s:s + chunk].astype(np.float32), batch_size=batch)
            parts.append(outs if isinstance(outs, list) else [outs])
    return [np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))]   # 8 個 (N, H, W, C)，順序交給 Post 用形狀對應


def compare(har, spec, args, full, post, names, outputs=None):
    """outputs：給 data、回傳前段 8 個輸出的函式；不給就用 DFC 模擬器跑 har（sim_quant.py 給的是 ONNX 上的模擬量化，
    --hef 給的是晶片，這時 har 是 .hef 的路徑）。"""
    data, meta = load(spec)
    q = outputs(data) if outputs else quantized_outputs(har, data, args.context, args.batch, args.chunk)
    rows, kd_all, kd_head = [], [], []
    snr_num, snr_den = defaultdict(float), defaultdict(float)
    for k in range(len(data)):
        outs = full.run(None, {"input": to_nchw(data[k])})
        ref_feed = post.feed(outs[2:])
        got_feed = post.feed([o[k] for o in q])
        for name in ref_feed:
            snr_num[names[name]] += float((ref_feed[name] ** 2).sum())
            snr_den[names[name]] += float(((ref_feed[name] - got_feed[name]) ** 2).sum())
        rb, rs, rk = people(outs[0], outs[1])
        gb, gs, gk = people(*post.sess.run(["dets", "keypoints"], got_feed))
        pairs = match(rb, gb)
        sd = [abs(float(rs[i] - gs[j])) for i, j in pairs]
        bd = [float(np.abs(rb[i] - gb[j]).max()) for i, j in pairs]
        kd, hd, flips = [], [], 0
        for i, j in pairs:
            vr, vg = rk[i][:, 2] >= KPT_TH, gk[j][:, 2] >= KPT_TH
            flips += int((vr != vg).sum())
            d = np.linalg.norm(rk[i][:, :2] - gk[j][:, :2], axis=1)
            kd += d[vr & vg].tolist()
            hd += d[:5][(vr & vg)[:5]].tolist()
        kd_all += kd
        kd_head += hd
        m = meta[k]
        rows.append({"i": k, "source": m.get("source", ""), "cond": m.get("cond", "orig"),
                     "people_onnx": len(rb), "people_hailo": len(gb), "matched": len(pairs),
                     "max_score_diff": round(max(sd), 4) if sd else "", "max_box_diff_px": round(max(bd), 2) if bd else "",
                     "kpt_med_px": round(float(np.median(kd)), 2) if kd else "", "kpt_max_px": round(max(kd), 2) if kd else "",
                     "head_kpt_max_px": round(max(hd), 2) if hd else "", "kpt_vis_flips": flips,
                     "scores_onnx": " ".join(f"{s:.3f}" for s in rs), "scores_hailo": " ".join(f"{s:.3f}" for s in gs),
                     "_kd": kd, "_hd": hd, "_sd": sd, "_bd": bd})
    snr = {r: 10 * np.log10(snr_num[r] / max(snr_den[r], 1e-12)) for r in ROLES}
    tag = Path(har).stem.replace('_quantized', '') + ("_chip" if Path(har).suffix == ".hef" else "")
    out = args.out / f"compare_{tag}_{Path(spec.split(':')[0]).stem}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=[f for f in rows[0] if not f.startswith("_")], extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return rows, snr, out


def summarize(rows):
    pct = lambda v, p: f"{np.percentile(v, p):.2f}" if v else "-"   # noqa: E731
    pool = lambda key: [x for r in rows for x in r[key]]              # noqa: E731
    sd, bd, kd, hd = pool("_sd"), pool("_bd"), pool("_kd"), pool("_hd")
    flips = sum(r["people_onnx"] != r["people_hailo"] for r in rows)
    return (f"{len(rows):4d} 張｜ONNX 人 {sum(r['people_onnx'] for r in rows):4d}｜人數不同 {flips:3d} 張｜"
            f"分數差 中位 {pct(sd, 50)}、最大 {pct(sd, 100)}｜框差 px 中位 {pct(bd, 50)}、95% {pct(bd, 95)}｜"
            f"關鍵點 px 中位 {pct(kd, 50)}、95% {pct(kd, 95)}、最大 {pct(kd, 100)}｜頭部 95% {pct(hd, 95)}｜"
            f"可見度翻轉 {sum(r['kpt_vis_flips'] for r in rows)}／{len(kd) + sum(r['kpt_vis_flips'] for r in rows)} 點")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--har", nargs="+", default=[], type=Path, help="DFC 模擬器跑的 HAR（WSL）")
    ap.add_argument("--hef", nargs="+", default=[], type=Path, help="晶片上跑的 HEF（Windows，UGen300）")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--context", default="quantized", choices=("quantized", "native", "fp_optimized"),
                    help="native＝parse 後的浮點模擬（用來確認輸出對應沒弄錯，應該跟 ONNX 幾乎一樣）")
    ap.add_argument("--out", type=Path, default=Path.home() / "work" / "rtmo")
    ap.add_argument("--batch", type=int, default=4, help="模擬器的 batch")
    ap.add_argument("--chunk", type=int, default=16, help="每次餵給模擬器幾張（見 quantized_outputs）")
    args = ap.parse_args()
    if not args.har and not args.hef:
        ap.error("--har 或 --hef 至少要給一個")
    args.out.mkdir(parents=True, exist_ok=True)

    import onnx
    import onnxruntime as ort
    model = onnx.load(str(ONNX))
    full = ort.InferenceSession(with_cut_outputs(model).SerializeToString(), providers=["CPUExecutionProvider"])
    post = Post()
    names = dict(zip(cut_tensors(model), ROLES))
    def report(model, spec, label, outputs=None):
        rows, snr, out = compare(model, spec, args, full, post, names, outputs)
        print(f"\n=== {model.name} × {spec}（{label}）→ {out.name}")
        print("  全部    " + summarize(rows))
        for cond in sorted({r["cond"] for r in rows}):
            print(f"  {cond:6s}  " + summarize([r for r in rows if r["cond"] == cond]))
        print("  SNR dB：" + "、".join(f"{n} {v:.1f}" for n, v in snr.items()))

    for har in args.har:
        for spec in args.data:
            report(har, spec, args.context)
    for hef in args.hef:
        from hailo_infer import Hef
        with Hef(hef) as chip:
            for spec in args.data:
                report(hef, spec, "晶片", chip)


if __name__ == "__main__":
    main()
