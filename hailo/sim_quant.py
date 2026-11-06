"""在 ONNX 上模擬 DFC 的 8 位元量化（RTMO-m 前段），不跑 DFC 就能預估量化掉多少。

規則（照 DFC 對 RTMO 的預設：SiLU 做不了 equalization，大部分層都是「一層一組範圍」）：
- 權重：8 位元對稱，每層一組範圍（最大絕對值）；--w channel 改成每個輸出通道一組。bias 不量化。
- 啟動值：每個「層的輸出」8 位元非對稱、整層一組範圍＝校正圖上的最小～最大（含 0）。
  層的輸出＝卷積（後面接 SiLU 就取 SiLU 之後）、相加、串接、池化、放大、softmax、矩陣乘法（qat_rtmo.front_onnx 同一套）。
- --float-convs 1,2,4：這幾個卷積（ONNX 裡的順序，1 起算）的權重和輸出都不量化，近似 DFC 的 16 位元（a16_w16）。
用 QuantizeLinear／DequantizeLinear 插在每個層的輸出後面，onnxruntime 關掉圖最佳化執行（不會被換成整數運算）。

跟 DFC 實測對得上（2026/11/05，前段 8 個輸出的 SNR）：
- 原本的 RTMO-m（A）：模擬 0.4–6.7 dB、DFC 0.4–6.7 dB
- 前三層 16 位元（C1）：模擬 2.4–10.4 dB、DFC 4–11 dB
模擬沒有包含的：DFC 用分段直線近似 SiLU、16 位元的 bias、equalization（RTMO 只有 25 層做得了）。

標準答案一律是原本的 RTMO-m（rtmo_post.ONNX），--onnx 換成量化感知訓練後的模型（qat_rtmo.py）時，
比的是「訓練後的模型量化之後」跟「原本的模型」差多少。

  uv run --with onnx python hailo/sim_quant.py                                    # 原本的 RTMO-m，SNR
  uv run --with onnx python hailo/sim_quant.py --onnx <qat 模型> --compare rtmo_doll_hold.npy rtmo_coco_hold.npy
--compare 用 compare_rtmo.py 的指標（人數不同幾張、框差、關鍵點差…），結果存 <out>/compare_sim_<模型>_<資料>.csv。
--dfc-ranges <json>：量化點和範圍照 DFC 量化後的 HAR（hailo/har_ranges.py 匯出；規則見 dfc_quant.py），
  模擬就跟 DFC 實測對得上（2026/11/06，結果.md）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmo_post import ONNX, cut_tensors, load, to_nchw, with_cut_outputs   # noqa: E402

ROLES = ("cls16", "box16", "vis16", "feat16", "cls32", "box32", "vis32", "feat32")
LAYER_OPS = ("Add", "Concat", "MaxPool", "Resize", "Softmax", "MatMul", "Gemm")


def session(model, optimize=True):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def front(path):
    """ONNX → 前段（輸入到 8 個切點）、batch 固定 1；回傳 (模型, 前段節點, 卷積節點, 要量化的張量)。"""
    m = onnx.load(str(path))
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    cut = cut_tensors(m)
    fm = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(m)).extract_model(["input"], cut)
    nodes = list(fm.graph.node)
    consts = {i.name for i in fm.graph.initializer} | {n.output[0] for n in nodes if n.op_type == "Constant"}
    users = {}
    for n in nodes:
        for x in n.input:
            users.setdefault(x, []).append(n)

    def silu_of(t):
        sig = [u for u in users.get(t, []) if u.op_type == "Sigmoid"]
        mul = [u for u in users.get(t, []) if sig and u.op_type == "Mul" and sig[0].output[0] in u.input]
        return mul[0].output[0] if mul else None

    convs = [n for n in nodes if n.op_type == "Conv"]
    q = [silu_of(c.output[0]) or c.output[0] for c in convs]
    q += [n.output[0] for n in nodes if n.op_type in LAYER_OPS and not any(x in consts for x in n.input)]
    inf = onnx.shape_inference.infer_shapes(fm)
    etype = {v.name: v.type.tensor_type.elem_type for v in list(inf.graph.value_info) + list(inf.graph.output)}
    q = [t for t in dict.fromkeys(q) if etype.get(t) == TensorProto.FLOAT]
    return fm, convs, q


def build(path, calib, w="tensor", a="8", float_convs=(), dfc_json=None):
    """回傳模擬量化的前段（onnxruntime session，輸出 8 個切點的張量）。
    dfc_json：照 DFC 量化後的 HAR（har_ranges.py 匯出的 JSON）決定量化點和範圍（dfc_quant.dfc_points）。"""
    fm, convs, q = front(path)
    act_of = {i: t for i, t in enumerate(q[:len(convs)], 1)}   # 第 i 個卷積的輸出
    skip = {act_of[i] for i in float_convs}
    fixed = {}
    if dfc_json:
        from dfc_quant import dfc_points
        inf = onnx.shape_inference.infer_shapes(fm)
        floats = {v.name for v in list(inf.graph.value_info) + list(inf.graph.output)
                  if v.type.tensor_type.elem_type == TensorProto.FLOAT}
        raw = {c.output[0] for i, c in enumerate(convs, 1) if i in set(float_convs)}
        fixed, add, drop = dfc_points(dfc_json, list(fm.graph.node), q, floats, skip | raw)
        n0 = len(q)
        q = [t for t in q if t not in drop] + add
        print(f"  照 DFC：量化點 {n0} → {len(q)}（加 {len(add)}、拿掉 {len(drop)}），用 DFC 範圍的 {len(fixed)} 個", flush=True)
    lo, hi = {}, {}
    if a == "8":
        probe = onnx.ModelProto()
        probe.CopyFrom(fm)
        del probe.graph.output[:]
        for t in q:
            probe.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
        sp = session(probe)
        for x in calib:
            for t, v in zip(q, sp.run(None, {"input": to_nchw(x)})):
                lo[t] = min(lo.get(t, 0.0), float(v.min()))
                hi[t] = max(hi.get(t, 0.0), float(v.max()))
        for t, (f_lo, f_hi) in fixed.items():
            if t in lo:
                lo[t], hi[t] = f_lo, f_hi
    qset = {t for t in q if t in lo and t not in skip}
    m = onnx.ModelProto()
    m.CopyFrom(fm)
    g = m.graph
    if w != "float":
        keep = {c.input[1] for i, c in enumerate(convs, 1) if i in set(float_convs)}
        wnames = ({c.input[1] for c in convs} - keep) | {x for n in g.node if n.op_type in ("MatMul", "Gemm")
                                                          for x in n.input if x in {i.name for i in g.initializer}}
        for init in g.initializer:
            if init.name in wnames:
                W = numpy_helper.to_array(init).astype(np.float32)
                if w == "channel" and W.ndim == 4:
                    mx = np.abs(W).reshape(W.shape[0], -1).max(1).reshape(-1, 1, 1, 1)
                else:
                    mx = np.abs(W).max()
                mx = np.maximum(mx, 1e-12)
                init.CopyFrom(numpy_helper.from_array((np.round(W / mx * 127) * mx / 127).astype(np.float32), init.name))
    new = []
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in qset:
                n.input[i] = x + "__dq"
        new.append(n)
        for o in n.output:
            if o in qset:
                sc = max(hi[o] - lo[o], 1e-8) / 255
                zp = int(np.clip(round(-lo[o] / sc), 0, 255))
                g.initializer.extend([numpy_helper.from_array(np.array(sc, np.float32), o + "__s"),
                                      numpy_helper.from_array(np.array(zp, np.uint8), o + "__z")])
                new.append(helper.make_node("QuantizeLinear", [o, o + "__s", o + "__z"], [o + "__q"]))
                new.append(helper.make_node("DequantizeLinear", [o + "__q", o + "__s", o + "__z"], [o + "__dq"]))
    del g.node[:]
    g.node.extend(new)
    for o in g.output:
        if o.name in qset:
            o.name += "__dq"
    return session(m, optimize=False), len(qset), m


def full_model(front_q):
    """模擬量化的前段＋主機端的後段（rtmo_post.POST）接成一個完整的 ONNX：輸入 input、輸出 dets、keypoints，
    主流程（infant_face_v3.load_rtmo）可以直接載入，用來估「上晶片之後」三套評估會變怎樣。"""
    from onnx import compose
    from rtmo_post import POST
    post = compose.add_prefix(onnx.load(str(POST)), "post_")
    io_map = [(o.name, "post_" + o.name.removesuffix("__dq")) for o in front_q.graph.output]
    for m in (front_q, post):
        m.ir_version = min(front_q.ir_version, post.ir_version)
    merged = compose.merge_models(front_q, post, io_map=io_map)
    for o in merged.graph.output:      # dets 在後段裡還會用到（DCC 要用 NMS 後的框），接 Identity 改名，不直接改
        merged.graph.node.append(helper.make_node("Identity", [o.name], [o.name.removeprefix("post_")]))
        o.name = o.name.removeprefix("post_")
    return merged


def outputs(sess, data):
    """模擬量化的前段跑 data（N×640×640×3）→ 8 個 (N, C, H, W)。"""
    outs = [sess.run(None, {"input": to_nchw(x)}) for x in data]
    return [np.concatenate([o[k] for o in outs]) for k in range(8)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=ONNX, help="要量化的模型（預設原本的 RTMO-m）；標準答案一律是原本的")
    ap.add_argument("--w", choices=("float", "tensor", "channel"), default="tensor")
    ap.add_argument("--a", choices=("float", "8"), default="8")
    ap.add_argument("--float-convs", default="", help="例如 1,2,4")
    ap.add_argument("--calib", nargs="+", default=["rtmo_coco.npy:16", "rtmo_doll_calib.npy:16"])
    ap.add_argument("--data", nargs="+", default=["rtmo_doll_hold.npy:12", "rtmo_coco_hold.npy:12"], help="算 SNR 用")
    ap.add_argument("--compare", nargs="+", help="改用 compare_rtmo.py 的指標比這些資料")
    ap.add_argument("--export-full", type=Path, help="存成完整的 ONNX（模擬量化的前段＋主機後段），給主流程跑評估用")
    ap.add_argument("--dfc-ranges", type=Path, help="量化點和範圍照 DFC 量化後的 HAR（hailo/har_ranges.py 匯出的 JSON）")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "hailo")
    args = ap.parse_args()
    calib = np.concatenate([load(s)[0] for s in args.calib])
    sess, n, front_q = build(args.onnx, calib, args.w, args.a, [int(k) for k in args.float_convs.split(",") if k],
                             args.dfc_ranges)
    print(f"{args.onnx.name}｜權重 {args.w}、啟動值 {args.a}、不量化的卷積 [{args.float_convs}]｜量化 {n} 個張量"
          + (f"｜範圍用 DFC 的（{args.dfc_ranges.name}）" if args.dfc_ranges else ""), flush=True)
    if args.export_full:
        merged = full_model(front_q)
        onnx.save(merged, str(args.export_full))
        ref = session(onnx.load(str(ONNX)))
        x = to_nchw(load("rtmo_doll_hold.npy:1")[0][0])
        got = session(merged, optimize=False).run(None, {"input": x})
        print(f"  存成 {args.export_full}：輸出 {[o.name for o in merged.graph.output]}，形狀 {[g.shape for g in got]}"
              f"（原本 {[r.shape for r in ref.run(None, {'input': x})]}）")
        return
    full = session(with_cut_outputs(onnx.load(str(ONNX))))
    if not args.compare:
        data = np.concatenate([load(s)[0] for s in args.data])
        got = outputs(sess, data)
        num, den = np.zeros(8), np.zeros(8)
        for k, x in enumerate(data):
            for i, r in enumerate(full.run(None, {"input": to_nchw(x)})[2:]):
                num[i] += float((r ** 2).sum())
                den[i] += float(((r - got[i][k:k + 1]) ** 2).sum())
        s = 10 * np.log10(num / np.maximum(den, 1e-20))
        print("  SNR dB：" + "、".join(f"{r} {v:.1f}" for r, v in zip(ROLES, s)) + f"｜平均 {s.mean():.1f}")
        return
    import compare_rtmo
    from rtmo_post import Post
    args.out.mkdir(parents=True, exist_ok=True)
    post = Post()
    names = dict(zip(cut_tensors(onnx.load(str(ONNX))), ROLES))
    for spec in args.compare:
        rows, snr, out = compare_rtmo.compare(Path(f"sim_{args.onnx.stem}"), spec, args, full, post, names,
                                              outputs=lambda d: outputs(sess, d))
        print(f"\n=== 模擬量化 {args.onnx.name} × {spec} → {out}")
        print("  全部    " + compare_rtmo.summarize(rows))
        for cond in sorted({r["cond"] for r in rows}):
            print(f"  {cond:6s}  " + compare_rtmo.summarize([r for r in rows if r["cond"] == cond]))
        print("  SNR dB：" + "、".join(f"{k} {v:.1f}" for k, v in snr.items()))


if __name__ == "__main__":
    main()
