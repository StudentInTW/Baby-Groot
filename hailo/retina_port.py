"""
hailo/retina_port.py
v7 的 RetinaFace：把 Hailo Model Zoo 給的 retinaface_mobilenet_v1 TFLite（輸入固定 736×1280）照運算圖一層一層搬到 PyTorch
（權重原封不動，BN 已經併在卷積裡），再匯出固定 640×480（高×寬）的 ONNX。
為什麼要搬：DFC 5.x 不接受 TFLite 改輸入大小（translate_tf_model 的 tensor_shapes 已停用），
而床位轉正、縮到長邊 640 後多半是寬 320–480、高 640，736×1280 的輸入大半是補 0，晶片上一次 38 ms；640×480 是 16 ms。

TFLite 下載（Hailo Model Zoo 的 retinaface_mobilenet_v1.yaml 指的檔案，MIT），解壓到 model/retinaface_mobilenet_v1_hd/：
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceDetection/retinaface_mobilenet_v1_hd/2023-07-18/retinaface_mobilenet_v1_hd.zip

  .venv-vlm/Scripts/python -m hailo.retina_port            # 匯出（要 PyTorch、onnx 和 tflite（只解析檔案，不跑 TensorFlow））
  uv run --with ai-edge-litert python -m hailo.retina_port --check   # 核對：736×1280 的 ONNX 跟 TFLite 逐個輸出比
輸出：model/retinaface_v1_640x480.onnx（主流程、DFC 用）、model/retinaface_v1_736x1280.onnx（只給 --check 核對）。
輸入 1×3×H×W，BGR 減 (104, 117, 123)（跟 TFLite 一樣；HEF 的正規化寫在 model script，見 compile_hef.py）；
輸出 9 個卷積（NCHW；每層框 8、分類 4、特徵點 20 通道），解碼在 pipeline/infant_face_v7.py。
2026/11/27 核對：16 張床位畫面、9 個輸出最大差 1.2e-4（除以該輸出最大值 1.3e-5），就是浮點誤差。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TFL = ROOT / "model" / "retinaface_mobilenet_v1_hd" / "retinaface_mobilenet720p.tflite"
OUT = {(640, 480): ROOT / "model" / "retinaface_v1_640x480.onnx", (736, 1280): ROOT / "model" / "retinaface_v1_736x1280.onnx"}


def export():
    import tflite
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ops = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
    dtypes = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.INT32: np.int32}
    act = {0: None, 1: "relu", 3: "relu6"}
    m = tflite.Model.GetRootAsModel(TFL.read_bytes(), 0)
    g = m.Subgraphs(0)

    def name(i):
        return g.Tensors(i).Name().decode()

    def const(i):
        t = g.Tensors(i)
        return np.frombuffer(m.Buffers(t.Buffer()).DataAsNumpy().tobytes(), dtypes[t.Type()]).reshape(t.ShapeAsNumpy())

    def options(kind, op):
        cls = {"CONV_2D": tflite.Conv2DOptions, "DEPTHWISE_CONV_2D": tflite.DepthwiseConv2DOptions,
               "LEAKY_RELU": tflite.LeakyReluOptions, "CONCATENATION": tflite.ConcatenationOptions,
               "ADD": tflite.AddOptions}.get(kind)
        if cls is None or op.BuiltinOptions() is None:
            return None
        o, t = cls(), op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        return o

    class Port(nn.Module):
        """TFLite 的運算照順序跑。卷積前的 PAD 都是上下左右各 1，併成卷積的 padding=1（運算一樣，DFC 比較好接）。"""

        def __init__(self):
            super().__init__()
            self.steps, self.convs, pads = [], nn.ModuleList(), {}
            self.inp, self.outs = int(g.InputsAsNumpy()[0]), [int(i) for i in g.OutputsAsNumpy()]
            self.out_names = [name(i).replace("/", "_").replace(".", "_") for i in self.outs]
            for k in range(g.OperatorsLength()):
                op = g.Operators(k)
                oc = m.OperatorCodes(op.OpcodeIndex())
                kind = ops[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]
                ins, outs, o = list(op.InputsAsNumpy()), list(op.OutputsAsNumpy()), options(kind, op)
                if kind == "PAD":
                    p = const(ins[1])
                    if not ((p[1] == 1).all() and (p[2] == 1).all() and not p[0].any() and not p[3].any()):
                        raise SystemExit(f"PAD 不是上下左右各 1：{name(outs[0])} {p.tolist()}")
                    pads[outs[0]] = ins[0]
                elif kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                    w, b = const(ins[1]), const(ins[2])
                    if kind == "CONV_2D":                       # [O, H, W, I]
                        co, kh, kw, ci = w.shape
                        conv = nn.Conv2d(ci, co, (kh, kw), (o.StrideH(), o.StrideW()))
                        conv.weight.data = torch.from_numpy(w.transpose(0, 3, 1, 2).copy())
                    else:                                       # [1, H, W, C]
                        _, kh, kw, c = w.shape
                        assert o.DepthMultiplier() in (0, 1)
                        conv = nn.Conv2d(c, c, (kh, kw), (o.StrideH(), o.StrideW()), groups=c)
                        conv.weight.data = torch.from_numpy(w.transpose(3, 0, 1, 2).copy())
                    conv.bias.data = torch.from_numpy(b.copy())
                    assert o.Padding() == tflite.Padding.VALID or kh == kw == 1, name(outs[0])
                    src = ins[0]
                    if src in pads:
                        src, conv.padding = pads[src], (1, 1)
                    elif kh > 1:
                        raise SystemExit(f"3×3 卷積前面沒有 PAD：{name(outs[0])}")
                    self.convs.append(conv)
                    self.steps.append(("CONV", [src], outs[0], len(self.convs) - 1, act[o.FusedActivationFunction()]))
                elif kind == "LEAKY_RELU":
                    self.steps.append(("LEAKY", ins[:1], outs[0], float(o.Alpha())))
                elif kind == "RELU":
                    self.steps.append(("RELU", ins[:1], outs[0]))
                elif kind == "ADD":
                    self.steps.append(("ADD", ins[:2], outs[0], act[o.FusedActivationFunction()]))
                elif kind == "CONCATENATION":
                    assert o.Axis() in (3, -1)                  # NHWC 的通道 → NCHW 的第 1 維
                    self.steps.append(("CAT", ins, outs[0], act[o.FusedActivationFunction()]))
                elif kind == "RESIZE_NEAREST_NEIGHBOR":         # FPN 上採樣：兩倍（各層都整除，等於對齊到下一層的大小）
                    self.steps.append(("UP2", ins[:1], outs[0]))
                else:
                    raise SystemExit(f"沒處理的運算 {kind}（{name(outs[0])}）")

        @staticmethod
        def act(x, a):
            return F.relu(x) if a == "relu" else (F.relu6(x) if a == "relu6" else x)

        def forward(self, x):
            v = {self.inp: x}
            for kind, ins, out, *arg in self.steps:
                xs = [v[i] for i in ins]
                if kind == "CONV":
                    v[out] = self.act(self.convs[arg[0]](xs[0]), arg[1])
                elif kind == "LEAKY":
                    v[out] = F.leaky_relu(xs[0], arg[0])
                elif kind == "RELU":
                    v[out] = F.relu(xs[0])
                elif kind == "ADD":
                    v[out] = self.act(xs[0] + xs[1], arg[0])
                elif kind == "CAT":
                    v[out] = self.act(torch.cat(xs, 1), arg[0])
                elif kind == "UP2":
                    v[out] = F.interpolate(xs[0], scale_factor=2.0, mode="nearest")
            return tuple(v[i] for i in self.outs)

    net = Port().eval()
    print(f"{TFL.name}：{len(net.steps)} 步、卷積 {len(net.convs)} 層、輸出 {len(net.outs)} 個")
    for (H, W), path in OUT.items():
        with torch.no_grad():
            torch.onnx.export(net, (torch.zeros(1, 3, H, W),), str(path), input_names=["input"], output_names=net.out_names,
                              opset_version=13, dynamo=False, do_constant_folding=True)
        print(f"→ {path}（{path.stat().st_size / 1e6:.1f} MB）")


def check(n=16):
    """736×1280 的 ONNX 跟 TFLite 逐個輸出比（輸入用 hailo/calib/retina_calib.npy 的床位畫面，沒有就用隨機圖）。"""
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    from pipeline.infant_face_v7 import MEAN_BGR, letterbox
    it = Interpreter(str(TFL))
    it.allocate_tensors()
    inp = it.get_input_details()[0]["index"]
    tfl_out = {d["name"].replace("/", "_").replace(".", "_"): d["index"] for d in it.get_output_details()}
    sess = ort.InferenceSession(str(OUT[(736, 1280)]), providers=["CPUExecutionProvider"])
    calib = ROOT / "hailo" / "calib" / "retina_calib.npy"
    imgs = (np.load(calib)[:n, :, :, ::-1] if calib.exists()
            else np.random.default_rng(0).integers(0, 256, (n, 640, 480, 3), dtype=np.uint8))
    worst = rel = 0.0
    for img in imgs:
        x = letterbox(np.ascontiguousarray(img), 1280, 736)[0].astype(np.float32) - MEAN_BGR
        it.set_tensor(inp, x[None])
        it.invoke()
        for o, out in zip(sess.get_outputs(), sess.run(None, {"input": x.transpose(2, 0, 1)[None].copy()})):
            ref = it.get_tensor(tfl_out[o.name])
            d = float(np.abs(out.transpose(0, 2, 3, 1) - ref).max())
            worst, rel = max(worst, d), max(rel, d / (float(np.abs(ref).max()) + 1e-9))
    print(f"{len(imgs)} 張、{len(tfl_out)} 個輸出：最大差 {worst:.2e}（除以該輸出最大值 {rel:.2e}）")


if __name__ == "__main__":
    if not TFL.exists():
        sys.exit(f"找不到 {TFL}（下載網址見檔頭）")
    check() if "--check" in sys.argv else export()
