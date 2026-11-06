"""
hailo/rtmo_post.py
RTMO-m 切成兩段：前段（骨幹、neck、偵測頭的卷積）編成 HEF 放晶片；後段（框的解碼、NMS、關鍵點的座標分類 DCC）
在主機用 onnxruntime 跑。這個檔案負責後段：從原本的 ONNX 取出後段（--split），驗證切開後結果不變（--check），
以及主機上執行後段的 Post 類別（給 compare_rtmo.py 和之後的主流程用）。

為什麼這樣切：RTMO 的 ONNX 把框的解碼、NMS 都包在裡面，NMS 之後的人數 N 是動態的，Hailo 要固定形狀。
NMS 之前，偵測頭每個 stride（16、32）各有 4 個卷積輸出（分數、框、關鍵點可見度、給 DCC 的特徵），形狀固定，
所以切在這 8 個卷積。後段直接從原本的 ONNX 取出來、不自己重寫：關鍵點的 DCC 要用 NMS 後的框產生座標分類的格子，
重寫容易出錯；取出來的後段跟原本的模型是同一份運算。

  uv run --with onnx python -m hailo.rtmo_post --split --check rtmo_doll_hold.npy:64 rtmo_coco_hold.npy:32
  （也可以在 WSL 的 DFC 虛擬環境跑：python ".../hailo/rtmo_post.py" ...；要 onnx、onnxruntime）

--split 把後段存成 model/rtmo-m_post.onnx（模型權重，不進 repo）；--check 用 prep_calib.py 存的圖驗證：
原本的 ONNX 直接跑 vs 切點的張量餵給後段，dets、keypoints 要一樣。
"""

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# rtmlib 下載的位置（Windows 的 %USERPROFILE%\.cache；WSL 裡是 /mnt/c/Users/<名字>/.cache）。repo 放在使用者資料夾下面
ONNX = REPO.parent / ".cache" / "rtmlib" / "hub" / "checkpoints" / "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.onnx"
POST = REPO / "model" / "rtmo-m_post.onnx"
CALIB_DIR = REPO / "hailo" / "calib"
# 偵測頭最後的 8 個卷積（ONNX 的節點名稱）：stride 16 的分數、框、可見度、DCC 特徵，再來是 stride 32 的
CUT_NODES = ("Conv_412", "Conv_413", "Conv_414", "Conv_415", "Conv_435", "Conv_436", "Conv_437", "Conv_438")
OUTPUTS = ("dets", "keypoints")


class Post:
    """主機端的後段：吃前段的 8 個輸出，回傳跟原本 ONNX 一樣的 dets [1,N,5]、keypoints [1,N,17,3]（640×640 輸入的座標）。"""

    def __init__(self, path=POST):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = [(i.name, tuple(i.shape[1:])) for i in self.sess.get_inputs()]   # (C, H, W)

    def feed(self, outs):
        """前段的輸出 → 後段的輸入。outs 可以是 NCHW（onnxruntime）或 NHWC（DFC 模擬器、HailoRT），
        有沒有 batch 維度都可以，順序不拘：用形狀對應（8 個張量的形狀都不一樣）。"""
        pool, feed = [np.asarray(o, np.float32) for o in outs], {}
        for name, (c, h, w) in self.inputs:
            for k, o in enumerate(pool):
                a = o[0] if o.ndim == 4 else o
                if a.shape == (c, h, w):
                    feed[name] = a[None]
                    break
                if a.shape == (h, w, c):
                    feed[name] = np.ascontiguousarray(a.transpose(2, 0, 1))[None]
                    break
            else:
                raise ValueError(f"前段輸出裡找不到形狀 {(c, h, w)} 的張量：{[o.shape for o in pool]}")
            pool.pop(k)
        return feed

    def __call__(self, outs):
        return self.sess.run(list(OUTPUTS), self.feed(outs))


def cut_tensors(model):
    nodes = {n.name: n for n in model.graph.node}
    for name in CUT_NODES:
        if nodes[name].op_type != "Conv":
            raise SystemExit(f"{name} 不是 Conv（ONNX 版本不對？）")
    return [nodes[name].output[0] for name in CUT_NODES]


def with_cut_outputs(model):
    """原本的 ONNX 多輸出 8 個切點的張量（輸入固定成 batch 1）。"""
    import copy
    from onnx import TensorProto, helper
    m = copy.deepcopy(model)
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    for t in cut_tensors(m):
        m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    return m


def const_value(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            from onnx import numpy_helper
            return numpy_helper.to_array(init)
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            from onnx import numpy_helper
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def split(sample):
    """取出後段存成 POST；sample 是一張 640×640 的圖，用來量切點張量的形狀。"""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    model = onnx.load(str(ONNX))
    full = with_cut_outputs(model)
    sess = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": to_nchw(sample)})
    names = [o.name for o in sess.get_outputs()]
    shapes = {n: o.shape for n, o in zip(names, outs)}
    cut = cut_tensors(model)

    # 切點張量補上形狀（Extractor 要），再從原本的模型取出「切點 → dets、keypoints」這一段
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    m = onnx.shape_inference.infer_shapes(model)
    keep = [v for v in m.graph.value_info if v.name not in cut]
    del m.graph.value_info[:]
    m.graph.value_info.extend(keep)
    m.graph.value_info.extend(helper.make_tensor_value_info(t, TensorProto.FLOAT, list(shapes[t])) for t in cut)
    post = onnx.utils.Extractor(m).extract_model(cut, list(OUTPUTS))
    used = {i for n in post.graph.node for i in n.input}
    if "input" in used or any(n.op_type == "Conv" for n in post.graph.node):
        raise SystemExit("後段還用到前段的東西（切點不完整）")
    onnx.checker.check_model(post)
    POST.parent.mkdir(exist_ok=True)
    onnx.save(post, str(POST))

    print(f"後段存在 {POST}：{len(post.graph.node)} 個節點（原本 {len(model.graph.node)} 個）")
    total = 0
    for node, t in zip(CUT_NODES, cut):
        s = shapes[t]
        total += int(np.prod(s))
        print(f"  切點 {node:9s} {t:22s} {list(s)}")
    print(f"  前段每張圖輸出 {total} 個數值（8 位元約 {total / 1024:.0f} KB）")
    for n in post.graph.node:   # 後段裡寫死的參數
        if n.op_type == "NonMaxSuppression":
            vals = [const_value(post, i) for i in n.input[2:]]
            print("  NMS：每類最多留 {}、IoU 門檻 {}、分數門檻 {}".format(*[None if v is None else v.tolist() for v in vals]))
        if n.op_type == "TopK":
            print(f"  {n.name}：k 來自 {n.input[1]}")
    for n in post.graph.node:
        if n.op_type in ("Less", "Where", "Add") and any(const_value(post, i) is not None and const_value(post, i).size == 1
                                                         for i in n.input):
            vals = {i: const_value(post, i).tolist() for i in n.input if const_value(post, i) is not None
                    and const_value(post, i).size == 1}
            print(f"  {n.name}（{n.op_type}）常數 {vals}")
    return post


def to_nchw(img):
    """prep_calib 存的 640×640 BGR uint8 → RTMO 的輸入（rtmlib 不做正規化）。"""
    return np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]


def load(spec):
    name, _, n = spec.partition(":")
    path = CALIB_DIR / name if not Path(name).is_absolute() else Path(name)
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr[:int(n)] if n else arr)
    with open(path.with_name(path.stem + "_meta.csv"), encoding="utf-8-sig") as fh:
        meta = list(csv.DictReader(fh))[:len(arr)]
    return arr, meta


def check(specs):
    """原本的 ONNX vs 「原本的 ONNX 輸出切點張量 → 後段」：dets、keypoints 要一樣。"""
    import onnx
    import onnxruntime as ort
    model = onnx.load(str(ONNX))
    orig = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    full = ort.InferenceSession(with_cut_outputs(model).SerializeToString(), providers=["CPUExecutionProvider"])
    post = Post()
    for spec in specs:
        data, _ = load(spec)
        worst, n_same, people = 0.0, 0, 0
        for img in data:
            x = to_nchw(img)
            ref = orig.run(list(OUTPUTS), {"input": x})
            outs = full.run(None, {"input": x})
            got = post(outs[2:])
            same = all(r.shape == g.shape for r, g in zip(ref, got))
            n_same += same
            people += int((ref[0][0, :, 4] >= 0.3).sum())
            if same and ref[0].size:
                worst = max(worst, max(float(np.abs(r - g).max()) for r, g in zip(ref, got)))
        print(f"  {spec}：{len(data)} 張，人數（N）一樣 {n_same} 張，分數 ≥ 0.3 的人 {people} 個，"
              f"dets／keypoints 最大差 {worst:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="store_true", help="取出後段存成 model/rtmo-m_post.onnx")
    ap.add_argument("--check", nargs="*", default=[], help="驗證用的圖（prep_calib.py 存的 .npy，「檔名:張數」）")
    args = ap.parse_args()
    if args.split:
        sample = load(args.check[0])[0][0] if args.check else np.full((640, 640, 3), 114, np.uint8)
        split(sample)
    if args.check:
        check(args.check)


if __name__ == "__main__":
    main()
