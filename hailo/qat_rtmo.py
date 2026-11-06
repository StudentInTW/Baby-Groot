"""RTMO-m 的量化感知訓練（QAT）：讓前段量化成 8 位元之後，輸出還是貼近原本的 RTMO-m。

為什麼：直接用 DFC 量化，前段 8 個輸出的 SNR 只有 0.4–6.7 dB；前幾層改 16 位元、DFC 內建的微調（finetune）、
啟動值裁切、權重分組都救不回來（hailo/結果.md「RTMO-m」）。在 ONNX 上逐層加誤差的測試顯示，網路從頭到尾都對
誤差很敏感（前幾層最嚴重），不是少數幾層的問題，所以要讓網路自己學會耐得住 8 位元。

做法（不用標註，只對齊原本模型的輸出）：
- 老師：原本的 RTMO-m 前段（輸入 → 偵測頭最後 8 個卷積，跟 HEF 切的位置一樣），浮點、不訓練。
- 學生：同一份權重，前向時模擬 DFC 的 8 位元量化（規則同 sim_quant.py，在 ONNX 上跟 DFC 的實測對得上）：
  權重每層一組範圍、對稱 8 位元；每一層的輸出一組範圍、非對稱 8 位元，範圍＝校正圖上的最小～最大。
  反向傳播用 straight-through estimator。每個 epoch 開始時，用 DFC 會用的同一批校正圖重算範圍。
- 損失：8 個輸出各自的相對平方誤差 Σ(學生−老師)²／Σ老師²（＝最大化 SNR），另外在「老師或學生認為有人」的
  anchor 上加權再算一次（人數、框、關鍵點都只看這些 anchor）。
- 只訓練卷積的權重和 bias；transformer 的線性層不動（量化後本來就沒問題）。
- 資料只用允許商用的：COCO train2017 裡 CC BY、No known copyright 的圖（--coco-list），加上校正那一半的娃娃。
  比對那一半的娃娃、COCO val 的比對圖只用來看進度，不拿來訓練。
- 訓練完把權重寫回原本的 ONNX（節點名稱、切點都不變）：DFC 照原本的流程 parse → optimize → compile，
  主流程換成這個 ONNX 也能直接跑。

需要 PyTorch（CUDA）、onnx2torch、onnx、onnxruntime、opencv；在國網中心 nano4 的 H200 上跑，步驟見 hailo/結果.md。
  python hailo/qat_rtmo.py --coco-dir <COCO>/train2017 --coco-list <COCO>/ok_train2017.txt --out <輸出資料夾>
  python hailo/qat_rtmo.py --check            # 只檢查：轉成 PyTorch 後跟 ONNX 一不一樣、模擬量化掉多少
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from torch.nn.utils import parametrize

from rtmo_post import ONNX, cut_tensors, load

ROLES = ("cls16", "box16", "vis16", "feat16", "cls32", "box32", "vis32", "feat32")
LAYER_OPS = ("Add", "Concat", "MaxPool", "Resize", "Softmax", "MatMul", "Gemm")
SIZE = 640


# ---------- 模型：ONNX 前段 → PyTorch，量化的位置插標記 ----------

def front_onnx(path, dfc=None, float_convs=()):
    """原本的 ONNX → 前段（輸入到 8 個切點），每個「層的輸出」後面插一個 Identity 標記（FQ0、FQ1…）。

    哪些算「層的輸出」：卷積（後面接 SiLU 就取 SiLU 之後）、相加、串接、池化、放大、softmax、矩陣乘法，
    輸入不含常數的才算（排除算形狀用的運算）。DFC 把卷積＋SiLU 當成一層、輸出量化一次，其他也一樣。
    dfc：DFC 量化後的 HAR 匯出的 JSON（har_ranges.py）→ 量化點照 DFC 加減（dfc_quant.dfc_points：transformer 的線性層、
    head 拆成兩塊的卷積…），也回傳 DFC 的範圍。
    回傳 (ONNX 模型, 量化的張量名稱, 前段卷積節點名稱, 每個卷積的輸出在量化清單裡的位置（沒量化就是 None）, DFC 的範圍)。
    """
    m = onnx.load(str(path))
    cut = cut_tensors(m)
    front = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(m)).extract_model(["input"], cut)
    g = front.graph
    nodes = list(g.node)
    consts = {i.name for i in g.initializer} | {n.output[0] for n in nodes if n.op_type == "Constant"}
    users = {}
    for n in nodes:
        for x in n.input:
            users.setdefault(x, []).append(n)

    def silu_of(t):
        sig = [u for u in users.get(t, []) if u.op_type == "Sigmoid"]
        mul = [u for u in users.get(t, []) if sig and u.op_type == "Mul" and sig[0].output[0] in u.input]
        return mul[0].output[0] if mul else None

    q = []
    for n in nodes:
        if n.op_type == "Conv":
            q.append(silu_of(n.output[0]) or n.output[0])
        elif n.op_type in LAYER_OPS and not any(x in consts for x in n.input):
            q.append(n.output[0])
    inf = onnx.shape_inference.infer_shapes(front)
    etype = {v.name: v.type.tensor_type.elem_type for v in list(inf.graph.value_info) + list(inf.graph.output)}
    q = [t for t in dict.fromkeys(q) if etype.get(t) == TensorProto.FLOAT]
    ranges = {}
    if dfc:
        from dfc_quant import dfc_points
        conv_nodes = [n for n in nodes if n.op_type == "Conv"]
        skip = {t for i, c in enumerate(conv_nodes, 1) if i in float_convs
                for t in (c.output[0], silu_of(c.output[0]) or c.output[0])}
        floats = {t for t, e in etype.items() if e == TensorProto.FLOAT}
        ranges, add, drop = dfc_points(dfc, nodes, q, floats, skip)
        q = [t for t in q if t not in drop] + add
    idx = {t: i for i, t in enumerate(q)}
    conv_q = [idx.get(silu_of(n.output[0]) or n.output[0]) for n in nodes if n.op_type == "Conv"]
    new = []
    for n in nodes:
        for i, x in enumerate(n.input):
            if x in idx:
                n.input[i] = x + "__fq"
        new.append(n)
        for o in n.output:
            if o in idx:
                new.append(helper.make_node("Identity", [o], [o + "__fq"], name=f"FQ{idx[o]}"))
    del g.node[:]
    g.node.extend(new)
    for o in g.output:
        if o.name in idx:
            o.name += "__fq"
    return front, q, [n.name for n in nodes if n.op_type == "Conv"], conv_q, ranges


def widen(t, training):
    """訓練時把量化範圍隨機放大 0～JITTER 倍（每次前向都不一樣）。

    不抖動的話，學生會學成「配合這一套固定的四捨五入」去抵消誤差（2026/11/05 的第一版：
    權重改成不量化、或前幾層不量化，SNR 就掉回訓練前），DFC 的範圍、SiLU 近似只要跟模擬差一點就失效。
    範圍只放大不縮小：不會把校正圖上出現過的最大值裁掉。"""
    if not training or JITTER <= 0:
        return t
    return t * (1 + torch.rand_like(t) * JITTER)


JITTER = 0.0


class ActQ(nn.Module):
    """一層輸出的 8 位元非對稱量化（整層一組範圍）。mode：off＝不動、observe＝記錄最小／最大值、quant＝量化。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("lo", torch.zeros(1))
        self.register_buffer("hi", torch.zeros(1))
        self.mode = "off"

    def forward(self, x):
        if self.mode == "observe":
            self.lo = torch.minimum(self.lo, x.detach().min().reshape(1))
            self.hi = torch.maximum(self.hi, x.detach().max().reshape(1))
            return x
        if self.mode == "quant":
            lo, hi = widen(self.lo, self.training), widen(self.hi, self.training)
            scale = ((hi - lo) / 255).clamp_min(1e-8)
            zp = torch.round(-lo / scale).clamp(0, 255).to(torch.int32)
            return torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)
        return x


class WeightQ(nn.Module):
    """權重的 8 位元對稱量化（整層一組範圍＝最大絕對值，DFC 的預設；DFC 的 HAR 裡 conv1 的權重是 −101～127，同一套）。"""
    enabled = True

    def forward(self, w):
        if not WeightQ.enabled:
            return w
        scale = widen((w.detach().abs().max() / 127).clamp_min(1e-12).reshape(1), self.training)
        return torch.fake_quantize_per_tensor_affine(w, scale, torch.zeros(1, dtype=torch.int32, device=w.device), -127, 127)


def build(path, quantize, float_convs=(), dfc=None):
    """回傳 (PyTorch 模型, 量化標記模組 list, 卷積 {ONNX 節點名稱: 模組})。quantize=False 就是老師。
    float_convs：第幾個卷積（ONNX 裡的順序，1 起算）不量化，對應 DFC 的 quantization_param(..., precision_mode=a16_w16)。
    dfc：DFC 量化後的 HAR 匯出的 JSON。量化點照 DFC（dfc_quant.py），校正後改用 DFC 的範圍（calibrate）：
    只用校正圖算範圍的話，模擬比 DFC 樂觀 2–5 dB，訓練越久越嚴重（2026/11/06，結果.md）。"""
    from onnx2torch import convert
    front, q, conv_names, conv_q, fixed = front_onnx(path, dfc, float_convs if quantize else ())
    gm = convert(front)
    mods = dict(gm.named_modules())
    acts = []
    for i in range(len(q)):
        name = next(k for k in mods if k.split(".")[-1] == f"FQ{i}")
        parent = gm.get_submodule(name.rpartition(".")[0]) if "." in name else gm
        a = ActQ()
        a.fixed = fixed.get(q[i])
        setattr(parent, name.rpartition(".")[2], a)
        acts.append(a)
    if dfc:
        print(f"照 DFC（{Path(dfc).name}）：量化位置 {len(acts)} 個，用 DFC 範圍的 {sum(a.fixed is not None for a in acts)} 個", flush=True)
    gm.recompile()
    mods = dict(gm.named_modules())
    convs = {}
    for n in conv_names:
        key = next((k for k in mods if k.split(".")[-1] == n or k == n), None)
        if key is None or not isinstance(mods[key], nn.Conv2d):
            raise SystemExit(f"找不到 {n} 對應的 Conv2d（onnx2torch 的命名改了？）")
        convs[n] = mods[key]
    for p in gm.parameters():
        p.requires_grad_(False)
    if quantize:
        for i, c in enumerate(convs.values(), 1):
            if i in float_convs:     # DFC 的 16 位元（a16_w16）：權重和輸出都不量化，近似成浮點
                if conv_q[i - 1] is not None:
                    acts[conv_q[i - 1]].keep_float = True
            else:
                parametrize.register_parametrization(c, "weight", WeightQ())
            for p in c.parameters():
                p.requires_grad_(True)
        for lin in (m for m in gm.modules() if isinstance(m, nn.Linear)):
            parametrize.register_parametrization(lin, "weight", WeightQ())
    return gm, acts, convs


def set_mode(acts, mode):
    for a in acts:
        a.mode = "off" if mode == "quant" and getattr(a, "keep_float", False) else mode


@torch.no_grad()
def calibrate(model, acts, images, batch):
    """DFC 的做法：浮點跑一遍校正圖，每層的範圍＝最小～最大（含 0）。"""
    for a in acts:
        a.lo.zero_()
        a.hi.zero_()
    set_mode(acts, "observe")
    WeightQ.enabled = False
    for i in range(0, len(images), batch):
        model(to_gpu(images[i:i + batch]))
    WeightQ.enabled = True
    for a in acts:
        if getattr(a, "fixed", None) is not None:   # --dfc-ranges：改用 DFC 量化後的範圍
            a.lo.fill_(a.fixed[0])
            a.hi.fill_(a.fixed[1])
    set_mode(acts, "quant")


# ---------- 資料 ----------

def letterbox(img, shrink=1.0):
    """同 rtmlib 的 RTMO.preprocess：等比例縮放、放左上角、其餘補 114；shrink < 1 讓內容變小（資料增強）。"""
    h, w = img.shape[:2]
    r = min(SIZE / h, SIZE / w) * shrink
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    rs = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_LINEAR)
    out[:rs.shape[0], :rs.shape[1]] = rs
    return out


def degrade(img, rng):
    """夜視鏡頭的近似：灰階、變暗（evaluation/degrade.py 的條件）。"""
    if rng.random() < 0.25:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.2:
        img = (img.astype(np.float32) * rng.uniform(0.3, 0.8)).astype(np.uint8)
    return img


class Images(torch.utils.data.Dataset):
    def __init__(self, coco_dir, coco_list, doll, doll_repeat):
        names = [ln.split()[0] for ln in open(coco_list, encoding="utf-8") if ln.strip()]
        self.coco = [Path(coco_dir) / n for n in names if (Path(coco_dir) / n).exists()]
        self.doll = doll
        self.items = [("coco", i) for i in range(len(self.coco))] + [("doll", i) for i in range(len(doll))] * doll_repeat

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, j = self.items[i]
        rng = random.Random()
        if kind == "coco":
            img = cv2.imread(str(self.coco[j]))
            if rng.random() < 0.5:
                img = img[:, ::-1]
            x = letterbox(degrade(np.ascontiguousarray(img), rng), rng.uniform(0.4, 1.0) if rng.random() < 0.5 else 1.0)
        else:
            x = np.asarray(self.doll[j])
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def to_gpu(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
    return x.cuda(non_blocking=True).float()


class precise:
    """老師用完整的 FP32 算。GPU 的卷積預設用 TF32（尾數 10 位元），RTMO 對誤差敏感，輸出會跟 onnxruntime 差到 0.1–0.3；
    學生本來就帶 8 位元的誤差，用 TF32 比較快。"""

    def __enter__(self):
        self.old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = self.old


# ---------- 損失與評估 ----------

def person_weight(t_cls, s_cls):
    return torch.maximum(torch.sigmoid(t_cls), torch.sigmoid(s_cls.detach()))


def loss_fn(s_outs, t_outs):
    total, parts = 0.0, []
    w = {0: person_weight(t_outs[0], s_outs[0]), 4: person_weight(t_outs[4], s_outs[4])}
    for k, (s, t) in enumerate(zip(s_outs, t_outs)):
        d, t2 = (s - t) ** 2, t ** 2
        rel = (d.flatten(1).sum(1) / t2.flatten(1).sum(1).clamp_min(1e-6)).mean()
        wk = w[0 if k < 4 else 4]
        relw = ((d * wk).flatten(1).sum(1) / (t2 * wk).flatten(1).sum(1).clamp_min(1e-6)).mean()
        total = total + rel + relw
        parts.append(float(rel))
    return total, parts


class float_mode:
    """學生暫時不量化（啟動值、權重都用浮點），算浮點那一路的損失和 SNR 用（--float-weight）。"""

    def __init__(self, acts):
        self.acts = acts

    def __enter__(self):
        set_mode(self.acts, "off")
        WeightQ.enabled = False

    def __exit__(self, *exc):
        WeightQ.enabled = True
        set_mode(self.acts, "quant")


@torch.no_grad()
def snr(student, teacher, images, batch):
    num, den = np.zeros(8), np.zeros(8)
    for i in range(0, len(images), batch):
        x = to_gpu(images[i:i + batch])
        with precise():
            t_outs = teacher(x)
        for k, (s, t) in enumerate(zip(student(x), t_outs)):
            num[k] += float((t.double() ** 2).sum())
            den[k] += float(((s.double() - t.double()) ** 2).sum())
    return 10 * np.log10(num / np.maximum(den, 1e-20))


def fmt(v):
    return " ".join(f"{r} {x:.1f}" for r, x in zip(ROLES, v)) + f"｜平均 {np.mean(v):.1f}"


# ---------- 存檔：權重寫回原本的 ONNX ----------

def export(convs, src, dst):
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    nodes = {n.name: n for n in m.graph.node}
    for name, conv in convs.items():
        n = nodes[name]
        w = conv.parametrizations.weight.original if parametrize.is_parametrized(conv, "weight") else conv.weight
        inits[n.input[1]].CopyFrom(numpy_helper.from_array(w.detach().cpu().numpy().astype(np.float32), n.input[1]))
        if len(n.input) > 2 and n.input[2]:
            inits[n.input[2]].CopyFrom(numpy_helper.from_array(conv.bias.detach().cpu().numpy().astype(np.float32), n.input[2]))
    onnx.save(m, str(dst))


def check(args):
    """轉成 PyTorch 後（不量化）跟 onnxruntime 一不一樣；模擬量化（訓練前）掉多少。"""
    import onnxruntime as ort
    from rtmo_post import with_cut_outputs
    teacher, t_acts, _ = build(ONNX, quantize=False)
    student, s_acts, convs = build(ONNX, quantize=True, float_convs=args.float_convs)
    teacher.cuda().eval()
    student.cuda().eval()
    print(f"量化的位置 {len(s_acts)} 個、卷積 {len(convs)} 個", flush=True)
    sess = ort.InferenceSession(with_cut_outputs(onnx.load(str(ONNX))).SerializeToString(), providers=["CPUExecutionProvider"])
    x, _ = load("rtmo_doll_hold.npy:4")
    for label, ctx in (("FP32", precise), ("TF32", torch.no_grad)):
        num, den, mx = np.zeros(8), np.zeros(8), np.zeros(8)
        for xi in x:
            ref = sess.run(None, {"input": xi.transpose(2, 0, 1)[None].astype(np.float32)})[-8:]
            with torch.no_grad(), ctx():
                out = [o.cpu().numpy() for o in teacher(to_gpu(xi[None]))]
            for k, (r, o) in enumerate(zip(ref, out)):
                num[k] += float((r.astype(np.float64) ** 2).sum())
                den[k] += float(((r.astype(np.float64) - o) ** 2).sum())
                mx[k] = max(mx[k], float(np.abs(r - o).max()))
        print(f"PyTorch（{label}）vs onnxruntime：SNR", fmt(10 * np.log10(num / np.maximum(den, 1e-30))),
              "｜最大差", " ".join(f"{v:.1e}" for v in mx), flush=True)
    calib = np.concatenate([load(f"rtmo_coco.npy:{args.n_check}")[0], load(f"rtmo_doll_calib.npy:{args.n_check}")[0]])
    calibrate(student, s_acts, calib, args.batch)
    hold = np.concatenate([load("rtmo_doll_hold.npy:12")[0], load("rtmo_coco_hold.npy:12")[0]])
    print("訓練前、模擬 8 位元：", fmt(snr(student, teacher, hold, args.batch)), flush=True)


def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    teacher, _, _ = build(ONNX, quantize=False)
    student, acts, convs = build(args.init or ONNX, quantize=True, float_convs=args.float_convs,   # 老師一律是原本的
                                 dfc=args.dfc_ranges)
    teacher.cuda().eval()
    student.cuda().train()
    calib = np.concatenate([load("rtmo_coco.npy")[0], load("rtmo_doll_calib.npy")[0]])    # DFC 用的同一批校正圖
    doll_hold, coco_hold = load("rtmo_doll_hold.npy")[0], load("rtmo_coco_hold.npy")[0]
    data = Images(args.coco_dir, args.coco_list, load("rtmo_doll_calib.npy")[0], args.doll_repeat)
    print(f"訓練圖：COCO {len(data.coco)} 張＋娃娃 {len(data.doll)} 張 ×{args.doll_repeat}；校正 {len(calib)} 張", flush=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         pin_memory=True, drop_last=True, persistent_workers=True)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log = open(args.out / "train_log.jsonl", "a", encoding="utf-8")
    calibrate(student, acts, calib, args.batch)
    base = snr(student, teacher, doll_hold, args.batch), snr(student, teacher, coco_hold, args.batch)
    print("訓練前 娃娃比對：", fmt(base[0]), "\n訓練前 COCO 比對：", fmt(base[1]), flush=True)
    best, step = -1e9, 0
    for ep in range(1, args.epochs + 1):
        t0, run = time.time(), []
        calibrate(student, acts, calib, args.batch)
        student.train()
        for x in loader:
            x = x.cuda(non_blocking=True).float()
            with torch.no_grad(), precise():
                t_outs = teacher(x)
            loss, parts = loss_fn(student(x), t_outs)
            if args.float_weight > 0:     # 浮點那一路也要貼近老師：不讓權重跑去「只靠這一套四捨五入抵消」的解
                with float_mode(acts):
                    loss_f, _ = loss_fn(student(x), t_outs)
                loss = loss + args.float_weight * loss_f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            run.append(float(loss))
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {np.mean(run[-100:]):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        calibrate(student, acts, calib, args.batch)
        student.eval()
        sd, sc = snr(student, teacher, doll_hold, args.batch), snr(student, teacher, coco_hold, args.batch)
        with float_mode(acts):
            fd = snr(student, teacher, doll_hold, args.batch)
        score = float(np.mean(sd) + np.mean(sc)) / 2
        print(f"epoch {ep}（{time.time() - t0:.0f}s）loss {np.mean(run):.4f}\n  娃娃比對：{fmt(sd)}\n  COCO 比對：{fmt(sc)}"
              f"\n  浮點（娃娃）：{fmt(fd)}", flush=True)
        log.write(json.dumps({"epoch": ep, "loss": float(np.mean(run)), "snr_doll": sd.tolist(), "snr_coco": sc.tolist(),
                              "snr_doll_float": fd.tolist()}) + "\n")
        log.flush()
        export(convs, ONNX, args.out / "rtmo-m_qat_last.onnx")
        if score > best:
            best = score
            export(convs, ONNX, args.out / "rtmo-m_qat.onnx")
            print(f"  → 存成 rtmo-m_qat.onnx（目前最好，平均 {score:.1f} dB）", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只檢查轉換和訓練前的模擬量化")
    ap.add_argument("--coco-dir", type=Path)
    ap.add_argument("--coco-list", type=Path, help="允許商用的 COCO 圖清單（每行第一欄是檔名）")
    ap.add_argument("--out", type=Path, default=Path("qat_out"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--doll-repeat", type=int, default=4, help="校正那一半的娃娃重複幾次（COCO 約 2 萬張、娃娃 512 張）")
    ap.add_argument("--workers", type=int, default=10, help="nano4 的 dev 分區每張 GPU 最多 12 顆 CPU")
    ap.add_argument("--n-check", type=int, default=64)
    ap.add_argument("--float-convs", default="", help="例如 1,2,4：這幾個卷積在 DFC 用 16 位元，訓練時不量化")
    ap.add_argument("--jitter", type=float, default=0.0, help="訓練時量化範圍隨機放大 0～這個倍數（見 widen）")
    ap.add_argument("--init", type=Path, help="學生從這個 ONNX 的權重開始（例如上一輪的 rtmo-m_qat.onnx）；預設原本的 RTMO-m")
    ap.add_argument("--float-weight", type=float, default=0.0,
                    help="浮點那一路損失的權重（0＝只看量化那一路）。只看量化的話，學生的浮點版會偏離原本的 RTMO（Q1 只剩 3.2 dB），"
                         "量化方式跟訓練時差一點就失效，DFC 的微調（老師是浮點版）也不能用")
    ap.add_argument("--dfc-ranges", type=Path,
                    help="啟動值範圍改用 DFC 量化後的（hailo/har_ranges.py 從 --init 那個模型的 HAR 匯出的 JSON）")
    args = ap.parse_args()
    args.float_convs = {int(k) for k in args.float_convs.split(",") if k}
    global JITTER
    JITTER = args.jitter
    check(args) if args.check else train(args)


if __name__ == "__main__":
    main()
