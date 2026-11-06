"""照 DFC 實際的量化方式，決定 RTMO 前段「在哪些張量量化、用什麼範圍」。sim_quant.py 和 qat_rtmo.py 共用。

輸入是 har_ranges.py 從 DFC 量化後的 HAR 匯出的 JSON（每層的 limvals_out 和 original_names）。
2026/11/06：只用校正圖算範圍、只量化卷積／相加／串接…的輸出時，模擬比 DFC 樂觀 2–5 dB，差在三處（結果.md）：
1. 範圍：DFC 讓同一個 concat 的輸入共用 zero point 等規則，有些層的範圍比實際數值寬 1.3–3 倍 → 用 HAR 裡的範圍。
2. head 裡接 Split 的兩個 1×1 卷積，DFC 拆成兩塊（feature splitter）、各自量化 → 量化 Split 的兩個輸出，卷積的輸出不量化。
3. transformer（AIFI）的線性層（q／k／v、輸出投影、FFN）、位置編碼相加、layer norm 的輸出，DFC 都量化成 8 位元；
   原本的規則把「輸入有常數」的運算都排除（為了排除算形狀的運算），這些就漏掉了 → 照 DFC 的層加量化點。

DFC 的一層常對到好幾個 ONNX 節點（卷積＋SiLU、MatMul＋Add＋Reshape＋Transpose…），量化點取這組節點的出口
（組內沒有別的節點用到的那個輸出）。好幾層 DFC 對到同一組節點時（layer norm、softmax 拆成的內部運算），
用最後一層（HN 的順序）的範圍：那一層才是整組的輸出（softmax 的 reduce_max 範圍是 −13～36，輸出才是 0～1）。
layer norm、softmax 內部的量化沒有模擬（DFC 內部看起來用較高精度：輸出前有 precision_change 層）。
"""
import json


def dfc_points(json_path, nodes, q, float_tensors, skip=()):
    """回傳 (ranges, add, drop)：
    ranges：{張量: (lo, hi)}，DFC 的範圍；add：要多量化的張量（照 DFC 的層，原本的 q 沒有）；drop：q 裡不該量化的張量。
    nodes：前段 ONNX 的節點（拓撲順序）；q：原本的量化張量；float_tensors：前段裡浮點張量的名稱；
    skip：不量化的張量（例如 16 位元的卷積，見 sim_quant／qat_rtmo 的 float_convs），DFC 的層對到這些就略過。"""
    pos = {n.name: (i, n) for i, n in enumerate(nodes)}
    layers = json.load(open(json_path, encoding="utf-8"))
    by_layer = {L["name"]: L for L in layers}
    qset, skip = set(q), set(skip)
    ranges, add, drop = {}, [], set()
    for L in layers:
        if L.get("limvals_out") is None or not L.get("original_names") \
                or L["type"] in ("input_layer", "const_input", "output_layer", "format_conversion"):
            continue
        grp = [pos[nm] for nm in L["original_names"] if nm in pos]
        if not grp:
            continue
        rng = tuple(L["limvals_out"])
        if "feature_splitter" in L["name"]:
            # 這一塊＝Split 的其中一個輸出：下游那層的第一個節點吃的那個
            split = next((n for _, n in grp if n.op_type == "Split"), None)
            down = by_layer.get((L.get("outputs") or [""])[0].split("/")[-1])
            first = next((pos[nm][1] for nm in (down or {}).get("original_names") or [] if nm in pos), None)
            part = next((x for x in (first.input if first else []) if split and x in split.output), None)
            if part is None:
                continue
            ranges[part] = rng
            add.append(part)
            drop |= {o for _, n in grp if n.op_type == "Conv" for o in n.output}
            continue
        used = {x for _, n in grp for x in n.input}
        sinks = [(i, o) for i, n in grp for o in n.output if o not in used and o in float_tensors]
        if not sinks:
            continue
        t = max(sinks)[1]
        if t in skip:
            continue
        ranges[t] = rng          # 同一個出口有好幾層時，後面的蓋掉前面的（HN 的順序＝拓撲順序）
        if t not in qset:
            add.append(t)
        drop |= {o for _, n in grp for o in n.output if o in qset and o != t}   # 組內其他已在 q 的點（例如 MatMul 後面還有 Reshape）
    add = [t for t in dict.fromkeys(add) if t not in drop]
    return ranges, add, drop
