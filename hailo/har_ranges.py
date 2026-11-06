"""DFC 量化後的 HAR → 每層的輸出範圍和對應的 ONNX 節點，存成 JSON。

用途：跟 sim_quant.py 的模擬逐層比，或用 `sim_quant.py --dfc-ranges <json>` 直接拿 DFC 的範圍模擬，
看模擬和 DFC 實測的差距有多少來自範圍（2026/11/05：DFC 有些層的範圍比實際數值寬 1.3–2.9 倍，見 結果.md）。

  python hailo/har_ranges.py ~/work/rtmo/rtmo_D1_quantized.har outputs/hailo/ranges_D1.json   # WSL 的 DFC 環境

每層記：名稱、種類、activation、original_names（ONNX 節點）、上下游、limvals_out／limvals_in（輸出／輸入範圍）、
輸出的 zero point、輸出 scale 有幾組（1＝整層一組）、權重的整數範圍、有沒有用分段直線的輸出階段（piecewise）。
"""
import json
import sys

import numpy as np
from hailo_sdk_client import ClientRunner


def main():
    har, out_path = sys.argv[1], sys.argv[2]
    r = ClientRunner(har=har)
    hn = r.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    p = r.get_params_translated()
    net = next(k for k in p.keys() if "/" in k).split("/")[0]   # 第一個鍵是 params_kind
    out = []
    for full, v in hn["layers"].items():
        name = full.split("/")[-1]

        def get(k):
            key = f"{net}/{name}/{k}:0"
            return np.asarray(p[key]) if key in p else None

        sc, kern = get("layer_params/output_scales"), get("kernel")
        out.append({
            "name": name, "type": v["type"],
            "activation": v.get("params", {}).get("activation"),
            "original_names": v.get("original_names"),
            "inputs": v.get("input"), "outputs": v.get("output"),
            "limvals_out": None if get("limvals_out") is None else get("limvals_out").tolist(),
            "limvals_in": None if get("limvals_in") is None else get("limvals_in").tolist(),
            "zp_out": None if get("zero_point_out") is None else get("zero_point_out").tolist(),
            "n_out_scales": None if sc is None else len(set(np.round(sc.ravel(), 8))),
            "kernel_range": [int(kern.min()), int(kern.max())] if kern is not None and kern.size else None,
            "piecewise": get("output_stage/piecewise/x_points") is not None,
        })
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=0)
    types = {}
    for o in out:
        types[o["type"]] = types.get(o["type"], 0) + 1
    print(f"{har}：{len(out)} 層 {sorted(types.items(), key=lambda t: -t[1])}；"
          f"有範圍的 {sum(o['limvals_out'] is not None for o in out)} 層、用分段直線的 {sum(o['piecewise'] for o in out)} 層 → {out_path}")


if __name__ == "__main__":
    main()
