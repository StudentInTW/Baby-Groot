"""
hailo/compile_hef.py
ONNX → HEF（parse → optimize 量化 → compile），在 WSL 的 DFC 虛擬環境執行。YuNet、RTMO-m 共用。

  . ~/hailo_dfc/bin/activate && cd ~/work        # DFC 把 log 寫在目前的資料夾，路徑不能有空白
  python "/mnt/c/Users/user/ASUS hackathon/hailo/compile_hef.py" --model yunet --name yunet_A --calib coco.npy
  python ".../compile_hef.py" --model rtmo --name rtmo_A --calib rtmo_coco.npy

--calib 是 prep_calib.py 存的 .npy（預設在 hailo/calib/），「檔名:張數」只取前幾張；多個檔案依序接起來。
輸出放在 --out（預設 ~/work/<model>/）：<name>_parsed.har、<name>_quantized.har、<name>.hef、<name>.alls（實際用的 model script）。

模型：
- yunet：整個 ONNX 都放晶片；12 個輸出是原始特徵，解碼在主機（yunet_decode.py）。
- rtmo：end node 切在偵測頭最後的 8 個卷積（rtmo_post.CUT_NODES），框的解碼、NMS、關鍵點的 DCC 在主機跑
  （rtmo_post.py 從原本的 ONNX 取出的後段）。ONNX 的 batch 是動態的，parse 時固定成 1×3×640×640。

量化設定（model script）：
- 校正張數明確設成實際給的張數：DFC 預設只拿前 64 張算各層範圍（calibration 的 calibset_size 預設 64）。
- optimization_level 明確設成 0（只做 equalization）。不設的話 DFC 依硬體決定：只有 CPU 時是 0，
  有 GPU、校正圖少於 1024 張時是 1（再加偏差校正）——同一份腳本在不同電腦會量化得不一樣。
  等級 1 的偏差校正會在目前的資料夾寫暫存檔，768 張約 13 GB（2026/11/05 在筆電上把 C 槽寫滿過）。
  等級 1 以上會用校正圖做偏差校正或微調，目標是 ONNX 自己的輸出，不看標註。
- 不做正規化：YuNet、RTMO 都吃 BGR 0–255（OpenCV、rtmlib 都不做正規化），晶片的輸入就是原始像素。
"""

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CALIB_DIR = REPO / "hailo" / "calib"
HW_ARCH = "hailo10h"


def model_spec(name, onnx_path=None):
    """回傳 (ONNX 路徑, translate_onnx_model 的額外參數)。onnx_path：換成別的 ONNX（例如 qat_rtmo.py 訓練後的 RTMO）。"""
    if name == "yunet":
        return onnx_path or REPO / "model" / "face_detection_yunet_2023mar.onnx", {}
    from rtmo_post import CUT_NODES, ONNX
    return onnx_path or ONNX, {"start_node_names": ["input"], "end_node_names": list(CUT_NODES),
                               "net_input_shapes": {"input": [1, 3, 640, 640]}}


def load_calib(specs):
    parts = []
    for spec in specs:
        name, _, n = spec.partition(":")
        arr = np.load(CALIB_DIR / name if not Path(name).is_absolute() else name, mmap_mode="r")
        parts.append(np.asarray(arr[:int(n)] if n else arr))
        print(f"  校正圖 {name}：{len(parts[-1])} 張")
    return np.concatenate(parts).astype(np.float32)   # DFC 的校正資料是 NHWC float


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("yunet", "rtmo"))
    ap.add_argument("--onnx", type=Path, help="不用預設的 ONNX（RTMO 量化感知訓練後的模型：節點名稱、切點跟原本一樣）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--calib", nargs="+", help="不給就只做 parse")
    ap.add_argument("--opt-level", type=int, default=0, help="optimization_level（預設 0，見上面的說明）")
    ap.add_argument("--extra", nargs="+", default=[],
                    help="額外的 model script 指令，例如 'quantization_param([conv55, conv56], precision_mode=a16_w16)'")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-compile", action="store_true", help="只做到量化（比對模擬器用），不產生 HEF")
    ap.add_argument("--from-quantized", action="store_true",
                    help="直接拿 <out>/<name>_quantized.har 編譯（跳過 parse 和量化；先 --no-compile 比過模擬器再編）")
    args = ap.parse_args()
    out = args.out or Path.home() / "work" / args.model

    from hailo_sdk_client import ClientRunner
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.from_quantized:
        runner = ClientRunner(har=str(out / f"{args.name}_quantized.har"))
    else:
        onnx_path, kw = model_spec(args.model, args.onnx)
        print(f"  ONNX：{onnx_path}")
        runner = ClientRunner(hw_arch=HW_ARCH)
        runner.translate_onnx_model(str(onnx_path), net_name=args.name, **kw)
        runner.save_har(str(out / f"{args.name}_parsed.har"))
        print(f"[parse] 完成（{time.perf_counter() - t0:.0f}s）")
        if not args.calib:
            return

        calib = load_calib(args.calib)
        script = [f"model_optimization_config(calibration, calibset_size={len(calib)})",
                  f"model_optimization_flavor(optimization_level={args.opt_level})"] + args.extra
        (out / f"{args.name}.alls").write_text("\n".join(script) + "\n")
        runner.load_model_script("\n".join(script) + "\n")
        t1 = time.perf_counter()
        runner.optimize(calib)
        runner.save_har(str(out / f"{args.name}_quantized.har"))
        print(f"[optimize] 完成：{len(calib)} 張校正圖（{time.perf_counter() - t1:.0f}s）")

    if not args.no_compile:
        t2 = time.perf_counter()
        hef = runner.compile()
        (out / f"{args.name}.hef").write_bytes(hef)
        print(f"[compile] 完成：{out / args.name}.hef，{len(hef) / 1024:.0f} KB（{time.perf_counter() - t2:.0f}s）")
    print(f"總共 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
