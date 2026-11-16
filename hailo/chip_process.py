"""
hailo/chip_process.py
把 UGen300 放在另一個程序（實驗，2026/11/16 寫好，還沒在晶片上測過）。

為什麼：同一個程序裡，晶片斷線後釋放舊連線可能卡死（pyHailoRT 的回呼要 GIL，VDevice.release() 卻抓著 GIL）。
11/15 整夜測試（RTMO 單獨在晶片、8 床）約 3.5 小時斷 9 次，第 9 次就這樣卡死，之後 3.5 小時沒有判讀。
在同一個程序裡救不回來，所以把晶片搬到子程序：主流程送圖給它，超過 CALL_TIMEOUT_S 沒回、或子程序死掉，
就直接把子程序砍掉、重開一個（重新開 VDevice 和所有模型），這一批重送一次。主流程本身不會卡住。

用法：設環境變數 FG_CHIP_PROCESS=1，其他照舊（FG_RTMO_HEF、FG_YUNET_HEF）。沒設就跟以前一樣在同一個程序裡跑。
pipeline 透過 hailo_infer.open_hef() 拿模型，設了就拿到這裡的 RemoteHef，介面跟 hailo_infer.Hef 一樣。

已知限制：
- 所有模型、所有執行緒共用一個子程序，一次只送一批（用鎖排隊），多床同時送時吞吐量會比在同一個程序裡低。
- 每次呼叫要把圖片從主程序複製到子程序（640×640×3 一張約 1.2 MB），多幾毫秒。
- 子程序重開時（等裝置恢復、重新載模型）那幾秒沒有判讀，次數在 chip_process.restarts。
"""

import atexit
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

CALL_TIMEOUT_S = 30        # 一次呼叫超過這麼久沒回 → 當作卡死，重開子程序
START_TIMEOUT_S = 180      # 子程序開好所有模型最多等這麼久（裝置剛重開時要等它恢復）
restarts = 0               # 這個程序裡重開子程序的次數（評估 log 要記下來）

_LOCK = threading.Lock()   # 一次只讓一條執行緒跟子程序講話
_SPECS = []                # 開過的模型 (path, batch)；子程序重開時照這個順序全部重開
_proc = _conn = None


def _child(conn, specs):
    """子程序：開 VDevice 和所有模型，回報每個模型的介面，然後一直收圖、跑、回傳。"""
    os.environ["FG_CHIP_PROCESS"] = "0"
    from hailo.hailo_infer import Hef
    hefs = []
    try:
        for path, batch in specs:
            hefs.append(Hef(path, batch))
        conn.send(("ready", [(h.input_name, h.input_shape, h.outputs, h.original_names) for h in hefs]))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            break
        k, images = msg
        try:
            conn.send(("ok", hefs[k](images)))
        except Exception as e:      # hailo_infer 自己重連不回來（例如等太久）→ 回報，主程序決定重開
            conn.send(("error", f"{type(e).__name__}: {e}"))


def _stop():
    global _proc, _conn
    if _proc is not None:
        try:
            _conn.send(None)
            _proc.join(5)
        except Exception:
            pass
        if _proc.is_alive():
            _proc.kill()
            _proc.join(5)
    _proc = _conn = None


def _start():
    """開子程序、等它載好所有模型；回傳每個模型的介面。"""
    global _proc, _conn
    parent, child = mp.get_context("spawn").Pipe()
    _proc = mp.get_context("spawn").Process(target=_child, args=(child, list(_SPECS)), daemon=True)
    _proc.start()
    _conn = parent
    if not parent.poll(START_TIMEOUT_S):
        _stop()
        raise RuntimeError(f"晶片子程序 {START_TIMEOUT_S} 秒內沒有開好模型")
    status, info = parent.recv()
    if status != "ready":
        _stop()
        raise RuntimeError(f"晶片子程序開不起來：{info}")
    return info


def _restart(reason):
    global restarts
    t0 = time.time()
    print(f"[chip_process] {reason}：重開晶片子程序…", flush=True)
    _stop()
    while True:
        try:
            _start()
            break
        except Exception as e:     # 裝置還在重開
            if time.time() - t0 > START_TIMEOUT_S:
                raise RuntimeError(f"晶片子程序 {START_TIMEOUT_S} 秒內重開不了") from e
            time.sleep(2)
    restarts += 1
    print(f"[chip_process] {time.time() - t0:.0f} 秒後重開好了（這個程序第 {restarts} 次）", flush=True)


def _call(k, images):
    """送一批給子程序；卡住或子程序死掉就重開、重送一次。"""
    with _LOCK:
        for attempt in (1, 2):
            try:
                _conn.send((k, images))
                if not _conn.poll(CALL_TIMEOUT_S):
                    raise TimeoutError(f"{CALL_TIMEOUT_S} 秒沒有回應")
                status, out = _conn.recv()
                if status == "ok":
                    return out
                raise RuntimeError(out)
            except Exception as e:
                if attempt == 2:
                    raise
                _restart(f"晶片呼叫失敗（{type(e).__name__}: {str(e)[:80]}）")


class RemoteHef:
    """跟 hailo_infer.Hef 一樣的介面，但真正跑在子程序裡。"""

    def __init__(self, path, batch=1):
        import numpy as np
        self._np = np
        self.path, self.batch = Path(path), batch
        with _LOCK:
            _SPECS.append((str(self.path), batch))
            self._k = len(_SPECS) - 1
            if _proc is None:
                atexit.register(_stop)
            else:
                _stop()            # 已經有子程序（開著別的模型）：全部一起重開，模型才會在同一個 VDevice 上
            info = _start()
        self.input_name, self.input_shape, self.outputs, self.original_names = info[self._k]

    def __call__(self, images):
        images = self._np.ascontiguousarray(images, dtype=self._np.uint8)
        if images.ndim == 3:
            images = images[None]
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"輸入形狀 {images.shape[1:]}，HEF 要 {self.input_shape}")
        return _call(self._k, images)

    def close(self):
        pass                       # 子程序在程式結束時一起關（atexit）

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
