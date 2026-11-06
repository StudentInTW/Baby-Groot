"""
hailo/hailo_infer.py
在 UGen300（Hailo-10H）上跑 HEF。包一層 HailoRT 的 Python 套件（hailo_platform，也就是 pyHailoRT），
給比對腳本（compare_rtmo.py --hef）和主流程（RTMO：infant_face_v3.load_rtmo；YuNet：yunet_hef.py）用。

- 輸入：uint8 NHWC 640×640×3（BGR，跟 prep_calib.py 存的、rtmlib 的前處理一樣），可以一張或一批 (N, 640, 640, 3)。
- 輸出：HailoRT 自己反量化成 float32，依 HEF 的輸出順序回傳 list，每個是 (N, H, W, C)。RTMO 的 8 個輸出交給
  rtmo_post.Post 用形狀對應；YuNet 的 12 個由 yunet_hef.py 依原本 ONNX 的節點名稱對應。
- 多個模型共用一顆晶片：同一個程序裡的 Hef 都開在同一個 VDevice（shared_vdevice），HailoRT 的排程器輪流執行。
- 裝置斷線（UGen300 自己重開過，`HAILO_COMMUNICATION_CLOSED`，11/06）：等它恢復（服務會自動把系統重傳進去，
  通常 1–3 秒）、重開 VDevice 和所有模型、把這一批重送一次；印一行記錄，次數在 hailo_infer.reconnects。
  多條執行緒共用晶片時，重連前先等其他執行緒手上的晶片呼叫結束（_drain），不然釋放舊連線一定卡死。
  **已知限制**：還有送出去、沒回報的晶片工作時，VDevice.release() 仍可能卡死（pyHailoRT 的回呼要 GIL，release()
  等它們時卻抓著 GIL；11/06 8 條執行緒時 3 次重連卡 1 次）。要可靠就得把晶片放在另一個程序，斷線就重開那個程序。

pyHailoRT 不在 pyproject.toml 裡（Windows 版的 wheel 只在 HailoRT 安裝程式裡，見 安裝筆記.md），要另外裝進 .venv。

  uv run python -m hailo.hailo_infer model/rtmo_D1.hef --data rtmo_doll_hold.npy:32
"""

import argparse
import atexit
import faulthandler
import threading
import time
from pathlib import Path

import numpy as np

TIMEOUT_MS = 10000
RECONNECT_WAIT_S = 120      # 斷線後最多等裝置恢復這麼久
DRAIN_WAIT_S = 30           # 重連前最多等別的執行緒手上的晶片工作結束這麼久
_VDEVICE = None
_HEFS = []                  # 這個程序開的所有模型；裝置重開後要全部重建
reconnects = 0              # 這個程序裡重新連線的次數（評估 log 要記下來）
_LOCK = threading.Lock()    # 多條執行緒同時碰到斷線時，只讓一條去重連
_generation = 0             # 每重連一次加一；拿來判斷別的執行緒是不是已經重連過了
_INFLIGHT = threading.Condition()
_inflight = 0               # 正在晶片上跑的呼叫數（_INFLIGHT 保護）
_ABANDONED = []             # 釋放不掉的舊連線：留著不讓它被回收（回收時一樣會卡住）


def shared_vdevice():
    """這個程序共用的 VDevice（第一次呼叫時開）。程式結束時最後才釋放：atexit 後登記的先執行，
    各個 Hef 的 close 都登記在它之後，所以會先 shutdown 模型、再釋放 VDevice。"""
    global _VDEVICE
    if _VDEVICE is None:
        from hailo_platform import VDevice
        _VDEVICE = VDevice()
        atexit.register(_release_vdevice)
    return _VDEVICE


def _release_vdevice():
    global _VDEVICE
    if _VDEVICE is not None:
        try:
            _VDEVICE.release()
        except Exception:      # 裝置已經斷線時釋放會報錯，不影響
            pass
        _VDEVICE = None


def _enter():
    global _inflight
    with _INFLIGHT:
        _inflight += 1


def _leave():
    global _inflight
    with _INFLIGHT:
        _inflight -= 1
        _INFLIGHT.notify_all()


def _drain():
    """等別的執行緒手上的晶片呼叫都結束（失敗）；回傳是不是都結束了。
    11/06 多條執行緒測速時，有別的執行緒還在晶片呼叫裡就 VDevice.release()，整個程序卡死過兩次：release() 等那些
    工作結束，等的時候抓著 GIL，那些執行緒拿不到 GIL 就結束不了。在這裡用 Python 等，GIL 是放開的。"""
    with _INFLIGHT:
        return _INFLIGHT.wait_for(lambda: _inflight == 0, timeout=DRAIN_WAIT_S)


def _reconnect(err_name):
    """裝置斷線：丟掉舊的連線，等裝置恢復，重開 VDevice 和所有模型（呼叫前要拿 _LOCK）。"""
    global reconnects, _generation, _VDEVICE
    t0 = time.time()
    print(f"[hailo_infer] UGen300 斷線（{err_name}），等它恢復…", flush=True)
    old = [h.configured for h in _HEFS if h.configured is not None]
    for h in _HEFS:
        h._drop()
    if _drain():
        # 送出去、還沒回報的工作（pyHailoRT 每批留一份緩衝區直到回呼）：診斷用
        waiting = sum(len(getattr(m, "_buffer_guards", ())) for m in old)
        del old
        if waiting:
            print(f"[hailo_infer] 還有 {waiting} 批晶片工作沒有回報，照樣釋放舊的連線", flush=True)
        _release_vdevice()
    else:
        print(f"[hailo_infer] {DRAIN_WAIT_S} 秒後還有 {_inflight} 個晶片呼叫沒結束：舊的連線不釋放（會卡住），"
              "直接開新的", flush=True)
        faulthandler.dump_traceback(all_threads=True)    # 卡在哪裡，留給之後查
        _ABANDONED.append((_VDEVICE, old))
        _VDEVICE = None
    while True:
        try:
            shared_vdevice()
            for h in _HEFS:
                h._open()
            break
        except Exception as e:  # 裝置還在重開（不在、或停在 loader 模式）
            for h in _HEFS:
                h._drop()
            _release_vdevice()
            if time.time() - t0 > RECONNECT_WAIT_S:
                raise RuntimeError(f"UGen300 斷線後 {RECONNECT_WAIT_S} 秒還沒恢復") from e
            time.sleep(2)
    reconnects += 1
    _generation += 1
    print(f"[hailo_infer] {time.time() - t0:.0f} 秒後重新連上（這個程序第 {reconnects} 次）", flush=True)


class Hef:
    def __init__(self, path, batch=1):
        from hailo_platform import HEF
        self.path, self.batch = Path(path), batch
        self.model = self.configured = None
        self._open()
        self.input_name = self.model.input().name
        self.input_shape = tuple(self.model.input().shape)
        self.outputs = [(o.name, tuple(o.shape)) for o in self.model.outputs]
        hef = HEF(str(path))   # 每個輸出對應原本 ONNX 的哪個節點（YuNet 用來分 cls／obj）
        self.original_names = {n: list(hef.get_original_names_from_vstream_name(n)) for n, _ in self.outputs}
        _HEFS.append(self)
        # 程式結束時照順序關：先 shutdown 模型、再釋放 VDevice。交給 Python 自己回收的話順序不一定，
        # HailoRT 會印「Lost communication with the server」、「Failed to close session」
        atexit.register(self.close)

    def _open(self):
        from hailo_platform import FormatType
        self.model = shared_vdevice().create_infer_model(str(self.path))
        self.model.set_batch_size(self.batch)
        for o in self.model.outputs:
            o.set_format_type(FormatType.FLOAT32)
        self.configured = self.model.configure()

    def _drop(self):
        """裝置斷線後丟掉舊的模型（不能 shutdown，連線已經斷了）。"""
        self.configured = self.model = None

    def __call__(self, images):
        """images：(640, 640, 3) 或 (N, 640, 640, 3) uint8 → [每個輸出 (N, H, W, C) float32]。"""
        from hailo_platform.pyhailort.pyhailort import HailoRTException
        images = np.ascontiguousarray(images, dtype=np.uint8)
        if not images.flags.writeable:                # HailoRT 只收可寫的陣列（np.load 的 mmap 是唯讀）
            images = images.copy()
        if images.ndim == 3:
            images = images[None]
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"輸入形狀 {images.shape[1:]}，HEF 要 {self.input_shape}")
        with _LOCK:                         # 別的執行緒正在重連的話，等它連好再送
            gen, cm = _generation, self.configured
            _enter()
        try:
            return self._run(cm, images)
        except HailoRTException as e:
            failed = type(e).__name__       # 只留名稱：例外物件連著這次送出的工作和舊的模型
        finally:
            cm = None                       # 先放掉舊的模型，再說「我不在晶片上了」（_drain 在等）
            _leave()
        with _LOCK:
            if gen == _generation:          # 還沒有別的執行緒重連過
                _reconnect(failed)
            cm = self.configured
            _enter()
        try:
            return self._run(cm, images)
        finally:
            cm = None
            _leave()

    def _run(self, cm, images):
        outs = {n: np.empty((len(images),) + s, np.float32) for n, s in self.outputs}
        jobs = []
        for k in range(len(images)):
            b = cm.create_bindings(input_buffers={self.input_name: images[k]},
                                   output_buffers={n: outs[n][k] for n, _ in self.outputs})
            cm.wait_for_async_ready(timeout_ms=TIMEOUT_MS)
            jobs.append(cm.run_async([b]))
        for j in jobs:
            j.wait(TIMEOUT_MS)
        return [outs[n] for n, _ in self.outputs]

    def close(self):
        if self.configured is not None:
            try:
                self.configured.shutdown()
            except Exception:  # 裝置已經斷線
                pass
            self.configured = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hef", type=Path)
    ap.add_argument("--data", default=None, help="prep_calib.py 存的 .npy（「檔名:張數」）；不給就用隨機圖")
    args = ap.parse_args()
    if args.data:
        from hailo.rtmo_post import load
        data, _ = load(args.data)
    else:
        data = np.random.default_rng(0).integers(0, 256, (16, 640, 640, 3), dtype=np.uint8)
    with Hef(args.hef) as h:
        print(f"{args.hef.name}：輸入 {h.input_name} {h.input_shape}")
        h(data[:1])                                   # 第一次呼叫比較慢，不算進時間
        t0 = time.perf_counter()
        outs = h(data)
        dt = time.perf_counter() - t0
        for (n, s), o in zip(h.outputs, outs):
            print(f"  {n:32s} {str(s):16s} 範圍 {o.min():9.3f} ～ {o.max():9.3f}")
        print(f"  {len(data)} 張 {dt * 1000:.0f} ms（每張 {dt * 1000 / len(data):.1f} ms，含 USB 傳輸和反量化）")


if __name__ == "__main__":
    main()
