"""
hailo/chip_process.py
把 UGen300 放在另一個程序（11/16 寫好；11/22–23 在晶片上測：一次一張 18 小時 0 次斷線；合批 4 8 小時真的斷線 5 次，
每次 1–2 秒救回；見 結果.md「裝置斷線」）。

為什麼：同一個程序裡，晶片斷線後釋放舊連線可能卡死（pyHailoRT 的回呼要 GIL，VDevice.release() 卻抓著 GIL）。
11/15 整夜測試（RTMO 單獨在晶片、8 床）約 3.5 小時斷 9 次，第 9 次就這樣卡死，之後 3.5 小時沒有判讀。
在同一個程序裡救不回來，所以把晶片搬到子程序：主流程送圖給它；子程序碰到晶片錯誤就直接結束（不在裡面重連），
超過 CALL_TIMEOUT_S 沒回、或子程序死掉，主程序就把它砍掉、重開一個（重新開 VDevice 和所有模型），這一批重送一次。
主流程本身不會卡住。

用法：設環境變數 FG_CHIP_PROCESS=1，其他照舊（FG_RTMO_HEF、FG_YUNET_HEF）。沒設就跟以前一樣在同一個程序裡跑。
pipeline 透過 hailo_infer.open_hef() 拿模型，設了就拿到這裡的 RemoteHef，介面跟 hailo_infer.Hef 一樣。

多床合批（11/22，FG_CHIP_BATCH=N，N>1）：多條執行緒（多床）送來的圖先排隊，由一條送件執行緒把同一個模型的最多 N 張
合成一批送給子程序；子程序裡的模型也設成 batch N（HailoRT 連著跑完這一批再換下一段，RTMO 有 5 段 context，
hailortcli 量 batch 1 每秒 58 張、batch 8 每秒 95 張）。不等湊滿：上一批還在晶片上時排進來的，下一批一起送；
只有一床時跟一次一張一樣。沒設或設 1 就是一次一張：8 床慢約四分之一（21.6 對 28.8 幀/秒），但 18 小時沒斷過；
合批 4 半夜斷線跟同一個程序差不多多（都救得回來）。長時間跑、展演用一次一張。

傳圖用共用記憶體（11/22）：每個模型一塊（輸入 batch 張 uint8、輸出 batch 張 float32），主程序把圖寫進去、只傳一個短訊息，
子程序的晶片直接讀寫這塊記憶體，做完主程序再把輸出複製出來。原本整批圖用管線（pickle）來回傳，RTMO 一張要搬 1.2 MB
進去、3.2 MB 回來，8 床時這一段是瓶頸。一次送超過 batch 張的呼叫（例如比對工具一次給 32 張）照舊用管線傳。

已知限制：
- 一次只有一批在子程序裡（送出、等結果、再送下一批），傳圖和晶片運算沒有重疊。
- 子程序重開時（等裝置恢復、重新載模型）那幾秒沒有判讀，次數在 chip_process.restarts。
  晶片斷線時子程序馬上報錯，重開約 1.4 秒；子程序卡住（沒結束也沒回）要等 CALL_TIMEOUT_S 才發現，約 13.5 秒。
"""

import atexit
import collections
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

CALL_TIMEOUT_S = 12        # 一次呼叫超過這麼久沒回 → 當作卡住，砍掉重開子程序。比子程序裡等晶片的 10 秒
                           # （hailo_infer.TIMEOUT_MS）長一點：晶片沒反應時子程序自己先報錯；子程序本身卡住時，
                           # 所有床約 13.5 秒沒判讀（11/23 把子程序暫停來測；原本 30 秒＋先等它自己結束 5 秒是 36.5 秒）
START_TIMEOUT_S = 180      # 子程序開好所有模型最多等這麼久（裝置剛重開時要等它恢復）
MAX_BATCH = max(1, int(os.environ.get("FG_CHIP_BATCH") or 1))   # 一批最多幾張；1＝不合批
WAIT_S = 2 * CALL_TIMEOUT_S + START_TIMEOUT_S + 60              # 合批時，一個請求最多等這麼久（送兩次、中間重開一次）
restarts = 0               # 這個程序裡重開子程序的次數（評估 log 要記下來）
batches = batched_frames = 0   # 合批時送了幾批、共幾張（平均一批幾張＝batched_frames / batches）

_LOCK = threading.Lock()   # 一次只讓一條執行緒跟子程序講話
_SPECS = []                # 開過的模型 (path, batch)；子程序重開時照這個順序全部重開
_proc = _conn = None
_COND = threading.Condition()   # 保護 _PENDING、_sender
_PENDING = collections.deque()  # 合批時排隊等著送的請求（_Req）
_sender = None
_SHM = {}                       # 主程序：每個模型的共用記憶體（_Shm），模型編號 → _Shm


class _Shm:
    """一個模型的共用記憶體：輸入 cap 張 uint8、接著每個輸出 cap 張 float32。主程序建立（name=None），子程序用名字接上。
    每一段的長度都是 64 位元組的倍數（RTMO、YuNet 的形狀都是），陣列都對齊。"""

    def __init__(self, cap, input_shape, outputs, name=None):
        import numpy as np
        from multiprocessing import shared_memory
        shapes = [(np.uint8, (cap,) + tuple(input_shape))] + [(np.float32, (cap,) + tuple(s)) for _, s in outputs]
        sizes = [int(np.prod(s)) * np.dtype(t).itemsize for t, s in shapes]
        self.cap, self.owner = cap, name is None
        self.shm = (shared_memory.SharedMemory(create=True, size=sum(sizes)) if self.owner
                    else shared_memory.SharedMemory(name=name))
        views, off = [], 0
        for (t, s), size in zip(shapes, sizes):
            views.append(np.ndarray(s, t, self.shm.buf, off))
            off += size
        self.inp, self.outs = views[0], views[1:]

    def close(self):
        self.inp = self.outs = None
        try:
            self.shm.close()
        except BufferError:        # 還有人拿著指向這塊記憶體的陣列：程式要結束了，不管它
            pass


def _close_shms():
    for s in _SHM.values():
        s.close()
    _SHM.clear()


def _child(conn, specs):
    """子程序：開 VDevice 和所有模型，回報每個模型的介面，然後一直收圖、跑、回傳。
    訊息：("shm", 模型, 張數, 共用記憶體名稱, 容量) → 圖在共用記憶體裡，輸出也寫回去，只回 ("ok", None)；
          (模型, 圖) → 圖跟著訊息來，輸出也跟著回覆回去（一次超過容量時）。
    晶片出錯就回報後直接結束：在這裡重連的話，還有送出去的工作時 VDevice.release() 可能卡死（合批時一定有）；
    整個程序結束比較乾淨，主程序會重開一個新的。"""
    os.environ["FG_CHIP_PROCESS"] = "0"
    import hailo.hailo_infer as hi
    hi.RECONNECT = False
    hefs, shms = [], {}
    try:
        for path, batch in specs:
            hefs.append(hi.Hef(path, batch))
        conn.send(("ready", [(h.input_name, h.input_shape, h.outputs, h.original_names) for h in hefs]))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:           # 主程序結束了
            msg = None
        if msg is None:
            for s in shms.values():
                s.close()
            return
        try:
            if msg[0] == "shm":
                _, k, n, name, cap = msg
                if name not in shms:
                    shms[name] = _Shm(cap, hefs[k].input_shape, hefs[k].outputs, name)
                s = shms[name]
                hefs[k](s.inp[:n], out=[o[:n] for o in s.outs])
                out = None
            else:
                k, images = msg
                out = hefs[k](images)
        except Exception as e:
            print(f"[chip_process] 子程序 {os.getpid()} 晶片出錯（{type(e).__name__}: {str(e)[:80]}），結束、讓主程序重開",
                  flush=True)
            try:
                conn.send(("error", f"{type(e).__name__}: {e}"))
            finally:
                os._exit(1)        # 不跑任何清理（釋放 VDevice 可能卡死）
        conn.send(("ok", out))


def _stop(kill=False):
    """關掉子程序。kill=True（出錯要重開時）：舊的不是已經結束就是卡住，直接砍，不先送結束訊息、等它 5 秒。"""
    global _proc, _conn
    if _proc is not None:
        if not kill:
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
        _stop(kill=True)
        raise RuntimeError(f"晶片子程序 {START_TIMEOUT_S} 秒內沒有開好模型")
    status, info = parent.recv()
    if status != "ready":
        _stop(kill=True)
        raise RuntimeError(f"晶片子程序開不起來：{info}")
    return info


def _restart(reason):
    global restarts
    t0 = time.time()
    print(f"[chip_process] {reason}：重開晶片子程序…", flush=True)
    _stop(kill=True)
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


def _reply():
    if not _conn.poll(CALL_TIMEOUT_S):
        raise TimeoutError(f"{CALL_TIMEOUT_S} 秒沒有回應")
    status, out = _conn.recv()
    if status != "ok":
        raise RuntimeError(out)
    return out


def _send_recv(k, images):
    """送一批、等結果（要拿著 _LOCK）。放得進共用記憶體就走共用記憶體，輸出複製出來（下一批會蓋掉那塊記憶體）。"""
    s = _SHM.get(k)
    if s is not None and len(images) <= s.cap:
        n = len(images)
        s.inp[:n] = images
        _conn.send(("shm", k, n, s.shm.name, s.cap))
        _reply()
        return [o[:n].copy() for o in s.outs]
    _conn.send((k, images))
    return _reply()


def _call(k, images):
    """送一批給子程序；卡住或子程序死掉就重開、重送一次。"""
    with _LOCK:
        for attempt in (1, 2):
            try:
                return _send_recv(k, images)
            except Exception as e:
                if attempt == 2:
                    raise
                _restart(f"晶片呼叫失敗（{type(e).__name__}: {str(e)[:80]}）")


class _Req:
    """合批時排隊的一個請求：哪個模型、幾張圖；送件執行緒做完會填 out 或 err，再 set done。"""
    __slots__ = ("k", "images", "done", "out", "err")

    def __init__(self, k, images):
        self.k, self.images, self.done, self.out, self.err = k, images, threading.Event(), None, None


def _take():
    """（拿著 _COND）照排隊順序取出下一批：最前面那個請求的模型、最多 MAX_BATCH 張；其他的留在隊伍裡、順序不變。"""
    k, batch, rest, n = _PENDING[0].k, [], [], 0
    while _PENDING:
        r = _PENDING.popleft()
        if r.k == k and (not batch or n + len(r.images) <= MAX_BATCH):
            batch.append(r)
            n += len(r.images)
        else:
            rest.append(r)
    _PENDING.extend(rest)
    return k, batch


def _send_loop():
    """送件執行緒：隊伍裡有請求就合成一批送給子程序（_call 管重開、重送），結果切開還給各個請求。"""
    global batches, batched_frames
    import numpy as np
    while True:
        with _COND:
            _COND.wait_for(lambda: _PENDING)
            k, batch = _take()
        try:
            images = batch[0].images if len(batch) == 1 else np.concatenate([r.images for r in batch])
            outs = _call(k, images)
            i = 0
            for r in batch:
                r.out = [o[i:i + len(r.images)] for o in outs]
                i += len(r.images)
            batches += 1
            batched_frames += len(images)
        except Exception as e:     # 重開、重送之後還是失敗：這一批每個請求都報錯
            for r in batch:
                r.err = e
        finally:
            for r in batch:
                r.done.set()


def _submit(k, images):
    """合批時：排進隊伍、等送件執行緒做完。"""
    global _sender
    r = _Req(k, images)
    with _COND:
        if _sender is None:
            _sender = threading.Thread(target=_send_loop, name="chip_sender", daemon=True)
            _sender.start()
        _PENDING.append(r)
        _COND.notify()
    if not r.done.wait(WAIT_S):
        raise TimeoutError(f"晶片子程序 {WAIT_S} 秒沒有回應")
    if r.err is not None:
        raise RuntimeError(f"晶片呼叫失敗：{type(r.err).__name__}: {r.err}") from r.err
    return r.out


class RemoteHef:
    """跟 hailo_infer.Hef 一樣的介面，但真正跑在子程序裡。合批時子程序裡的模型設成 batch MAX_BATCH。"""

    def __init__(self, path, batch=1):
        import numpy as np
        self._np = np
        self.path, self.batch = Path(path), max(batch, MAX_BATCH)
        with _LOCK:
            _SPECS.append((str(self.path), self.batch))
            self._k = len(_SPECS) - 1
            if _proc is None:
                atexit.register(_stop)
                atexit.register(_close_shms)   # 比 _stop 先執行（atexit 後登記的先跑）
            else:
                _stop()            # 已經有子程序（開著別的模型）：全部一起重開，模型才會在同一個 VDevice 上
            info = _start()
            self.input_name, self.input_shape, self.outputs, self.original_names = info[self._k]
            _SHM[self._k] = _Shm(self.batch, self.input_shape, self.outputs)

    def __call__(self, images):
        images = self._np.ascontiguousarray(images, dtype=self._np.uint8)
        if images.ndim == 3:
            images = images[None]
        if images.shape[1:] != self.input_shape:
            raise ValueError(f"輸入形狀 {images.shape[1:]}，HEF 要 {self.input_shape}")
        return _submit(self._k, images) if MAX_BATCH > 1 else _call(self._k, images)

    def close(self):
        pass                       # 子程序在程式結束時一起關（atexit）

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
