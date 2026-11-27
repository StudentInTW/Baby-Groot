"""
ntfy_push.py
紅燈推播到護理師手機（決賽展演用，2026/11/28）：live_wall.py 某床變紅燈時 POST 到 ntfy（預設 https://ntfy.sh），
手機裝 ntfy app、訂閱同一個 topic 就會跳通知（iPhone 也可以；Harry 11/28 同意訊息經過 ntfy.sh）。

- topic 只從環境變數 FG_NTFY_TOPIC 讀，不能寫進 repo：repo 是公開的，知道 topic 的人都能訂閱、也能往裡面發訊息。
  沒設就不推播（畫面照常）。伺服器可以用 FG_NTFY_SERVER 換（例如自架的 ntfy），預設 https://ntfy.sh。
- 只送文字：標題「FirstGlance · Crib 03」、內容「Nose and mouth hidden for 3 s. Check now.」。不送畫面、不送任何影像。
  用 ntfy 的 JSON 發布（POST 到伺服器根目錄），標題裡的「·」才不會因為 HTTP 標頭只能 latin-1 而變亂碼。
- 優先度 5（ntfy 的 urgent／max）。手機會不會震動看手機的設定（iPhone：設定 → 聲音與觸覺 → 觸覺「一律播放」、
  ntfy 的通知要開聲音；展演時關專注模式）。
- 背景一條執行緒送，判讀和畫面不等它：每次逾時 TIMEOUT_S 秒、失敗隔 RETRY_GAP_S 秒再試一次；還是送不出去只記下來
  （on_result 回呼、終端機），判讀照常。佇列滿了（網路斷很久）就丟掉新的，不會越積越多。

單獨測試（量從送出到手機跳通知要幾秒）：
    $env:FG_NTFY_TOPIC="<topic>"; uv run python -m demo.ntfy_push               # 送一則，手機跳通知時在終端機按 Enter
    $env:FG_NTFY_TOPIC="<topic>"; uv run python -m demo.ntfy_push --repeat 3    # 連測 3 次，印每次秒數和中位數
    uv run python -m demo.ntfy_push --health                                     # 只測連得到 ntfy 伺服器（不發訊息）
"""

import argparse
import json
import os
import queue
import statistics
import threading
import time
import urllib.error
import urllib.request

TOPIC_ENV, SERVER_ENV = "FG_NTFY_TOPIC", "FG_NTFY_SERVER"
DEFAULT_SERVER = "https://ntfy.sh"
TIMEOUT_S = 4        # 一次 POST 最多等幾秒（逾時短：網路不通時很快放棄，不會一直佔著）
RETRY_GAP_S = 1.0    # 失敗後隔幾秒再試一次（只試一次）
PRIORITY = 5         # ntfy 的 urgent／max
PRODUCT = "FirstGlance"
ALERT_TEXT = "Nose and mouth hidden for 3 s. Check now."   # 跟 demo/ui/app.js 的 ALERT_TEXT 同一句話（推播版加了 Check now）


def crib_title(crib):
    """推播標題：FirstGlance · Crib 03。"""
    return f"{PRODUCT} · Crib {crib}"


class Notifier:
    """背景推播。send() 馬上回來（只放進佇列）；送完呼叫 on_result(item, ok, 秒數, 說明)。
    item 是 send() 給的 dict，另外帶 queued_at（time.monotonic()）。"""

    def __init__(self, topic=None, server=None, on_result=None, maxsize=32):
        self.topic = (topic if topic is not None else os.environ.get(TOPIC_ENV, "")).strip()
        self.server = (server or os.environ.get(SERVER_ENV) or DEFAULT_SERVER).rstrip("/")
        self.on_result = on_result
        self.q = queue.Queue(maxsize=maxsize)
        self.sent = self.failed = self.dropped = 0
        self.latencies = []   # 每則從 send() 到伺服器回 200 的秒數
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="ntfy").start()

    @property
    def enabled(self):
        return bool(self.topic)

    def describe(self):
        """畫面、終端機上顯示的狀態（不印 topic：錄螢幕、投影時才不會露出來）。"""
        host = self.server.split("://", 1)[-1]
        return f"on (ntfy · {host})" if self.enabled else f"off ({TOPIC_ENV} not set)"

    def send(self, title, message, **extra):
        """放進佇列就回來；沒開推播或佇列滿了回傳 False。extra 會原樣帶到 on_result 的 item。"""
        if not self.enabled:
            return False
        item = {"title": title, "message": message, "queued_at": time.monotonic(), **extra}
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            print(f"[推播] 佇列滿了，丟掉：{title}", flush=True)
            return False

    def post(self, title, message):
        """同步送一則（背景執行緒、--push-test 用）。回傳 (成功與否, 說明)。"""
        body = json.dumps({"topic": self.topic, "title": title, "message": message, "priority": PRIORITY}).encode("utf-8")
        req = urllib.request.Request(self.server + "/", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                info = json.loads(r.read().decode("utf-8") or "{}")
                return r.status == 200, f"id {info.get('id', '?')}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}"
        except Exception as e:   # 逾時、DNS、連不上、憑證……都只記下來
            return False, f"{type(e).__name__}: {str(e)[:80]}"

    def _run(self):
        while True:
            item = self.q.get()
            ok, info = self.post(item["title"], item["message"])
            if not ok:
                time.sleep(RETRY_GAP_S)
                ok, info = self.post(item["title"], item["message"])
            dt = time.monotonic() - item["queued_at"]
            if ok:
                self.sent += 1
                self.latencies.append(dt)
            else:
                self.failed += 1
            print(f"[推播] {'送出' if ok else '失敗'} {dt:.2f} 秒：{item['title']}（{info}）", flush=True)
            if self.on_result is not None:
                try:
                    self.on_result(item, ok, dt, info)
                except Exception as e:   # 回呼出錯也不能讓推播執行緒死掉
                    print(f"[推播] 回呼出錯：{type(e).__name__}: {e}", flush=True)


def health(server=None):
    """只問 ntfy 伺服器的 /v1/health（不發訊息），回傳 (成功與否, 秒數, 說明)。"""
    server = (server or os.environ.get(SERVER_ENV) or DEFAULT_SERVER).rstrip("/")
    tic = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(server + "/v1/health"),
                                    timeout=TIMEOUT_S) as r:
            return r.status == 200, time.monotonic() - tic, r.read().decode("utf-8", "replace").strip()
    except Exception as e:
        return False, time.monotonic() - tic, f"{type(e).__name__}: {str(e)[:80]}"


def main():
    ap = argparse.ArgumentParser(description="送一則測試推播，量到手機跳通知要幾秒")
    ap.add_argument("--crib", default="03", help="標題裡的床號（預設 03）")
    ap.add_argument("--repeat", type=int, default=1, help="連測幾次")
    ap.add_argument("--health", action="store_true", help="只測連得到 ntfy 伺服器（不發訊息）")
    args = ap.parse_args()
    if args.health:
        ok, dt, info = health()
        print(f"{'連得到' if ok else '連不到'}（{dt:.2f} 秒）：{info}")
        return
    n = Notifier()
    if not n.enabled:
        raise SystemExit(f"要先設環境變數 {TOPIC_ENV}（手機 ntfy app 訂閱的 topic）")
    results = []
    for i in range(args.repeat):
        input(f"[{i + 1}/{args.repeat}] 手機準備好（鎖定畫面、專注模式關掉）後按 Enter 送出…")
        tic = time.monotonic()
        ok, info = n.post(crib_title(args.crib), ALERT_TEXT)
        sent = time.monotonic() - tic
        print(f"  伺服器{'收到' if ok else '沒收到'}：{sent:.2f} 秒（{info}）")
        if not ok:
            continue
        input("  手機一跳通知（或一震動）就按 Enter…")
        total = time.monotonic() - tic
        results.append(total)
        print(f"  從送出到手機跳通知：{total:.2f} 秒（含按鍵反應約 0.2–0.3 秒）")
    if results:
        print(f"共 {len(results)} 次：中位數 {statistics.median(results):.2f} 秒、最慢 {max(results):.2f} 秒")


if __name__ == "__main__":
    main()
