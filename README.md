# 月子中心嬰兒室安全輔助 · ASUS UGen AI League

**English summary.** Many postpartum care centres in Taiwan already film each crib in the nursery and stream it to parents.
Nobody watches those streams continuously: one nurse looks after several newborns. Our edge box (ASUS UGen300 /
Hailo-10H) is designed to read those existing per-crib cameras and, when a baby's nose and mouth stay out of sight for 3 seconds
(face-down, or covered by a blanket or swaddle), quietly tells the nurse first. It is an observation aid, not a medical
device, and it makes no claims about SIDS.

- **How it works**: RTMO finds people and keypoints → each head is rotated upright → YuNet looks for a face; in crib mode
  the whole crib is also searched directly, because a blanket-covered body often hides the person from the pose model.
  A per-crib state machine raises an alert after 3 s without a visible airway. A second, independent check, a 2B
  vision-language model (Qwen2-VL-2B, which Hailo builds for the 10H), looks at the head crop: it can add a
  "please look", never cancel an alert. Caregiving (an adult at the crib) pauses the timer, and an empty crib turns gray
  instead of red when it matches its empty reference image.
- **Results so far (proxies, no real nursery data yet)**: real infant photos (SyRIP, internal only) 6.1% false alerts,
  98.6% alerted when the airway is hidden; six doll videos, 3-s rule at 1 FPS per crib: 31/41 danger segments caught (76%),
  7/40 false alerts; two new per-crib clips: 3/7 caught, 0/7 false alerts, the misses include an ear taken for a face;
  laptop CPU handles about 8 cribs at 1 FPS; the VLM separated safe from dangerous head crops with AUC 0.98 on clips it
  had never seen. Details: [`docs/architecture.md`](./docs/architecture.md), [`docs/影片測試結果.md`](./docs/影片測試結果.md).
- **Run it**: `uv sync`, then `uv run python infant_face_v3.py your_image.jpg`. Evaluation scripts are listed below.
  The datasets and videos are not in the repo (licensing and privacy); doll images are our own.

---

很多產後護理之家（月子中心）的嬰兒室，本來就有每床一支的直播鏡頭（給家長看）。一台邊緣裝置（ASUS UGen300／Hailo-10H）
判讀這些鏡頭：寶寶臉朝下、口鼻被被子或包巾蓋住，連續 3 秒看不到口鼻，就先安靜通知護理師——**讓護理師比家長先知道**。
AI 判讀在本地、不另存影像；定位是輔助觀察，不是醫療器材，不宣稱預防 SIDS。

參賽賽道：Battlefield Lightning。Stage I 截止 **2026/10/14 17:00**（英文簡報＋≤3 分鐘英文影片，用筆電做，不需硬體）。

> **2026/10/8 方向轉向**：原題目是「托嬰中心午睡安全監看」（一台裝置看整間午睡室）。訪談與參訪發現托嬰中心的監視器裝在高處、
> 物理上看不到臉，孩子也多半已經 7–8 個月大；月子中心的每床鏡頭從正上方拍新生兒，正好看得到臉。程式不受影響：
> 每支鏡頭只拍一張床，就是原本的「床位模式」。托嬰時期的問卷與訪談當成前導研究。

## 先讀這個

[`docs/architecture.md`](./docs/architecture.md)：判斷流程、床位模式、床位情境（照護中、空床）、VLM 第二道檢查、v1→v5 的演進。
測試數字：[`docs/娃娃測試結果.md`](./docs/娃娃測試結果.md)（擺拍照片）、[`docs/影片測試結果.md`](./docs/影片測試結果.md)（影片）。
團隊內部的決策紀錄與現況在 `docs/private/`（不在 GitHub）。

## 資料夾結構

```
.
├── README.md
├── pyproject.toml / uv.lock     # uv 專案設定與鎖定的依賴
├── infant_face_v3.py            # 主力：RTMO 找人 → 頭部轉正 → YuNet 找臉 → 每個寶寶口鼻看不看得到（真實嬰兒基準）
├── infant_face_v4.py            # 實驗：v3＋口鼻膚色檢查（擋「只露眼睛」的假臉）
├── infant_face_v5.py            # 實驗：v4＋臉分三級（正臉 ok／側臉 warn／alert）
├── face_common.py               # 共用：門檻、YuNet 找臉、配對、畫圖（不依賴 YOLOv8）
├── infant_face_spike.py         # v2（留著對照；要先 uv sync --group yolo）
├── infant_prone_spike.py        # v1：關鍵點信心度規則（已知會漏報，留著對照；要先 uv sync --group yolo）
├── eval_syrip.py                # SyRIP 真實嬰兒圖：偵測率、誤報率、姿態模型比較
├── make_relabel_set.py / score_relabel.py   # SyRIP 誤報圖人工重標、用重標結果重算
├── eval_doll.py                 # 娃娃擺拍照片評估（計分單位是每隻娃娃）
├── doll_labels.csv              # 娃娃照片的人工標註
├── eval_video.py                # 娃娃影片評估：每個床位一個警示狀態機，跟時間軸比對
├── crib_context.py              # 每床鏡頭的床位情境（原型）：照護中暫停計時、空床灰燈
├── video_labels.py              # 影片標註的共用讀取
├── doll_video_slots.csv         # 影片的床位：範圍、頭朝哪邊、床頭範圍（VLM 看這裡）、空床參考畫面
├── doll_video_labels.csv        # 影片的時間軸：每段狀態的開始、結束秒數
├── vlm_spike.py                 # VLM 第二道檢查的實驗（另一個虛擬環境，見下）
├── bench_multistream.py         # CPU 多路負載測試：一台主機能同時判讀幾支每床鏡頭
├── degrade.py                   # 模擬暗光／IR 夜視（灰階、關燈）
├── sideface_spike.py / spike_posture.py     # 早期 spike：側臉幾何、關鍵點判睡姿（結論見 docs）
├── model/face_detection_yunet_2023mar.onnx  # 人臉偵測（OpenCV YuNet，MIT）；其他權重第一次執行時自動下載
├── docs/
│   ├── architecture.md          # 技術架構（先讀這份）
│   ├── 娃娃測試結果.md / 影片測試結果.md   # 測試數字、失敗模式、限制
│   ├── 資料收集說明.md          # 娃娃照片與影片的拍攝規範
│   ├── demo規格.md              # 托嬰中心時期的通知方式與 demo 腳本（待改成月子版）
│   ├── 任務_HEF轉換測試.md      # 模型轉 Hailo HEF 的可行性測試
│   └── GitHub教學.md            # 給隊友：從註冊 GitHub 到上傳自己的工作
├── dataset/                     # 不進 repo：SyRIP、娃娃照片、娃娃影片（doll_video/，含 Harry 的時間軸 .txt）
└── outputs/                     # 不進 repo：評估結果
```

**不在 GitHub 上的東西**（授權或隱私）：`dataset/`、`outputs/`、YuNet 以外的模型權重、`docs/private/`、`docs/archive/`、
官方同意書、`local_video_*.csv`（不能公開的素材的標註，例如網路上找到的影片，只在本機評估）。需要的話跟 Harry 要。

## 安裝

1. `git clone https://github.com/StudentInTW/infant-nap-guard.git`，進到資料夾。第一次用 GitHub 的隊友看 [`docs/GitHub教學.md`](./docs/GitHub教學.md)。
2. 安裝 [uv](https://docs.astral.sh/uv/)，執行 `uv sync`（約 200 MB，不含 YOLOv8 和 PyTorch）。
3. 試跑：`uv run python infant_face_v3.py 你的圖片.jpg`。第一次執行會自動下載 RTMO 模型（約 80 MB）。

輸出在 `outputs/<檔名>_v3.jpg`：紅框＝臉看不見（ALERT）、綠框＝臉看得到、黃框＝偵測到的臉。
舊版 v1、v2 要用 YOLOv8（AGPL），先 `uv sync --group yolo`（會多裝 PyTorch）。

## 評估

### 娃娃影片：床位模式＋3 秒警示

```bash
uv run python eval_video.py                  # 全部娃娃影片（每支約 2–5 分鐘）
uv run python eval_video.py IMG_8429         # 只跑一支
uv run python eval_video.py --reuse --on 2   # 不重跑偵測，只換警示規則重算
```

每秒取 5 幀，每個床位一個警示狀態機（預設連續 3 秒看不到口鼻才警示、連續 2 秒看得到才解除），比較 v3、v3＋床位、v4＋床位、v5＋床位。
時間軸只列手不在畫面裡的乾淨片段；`label` 另有 `empty`（空床），`staged` 另有 `prone`（趴睡但口鼻看得到），都另外統計、不算進誤報。
結果在 `outputs/video_eval/`：時間軸對照圖、標註影片、每幀判定。非娃娃的 `嬰兒翻身`（AI 生成的灰階夜視畫面）要指定名稱才跑。

### 每床鏡頭的床位情境：照護中、空床

```bash
uv run python crib_context.py                # 要先跑過 eval_video.py（讀它存的每幀判定）
```

床位邊緣有動作（大人伸手進來）→ 照護中、暫停計時；大人離開後臉不見、畫面又跟這張床的空床參考畫面相符 → 空床（灰燈），不響。
空床參考在 `doll_video_slots.csv` 的 `empty_ref` 欄。結果在 `outputs/crib_context/`，說明見 [`docs/影片測試結果.md`](./docs/影片測試結果.md) §3h。

### SyRIP 真實嬰兒圖（誤報率）

```bash
uv run python eval_syrip.py --face v3        # 讀 dataset/SyRIP_Posture.zip，不用解壓；CPU 約 10 分鐘
uv run python score_relabel.py               # 用人工重標結果重算誤報率、漏判率
```

**SyRIP 的圖只能內部評估**：`outputs/syrip_eval/` 的東西不可放進簡報、影片或 GitHub。用 `--limit` 試跑會覆蓋同名的 CSV。
誤報圖人工重標：`uv run python make_relabel_set.py` 產生資料包，填好的 `labels.csv` 存成 `outputs/relabel_syrip_labels.csv` 後再跑 `score_relabel.py`。

### 娃娃擺拍照片

```bash
uv run python eval_doll.py --det-th 0.15             # 建議參數：找人門檻降到 0.15
uv run python eval_doll.py --face v4 --det-th 0.15   # v4（實驗版）
```

**計分單位是每隻娃娃**：一張照片兩隻都危險、只抓到一隻時，用整張圖算會被判對。數字和失敗模式見 [`docs/娃娃測試結果.md`](./docs/娃娃測試結果.md)。

### CPU 多路負載

```bash
uv run python bench_multistream.py                         # 720p、1080p，1–16 個工作程序
uv run python bench_multistream.py --res 720 --workers 1 8 --sec 10   # 試跑
```

### VLM 第二道檢查（實驗，需要 NVIDIA GPU）

另開虛擬環境，主程式不需要 PyTorch：

```bash
uv venv .venv-vlm --python 3.12
uv pip install --python .venv-vlm torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-vlm transformers accelerate bitsandbytes opencv-python pillow
.venv-vlm/Scripts/python vlm_spike.py IMG_8429 IMG_8432    # 要先跑過 eval_video.py
```

Qwen2-VL-2B-Instruct（Apache-2.0），語言模型 4-bit；每幀問「看得到鼻子和嘴巴嗎」等三題，取 Yes／No 的機率。

### 模擬暗光／IR 夜視

三支評估都能加 `--gray`（去掉顏色）和 `--dim 0.3`（亮度 ×0.3 加雜訊），結果另外放、不蓋掉原圖基準。這只是近似，要驗證還是得用夜視攝影機實拍。

## 現在進度（2026/10/10）

- **真實嬰兒基準是 v3**：SyRIP 700 張（人工重標）偵測率 97.7%、口鼻看得到時誤報 6.1%、口鼻看不到時 98.6% 會警示。SyRIP 是清醒的居家照片，不是嬰兒室。
- **娃娃擺拍**（114 張、185 隻）：臉朝下趴睡 25/25 會警示、誤報 4.4%；床位模式把警示召回從 62% 拉到 90%。
- **娃娃影片**（6 支，3 秒規則）：v3＋床位危險片段抓到 31/41（76%，每床 1 FPS）、安全片段誤報 7/40。
  10/10 補拍的兩支每床鏡頭影片：3/7、誤報 0/7。新發現的漏抓：頭轉開時偵測器把**耳朵當成臉**；只蓋住嘴、眼睛露出也還會漏。
- **床位情境（原型）**：兩段空床結束時都是灰燈（原本一直響）、沒有危險片段被判成空床、沒有新增誤報。
- **運算量**：筆電 CPU（i7-11800H）完整流程每床 1 FPS 可同時判讀約 8 床，其中 7–8 成運算是姿態模型（要搬上 UGen300 的部分）。
- **VLM 第二道檢查**（筆電 GPU）：在沒看過的兩支影片上分辨安全／危險的頭部畫面 AUC 0.98；偵測器漏掉的 13 幀全部補到，安全畫面多 9 次提醒；判不出趴睡。只能加「請看一眼」，不能取消警示。
- **還沒有**：真實嬰兒室畫面、新生兒影片、實機（晶片 10/26 入圍後才寄到）。真實場域測試要等 IRB 核准（Stage II）。

細節見 [`docs/architecture.md`](./docs/architecture.md)、[`docs/影片測試結果.md`](./docs/影片測試結果.md)。

## 範圍護欄（Stage I 不要做）

晶片 10/26 才到，任何「跑在 UGen300 實機上」的事（部署、量化、實機效能）**全部等入圍**；Stage I 只在筆電上做概念驗證，唯一例外是 HEF 轉換可行性測試（不需要晶片）。Stage I 不訓練模型。交付物是**英文**。
