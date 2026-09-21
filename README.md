# 托嬰中心午睡安全監看 · ASUS UGen AI League

一顆邊緣 AI 晶片（ASUS UGen300 / Hailo-10H），同時監測整間托嬰中心午睡室的多名嬰兒，
即時標記出現**趴睡、口鼻遮蔽或異常姿勢**的孩子。影像全程不出教室——隱私是硬約束也是賣點。
定位為輔助托育人員的 co-pilot，不是醫療器材。

參賽賽道：Battlefield Lightning（輕量運算組）。Stage I 截止：**2026/10/14**（英文簡報 + ≤3 分鐘英文影片，用筆電做，不需硬體）。

---

## 先讀這個

**接手前先讀 [`docs/context.md`](./docs/context.md)** — 裡面有題目、**關鍵決策與理由**、現況、下一步。
決策背後的權衡都在那裡，先讀可避免走回已淘汰的方向。

## 資料夾結構

```
.
├── README.md                    # 操作入口
├── pyproject.toml               # uv 專案設定與依賴
├── uv.lock                      # 鎖定依賴版本
├── infant_face_v3.py            # 風險前置實驗 v3：頭部轉正再找臉、合併重複框，姿態模型改用 RTMO（主力）
├── infant_face_v4.py            # v4：v3 再加「口鼻檢查」——找到的臉，下半臉不像皮膚（被衣領、玩偶、被子蓋住）就不算數
├── face_common.py               # v2、v3 共用：門檻、YuNet 找臉、配對、畫圖（不依賴 YOLOv8）
├── infant_face_spike.py         # v2：整張圖找臉、YOLOv8（留著對照；要先 uv sync --group yolo）
├── infant_prone_spike.py        # v1：關鍵點信心度規則（已知會漏報，留著對照；要先 uv sync --group yolo）
├── eval_syrip.py                # 用 SyRIP 算偵測率／誤報率，比較 YOLOv8、RTMPose、RTMO（--face v2／v3／v4）
├── eval_doll.py                 # 用娃娃擺拍照片評估，計分單位是「每隻娃娃」（偵測率、警示召回率、誤報率）
├── doll_labels.csv              # 娃娃照片的人工標註：每隻娃娃的位置和口鼻看不看得到
├── eval_video.py                # 用娃娃影片評估床位模式：每個床位的警示狀態機，跟時間軸比對
├── doll_video_labels.csv        # 娃娃影片的時間軸標註：每段狀態的開始、結束秒數
├── doll_video_slots.csv         # 娃娃影片的床位：每隻娃娃的頭部範圍和頭朝哪邊
├── make_relabel_set.py          # 產生 SyRIP 誤報圖人工重標的資料包
├── score_relabel.py             # 用人工重標結果重算誤報率、漏判率
├── main.py                      # uv init 產生的基本入口，目前非主流程
├── dataset/                     # 測試影像與資料集（只供內部測試，不可放進成品）
│   ├── SyRIP.zip                # ACLab 提供：嬰兒姿態（COCO 關鍵點）
│   ├── SyRIP_Posture.zip        # ACLab 提供：700 張真實圖＋睡姿標籤
│   ├── infant_jpg_without_video/  # 娃娃擺拍照片（自己拍的，可放進成品；照片本身不進 Git）
│   ├── doll_video/              # 娃娃影片和 Harry 的時間軸說明（.txt）
│   ├── infant_1.jpg
│   ├── infant_2.png
│   └── infant_3.png
├── model/                       # 模型權重
│   ├── yolov8n-pose.pt                    # 姿態模型（Ultralytics，AGPL-3.0）
│   └── face_detection_yunet_2023mar.onnx  # 人臉偵測（OpenCV YuNet，MIT）
├── outputs/                     # v2 輸出圖（<檔名>_face.jpg）；syrip_eval/ 是 SyRIP 評估結果（只供內部）；doll_eval/ 是娃娃測試結果
├── docs/                        # 專案決策與提案文件
│   ├── context.md               # 背景、決策理由、現況、下一步
│   ├── StageI行動清單.md        # 到 10/14 的逐項清單與每週節奏
│   ├── 任務_SyRIP重標.md        # 給宜臻的誤報圖重標任務說明
│   ├── 娃娃測試結果.md          # 娃娃擺拍測試的數字、失敗模式、補拍建議
│   ├── 影片測試結果.md          # 娃娃影片測試：床位模式的結果、實作細節、限制
│   ├── 資料收集說明.md          # 娃娃照片、AI 生成圖、影片要收什麼、怎麼分類
│   ├── demo規格.md              # 偵測到危險後怎麼通知、情報版、demo 影片腳本、訪談題目
│   ├── 任務_AI生成圖.md         # 給嘉晏哥哥：用 Stable Diffusion 生成測試圖
│   ├── 任務_HEF轉換測試.md      # 給士丞：模型轉 HEF 的可行性測試
│   ├── GitHub教學.md            # 給隊友：從註冊 GitHub 到上傳自己的工作
│   ├── ASUS UGen AI League_著作財產權授權暨肖像授權同意書.docx  # 官方同意書，每位隊員各簽一張
│   └── archive/
│       ├── 招募版提案.md        # 已歸檔；痛點數據與來源在這份
│       └── 拍攝計畫_娃娃與影片.md  # 舊版拍攝清單（太細），已由資料收集說明取代
└── .venv/                       # uv 建立的虛擬環境（不提交 Git）

```

## 從 GitHub 抓下來之後（隊友看這裡）

第一次用 GitHub 的話，照 [`docs/GitHub教學.md`](./docs/GitHub教學.md) 一步一步做（從註冊帳號開始）。

1. 接受 Harry 寄的 GitHub 邀請（repo 是私人的）。
2. `git clone https://github.com/StudentInTW/infant-nap-guard.git`（或照 GitHub 教學用 SSH），進到資料夾。
3. 安裝 [uv](https://docs.astral.sh/uv/)，然後執行 `uv sync` 裝好套件（約 200 MB，不含 YOLOv8 和 PyTorch）。
4. 用自己的圖片試跑：`uv run python infant_face_v3.py 你的圖片.jpg`。第一次執行會自動下載 RTMO 模型（約 80 MB）。

**不在 GitHub 上的東西**（授權或隱私的關係）：`dataset/`（SyRIP、測試圖、娃娃照片）、`outputs/`（評估結果）、`model/` 裡的 YOLOv8 權重、`docs/context.md`、`docs/archive/`、官方同意書。需要的話跟 Harry 要。

## 怎麼跑風險前置實驗

目的：驗證「每個嬰兒的臉看不看得到」——看不到就標 ALERT，產出可以給隊友看的截圖。

```bash
uv sync
uv run python infant_face_v3.py
```

不給參數就跑 `dataset/` 裡所有圖片，也可以指定圖片，例如 `uv run python infant_face_v3.py dataset/infant_2.png`。
輸出在 `outputs/<檔名>_v3.jpg`：紅框＝臉看不見（ALERT）、綠框＝臉看得到、黃框＝偵測到的臉、紅／綠圈＝用來配對臉的頭部範圍。

舊版保留作對照（要用 YOLOv8，先執行 `uv sync --group yolo`，會多裝 PyTorch）：
- v2（`infant_face_spike.py`，輸出 `<檔名>_face.jpg`）：整張圖找臉、姿態模型用 YOLOv8；躺著時橫的、倒的臉常找不到。
- v1（`infant_prone_spike.py`，輸出 `spike_out.jpg`）：用姿態模型的臉部關鍵點信心度判斷，在趴睡圖上會漏報，原因見 [`docs/context.md`](./docs/context.md) §3。

## 用 SyRIP 評估（偵測率、誤報率、模型比較）

```bash
uv run python eval_syrip.py
```

直接讀 `dataset/SyRIP_Posture.zip`（不用解壓），在 700 張真實嬰兒圖上比較 YOLOv8、RTMPose、RTMO 三個姿態模型；人臉偵測固定用 YuNet。比較 YOLOv8 要先 `uv sync --group yolo`；只比 RTMPose、RTMO 的話加 `--models rtmpose rtmo`。
**注意**：用 `--limit` 試跑也會覆蓋 `outputs/syrip_eval/` 裡同名的 CSV，試跑完要再跑一次完整的，`score_relabel.py` 才算得準。
CPU 上全部跑完約 10 分鐘，可以先用 `--limit 10` 試跑；加 `--face v3` 改用 v3 的人臉判斷（輸出檔名加 `_v3`）。RTMPose／RTMO 的權重第一次執行時會自動下載到 `~/.cache/rtmlib`（約 220 MB）。
結果在 `outputs/syrip_eval/`：每個模型一份 CSV，加上誤報縮圖供人工複查。**SyRIP 的圖只能內部評估，這個資料夾的東西不可放進簡報、影片或 GitHub。**

## 用娃娃擺拍照片評估（每隻娃娃各算一次）

```bash
uv run python eval_doll.py --det-th 0.15             # 建議參數：找人門檻降到 0.15
uv run python eval_doll.py --face v4 --det-th 0.15   # v4（v3＋口鼻檢查，實驗版）
uv run python eval_doll.py --det-th 0.15 --dim 0.3   # 模擬關燈
uv run python eval_doll.py --det-th 0.15 --max-side 1280  # 模擬 1080p 攝影機
```

讀 `dataset/infant_jpg_without_video/` 的娃娃照片和 `doll_labels.csv` 的人工標註（三批共 114 張、185 隻娃娃），用 v3（或 `--face v4`）跑一遍。
**計分單位是「每隻娃娃」**：一張照片裡兩隻娃娃都危險、系統只抓到一隻時，用整張圖算會被判對，漏掉的那隻看不出來。
娃娃的標註點落在哪個人框裡就算配到那隻；配不到＝沒偵測到，沒配到娃娃的人框＝額外偵測。
結果在 `outputs/doll_eval/`：每隻娃娃一列的 CSV，加上判錯那幾張的標註圖。數字和失敗模式見 [`docs/娃娃測試結果.md`](./docs/娃娃測試結果.md)。

## 用娃娃影片評估床位模式

```bash
uv run python eval_video.py                          # 全部影片（每支約 2–3 分鐘）
uv run python eval_video.py --reuse --on 3 --off 3   # 不重跑偵測，只換警示規則重算
```

每秒取 5 幀，每個床位（`doll_video_slots.csv`）一個警示狀態機，跟時間軸（`doll_video_labels.csv`，只列手不在畫面裡的乾淨片段）比對，比較 v3、v3＋床位、v4＋床位。
結果在 `outputs/video_eval/`：時間軸對照圖、標註影片、每幀判定。數字見 [`docs/影片測試結果.md`](./docs/影片測試結果.md)。

## SyRIP 誤報圖人工重標

```bash
uv run python make_relabel_set.py
```

讀 `outputs/syrip_eval/` 的評估結果，挑出誤報圖、GT 臉看不到的圖和隨機 30 張判 OK 的圖，打亂後產生 `outputs/relabel_syrip.zip`（給標註的人）和 `outputs/relabel_syrip_key.csv`（編號對照表，自己留著）。任務說明見 [`docs/任務_SyRIP重標.md`](./docs/任務_SyRIP重標.md)。

拿回填好的 `labels.csv` 後存成 `outputs/relabel_syrip_labels.csv`，再執行 `uv run python score_relabel.py`，就會用人工標註重算每一次評估的誤報率和漏判率。

## 現在進度（一句話）

決策與提案已定案。v3（RTMO＋頭部轉正再找臉）在 SyRIP 700 張真實嬰兒圖上（經人工重標）：嬰兒偵測率 97.7%，口鼻看得到時誤報 6.1%（v2 是 19.1%），口鼻看不到時 98.6% 會警示。娃娃擺拍測試（三批 114 張、185 隻娃娃，每隻各算一次，含近似天花板的俯拍）：臉朝下趴睡 25/25 會警示、誤報 4.4%，但**口鼻被被子、玩偶、衣領蓋住的多半漏掉**——身體被蓋住時姿態模型找不到人，只露眼睛也被當成看得到臉。模擬「床位模式」（床位裡找不到人也警示）可把警示召回率從 62.1% 拉到 90.2%；口鼻檢查（v4）在娃娃上有效，但真實嬰兒誤報變 16.6%，還不採用。細節見 [`docs/娃娃測試結果.md`](./docs/娃娃測試結果.md)。娃娃影片上床位模式把危險片段抓到從 32% 拉到 84%、0 誤報（[`docs/影片測試結果.md`](./docs/影片測試結果.md)）。下一步＝把床位模式做進主程式（多嬰兒追蹤＋床位＋狀態機）、報名（Workplace AI／Lightning USB／實體組）。細節見 [`docs/context.md`](./docs/context.md) 第 7–8 節。

## 範圍護欄（Stage I 不要做）

晶片 10/26 才到，所以任何「跑在 UGen300 實機上」的事（部署、量化、實機效能）**全部等入圍**；Stage I 只在筆電上做概念驗證。交付物是**英文**。
