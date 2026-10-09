# CLAUDE.md

給在這個 repo 工作的 Claude（和新接手的人）。

## 這是什麼

**2026/10/8 起方向轉向（Harry 提案，全隊 10/9 確認）**：產後護理之家（月子中心）嬰兒室的即時安全輔助——「讓護理師比家長先知道」。一台邊緣裝置（ASUS UGen300／Hailo-10H）判讀嬰兒室既有的每床直播鏡頭，寶寶臉朝下、口鼻被蓋住時先安靜通知護理師，AI 判讀在本地、不另存影像。參加 ASUS UGen AI League（Battlefield Lightning 賽道）。

原題目是「托嬰中心午睡安全監看」（同一台裝置看整間午睡室的多名嬰兒）；10/5–10/8 的訪談與參訪發現托嬰中心的監視器在物理上看不到臉、孩子多半 7–8 個月以上，所以轉向。**轉向的理由、證據與淘汰的方向見 `docs/private/decisions.md`「2026/10/8 方向轉向」與 `docs/private/status.md` §0。** `README.md`、`docs/architecture.md` 目前仍是托嬰中心版本，要改寫成月子版。程式與評估不受影響：每支鏡頭只拍一張床，等於現有的床位模式。

**Stage I 截止 2026/10/14 17:00**：英文簡報（正文 ≤20 頁，要附這個 repo 的連結）＋ ≤3 分鐘英文影片（上傳 YouTube、設「知道連結者可觀看」）。

## 開場先讀

1. `docs/private/context.md`——薄的入口，指向其他文件。**`docs/private/` 只在 Harry 的電腦上，不在 GitHub**；沒有的話就讀 `docs/architecture.md`、`README.md` 和 `docs/` 裡其他文件。
   - 從那裡分流：`docs/private/decisions.md`（已定案與理由）、`docs/private/status.md`（現況數字與待辦）、`docs/architecture.md`（公開的技術架構與 v1→v5 演進）。
2. `README.md`——怎麼跑、資料夾結構。

## 慣例

- **回覆用繁體中文。**
- **主力程式是 `infant_face_v3.py`**：RTMO 找出所有人 → 頭部依身體方向轉正 → YuNet 找臉 → 判斷每個嬰兒的口鼻看不看得到。`infant_face_spike.py`（v2）和 `infant_prone_spike.py`（v1）留著對照，改版時不要直接覆蓋。
- **`infant_face_v4.py`（v3＋口鼻膚色檢查）、`infant_face_v5.py`（v4＋臉三級 ok/warn/alert）都是實驗版，還沒採用**：v4 擋「只露眼睛」的假臉（2026/9/23 改良口鼻膚色後 SyRIP 誤報 16.6%→10.9%）；v5 加側臉 warn（娃娃影片好，但真實嬰兒召回/誤報仍輸 v3）。**v3 仍是真實嬰兒基準。** 細節見 `docs/architecture.md`、`docs/影片測試結果.md` §3d、`docs/娃娃測試結果.md` §5。
- **YOLOv8 是選配**（AGPL，只有 v1、v2 和評估比較會用到）：要用時 `uv sync --group yolo`。平常 `uv sync` 不會裝它，也不需要 PyTorch。
- **這些不進 repo**：`dataset/`、`outputs/`、模型權重（YuNet 除外）、`docs/private/`（含 context/decisions/status）、`docs/*.docx` 同意書。`.gitignore` 已設定好。
- **SyRIP 只能內部評估**：它的圖、以及由它產生的輸出圖，不可放進簡報、影片或 repo。
- **評估流程**（改了 `infant_face_v3.py` 就三套都要重跑）：
  - SyRIP（測誤報）：`eval_syrip.py --face v3` 跑分 → `score_relabel.py` 用人工標註重算真實的誤報率。注意用 `--limit` 試跑會覆蓋 `outputs/syrip_eval/` 裡的 CSV。
  - 娃娃影片（測床位模式和警示狀態機）：`eval_video.py`，床位在 `doll_video_slots.csv`、時間軸在 `doll_video_labels.csv`（只列手不在畫面裡的片段）。結果見 `docs/影片測試結果.md`。
  - 娃娃擺拍（測危險情境召回）：`eval_doll.py`，標註在 `doll_labels.csv`。**計分單位是「每隻娃娃」**，不是每張照片——同框兩隻都危險、只抓到一隻時，用整張圖算會被判對。結果和失敗模式見 `docs/娃娃測試結果.md`。
  - 灰階／暗光（模擬 IR 夜視，選跑）：三支都有 `--gray`、`--dim 0.3`（`degrade.py`），結果另外放 `_<條件>` 的檔名／資料夾、不蓋掉原圖基準；SyRIP 用 `score_relabel.py --eval-dir outputs/syrip_eval_<條件>` 算。**灰階下口鼻膚色檢查（v4/v5）失效**，結果見 `docs/娃娃測試結果.md` §8。
- **Stage I 不訓練模型**：用現成、授權寬鬆的模型加上演算法設計。

## 範圍護欄

晶片 10/26 入圍後才寄到，所以實機部署、量化、實機效能測試都等入圍。唯一例外是 HEF 轉換可行性測試（不需要晶片，見 `docs/任務_HEF轉換測試.md`）。
