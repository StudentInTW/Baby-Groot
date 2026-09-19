# GitHub 教學（給士丞，其他隊友也適用）

我們的程式放在 GitHub 的**私人** repo，只有受邀的隊友看得到。這份教學從註冊帳號開始，一步一步帶你做到：看程式 → 抓到自己電腦 → 跑起來 → 把自己的工作傳上去。

**先認識幾個名詞**
- **repo**：放在 GitHub 上、大家共用的專案資料夾。
- **clone**：把 repo 整個複製到自己的電腦。
- **commit**：把你的修改存成一個「版本」，附一句說明。
- **push**：把你電腦上的 commit 傳到 GitHub；**pull**：把 GitHub 上別人的新版本抓下來。
- **branch（分支）**：你自己的工作線，在上面改東西不會影響主線（`main`）。
- **Pull Request（PR）**：請 Harry 把你分支上的修改合併進主線。

## 1. 註冊 GitHub、接受邀請

1. 到 https://github.com 按 **Sign up**，用 email 註冊，選一個帳號名稱（username）。
2. 去信箱收驗證信。
3. GitHub 會要求開啟**雙重驗證（2FA）**：用手機 app（Google Authenticator 或 Microsoft Authenticator）掃 QR code。**recovery codes 一定要存起來**，手機弄丟時要靠它登入。
4. 右上角頭像 → **Settings → Emails**，勾選「**Keep my email addresses private**」，記下下方那個 `數字+帳號@users.noreply.github.com`，第 3 節會用到。這樣之後 repo 公開時，別人看不到你的私人信箱。
5. 把你的**帳號名稱**傳給 Harry。Harry 邀請你之後會收到 email，按 **Accept invitation**。
6. 接受後打開 https://github.com/StudentInTW/infant-nap-guard ，就能在瀏覽器看所有檔案。

**只是要看程式的話，做到這裡就夠了。**

## 2. 在瀏覽器看程式（不用安裝任何東西）

- 點檔名就能看內容。在 repo 頁面按鍵盤的 `.`（句點），會用網頁版 VS Code 打開整個 repo，找檔案比較方便。
- 你最需要看的：

| 檔案 | 看什麼 |
|---|---|
| `docs/任務_HEF轉換測試.md` | 你的任務說明 |
| `infant_face_v3.py` | 主程式。`load_rtmo()` 是 RTMO 的前處理、後處理；`head_faces()`、`detect_faces_v3()` 是 YuNet 怎麼用 |
| `infant_face_spike.py` | `detect_faces()`：YuNet 對整張圖的用法 |
| `README.md` | 專案總覽、怎麼跑 |

- `dataset/`（資料）、`outputs/`（結果）、模型權重（YuNet 除外）不在 GitHub 上，這是正常的。

## 3. 把 repo 抓到電腦（在 WSL 的 Ubuntu 裡做）

Hailo 編譯器只能在 Linux 跑，所以 git 和程式也都放在 WSL 的 Ubuntu 裡，一個環境做到底。

**3-0. 還沒裝 WSL 的話**：在 Windows 開始選單搜尋 PowerShell → 按右鍵「以系統管理員身分執行」→ 輸入下面這行，完成後重開機。開機後 Ubuntu 會請你設定帳號和密碼（這組密碼之後打 `sudo` 指令時會用到）。

```
wsl --install
```

以下指令都在 **Ubuntu 終端機**裡輸入（開始選單搜尋「Ubuntu」）。

**3-1. 設定 git 身分**（只要做一次）

```
git config --global user.name "你的名字"
git config --global user.email "第 1 節記下的 noreply 信箱"
git config --global init.defaultBranch main
```

**3-2. 建立 SSH 金鑰，讓電腦可以連到 GitHub**（只要做一次）

```
ssh-keygen -t ed25519 -C "你的 noreply 信箱"
```

問問題時直接按 Enter 三次。接著印出公鑰：

```
cat ~/.ssh/id_ed25519.pub
```

複製印出來的整行（`ssh-ed25519` 開頭的那行）→ GitHub 右上角頭像 → **Settings → SSH and GPG keys → New SSH key** → 貼上 → **Add SSH key**。然後測試：

```
ssh -T git@github.com
```

第一次會問 `yes/no`，打 `yes`。看到 `Hi 你的帳號! You've successfully authenticated` 就成功了。

**3-3. clone**：

```
cd ~
git clone git@github.com:StudentInTW/infant-nap-guard.git
cd infant-nap-guard
```

**小技巧**：在 Windows 檔案總管的網址列輸入 `\\wsl$`，就能看到 Ubuntu 裡的檔案，可以直接拖曳圖片進去。

## 4. 把程式跑起來

1. 安裝系統套件（OpenCV 在 Ubuntu 上需要）：

   ```
   sudo apt update && sudo apt install -y libgl1 libglib2.0-0
   ```

2. 安裝 uv（Python 套件管理工具），**裝完關掉終端機再重開**：

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. 回到專案資料夾，安裝專案需要的套件。第一次會下載 Python 3.12 和需要的套件（約 200 MB），要等一下：

   ```
   uv sync
   ```

4. 拿一張有人的照片（自己拍的就好，**不要用網路上的嬰兒照片**）放進專案資料夾，執行：

   ```
   uv run python infant_face_v3.py 你的圖片.jpg
   ```

   第一次會自動下載 RTMO 模型（約 80 MB）。結果圖在 `outputs/你的圖片_v3.jpg`。

## 5. 開自己的分支做事

主線（`main`）由 Harry 管，你的東西放在自己的分支 `hef-test`：

```
git switch -c hef-test
```

- 你的腳本、筆記放在 `hailo/` 資料夾（自己建）。
- **不要放進 git**：模型檔（`.onnx`、`.hef`、`.har`）、Hailo 的安裝檔、校正用的圖片、log 檔。`.gitignore` 已經擋掉大部分，但 `git add` 前還是看一下。

**存一個版本並傳上去**（每做完一段就做一次）：

```
git status
git add hailo/
git commit -m "說明做了什麼，例如：YuNet 轉 HEF 成功"
git push -u origin hef-test
```

- `git status` 會列出改了哪些檔案；`git add` 只加你要傳的東西。
- `git push -u origin hef-test` 第一次這樣打，之後只要打 `git push`。

**請 Harry 合併**：push 之後打開 repo 頁面，會出現 **Compare & pull request** 按鈕 → 寫一下做了什麼 → **Create pull request**。

**拿 Harry 的新版本**：在 `hef-test` 分支上執行，會把主線的新東西合進來：

```
git pull origin main
```

## 6. 用 VS Code 比較好操作（選用）

在 Windows 裝 VS Code，再裝「**WSL**」擴充套件。在 Ubuntu 終端機進到專案資料夾，打 `code .`，就能用 VS Code 編輯 WSL 裡的檔案。左邊的「**原始檔控制**」可以用點的完成 add、commit、push。

## 7. 常見問題

| 看到什麼 | 怎麼辦 |
|---|---|
| `Permission denied (publickey)` | SSH 金鑰沒加到 GitHub，重做 3-2 |
| `Repository not found` | 還沒接受邀請，或網址複製錯了 |
| `ImportError: libGL.so.1` | 還沒裝系統套件，做第 4 節第 1 步 |
| push 被拒絕，出現 `rejected` 或 `fetch first` | 別人先傳了新版本：先 `git pull`，再 `git push` |
| 出現 `CONFLICT` | 你和別人改到同一個地方。不要亂刪，截圖問 Harry 或 Claude |
| 不小心 commit 了大檔案或不該傳的東西 | **先不要 push**，問 Harry；已經 push 了就馬上跟 Harry 說 |

卡住時把**完整的錯誤訊息**貼給 Claude 或 Harry，通常很快就能解決。

## 8. 絕對不要傳上 GitHub

- 密碼、token、Hailo 帳號資訊
- SyRIP、網路上的嬰兒照片、任何人的個人資料
- 超過 100 MB 的檔案（GitHub 會拒收）
