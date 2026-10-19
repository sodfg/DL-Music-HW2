# Optional 擴充實驗流程

這是本專案後續實驗的共同名稱。WAV 本身也是基本作業的繳交要求，
只是現在先用少量音訊，為選做的 SI 建立一致的分析條件。

| 階段 | 內容 | 狀態 |
|---|---|---|
| 1 | H1/H4/GS 總表與易讀 Markdown | 完成 |
| 2 | 統一鋼琴音色、批次 WAV、品質檢查與合成記錄 | 已實作，本機少量試轉 |
| 3 | SI 小量試算與匹配長度的批次比較 | 完成 32 秒 pilot 與 80 個 48 秒片段 |
| 4 | 模型深度比較，固定目前表示法 | 尚未開始 |
| 5 | MIDI-like 表示法比較，固定模型架構 | 尚未開始 |

Git 分支不是每個階段的必要條件。每個可獨立執行且通過驗證的功能做一次 commit；
大型表示法實驗可另開 branch。Git add/commit/push 由使用者操作。

## 第 2 步：統一 WAV 合成

這一步只把已有的 MIDI 樂譜演奏成 WAV，不重新生成音符、不載入模型、不訓練，
不需要 GPU。合成器為 FluidSynth，音色為同一份 SoundFont。
即使與 MIDI RealPlayer 用同一份 SoundFont，不同合成引擎／效果設定的聲音也可能不同。
正式比較應全部使用同一條渲染流程，不混用正弦波 preview 或手動改過音色的輸出。

目前預設：

- GM Acoustic Grand Piano，bank 0 / program 0（介面通常稱 Piano 1）。
- 44,100 Hz、雙聲道、16-bit PCM WAV。
- 固定 gain 0.1、polyphony 256、單核心合成、關閉 reverb / chorus。
- 不做各曲音量正規化；保留音符力度與來源音量控制。
- 不修改來源 MIDI；在暫存副本中統一各非打擊樂通道的音色。
- 有打擊樂通道時 piano 模式會報錯；本作業為鋼琴資料，若其他用途需保留樂器，可用 `--instrument midi`。

### 工具準備

需要可執行的 FluidSynth，及本地 `.sf2` SoundFont。本步不使用 requirements.txt，
不變更學校共用 PyTorch。Python 只使用既有的 `mido` 與標準函式庫。

本機這次使用 MIDI RealPlayer 已附帶的音色：

```text
/home/sodfg/.vscode/extensions/vaclis.midi-realplayer-0.2.5/media/GeneralUser-GS.sf2
```

來源：MIDI RealPlayer bundled GeneralUser GS，作者 S. Christian Collins。
授權文字同目錄的 `GeneralUser-GS-LICENSE.txt`，允許使用於音樂創作。
音色並未加入 Git；若之後搬到其他機器，使用同一個檔案並保存授權文字。
manifest 會記錄 SHA-256，確認它真的是同一個音色檔，不只檔名相同。

本機未全域安裝 FluidSynth。本次從 Ubuntu 套件取得並解壓到 `.tools/fluidsynth/`，
使用 `.tools/fluidsynth/bin/fluidsynth` 啟動，不修改系統套件。
該目錄已被 Git 忽略，是本機 runtime，不應直接搬到不同 Ubuntu 版本的 EDA 主機。
若系統已有 FluidSynth，省略 `--fluidsynth` 即可使用 PATH 裡的版本。

若換機器，需要該機器可執行的 FluidSynth 與相同 SoundFont；目前先使用
已驗證的本機環境，不在學校主機使用 sudo 安裝。

### 先試一首

以下指令從作業根目錄執行；本機使用 `.venv/bin/python`，已具備 mido。

```bash
.venv/bin/python baseline/render_audio.py \
  --input runs/experiments/basic_a/t08/sample_000.mid \
  --output runs/audio_trial/t08 \
  --soundfont /home/sodfg/.vscode/extensions/vaclis.midi-realplayer-0.2.5/media/GeneralUser-GS.sf2 \
  --fluidsynth .tools/fluidsynth/bin/fluidsynth
```

把 input/output 的 t08 改成 t10、t12，便可使用同一組設定試聽三組各一首。
同一輸入、音色、設定與完整輸出雜湊相符時，重跑會顯示 unchanged；
設定或輸入不同但輸出已存在時，會報錯避免不知情覆蓋。確實要重轉時加 `--overwrite`。
中途失敗可重跑，已完成且可核對的曲目會跳過。

### 批次指令（需要時才執行）

合成整個資料夾，保留子目錄對應關係，例如 t08/sample_000.mid → t08/sample_000.wav：

```bash
.venv/bin/python baseline/render_audio.py \
  --input runs/experiments/basic_a \
  --output runs/audio/basic_a \
  --recursive \
  --soundfont /home/sodfg/.vscode/extensions/vaclis.midi-realplayer-0.2.5/media/GeneralUser-GS.sf2 \
  --fluidsynth .tools/fluidsynth/bin/fluidsynth
```

測試批次時可加 `--limit 1`。Basic B 可改成 `--input runs/experiments/basic_b`
和 `--output runs/audio/basic_b`。全組 A 合成是為比較／SI；正式繳交只要求選定組的
20 首 A WAV，以及 B 的全部 9 首 WAV。

### 如何驗證與保存

每次執行在 output 目錄保存 `render_manifest.json`，內含每首：

- 來源 MIDI 與音色的 SHA-256、合成器版本及設定。
- WAV SHA-256、取樣率、聲道、秒數、峰值、RMS、到達 PCM 極值的樣本數。
- MIDI 時間長度與合成器記錄。WAV 含音色釋放尾音，比 MIDI 長是可能的。

程式會拒絕無聲、截斷、格式不符或短於 MIDI 時間軸的輸出。
若有 clipping 警告，先降低 gain，並以相同新 gain 重轉所有要比較的資料；
不能只為某一組改設定。這些檢查不保證樂曲本身好聽。

音訊與 manifest 都在 `runs/`，不加入 Git，另行備份。
程式、測試、說明文件與 .gitignore 才是本次 commit 的內容。
第 3 步已為小量試算固定 32 秒片段與 3/8/15 秒尺度；正式批次的抽樣和長度策略
尚未決定，不把少量試算當作全資料集評估。

### 測試與來源

```bash
.venv/bin/python -m unittest discover -s baseline -p 'test_render_audio.py'
```

FluidSynth CLI：<https://www.fluidsynth.org/wiki/UserManual/>

MIDI RealPlayer：<https://marketplace.visualstudio.com/items?itemName=vaclis.midi-realplayer>

## 第 3 步：SI 小量試算

SI 使用音訊的 chroma（音高類別特徵）建立自我相似矩陣，再計算 fitness scape plot，
偵測不同時間長度的重複結構。它不是旋律抄襲檢測，也不是聽感總分。
本程式呼叫官方 MusDr 的 SSM 與 scape plot 實作；只將舊版 librosa 的讀檔步驟
改為現代 keyword API，並在計算特徵前固定裁切音訊。上游原始碼保持不變。

### 本次 pilot 的明確範圍

- 真實曲：`POP909/training/001/001.mid`，以第 2 步完全相同的設定合成。
- 生成曲：t08/t10/t12 的 `sample_000.wav`，每組各一首。
- 每首取第 0 秒起的 **32 秒**，不包含 MIDI 結束後的 release tail。
- 特徵率 2 Hz；SI short 為 3–8 秒、mid 為 8–15 秒、long 為 15–32 秒。
- 範圍與 row indexing 沿用 MusDr；8 秒／15 秒邊界可被相鄰範圍共同包含。
- 繼承官方的速度／移調相似性增強及閾值設定，記錄於 `protocol.json`。
- 不隨機選曲，也不是代表性抽樣；只測試實作與時間成本。

這四個值**不能**代表各組 20 首的平均，更不能直接跟全曲 H4/GS 混成相同樣本的總分。
因此結果另外放在 `runs/metrics/si_trial/`。正式批次實驗之前，應先決定一致的抽樣、
曲子／片段長度與時間尺度；例如均取匹配的 32 小節，或同樣的固定秒數。
固定秒數方便這次控制成本，但不同速度的 32 秒含有不同小節數。

### 本機安裝（不影響原訓練環境）

音訊分析可能需要不同的 NumPy／librosa 版本，因此本機建立獨立 `.venv-si`。
這不是要求學校主機重裝環境；目前整個試算都在本機，不變更 EDA Union 的共用套件。

```bash
# 在本機作業根目錄執行，第一次才需下載：
git clone https://github.com/slSeanWU/MusDr.git HW2_scripts/MusDr
python3 -m venv .venv-si
.venv-si/bin/python -m pip install -r baseline/requirements-si.txt
```

若已經有 MusDr 或 `.venv-si`，不用重新 clone／建立。
不要安裝上游舊版 requirements，也不要更動原本的 PyTorch。
本次 MusDr commit：`b034cf86550963cf2021299eafb1001d0fe34853`。
程式還會保存實際上游原始碼 SHA-256 與分析套件版本，避免只憑版本名稱判斷相同。

### 執行

確定 `runs/audio_trial/` 已有 real、t08、t10、t12，各資料夾有 WAV 與 render manifest。
要建立 real 音訊，用第 2 步同一條指令，把 input 改為 `POP909/training/001/001.mid`，
output 改為 `runs/audio_trial/real`，其餘音色、gain、效果全部相同。

```bash
.venv-si/bin/python baseline/evaluate_si.py
```

輸出：

- `summary.md`：可在 VS Code 按 Ctrl+Shift+V 閱讀的試算表。
- `per_piece.csv`：逐首 short/mid/long SI 與原始計算耗時。
- `protocol.json`：裁切位置、音色與合成設定、分析參數、套件與原始碼資訊。
- `cache/`：每首的 SSM、fitness 矩陣與輸入音訊雜湊。

相同輸入／設定可重跑並使用 cache，不重算慢速 scape plot；改協定時指定新的
`--output`，避免蓋掉先前實驗。更換 input WAV 會依雜湊重算。
程式預先核對全部音訊 manifest，拒絕不同音色／gain／渲染設定、削波或內容不足 32 秒的資料。
若分析中斷，可重新執行；已完成的矩陣會保留。

預設是每組一首，不會自動展開 60 首或 896 首。Python scape plot 計算主要在 CPU，
原實作的成本隨長度快速增加，不是把 32 秒耗時乘幾倍就可預測完整曲子的耗時。
第一次讀音訊還可能有 librosa/Numba 編譯成本。

測試索引、時間尺度與檔案驗證：

```bash
.venv-si/bin/python -m unittest discover -s baseline -p 'test_evaluate_si.py'
```

來源：<https://github.com/slSeanWU/MusDr> 及其 `mueller_audio_tools/` 內註明的 FMP 教材。
上游程式內有完整作者及來源註記，本專案未複製上游演算法到受追蹤的檔案中。

### 本機試算驗證結果

四個片段均成功，第一次（含初始化／編譯）約 13.73 秒，其餘各約 1.8 秒。
再次執行全部使用快取，分數一致；SI window extraction 已與官方
`compute_structure_indicator` 對三個尺度核對一致。
本次 long SI 全為 0，32 秒片段對長段落重複的機會有限；這不證明整首歌沒有長期結構。
完整結果在 `runs/metrics/si_trial/summary.md`，套件快照在同目錄的 `environment.txt`。
該試算資料與原先全曲 H4/GS 的樣本數不同，因此維持獨立報告。

## 第 3 步延伸：48 秒批次 SI 比較

由於 60 首生成曲最短約 61.44 秒，這次採用每首開頭固定 **48 秒**，
增加長片段重複的觀察範圍，避免把 32 秒 pilot 的結果誤當成全組結果。

研究計畫：

- t08/t10/t12 各使用既有全部 20 首，不丟棄較差的生成樣本，共 60 首。
- 真實曲從 896 首中篩選長度至少 48 秒者，共 895 首，再以 seed 42 無放回均勻抽 20 首。
- 音色統一 GeneralUser GS，piano，44,100 Hz PCM16，關閉效果；本批次 gain **0.025**。
- 批次試轉時發現 gain 0.1 對部分密集曲子有削波，因此停止該次嘗試。
  舊的 `runs/metrics/si48/` 僅是未完成中間資料，不是正式結果。
  較低 gain 的新批次獨立存在 `runs/metrics/si48_g0025/`，不混用前批音訊。
- short：3–8 秒；mid：8–15 秒；long：15–48 秒。不同速度的 48 秒包含不同小節數。
- H4/GS 全曲參考仍是 896 首，與這次 SI 的 20 首參考不同，因此用獨立 SI 表格。

### 建立／檢視計畫

```bash
.venv/bin/python baseline/run_si_study.py
```

不加 `--execute` 只會建立或核對 `study_plan.json`，不會開始合成與 SI。
計畫保存所有來源 MIDI 的絕對路徑與 SHA-256，重跑不會重新抽選曲子。
換機器或更改來源時，應使用新的 output 重新建立計畫，不直接複用舊機器的絕對路徑。

### 執行（本機，不需學校 GPU）

```bash
.venv/bin/python baseline/run_si_study.py \
  --execute --gain 0.025 \
  --renderer-python .venv/bin/python \
  --si-python .venv-si/bin/python \
  --soundfont /home/sodfg/.vscode/extensions/vaclis.midi-realplayer-0.2.5/media/GeneralUser-GS.sf2 \
  --fluidsynth .tools/fluidsynth/bin/fluidsynth \
  --workers 4
```

四個 worker 各處理一組，使用 CPU subprocess，不是多個 AI agent，也不使用 GPU。
若電腦資源有限，改 `--workers 1` 或 2。先前 48 秒單首基準約 8.6 秒，但
平行計算與合成真實長曲的成本不同，整批通常需要數分鐘以上。

每組結果與進度獨立寫入同目錄的 `real.log`、`t08.log`、`t10.log`、`t12.log`。
程式只有在四組都完成、每組恰有計畫中的 20 首且算法協定一致時，才產生合併總表。
遇到削波／缺檔／設定不一致會停止，不會以部分成功結果當成完整 80 首。
中斷後重跑同一指令可核對已完成 WAV 並使用 SI 矩陣快取。

### 結果與保存

```text
runs/metrics/si48_g0025/
  study_plan.json       固定選曲、長度、來源與雜湊
  inputs/               選中 MIDI 的工作副本
  audio/                同設定 WAV 和 render manifests
  groups/               各組 SI 矩陣、協定與逐首結果
  per_piece.csv         全部 80 個片段的數值
  summary.csv           各組平均、樣本標準差、與真實參考的差距
  summary.md            易讀報告
```

SI 本身計算的是從 WAV 取出的 48 秒，WAV 檔可能保留整首演奏與 release tail。
研究計畫與所有 `runs/` 成果另外備份；不加入 Git。程式與說明可獨立提交。
這是有明確範圍的抽樣參考比較，不代表所有 896 首真實曲子的 SI，也不是全曲結構分析。

本次 48 秒批次已完整完成，四組各 20 首，共 80 個片段。
所有 WAV 來源與雜湊核對通過，無削波；合併表中的每個 SI 值皆有限。
在此抽樣參考下，t10 的三個 SI 平均值差距皆最小；這不是統計顯著性或整體音樂品質結論。
音色授權文字另保存在結果的 `audio/GeneralUser-GS-LICENSE.txt`。
