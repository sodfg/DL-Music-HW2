# Optional 擴充實驗流程

這是本專案後續實驗的共同名稱。WAV 本身也是基本作業的繳交要求，
只是現在先用少量音訊，為選做的 SI 建立一致的分析條件。

| 階段 | 內容 | 狀態 |
|---|---|---|
| 1 | H1/H4/GS 總表與易讀 Markdown | 完成 |
| 2 | 統一鋼琴音色、批次 WAV、品質檢查與合成記錄 | 已實作，本機少量試轉 |
| 3 | 少量 SI → 確認時間尺度、片段長度與成本 → 批次整合 | 尚未開始 |
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
下一步 SI 的片段長度、參考資料抽樣及時間尺度尚未決定；不把這幾首試轉結果
當作正式 SI 實驗資料，也不在總表填入未計算的 SI。

### 測試與來源

```bash
.venv/bin/python -m unittest discover -s baseline -p 'test_render_audio.py'
```

FluidSynth CLI：<https://www.fluidsynth.org/wiki/UserManual/>

MIDI RealPlayer：<https://marketplace.visualstudio.com/items?itemName=vaclis.midi-realplayer>
