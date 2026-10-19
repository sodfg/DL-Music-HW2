# REMI-like 多樣性試聽：top-k / top-p 小比較

使用既有六層 Bar/Pos（REMI-like）模型，不重新訓練、不使用 MIDI-like 模型。
三組各三首、各 32 小節，固定 T=1.0、seeds=123–125；先只做小量試聽。

| 組別 | Top-k | Top-p | 目的 |
|---|---:|---:|---|
| A | 32 | 0.95 | 原設定；沿用 t10 的 sample_000–002 |
| B | 64 | 0.95 | 相較 A 僅放寬 top-k |
| C | 64 | 0.98 | 相較 B 僅放寬 top-p |

所有設定仍受原版合法事件 grammar 限制。k/p 改變不保證每一步候選集合都不同，
候選比較分散時較可能發揮作用。不同設定的同 seed 不表示同一首歌的改編版本。

在作業根目錄，先檢視計畫：

```bash
python3 baseline/experiments/run_sampling_pilot.py
```

在有可用 GPU 的 EDA Union 執行：

```bash
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_sampling_pilot.py --execute
```

預設輸出 runs/sampling_pilot/，每組有三個 MIDI 與原始 token；comparison.md
提供配對連結、簡單 pitch 4-gram 重疊診斷及空白聽感表。啟動前會核對原始 checkpoint、
generator 及 t10 資料來源；來源不符或既有輸出未完成時會拒絕覆蓋。
此入口沿用已完成深度實驗 plan 的 baseline 身分雜湊，避免拿新 checkpoint
生成的 B/C 和舊 checkpoint 的 A 混比。計畫輸出固定，改條件需另開輸出。

試聽建議：同一個 sample 編號按 A→B→C 聽，先比較前四小節，再聽中段。
記錄不同首前奏是否相似、曲內重複是否突兀、節奏與旋律是否自然，及具體時間點。
不用僅因更隨機或重疊率較低就選它。

重疊診斷包含和弦／伴奏的 pitch token 順序，忽略時間、力度與移調，不是旋律辨識
或抄襲檢查。各組只有三首，不當成正式 20 首音樂評分，不與原本 H4/GS 均值混表。
本輪 temperature 固定；先前 t08/t10/t12 可另外作 temperature 試聽參考。
試聽確認偏好後，再決定是否擴充成 20 首及重新計算 H4/GS/SI。

本機查看資料夾不會自動看到學校新成果，完成後需另行下載。
這些 runs 檔案仍被忽略；程式和文件由使用者自行 Git commit。
