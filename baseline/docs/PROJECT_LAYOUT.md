> 2026-10-21 更新：runs已加上用途／版本標籤，10個歷史目錄移至_archive。請先讀 [最新runs整理規則](RUNS_LAYOUT.md) 與 [本機完整索引](../../runs/00_README.md)。下方是早期整理紀錄，smoke大小等數字不是目前盤點。

# 資料夾整理與 smoke 保存規則

## 程式與成果分開

`baseline/` 只放程式、測試與說明；`runs/` 放模型、MIDI、音訊、評分與實驗記錄。
`POP909/` 是原始資料，`data/baseline/` 是前處理資料，兩者不與工具程式混放。

| 程式位置 | 內容 | 指令例子（在作業根目錄） |
|---|---|---|
| baseline/ | 核心前處理、模型、訓練、生成 | `python3 baseline/train.py --help` |
| baseline/audio/ | WAV 合成 | `python3 baseline/audio/render_audio.py --help` |
| baseline/evaluation/ | H4/GS、SI、結果彙整 | `python3 baseline/evaluation/summarize_metrics.py` |
| baseline/experiments/ | 批次實驗入口 | `python3 baseline/experiments/run_depth_experiment.py --stage plan` |
| baseline/tests/ | 所有測試程式 | `python3 -m unittest discover -s baseline/tests -p 'test_*.py'` |
| baseline/docs/ | Optional 與各實驗說明 | Markdown 預覽 |

輔助工具的舊平面路徑已搬移，未保留一整排相容性 wrapper，以免根目錄再次混亂。
核心 `music.py`、`model.py`、`prepare.py`、`train.py`、`generate.py` 保持原路徑與內容，
既有 state_dict 模型 checkpoint 可由核心生成程式照常載入。
`HW2_scripts/` 助教程式未移動；正式 runs 與 data 的路徑不變。

## Smoke 是什麼，是否保留？

Smoke test 是低成本的程式試跑：少量步數、小模型、少量生成，確認整條流程能執行。
它不是正式模型，也不是音樂評分指標。

- 保留 `tests/` 的測試程式，以及 README 的小模型試跑指令。
- 過去的試跑產物先移到 `runs/_smoke/`；本次未刪除任何 smoke 檔案。
- 這些快照只有約 5 MB，不急著刪。以後確認不需除錯時可自行移除。
- 它們不作報告品質比較、不繳交，也不加入 Git。
- 快照內記錄的舊檔案路徑屬於歷史資訊，未被改寫；不把快照當成正式續訓來源。

新 smoke 建議輸出到 `runs/_smoke/<名稱>/`。
正式 `runs/baseline/`、`runs/depth_3_vs_6/`、`runs/experiments/`、`runs/metrics/`
及已完成的 SI 結果維持原位，不屬於可隨意刪除的測試產物。

## 歷史實驗計畫與來源雜湊

改程式位置或 import 會改變部分原始碼 SHA-256，不能因此改寫既有計畫，假裝舊結果
是新版本程式產生的。深度實驗的舊 plan 在資料、模型與固定參數都一致時：

- `--stage plan` / `--stage report` 可以查看或整理既有結果，顯示來源版本警告。
- 原本的 plan.json 保留，不寫成新程式雜湊。
- `train` / `generate` / `evaluate` / `all` 若來源不符仍會拒絕，需新的 `--output`。

SI 的協定與 cache 也保留原本來源資訊；明確重跑新版時可依新 wrapper 雜湊重新計算。
本次資料夾整理不重新訓練、不重新生成正式曲子、不重新計算正式評分。

## 本機與 EDA Union

這次整理只在本機。EDA Union 仍保有先前可運行的平面結構；不會因本機改檔案而自動同步。
舊的學校指令暫時仍對應舊位置，本機請使用新 README 的路徑。
要更新 EDA 時，先確認沒有程序正在使用程式，再同步程式目錄；不要連 runs、資料、
本機環境或 `.tools` 一起覆蓋，也不要混用舊、新入口各啟動一份實驗。
Git add/commit/push 仍由使用者操作；這次未改 Git 的分支或暫存區。
