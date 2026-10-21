# runs資料夾管理

本機完整目錄索引：[runs/00_README.md](../../runs/00_README.md)。每個目錄都有00_STATUS.md。runs依.gitignore排除，以下規則保存在Git追蹤的說明文件內。

## 正式來源

- corrected_boundary_v2：新版Basic／Optional表格及生成。
- task_b_aligned：正式九份Basic B MIDI。
- bonus_aligned_v2：正式十份Bonus MIDI與重新評分。
- baseline、depth_3_vs_6、representation_midi_like、bonus_accompaniment：模型或參考來源，勿視為可刪除測試。
- submission_state：選曲主記錄。
- upload_preparation_v2：39份新版WAV的打包來源。
- report_build_v5：最新報告建置；report_build/fonts.conf仍是字型依賴。

## 歷史及測試

_smoke集中小量試跑。_archive集中舊計畫、早期樣本、結尾診斷、對齊試跑、v1打包暫存、v2–v4報告建置。
2026-10-21搬移10個目錄，263個原檔案逐一SHA256驗證相同，對照表在runs/_archive/reorganization_2026-10-21.json。沒有刪除、重訓或改動正式音樂。

experiments是舊生成，metrics混有舊分數及真實參考，audio_trial與listening是舊音訊及程式預設來源，sampling_pilot是探索性小量試跑。這些保留原路徑並以00_STATUS.md標明，不能直接當正式結果。

歷史plan內來源路徑與雜湊不改寫。封存快照僅供追溯，不保證直接在新路徑執行。

新增小量測試請明確指定runs/_smoke/名稱，正式實驗用有意義的新目錄名並記錄設定與來源。雲端只上傳output/UPLOAD_HW2_B12901157，不上傳runs。

本次只整理本機，沒有自動同步EDA Union，Git仍由使用者自行操作。
