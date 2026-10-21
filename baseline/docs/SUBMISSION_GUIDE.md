# 繳交整理與目前版本

## 上傳包

本專案根目錄 output/UPLOAD_HW2_B12901157 是集中整理的雲端資料夾。
生成結果全部沿用已訓練模型，本次不重訓。

| 任務 | MIDI | WAV | 來源 |
|---|---:|---:|---|
| Task A | 20 | 20 | corrected_boundary_v2/basic_a/t12 |
| Task B | 9 | 9 | task_b_aligned，三提示各三設定 |
| Bonus | 10 | 10 | bonus_aligned_v2，十個16小節作品 |
| 合計 | 39 | 39 | generation_results/ |

其餘內容：code、models、report、metrics、audio_assets、verification、selections.json、README。
四個推論模型為6層Basic、3層Basic、6層MIDI-like、Bonus。匯出移除optimizer，逐張量核對權重相同。
不帶訓練資料、前處理資料、課程提示、快取、虛擬環境、Git歷史與他人的神經網路模型。
SoundFont是音色資產，連同授權放入audio_assets，方便重現音色。

## 中途問題與本版處理

1. Basic結尾生成：原遮罩只允許EOS結束最後小節，模型常用的Bar邊界遭排除，造成末小節堆音。boundary-v2接受最後Bar作為終止訊號，並禁止同一起音位置與音高重複。新Bar/Pos正式89首已重新生成及評估，舊分數保留作歷史紀錄。
2. Task B小節顯示：原提示MIDI的tempo刻度與beat_midi.txt不同，直接拼接造成顯示約39小節。align_task_b.py按標註拍點重新映射ticks，總共8＋24個4/4小節，事件內容不變，時間誤差小於0.05毫秒。
3. Bonus小節顯示：按16個標註小節重寫tempo／meter刻度。每個標註拍顯示為四分音符，分子依標註拍數，沒有從標註推測原譜分母。保留事件、音高、力度與順序，最大時間誤差小於0.027毫秒。末拍可延長不到1毫秒保留原note-off，僅略去舊EOT的極短尾端靜音。
4. Bonus重新評分：正式對齊MIDI全部重跑原始eval_bonus.py。刻度的微小誤差可能改變音長／IOI分箱，所以報告採新版DD0.7166、DIOI0.5670，CA0.1125、DP0.8549等其餘分數亦以新CSV為準。
5. WAV音量：提交用FluidSynth gain1.0，原SI評估gain0.025保持原樣。39份WAV皆44.1kHz／PCM16／stereo、無PCM削波，最高峰值0.7324。音量提高不代表改善模型或評估分數。

MIDI-like使用獨立NoteOn/NoteOff生成器，沒有Basic最後Bar遮罩問題。其原作品與評估來源核對後沿用，不需假稱全部重跑。

## 重現整理

從專案根目錄執行，需已備齊正式實驗輸出與SoundFont。

```bash
python3 baseline/experiments/align_task_b.py --output runs/task_b_aligned
python3 baseline/experiments/align_bonus.py --output runs/bonus_aligned_v2
python3 HW2_scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir runs/bonus_aligned_v2 --output_csv runs/bonus_aligned_v2/metrics.csv
python3 baseline/experiments/align_bonus.py --output runs/bonus_aligned_v2 --summarize-metrics
python3 baseline/experiments/prepare_upload.py --stage audio --gain 1 --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth /path/to/fluidsynth
python3 baseline/experiments/prepare_upload.py --stage package --soundfont /path/to/GeneralUser-GS.sf2
```

對齊與打包均拒絕覆寫，已完成不用再執行；要做新版本請指定新output，音訊來源變動也要用新work。
打包程式不會上傳或執行Git。report預設讀output/report/B12901157_report_draft_v5。
上傳包根目錄README提供不需重新訓練的各模型推論命令，外部課程提示由使用者另行取得。

## 最後人工確認

- Task A已選t12，提交完整20首。
- Task B每個提示選一版最佳，將選擇寫入主記錄、上傳包selections.json與報告。仍需保留全部九份MIDI與WAV。
- 最終聽感只記錄實際聽過的現象，勿將客觀分數當成音樂品質排名。
- 上傳整個UPLOAD資料夾，確認助教可下載；封面及COOL留言放分享連結。
- 更新報告後，將上傳包內report/同步為同版，重建file_manifest.json以反映變更。

Git由使用者自行add／commit／push。output與runs依.gitignore排除；引用中的正式來源保留原路徑；已封存的10個歷史／測試目錄另有搬移與SHA256記錄，詳見RUNS_LAYOUT.md。
