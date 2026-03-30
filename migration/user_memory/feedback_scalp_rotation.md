---
name: High-rotation scalp design
description: スキャルプは高速回転。計算はバックグラウンド、Claudeは判断だけ。ポジ管理を毎サイクル強制。長時間HOLDは禁止
type: feedback
---

スキャルプと長期ホールドを同じタスクでやるな。分けろ。

**Why:** 2026-03-19に+770 UPL→+91 realized。6ポジ全部方向は合ってたのにpartial closeゼロ、trailing stopゼロで利益蒸発。5分間隔の重いタスクでスキャルプは無理。

**How to apply:**
- テクニカル計算はバックグラウンドプロセス（ボット可）で常時更新→JSONに書き出す
- Claudeタスクは「モニター1枚読んで即判断」だけ。計算しない
- スキャルプは「見るところを減らす」ことで高速化。M5+S5+pricing → 3-5pip取って回転
- ポジション管理（partial/trail/close）はエントリー判断より先に毎サイクル実行
- +5pipでtrail必須、30分以上HOLDは閉じて次へ
- スキャルプ用タスクはグローバルロック不要にして、SKIPされない設計にする
