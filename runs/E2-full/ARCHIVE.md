# E2-full 保留紀錄(結束於 step 1550 = E1 early-stop 點,使用者決定)

| 檔案 | sha256 |
|---|---|
| `final_step1550.pt` | 04ddfea56a3a38afc53dddb784e805baf7fa49d492a64c83f00c09e591eeb87d |
| `dose_0.25.pt` | 090ac24df1616702fda90330cc769b88945721d6261302d5e42921051375db15 |
| `stop_0.2583Dstar.pt` | c890999b979384668a57536c68c50f5fb3e249bcee433a138cf4859e9c8a201d |

step 1500 停止後以 --stop-at-step 1550 自 checkpoint 續跑(lr 排程不變),於 step 1550 eval 後經 early-stop 路徑結束。
