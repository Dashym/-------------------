# -*- coding: utf-8 -*-
"""列出所有 python 进程的命令行 + 当前缓存规模，写入 _procs_out.txt。"""
import csv
import subprocess
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"

command = (
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
    "Select-Object ProcessId,CommandLine | Format-Table -AutoSize | Out-String -Width 400"
)
listing = subprocess.run(
    ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True
).stdout

lines = ["### python 进程", listing or "(无)"]

if CACHE.is_file():
    with CACHE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    codes = [row["股票代码"] for row in rows]
    lines.append("### 缓存：%d 行，去重前 %d 只，去重后 %d 只" % (len(rows), len(codes), len(set(codes))))
    lines.append("代码样例：%s" % sorted(set(codes))[:8])
    periods = [row["报告期"] for row in rows]
    lines.append("报告期样例：%s" % sorted(set(periods))[:5])
    lines.append("### 缓存文件最后 3 行")
    lines.extend(lines[-1:] if False else [])
    with CACHE.open(encoding="utf-8-sig") as handle:
        tail = handle.read().strip().splitlines()[-3:]
    lines.extend(tail)

(DEMO / "_procs_out.txt").write_text("\n".join(lines), encoding="utf-8")
print("done", flush=True)
