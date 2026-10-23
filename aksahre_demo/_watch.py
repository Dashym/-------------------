# -*- coding: utf-8 -*-
"""核对：_pipeline2.py 实例数、爬虫存活、缓存规模。追加写入 _watch_out.txt。"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"


def pids(keyword):
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' "
        "-and $_.CommandLine -like '*%s*' } | Select-Object -ExpandProperty ProcessId"
        % keyword
    )
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True
    ).stdout
    return [int(token) for token in output.split() if token.strip().isdigit()]


rows = stocks = 0
if CACHE.is_file():
    with CACHE.open(encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    rows, stocks = len(records), len({row["股票代码"] for row in records})

line = "[%s] pipeline2=%s | crawler=%s | cache=%d 行/%d 只" % (
    datetime.now().strftime("%H:%M:%S"),
    pids("_pipeline2.py"),
    pids("sales_receipt_cash_flow_crawler.py"),
    rows,
    stocks,
)
with (DEMO / "_watch_out.txt").open("a", encoding="utf-8") as handle:
    handle.write(line + "\n")
print("done", flush=True)
