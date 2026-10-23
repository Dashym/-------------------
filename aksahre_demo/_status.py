# -*- coding: utf-8 -*-
"""快照：进程存活 + 爬虫进度 + 缓存规模，追加写入 _status.txt。"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
OUTPUT = DEMO / "_status.txt"


def alive(pid):
    output = subprocess.run(
        ["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in output


def progress():
    if not CACHE.is_file():
        return 0, 0
    with CACHE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["股票代码"] for row in rows})


rows, stocks = progress()
lines = [
    "[%s] 流水线 pid 8768 存活=%s | 爬虫 pid 32976 存活=%s | 缓存 %d 行 / %d 只"
    % (datetime.now().strftime("%H:%M:%S"), alive(8768), alive(32976), rows, stocks)
]
crawler_log = DEMO / "_crawler_run2.log"
if crawler_log.is_file():
    tail = crawler_log.read_text(encoding="gbk", errors="replace").strip().splitlines()[-3:]
    lines.extend("    爬虫日志：" + line.strip() for line in tail)
with OUTPUT.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("done", flush=True)
