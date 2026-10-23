# -*- coding: utf-8 -*-
"""定位：构建脚本实际读的缓存文件路径 vs 爬虫实际写的文件路径。"""
import glob
import os
import time
from pathlib import Path

BUILDER = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子\sales_receipt_ratio_improvement.py")
out = []

patterns = [
    r"C:\Users\Heyuanhang\Desktop\**\*销售收现*.csv",
    r"C:\Users\Heyuanhang\Desktop\**\*汽车零部件*.csv",
]
found = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
for path in found:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            rows = sum(1 for _ in handle) - 1
    except Exception as exc:
        rows = "读取失败:%s" % exc
    out.append(
        "%s | rows=%s | mtime=%s"
        % (path, rows, time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))))
    )

out.append("")
out.append("---- 构建脚本中的路径常量 ----")
for number, line in enumerate(BUILDER.read_text(encoding="utf-8").splitlines(), start=1):
    if any(key in line for key in ("CRAWL_FILE", "POOL_FILE", "BASE_DIR", "SCRIPT_DIR", "raw_data")):
        out.append("%4d | %s" % (number, line.rstrip()))

Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_audit2_out.txt").write_text(
    "\n".join(out), encoding="utf-8"
)
print("done", flush=True)
