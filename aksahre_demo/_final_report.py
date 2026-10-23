# -*- coding: utf-8 -*-
"""把最终结果汇总到 _final_out.txt（修复后一轮的因子产出 + 三大检验关键指标）。"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
FACTOR_CSV = FACTOR_DIR / "销售收现比改善因子_raw.csv"
DETAIL_CSV = FACTOR_DIR / "销售收现比改善因子_明细.csv"

out = []


def add(text):
    out.append(text)


def section(title):
    add("")
    add("======== %s ========" % title)


def pids(keyword):
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' "
        "-and $_.CommandLine -like '*%s*' } | Select-Object -ExpandProperty ProcessId" % keyword
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True
    ).stdout
    return [int(t) for t in result.split() if t.strip().isdigit()]


def csv_stats(path, code="股票代码"):
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row[code] for row in rows}), list(rows[0].keys())


section("进程与缓存（%s）" % datetime.now().strftime("%H:%M:%S"))
add("爬虫 pid=%s | 抢跑脚本 pid=%s" % (pids("sales_receipt_cash_flow_crawler.py"), pids("_oneshot.py")))
if CACHE.is_file():
    rows, stocks, _ = csv_stats(CACHE)
    add("爬虫缓存：%d 行 / %d 只" % (rows, stocks))

section("修复后抢跑记录（_oneshot_status.txt 第 39 行起）")
status = (DEMO / "_oneshot_status.txt").read_text(encoding="utf-8").splitlines()
out.extend(status[38:])

section("因子构建 stdout（_oneshot_build.log）")
out.extend((DEMO / "_oneshot_build.log").read_text(encoding="utf-8").splitlines())

section("因子文件统计")
for path in (FACTOR_CSV, DETAIL_CSV):
    if path.is_file():
        rows, stocks, columns = csv_stats(path)
        add("%s：%d 行 / %d 只 | 列=%s" % (path.name, rows, stocks, columns))

section("三大报告关键结论")
report_keys = {
    "销售收现比改善因子_IC结果.txt": ("IC 均值", "ICIR", "有效月度截面", "平均有效股票数", "综合提示"),
    "销售收现比改善因子_单调性结果.txt": ("Q5-Q1", "单调性判断", "结论", "平均有效股票数", "多空胜率"),
    "销售收现比改善因子_回归结果.txt": ("变量", "销售收现比改善因子 ", "Fama-MacBeth", "结论", "平均有效股票数"),
}
for name, keys in report_keys.items():
    path = FACTOR_DIR / name
    add("---- %s" % name)
    if not path.is_file():
        add("（缺失）")
        continue
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(key in line for key in keys):
            add("   " + line.strip())

(DEMO / "_final_out.txt").write_text("\n".join(out), encoding="utf-8")
print("done", flush=True)
