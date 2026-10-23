# -*- coding: utf-8 -*-
"""抢跑：不等爬虫，直接用当前缓存跑通 因子构建 + 三大检验。

背景：爬虫每抓完一只股票就把"整张累积表"重写回同一个 CSV，
所以任何一次读取都可能是半张表（实测出现过 4217 行/130 只 → 380 行/10 只）。
本脚本因此做了三件事：
    1. 先杀掉可能残留的 _pipeline2.py 实例，避免并发写同一批结果文件；
    2. 对缓存连续采样 5 次，取行数/股票数最大的一次作为"参照快照"；
    3. 因子构建后校验产出覆盖率，若明显残缺（读到残表）则立即重跑，最多 3 次。
"""
import csv
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
CACHE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
POOL = DEMO / "汽车零部件_申万二级成分股.csv"
FACTOR_CSV = FACTOR_DIR / "销售收现比改善因子_raw.csv"
PY = DEMO / ".venv" / "Scripts" / "python.exe"
STATUS = DEMO / "_oneshot_status.txt"

BUILD = "sales_receipt_ratio_improvement.py"
TESTS = [
    ("IC-IR 检验", "sales_receipt_ratio_ic.py", "销售收现比改善因子_IC结果.txt"),
    ("分层单调性检验", "sales_receipt_ratio_monotonicity.py", "销售收现比改善因子_单调性结果.txt"),
    ("Fama-MacBeth 回归", "sales_receipt_ratio_regression.py", "销售收现比改善因子_回归结果.txt"),
]


def log(message):
    line = "%s | %s" % (datetime.now().strftime("%H:%M:%S"), message)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def kill_stale_pipelines():
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
        "$_.CommandLine -like '*_pipeline2.py*' } | ForEach-Object { Stop-Process -Id "
        "$_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)


def count_cache():
    with CACHE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["股票代码"] for row in rows})


def best_snapshot(samples=5, interval=2.0):
    best = (0, 0)
    for _ in range(samples):
        try:
            current = count_cache()
        except Exception as exc:
            log("读取缓存失败（计入下一轮）：%s" % exc)
            current = (0, 0)
        if current[1] > best[1]:
            best = current
        time.sleep(interval)
    return best


def run_script(script, tag):
    started = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [str(PY), script], cwd=str(FACTOR_DIR), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    log_file = DEMO / ("_oneshot_%s.log" % tag)
    log_file.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""), encoding="utf-8")
    cost = time.time() - started
    log("%s：rc=%s，%.1f 秒，日志=%s" % (script, proc.returncode, cost, log_file.name))
    return proc.returncode


def factor_coverage():
    if not FACTOR_CSV.is_file():
        return 0, 0
    with FACTOR_CSV.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["股票代码"] for row in rows})


def highlight(report_name):
    path = FACTOR_DIR / report_name
    if not path.is_file():
        return ["（报告缺失：%s）" % report_name]
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keys = ("Q5-Q1", "单调性判断", "结论", "ICIR", "IC 均值", "均值系数", "平均有效股票数",
            "有效月度截面", "有效回归截面", "t统计量")
    picked = [line.strip() for line in text if any(key in line for key in keys)]
    return picked[:8]


def main():
    log("=== 抢跑开始（不等爬虫）===")
    kill_stale_pipelines()
    log("已清理残留 _pipeline2.py 实例")
    pool = sum(1 for _ in POOL.open(encoding="utf-8-sig")) - 1
    log("股票池 %d 只" % pool)
    rows, stocks = best_snapshot()
    log("缓存最完整快照：%d 行 / %d 只" % (rows, stocks))

    for attempt in range(1, 4):
        run_script(BUILD, "build")
        f_rows, f_stocks = factor_coverage()
        log("因子产出：%d 行 / %d 只（第 %d 次尝试）" % (f_rows, f_stocks, attempt))
        if stocks == 0 or f_stocks >= 0.8 * stocks:
            break
        log("覆盖率不足（快照 %d 只 → 产出 %d 只），立即重跑因子" % (stocks, f_stocks))

    for label, script, report in TESTS:
        code = run_script(script, script.replace(".py", "").replace("sales_receipt_ratio_", ""))
        log("【%s】report=%s rc=%s" % (label, report, code))
        for line in highlight(report):
            log("    %s" % line)
    log("=== 抢跑结束 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
