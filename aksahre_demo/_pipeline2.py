# -*- coding: utf-8 -*-
"""无人值守流水线 v2：等爬虫进程退出 → 构建因子 → 三大检验。

相较 v1 的两处关键加固：
    1. 结束判据从"缓存行数 90 秒不增长"改为**等待爬虫进程真正退出**
       （爬虫每次会把整张累积表重写到同一个 CSV，任何时刻的读取都可能是"半张表"）；
    2. 读缓存前先做**稳定读取**：连续两次（间隔 6 秒）行数/股票数完全一致才采信。

用法（后台运行）：
    Start-Process -FilePath '.venv\\Scripts\\python.exe' -ArgumentList '_pipeline2.py' -WindowStyle Hidden
过程日志：_pipeline2_status.txt（UTF-8）
"""
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
CACHE_FILE = DEMO / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
POOL_FILE = DEMO / "汽车零部件_申万二级成分股.csv"
PYTHON = DEMO / ".venv" / "Scripts" / "python.exe"
STATUS_FILE = DEMO / "_pipeline2_status.txt"

CRAWLER_KEYWORD = "sales_receipt_cash_flow_crawler.py"
WAIT_DEADLINE_SECONDS = 3600
STALL_WARNING_SECONDS = 300

STEPS = [
    ("因子构建", "sales_receipt_ratio_improvement.py"),
    ("IC-IR 检验", "sales_receipt_ratio_ic.py"),
    ("分层单调性检验", "sales_receipt_ratio_monotonicity.py"),
    ("Fama-MacBeth 回归", "sales_receipt_ratio_regression.py"),
]


def log(message):
    line = "%s | %s" % (datetime.now().strftime("%H:%M:%S"), message)
    with STATUS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def crawler_pids():
    """匹配爬虫脚本的 python 进程（排除 powershell 自身，避免关键字自匹配）。"""
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' "
        "-and $_.CommandLine -like '*%s*' } | Select-Object -ExpandProperty ProcessId"
        % CRAWLER_KEYWORD
    )
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True
    ).stdout
    return [int(token) for token in output.split() if token.strip().isdigit()]


def read_cache():
    with CACHE_FILE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["股票代码"] for row in rows})


def stable_cache(attempts=8, interval=6):
    """连续两次读数一致才返回，避免读到爬虫重写过程中的"半张表"。"""
    previous = None
    for _ in range(attempts):
        try:
            current = read_cache()
        except Exception as exc:
            log("读取缓存失败（稍后重试）：%s" % exc)
            current = None
        if current is not None and current == previous:
            return current
        previous = current
        time.sleep(interval)
    return previous if previous is not None else (0, 0)


def pool_size():
    with POOL_FILE.open(encoding="utf-8-sig") as handle:
        return len(list(csv.DictReader(handle)))


def wait_for_crawler():
    deadline = time.time() + WAIT_DEADLINE_SECONDS
    last_progress, last_change = None, time.time()
    while time.time() < deadline:
        pids = crawler_pids()
        try:
            rows, stocks = read_cache()
        except Exception:
            rows, stocks = -1, -1
        if (rows, stocks) != last_progress:
            log("爬虫进度：%d 行 / %d 只（存活 pid=%s）" % (rows, stocks, pids or "无"))
            last_progress, last_change = (rows, stocks), time.time()
        elif time.time() - last_change > STALL_WARNING_SECONDS:
            log("注意：缓存已 %d 秒无增长，但爬虫进程仍在（pid=%s），继续等待" % (
                int(time.time() - last_change), pids or "无"))
            last_change = time.time()
        if not pids:
            log("爬虫进程已退出")
            return
        time.sleep(15)
    log("等待爬虫超时（%d 秒），按当前缓存继续" % WAIT_DEADLINE_SECONDS)


def run_step(label, script):
    started = time.time()
    log("开始 %s：%s" % (label, script))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [str(PYTHON), script],
        cwd=str(FACTOR_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    log_file = FACTOR_DIR / ("_%s.log" % Path(script).stem)
    log_file.write_text(
        (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""), encoding="utf-8"
    )
    cost = time.time() - started
    log("%s 结束：returncode=%s，用时 %.1f 秒，日志=%s" % (label, proc.returncode, cost, log_file.name))
    return proc.returncode, cost


def main():
    log("=== 流水线 v2 启动 ===")
    wait_for_crawler()
    rows, stocks = stable_cache()
    total = pool_size()
    log("稳定读取缓存：%d 行 / %d 只（股票池 %d 只，覆盖率 %.1f%%）" % (
        rows, stocks, total, 100.0 * stocks / total if total else 0.0))
    summary = []
    for label, script in STEPS:
        summary.append((label, script) + run_step(label, script))
    log("=== 汇总 ===")
    for label, script, code, cost in summary:
        log("  %-16s rc=%s  %.1f 秒  %s" % (label, code, cost, script))
    log("全部流程结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
