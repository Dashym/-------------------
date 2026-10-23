# -*- coding: utf-8 -*-
"""无人值守流水线：等待爬虫结束 → 构建销售收现比改善因子 → 三大检验。

用法（后台运行）：
    Start-Process -FilePath '.venv\\Scripts\\python.exe' -ArgumentList '_run_pipeline.py' -WindowStyle Hidden
所有过程写进 _pipeline_status.txt（UTF-8），可直接打开查看。
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
PYTHON = DEMO / ".venv" / "Scripts" / "python.exe"
STATUS_FILE = DEMO / "_pipeline_status.txt"
POOL_SIZE = 228

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


def crawl_progress():
    if not CACHE_FILE.is_file():
        return 0, 0
    with CACHE_FILE.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["股票代码"] for row in rows})


def wait_for_crawler():
    """股票数连续 ~90 秒不增长（或达到股票池规模）即视为爬虫结束。"""
    last_stocks, stable, deadline = -1, 0, time.time() + 3600
    while time.time() < deadline:
        try:
            rows, stocks = crawl_progress()
        except Exception as exc:  # 文件正被写入时会短暂失败
            rows, stocks = -1, -1
            log("读取缓存失败（稍后重试）：%s" % exc)
        if stocks != last_stocks:
            log("爬虫进度：%d 行 / %d 只" % (rows, stocks))
            last_stocks, stable = stocks, 0
        else:
            stable += 1
        if stocks >= POOL_SIZE:
            log("爬虫结束：%d 行 / %d 只（达到股票池规模）" % (rows, stocks))
            return rows, stocks
        if stable >= 30:
            log("爬虫结束：%d 行 / %d 只（90 秒无新增）" % (rows, stocks))
            return rows, stocks
        time.sleep(3)
    log("等待爬虫超时，按当前缓存继续")
    return crawl_progress()


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
    log(
        "%s 结束：returncode=%s，用时 %.1f 秒，日志=%s"
        % (label, proc.returncode, cost, log_file.name)
    )
    return proc.returncode, cost


def main():
    log("=== 流水线启动 ===")
    wait_for_crawler()
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
