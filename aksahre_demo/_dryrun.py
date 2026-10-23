# -*- coding: utf-8 -*-
"""彩排：用当前（可能不完整的）爬虫缓存跑一遍构建 + 三大检验，日志写入 _dryrun_*.txt。

彩排结果稍后会被全量流水线覆盖，仅用于提前暴露真实数据上的问题。
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEMO = Path(r"c:\Users\Heyuanhang\Desktop\aksahre_demo")
FACTOR_DIR = Path(r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子")
PYTHON = DEMO / ".venv" / "Scripts" / "python.exe"

SCRIPTS = [
    "sales_receipt_ratio_improvement.py",
    "sales_receipt_ratio_ic.py",
    "sales_receipt_ratio_monotonicity.py",
    "sales_receipt_ratio_regression.py",
]


def main():
    status = DEMO / "_dryrun_status.txt"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for script in SCRIPTS:
        started = datetime.now().strftime("%H:%M:%S")
        begin = time.time()
        proc = subprocess.run(
            [str(PYTHON), script],
            cwd=str(FACTOR_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        log_file = DEMO / ("_dryrun_%s.txt" % Path(script).stem)
        log_file.write_text(
            (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""), encoding="utf-8"
        )
        with status.open("a", encoding="utf-8") as handle:
            handle.write(
                "%s 起 %s | rc=%s | %.1f 秒 | %s\n"
                % (started, script, proc.returncode, time.time() - begin, log_file.name)
            )
    print("dryrun done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
