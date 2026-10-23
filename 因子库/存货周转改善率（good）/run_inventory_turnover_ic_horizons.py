import logging
import sys
from pathlib import Path

import pandas as pd

from inventory_turnover_ic import (
    ADJUSTED_QUOTE_FILE,
    FACTOR_FILE,
    UNADJUSTED_QUOTE_FILE,
    calculate_monthly_ic,
    render_report,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    for input_file in (FACTOR_FILE, UNADJUSTED_QUOTE_FILE, ADJUSTED_QUOTE_FILE):
        if not input_file.is_file():
            log.error("输入文件不存在: %s", input_file)
            return 1

    factor = pd.read_csv(FACTOR_FILE, dtype={"股票代码": str}, encoding="utf-8-sig")
    unadjusted = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    adjusted = pd.read_csv(
        ADJUSTED_QUOTE_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    factor_dates = pd.to_datetime(factor["可用日期"], errors="raise")
    market_dates = pd.to_datetime(unadjusted["日期"], errors="raise")
    adjusted_dates = pd.to_datetime(adjusted["日期"], errors="raise")

    results = calculate_monthly_ic(factor, unadjusted, adjusted)
    report = render_report(
        results,
        len(factor),
        (factor_dates.min().strftime("%Y-%m-%d"), factor_dates.max().strftime("%Y-%m-%d")),
        (market_dates.min().strftime("%Y-%m-%d"), market_dates.max().strftime("%Y-%m-%d")),
        (
            adjusted_dates.min().strftime("%Y-%m-%d"),
            adjusted_dates.max().strftime("%Y-%m-%d"),
        ),
    )
    output_file = SCRIPT_DIR / "存货周转改善率_IC_未来12个月.txt"
    output_file.write_text(report, encoding="utf-8")
    print(report, end="")
    log.info("未来 12 个月 IC 报告已输出：%s", output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
