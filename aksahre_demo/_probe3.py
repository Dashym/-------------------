# -*- coding: utf-8 -*-
"""探测 akshare 能否取到「销售商品、提供劳务收到的现金」，带硬超时保护。"""
import os
import sys
import threading
import traceback

RESULT = {}


def work():
    try:
        import akshare as ak

        df = ak.stock_cash_flow_sheet_by_report_em(symbol="SZ000030")
        RESULT["em_shape"] = tuple(df.shape)
        cols = [c for c in df.columns if "SALES" in str(c).upper()]
        RESULT["em_sales_cols"] = cols
        if cols:
            sub = df[["REPORT_DATE", cols[0]]].head(6)
            RESULT["em_head"] = sub.to_string(index=False)
        else:
            RESULT["em_all_cols_sample"] = [c for c in df.columns][:80]
    except Exception:  # noqa: BLE001
        RESULT["em_error"] = traceback.format_exc()[-1500:]

    try:
        import akshare as ak

        df2 = ak.stock_financial_report_sina(stock="sz000030", symbol="现金流量表")
        RESULT["sina_shape"] = tuple(df2.shape)
        RESULT["sina_sales_like"] = [
            c for c in df2.columns if ("销售" in str(c)) or ("现金" in str(c))
        ][:20]
    except Exception:  # noqa: BLE001
        RESULT["sina_error"] = traceback.format_exc()[-800:]


thread = threading.Thread(target=work, daemon=True)
thread.start()
thread.join(timeout=90)

lines = ["TIMEOUT" if thread.is_alive() else "FINISHED"]
for key, value in RESULT.items():
    lines.append("%s = %s" % (key, value))

with open(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_probe3.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("probe3 done")
sys.stdout.flush()
if thread.is_alive():
    os._exit(0)
