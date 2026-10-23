# -*- coding: utf-8 -*-
"""探测：因子目录现有文件 + 申万成分股数量口径。"""
import os
import sys
import threading
import traceback

import pandas as pd

OUT = []
FOLDER = r"C:\Users\Heyuanhang\Desktop\因子库\销售收现比改善因子"
for root, _dirs, files in os.walk(FOLDER):
    for f in sorted(files):
        fp = os.path.join(root, f)
        OUT.append("FILE %-60s %8d bytes" % (os.path.relpath(fp, FOLDER), os.path.getsize(fp)))
        if f.lower().endswith(".csv"):
            try:
                df = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
                OUT.append("     rows=%d cols=%s" % (len(df), list(df.columns)))
                if "股票代码" in df.columns:
                    OUT.append("     unique stocks=%d" % df["股票代码"].nunique())
                    OUT.append("     stocks sample=%s" % ", ".join(df["股票代码"].unique()[:40].tolist()))
                for col in df.columns:
                    if "日期" in col:
                        OUT.append("     %s: %s ~ %s (n=%d)" % (col, df[col].min(), df[col].max(), df[col].nunique()))
            except Exception as exc:  # noqa: BLE001
                OUT.append("     read error: %s" % exc)

for name in ["价值因子", "动量因子（无效）", "技术壁垒升级", "毛利率稳定性（可以，小心）", "盈利因子", "规模因子（很微弱）"]:
    fd = os.path.join(r"C:\Users\Heyuanhang\Desktop\因子库", name)
    if os.path.isdir(fd):
        OUT.append("")
        OUT.append(">>> %s" % name)
        for f in sorted(os.listdir(fd)):
            OUT.append("    %s (%d)" % (f, os.path.getsize(os.path.join(fd, f))))

RESULT = {}


def work():
    try:
        import akshare as ak

        info = ak.sw_index_second_info()
        hit = info.loc[info["行业名称"].astype(str).str.strip() == "汽车零部件"]
        RESULT["second_code"] = hit[["行业代码", "行业名称"]].to_dict("records")
        code = str(hit.iloc[0]["行业代码"]).split(".")[0]
        cons = ak.index_component_sw(symbol=code)
        RESULT["cons_shape"] = tuple(cons.shape)
        RESULT["cons_cols"] = list(cons.columns)
        RESULT["cons_head"] = cons.head(3).to_dict("records")
    except Exception:  # noqa: BLE001
        RESULT["error"] = traceback.format_exc()[-1200:]


t = threading.Thread(target=work, daemon=True)
t.start()
t.join(timeout=90)
OUT.append("")
OUT.append("akshare status: %s" % ("TIMEOUT" if t.is_alive() else "FINISHED"))
for k, v in RESULT.items():
    OUT.append("%s = %s" % (k, v))

with open(r"c:\Users\Heyuanhang\Desktop\aksahre_demo\_probe5.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
print("probe5 done")
sys.stdout.flush()
if t.is_alive():
    os._exit(0)
