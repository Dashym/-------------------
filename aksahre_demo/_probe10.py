# -*- coding: utf-8 -*-
"""校验现有财报数据口径：单季度值 or 累计值；并盘点行情/股票池范围。"""
import json
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\Heyuanhang\Desktop\aksahre_demo")
OUT = []


def p(s):
    OUT.append(str(s))


fin = pd.read_csv(BASE / "processed_data" / "财报_清洗完成.csv", dtype={"股票代码": str}, encoding="utf-8-sig")
fin["股票代码"] = fin["股票代码"].str.zfill(6)
fin["报告期"] = pd.to_datetime(fin["报告期"], errors="coerce")
p("=== 财报_清洗完成.csv ===")
p("shape=%s" % (fin.shape,))
p("cols=%s" % list(fin.columns))
p("报告期范围 %s ~ %s；股票数 %d" % (fin["报告期"].min().date(), fin["报告期"].max().date(), fin["股票代码"].nunique()))

sub = fin.loc[fin["股票代码"] == "000030"].sort_values("报告期")
p("\n000030 全部记录（看是否单季度）：")
p(sub[["报告期", "营业总收入", "经营活动产生的现金流量净额"]].to_string(index=False))
p("\n000030 2019 年 营业总收入 4 个季度求和 = %.2f" % sub.loc[sub["报告期"].dt.year == 2019, "营业总收入"].sum())
p("000030 2020 年 营业总收入 4 个季度求和 = %.2f" % sub.loc[sub["报告期"].dt.year == 2020, "营业总收入"].sum())

q = fin.assign(季度=fin["报告期"].dt.to_period("Q"))
dup = q.duplicated(["股票代码", "季度"]).sum()
p("\n同股票+同季度重复记录：%d" % dup)
p("每只股票季度数分布：%s" % q.groupby("股票代码")["季度"].nunique().value_counts().sort_index().to_dict())

# 连续季度缺口检查（以季度序号衡量）
q["序号"] = q["报告期"].dt.year * 4 + q["报告期"].dt.quarter
gaps = []
for code, grp in q.groupby("股票代码"):
    seq = sorted(grp["序号"].unique())
    expect = list(range(seq[0], seq[-1] + 1))
    missing = sorted(set(expect) - set(seq))
    if missing:
        gaps.append((code, len(seq), missing))
p("存在季度缺口的股票数：%d" % len(gaps))
p("缺口样例：%s" % gaps[:5])

p("\n缺失值统计：")
p(fin[["营业总收入", "经营活动产生的现金流量净额"]].isna().sum().to_string())

p("\n列名含“现金流”的字段：%s" % [c for c in fin.columns if "现金流" in c])

pool = pd.read_csv(BASE / "汽车零部件_申万二级成分股.csv", dtype=str, encoding="utf-8-sig")
pool2 = pd.read_csv(BASE / "汽车零部件_剔除ST_上市日期截至2024-12-31.csv", dtype=str, encoding="utf-8-sig")
p("\n=== 股票池 ===")
p("申万二级成分股 cols=%s rows=%d" % (list(pool.columns), len(pool)))
p("剔除ST(截至2024-12-31) cols=%s rows=%d" % (list(pool2.columns), len(pool2)))
p("池 ∩ 财报股票数 = %d；财报中不在池 = %d" % (
    len(set(pool["股票代码"]) & set(fin["股票代码"])),
    len(set(fin["股票代码"]) - set(pool["股票代码"])),
))
p("池中无财报记录的股票：%s" % sorted(set(pool["股票代码"]) - set(fin["股票代码"]))[:20])

adj = pd.read_csv(
    BASE / "clean_temp" / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv",
    dtype={"股票代码": str}, encoding="utf-8-sig",
)
adj["日期"] = pd.to_datetime(adj["日期"], errors="coerce")
p("\n=== 后复权行情 ===")
p("cols=%s rows=%d 日期 %s ~ %s 股票数=%d" % (
    list(adj.columns), len(adj), adj["日期"].min().date(), adj["日期"].max().date(), adj["股票代码"].nunique()))
days = sorted(adj.loc[(adj["日期"] >= "2024-10-20") & (adj["日期"] <= "2024-11-10"), "日期"].dt.strftime("%Y-%m-%d").unique())
p("2024-10-20~2024-11-10 交易日：%s" % days)

# 月末交易日（<=2024-11-01）
d = pd.DataFrame({"日期": sorted(adj["日期"].unique())})
d = d.loc[d["日期"] <= pd.Timestamp("2024-11-01")]
d["月"] = d["日期"].dt.to_period("M")
me = d.groupby("月")["日期"].max()
p("月末交易日数量(<=2024-11-01)=%d，首=%s 末=%s" % (len(me), me.iloc[0].date(), me.iloc[-1].date()))
p("最后 5 个月末：%s" % [x.date().isoformat() for x in me.tail(5)])

(BASE / "_probe10.txt").write_text("\n".join(OUT), encoding="utf-8")
print("probe10 done", flush=True)
