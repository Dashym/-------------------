"""
======================================================================
申万行业分类数据 - 汽车零部件三级分类爬取脚本 (最终版)
======================================================================
数据源：
  1. 申万官网历史分类变动Excel (含纳入日、更新日)
  2. 乐咕乐股实时成分股 (含纳入时间、股票名称)

输出文件(保存在因子库目录):
  - 申万_汽车零部件_三级分类.csv        (主文件，2026最新)
  - 申万_汽车零部件_三级分类_2024年底快照.csv
  - 申万_汽车零部件_全量历史变动.csv
======================================================================
"""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import requests
import io
import os
import time
from datetime import date

VERIFY_SSL = False
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# SW 2021版: 汽车零部件(2802) 三级行业代码 -> 名称
SW_THIRD_CODES = {
    "280201": "车身附件及饰件",
    "280202": "底盘与发动机系统",
    "280203": "轮胎轮毂",
    "280204": "其他汽车零部件",
    "280205": "汽车电子电气系统",
}

# 乐咕乐股对应代码 (仅作补充数据源)
LEGU_THIRD_CODES = {
    "850921.SI": "底盘与发动机系统",
    "850922.SI": "车身附件及饰件",
    "850923.SI": "轮胎轮毂",
    "850924.SI": "其他汽车零部件",
    "850925.SI": "汽车电子电气系统",
}

OUTPUT_DIR = r"c:\Users\Heyuanhang\Desktop\因子库"

# ===================================================================
# Part 1: 从乐咕乐股获取2026年最新成分股（含纳入时间、股票名称）
# ===================================================================
print("=" * 60)
print("[数据源1] 乐咕乐股 - 2026年最新成分股")
print("=" * 60)

all_le = []
for le_code, le_name in LEGU_THIRD_CODES.items():
    short_code = le_code.replace(".SI", "")
    url = f"https://legulegu.com/stockdata/index-composition?industryCode={le_code}"
    try:
        r = requests.get(url, headers=HEADERS, verify=VERIFY_SSL, timeout=30)
        r.encoding = 'utf-8'
        tables = pd.read_html(io.StringIO(r.text))
        if tables and len(tables[0]) > 0:
            df = tables[0]
            for _, row in df.iterrows():
                raw_code = str(row.get("股票代码", ""))
                stock_code = raw_code.replace(".SH","").replace(".SZ","").replace(".BJ","").strip()
                stock_name = str(row.get("股票简称", ""))
                incl_date = str(row.get("纳入时间", "")) if "纳入时间" in df.columns else ""
                
                all_le.append({
                    "股票代码": stock_code.zfill(6),
                    "股票名称": stock_name,
                    "二级行业": "汽车零部件",
                    "三级行业名称": le_name,
                    "纳入日": incl_date,
                    "数据来源": "乐咕乐股",
                })
            print(f"  {le_code} ({le_name}): {len(df)} 只")
        else:
            print(f"  {le_code} ({le_name}): 0 只 (可能动态加载)")
    except Exception as e:
        print(f"  {le_code} ({le_name}): 错误 - {e}")
    time.sleep(1.5)  # 防止请求过快被限制

df_le = pd.DataFrame(all_le)
print(f"\n乐咕乐股合计: {len(df_le)} 条, {df_le['股票代码'].nunique() if len(df_le)>0 else 0} 只独立股票")

# ===================================================================
# Part 2: 从申万官网获取历史分类变动数据(含纳入日/更新日)
# ===================================================================
print("\n" + "=" * 60)
print("[数据源2] 申万官网 - 历史分类变动数据")
print("=" * 60)

url_hist = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
r_hist = requests.get(url_hist, headers=HEADERS, verify=VERIFY_SSL, timeout=60)
df_raw = pd.read_excel(
    io.BytesIO(r_hist.content), dtype={"股票代码": "str", "行业代码": "str"}
)
print(f"全量数据: {len(df_raw)} 行")
print(f"列名: {df_raw.columns.tolist()}")

# 筛选汽车零部件三级行业
target_codes = list(SW_THIRD_CODES.keys())
df_auto = df_raw[df_raw["行业代码"].isin(target_codes)].copy()
print(f"汽车零部件记录: {len(df_auto)} 行")

# 重命名并标准化
df_auto = df_auto.rename(columns={
    "计入日期": "纳入日",
    "更新日期": "更新日",
})
df_auto["三级行业名称"] = df_auto["行业代码"].map(SW_THIRD_CODES)
df_auto["二级行业"] = "汽车零部件"
df_auto["数据来源"] = "申万官网"

# 日期列转换
df_auto["纳入日_dt"] = pd.to_datetime(df_auto["纳入日"], errors="coerce")
df_auto["更新日_dt"] = pd.to_datetime(df_auto["更新日"], errors="coerce")

print("\n各三级行业统计:")
for code, name in SW_THIRD_CODES.items():
    sub = df_auto[df_auto["行业代码"] == code]
    n_stocks = sub["股票代码"].nunique()
    n_records = len(sub)
    print(f"  {code} {name}: {n_records} 条记录, {n_stocks} 只股票")

# ===================================================================
# Part 3: 生成两个时间点的静态快照
# 
# 快照逻辑：对于每个(股票代码, 行业代码)组合，
# 找到纳入日 <= 快照日期的最后一条记录，该记录即为快照时点的有效分类
# （注意：更新日不是剔除日，不需要用更新日过滤）
# ===================================================================
print("\n" + "=" * 60)
print("生成时间点快照")
print("=" * 60)

AS_OF_2024 = pd.Timestamp("2024-12-31")
AS_OF_2026 = pd.Timestamp("2026-10-16")

def make_snapshot(df_hist, as_of_date):
    """
    从历史变动数据中抽取某一时点的静态分类快照。
    对于每个(stock, industry_code)，找纳入日 <= as_of 的最新记录即为有效分类。
    """
    results = []
    for (stock, ind_code), grp in df_hist.groupby(["股票代码", "行业代码"]):
        grp = grp.sort_values("纳入日_dt")
        valid = grp[grp["纳入日_dt"] <= as_of_date]
        if len(valid) == 0:
            continue
        latest = valid.iloc[-1]
        inc_date = latest["纳入日_dt"]
        upd_date = latest["更新日_dt"]
        
        results.append({
            "股票代码": stock,
            "二级行业": "汽车零部件",
            "三级行业代码": ind_code,
            "三级行业名称": SW_THIRD_CODES.get(ind_code, ""),
            "纳入日": str(inc_date.date()) if pd.notna(inc_date) else "",
            "更新日": str(upd_date.date()) if pd.notna(upd_date) else "",
            "快照日期": str(as_of_date.date()),
        })
    return pd.DataFrame(results)

# 2024年底快照
snap_2024 = make_snapshot(df_auto, AS_OF_2024)
print(f"\n2024年底快照 ({AS_OF_2024.date()}): {len(snap_2024)} 条")
for code, name in SW_THIRD_CODES.items():
    cnt = len(snap_2024[snap_2024["三级行业代码"] == code]) if len(snap_2024) > 0 else 0
    print(f"  {name}: {cnt} 只")

# 2026年最新快照
snap_2026 = make_snapshot(df_auto, AS_OF_2026)
print(f"\n2026年最新快照 ({AS_OF_2026.date()}): {len(snap_2026)} 条")
for code, name in SW_THIRD_CODES.items():
    cnt = len(snap_2026[snap_2026["三级行业代码"] == code]) if len(snap_2026) > 0 else 0
    print(f"  {name}: {cnt} 只")

# ===================================================================
# Part 4: 保存 CSV 文件
# ===================================================================
print("\n" + "=" * 60)
print("保存 CSV 文件")
print("=" * 60)

# --- 文件1: 主文件 - 三级分类.csv ---
# 整合思路：以乐咕乐股最新数据为主(有股票名称+纳入时间)，以SW 2026快照为辅
print("\n[生成] 申万_汽车零部件_三级分类.csv (主文件)")

# 合并数据
records = []

# 先从乐咕乐股获取（含股票名称）
if len(df_le) > 0:
    for _, r in df_le.iterrows():
        code = r["股票代码"]
        name = r["三级行业名称"]
        # 从SW快照中查找同一股票同一行业的纳入日
        sw_match = snap_2026[(snap_2026["股票代码"] == code) & 
                             (snap_2026["三级行业名称"] == name)] if len(snap_2026) > 0 else pd.DataFrame()
        inc_date = r.get("纳入日", "")
        upd_date = ""
        if len(sw_match) > 0:
            inc_date = inc_date or sw_match.iloc[0].get("纳入日", "")
            upd_date = sw_match.iloc[0].get("更新日", "")
        
        records.append({
            "股票代码": code,
            "股票名称": r.get("股票名称", ""),
            "二级行业": "汽车零部件",
            "三级行业代码": "",
            "三级行业名称": name,
            "纳入日": inc_date,
            "更新日": upd_date,
            "快照日期": "2026-10-16",
            "数据来源": "乐咕乐股+申万官网",
        })

# 从SW 2026快照补充乐咕乐股未覆盖的（如底盘与发动机系统可能缺失）
if len(snap_2026) > 0:
    existing_codes = set(r["股票代码"] + r["三级行业名称"] for r in records)
    for _, r in snap_2026.iterrows():
        key = r["股票代码"] + r["三级行业名称"]
        if key not in existing_codes:
            records.append({
                "股票代码": r["股票代码"],
                "股票名称": "",
                "二级行业": "汽车零部件",
                "三级行业代码": r["三级行业代码"],
                "三级行业名称": r["三级行业名称"],
                "纳入日": r.get("纳入日", ""),
                "更新日": r.get("更新日", ""),
                "快照日期": r["快照日期"],
                "数据来源": "申万官网",
            })

df_main = pd.DataFrame(records)
if len(df_main) > 0:
    main_cols = ["股票代码", "股票名称", "二级行业", "三级行业名称",
                 "纳入日", "更新日", "快照日期", "数据来源"]
    df_main[main_cols].sort_values(["三级行业名称", "股票代码"]).to_csv(
        os.path.join(OUTPUT_DIR, "申万_汽车零部件_三级分类.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"  已保存: {len(df_main)} 行")

# --- 文件2: 2024年底快照 ---
print("\n[生成] 申万_汽车零部件_三级分类_2024年底快照.csv")
if len(snap_2024) > 0:
    out_cols = ["股票代码", "二级行业", "三级行业代码",
                "三级行业名称", "纳入日", "更新日", "快照日期"]
    snap_2024[out_cols].sort_values(["三级行业代码", "股票代码"]).to_csv(
        os.path.join(OUTPUT_DIR, "申万_汽车零部件_三级分类_2024年底快照.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"  已保存: {len(snap_2024)} 行")

# --- 文件3: 全量历史变动 ---
print("\n[生成] 申万_汽车零部件_全量历史变动.csv")
out_cols = ["股票代码", "二级行业", "行业代码",
            "三级行业名称", "纳入日", "更新日", "数据来源"]
df_auto_out = df_auto[["股票代码", "二级行业", "行业代码",
                       "三级行业名称", "纳入日", "更新日", "数据来源"]].copy()
df_auto_out.rename(columns={"行业代码": "三级行业代码"}, inplace=True)
df_auto_out.sort_values(["三级行业代码", "股票代码", "纳入日"]).to_csv(
    os.path.join(OUTPUT_DIR, "申万_汽车零部件_全量历史变动.csv"),
    index=False, encoding="utf-8-sig"
)
print(f"  已保存: {len(df_auto)} 行")

# ===================================================================
# 汇总
# ===================================================================
print("\n" + "=" * 60)
print("汇总统计")
print("=" * 60)
for code, name in SW_THIRD_CODES.items():
    c24 = len(snap_2024[snap_2024["三级行业代码"] == code]) if len(snap_2024) > 0 else 0
    c26 = len(snap_2026[snap_2026["三级行业代码"] == code]) if len(snap_2026) > 0 else 0
    c_legu = len(df_le[df_le["三级行业名称"] == name]) if len(df_le) > 0 else 0
    print(f"  {name}: SW-2024={c24}, SW-2026={c26}, 乐咕={c_legu}")

total_2024 = snap_2024["股票代码"].nunique() if len(snap_2024) > 0 else 0
total_2026 = snap_2026["股票代码"].nunique() if len(snap_2026) > 0 else 0
print(f"\n合计: 2024年底 {len(snap_2024)} 条/{total_2024} 只, 2026最新 {len(snap_2026)} 条/{total_2026} 只")

print("\n全部完成!")