# -*- coding: utf-8 -*-
"""研发-毛利转化因子构建脚本
==================================
因子定义（方向哑变量，兜底策略）：
    研发强度 = TTM研发费用 / TTM营业总收入
    毛利率   = (TTM营业总收入 - TTM营业成本) / TTM营业总收入
    方向哑变量 = 1  若「研发强度_t-8Q 同比上升」且「毛利率_t 同比上升」
              = 0  否则
    （研发强度_t-8Q 同比上升 = 研发强度(t-8) > 研发强度(t-12)；
      毛利率_t 同比上升 = 毛利率(t) > 毛利率(t-4)）

同时保留基础连续因子（用于分组检验与回测排序）：
    基础因子 = 百分位排名(研发强度, t-8) × 百分位排名(毛利率, t)

构建流程（季度截面，样本内 2019-2024）：
    1. 财报单季度值 -> 连续季度网格 -> 滚动 4 季求和得 TTM
    2. 研发强度(TTM) 与 毛利率(TTM)
    3. 计算同比变化 -> 方向哑变量（0/1）
    4. 基础因子按申万三级行业中位数去均值（行业中性化）
    5. 截面内 1%/99% 缩尾

可用日期（防未来函数）：
    哑变量涉及 t-15 ~ t 共 16 个季度的财报，可用日期 = 这 16 个季度公告日中的最晚值，
    顺延至其后第一个交易日（不使用未来财报）。

运行环境：aksahre_demo/.venv（pandas / numpy）
    python rd_gm_conversion.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"
FACTOR_BASE = SCRIPT_DIR.parent

FINANCIAL_FILE = BASE_DIR / "processed_data" / "财报_清洗完成.csv"
UNADJUSTED_QUOTE_FILE = BASE_DIR / "A股日线_20190101-20260331_不复权.csv"
INDUSTRY_HISTORY_FILE = FACTOR_BASE / "申万_汽车零部件_全量历史变动.csv"

OUTPUT_RAW = SCRIPT_DIR / "研发毛利转化因子_raw.csv"

# 样本区间：财报起始 2019Q1，因子截面截止 2024Q4（样本内 2019-2024）
SAMPLE_START = pd.Period("2019Q1", freq="Q-DEC")
SAMPLE_END = pd.Period("2024Q4", freq="Q-DEC")

# 研发强度滞后季度数：t-2 年 = 8 个季度
RD_LAG_QUARTERS = 8
# 同比比较基期：研发强度(t-8) 去年同期为 t-12；毛利率(t) 去年同期为 t-4
RD_YOY_LAG_QUARTERS = 12
GM_YOY_LAG_QUARTERS = 4
# 哑变量涉及季度数：研发强度 t-8/t-12 的 TTM 用 t-15~t-8，毛利率 t/t-4 的 TTM 用 t-7~t
# -> 合并 t-15 ~ t 共 16 季
INVOLVED_QUARTERS = 16

# 缩尾分位数
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

FACTOR_COLUMN = "最终因子值"

FINANCIAL_COLUMNS = {
    "股票代码",
    "报告期",
    "首次公告日期",
    "研发费用",
    "营业总收入",
    "营业成本",
}


def normalize_codes(values):
    """统一股票代码为 6 位字符串。"""
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\.(SZ|SH|BJ)$", "", case=False, regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(6)
    )


def load_industry_history():
    """读取申万汽车零部件全量历史变动，归一化后返回（含纳入日/更新日）。"""
    df = pd.read_csv(INDUSTRY_HISTORY_FILE, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("申万行业历史分类文件为空")
    cols = {"股票代码": "股票代码", "三级行业名称": "三级行业名称",
            "纳入日": "纳入日", "更新日": "更新日"}
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"申万行业历史分类缺少字段: {sorted(missing)}")
    selected = df.loc[:, list(cols)].rename(columns=cols).copy()
    selected["股票代码"] = normalize_codes(selected["股票代码"])
    selected["纳入日"] = pd.to_datetime(selected["纳入日"], errors="coerce")
    selected["更新日"] = pd.to_datetime(selected["更新日"], errors="coerce")
    selected["三级行业名称"] = selected["三级行业名称"].fillna("").astype(str)
    return selected.sort_values(["股票代码", "纳入日"]).reset_index(drop=True)


def get_industry_map(industry_history, date):
    """返回 date 时刻每只股票 -> 三级行业名称 的映射。

    一条行业记录在「纳入日 <= date 且 (更新日为空 或 更新日 >= date)」时有效，
    取纳入日最新的一条作为该时刻的行业归属。
    """
    date = pd.Timestamp(date).normalize()
    mask = (industry_history["纳入日"] <= date) & (
        industry_history["更新日"].isna() | (industry_history["更新日"] >= date)
    )
    valid = industry_history.loc[mask].copy()
    if valid.empty:
        return {}
    valid = valid.sort_values(["股票代码", "纳入日"], ascending=[True, False])
    latest = valid.drop_duplicates(subset=["股票代码"], keep="first")
    return latest.set_index("股票代码")["三级行业名称"].to_dict()


def trading_calendar(quotes):
    """从行情数据提取排序去重的交易日历。"""
    calendar = quotes.loc[:, ["日期"]].copy()
    calendar["日期"] = pd.to_datetime(calendar["日期"], errors="raise").dt.normalize()
    calendar = calendar.dropna().drop_duplicates().sort_values("日期")
    if calendar.empty:
        raise ValueError("行情数据没有有效交易日")
    return calendar
    return calendar


def build_factor(financials, quotes):
    """由单季度财报构建研发-毛利转化因子（季度截面）。"""
    missing = FINANCIAL_COLUMNS.difference(financials.columns)
    if missing:
        raise ValueError(f"财报数据缺少必要字段: {', '.join(sorted(missing))}")
    if not {"股票代码", "日期"}.issubset(quotes.columns):
        raise ValueError("行情数据必须包含“股票代码”和“日期”字段")

    data = financials.loc[:, list(FINANCIAL_COLUMNS)].copy()
    data["股票代码"] = normalize_codes(data["股票代码"])
    data["报告期"] = pd.to_datetime(data["报告期"], errors="raise")
    data["首次公告日期"] = pd.to_datetime(data["首次公告日期"], errors="coerce")
    for column in ("研发费用", "营业总收入", "营业成本"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["季度"] = data["报告期"].dt.to_period("Q-DEC")
    data = data.drop_duplicates(["股票代码", "季度"], keep="last")

    quarters = pd.period_range(data["季度"].min(), data["季度"].max(), freq="Q-DEC")
    codes = sorted(data["股票代码"].unique())

    def to_grid(column):
        return (
            data.pivot(index="季度", columns="股票代码", values=column)
            .reindex(index=quarters, columns=codes)
        )

    rd = to_grid("研发费用")
    revenue = to_grid("营业总收入")
    cost = to_grid("营业成本")
    announcements = to_grid("首次公告日期")

    ttm_rd = rd.rolling(window=4, min_periods=4).sum()
    ttm_revenue = revenue.rolling(window=4, min_periods=4).sum()
    ttm_cost = cost.rolling(window=4, min_periods=4).sum()

    rd_intensity = ttm_rd.div(ttm_revenue.where(ttm_revenue > 0))
    gross_margin = (ttm_revenue - ttm_cost).div(ttm_revenue.where(ttm_revenue > 0))

    # 研发强度对齐到 t-2 年（8 个季度前）及其去年同期 t-12
    rd_intensity_lag = rd_intensity.shift(RD_LAG_QUARTERS)
    rd_intensity_lag12 = rd_intensity.shift(RD_YOY_LAG_QUARTERS)
    # 毛利率当期及其去年同期 t-4
    gross_margin_lag4 = gross_margin.shift(GM_YOY_LAG_QUARTERS)

    calendar = trading_calendar(quotes)

    quarter_index = {quarter: idx for idx, quarter in enumerate(quarters)}
    rows = []
    for quarter in quarters:
        if quarter > SAMPLE_END:
            break
        idx = quarter_index[quarter]
        start_idx = idx - (INVOLVED_QUARTERS - 1)
        if start_idx < 0:
            continue

        rd_lag = rd_intensity_lag.loc[quarter]        # 研发强度 t-8
        rd_lag12 = rd_intensity_lag12.loc[quarter]    # 研发强度 t-12（去年同期）
        gm_t = gross_margin.loc[quarter]              # 毛利率 t
        gm_lag4 = gross_margin_lag4.loc[quarter]      # 毛利率 t-4（去年同期）

        valid = rd_lag.notna() & rd_lag12.notna() & gm_t.notna() & gm_lag4.notna()
        if valid.sum() < 2:
            continue

        # 因子涉及的 16 个季度公告日期必须齐全，否则跳过（防未来函数）
        announcement_window = announcements.iloc[start_idx:idx + 1]
        latest_announcements = announcement_window.max(axis=0)
        announcements_complete = ~announcement_window.isna().any(axis=0)

        section_codes = valid.index[valid & announcements_complete]
        if len(section_codes) < 2:
            continue

        # 同比变化 -> 方向哑变量（0/1）
        rd_rise = (rd_lag > rd_lag12).loc[section_codes]
        gm_rise = (gm_t > gm_lag4).loc[section_codes]
        dummy = (rd_rise & gm_rise).astype(int)

        # 截面百分位排名（rank pct，值域 (0, 1]）——基础连续因子（用于分组/回测排序）
        rd_rank = rd_lag.loc[section_codes].rank(method="average", pct=True)
        gm_rank = gm_t.loc[section_codes].rank(method="average", pct=True)
        base_factor = rd_rank * gm_rank

        # 时点行业归属，按申万三级行业中位数去均值
        quarter_end = quarter.end_time.normalize()
        industry_map = get_industry_map(industry_history, quarter_end)
        industry_series = pd.Series(
            [industry_map.get(code, "") for code in section_codes],
            index=section_codes,
        )
        frame = pd.DataFrame({
            "股票代码": section_codes,
            "研发强度同比上升": rd_rise.values,
            "毛利率同比上升": gm_rise.values,
            "方向哑变量": dummy.values,
            "研发强度百分位": rd_rank.values,
            "毛利率百分位": gm_rank.values,
            "基础因子": base_factor.values,
            "三级行业名称": industry_series.values,
        })
        frame["中性化因子"] = (
            frame.groupby("三级行业名称")["基础因子"]
            .transform(lambda x: x - x.median())
        )

        # 最终因子值 = 行业中性化后的基础连续因子，截面内 1%/99% 缩尾
        frame["最终因子值"] = frame["中性化因子"]

        values = frame["最终因子值"].astype(float)
        lower, upper = values.quantile(WINSOR_LOWER), values.quantile(WINSOR_UPPER)
        frame["最终因子值"] = values.clip(lower=lower, upper=upper)

        # 可用日期：12 个季度公告日最晚值顺延至其后第一个交易日
        frame = frame.merge(
            pd.DataFrame({
                "股票代码": section_codes,
                "最晚公告日": latest_announcements.loc[section_codes].values,
            }),
            on="股票代码",
            how="left",
        )
        frame = frame.sort_values("最晚公告日")
        frame = pd.merge_asof(
            frame,
            calendar.rename(columns={"日期": "可用日期"}),
            left_on="最晚公告日",
            right_on="可用日期",
            direction="forward",
            allow_exact_matches=False,
        )
        frame = frame.dropna(subset=["可用日期"])
        if frame.empty:
            continue

        frame["截面季度"] = str(quarter)
        frame["报告期"] = quarter_end.strftime("%Y-%m-%d")
        frame["研发强度_TTM滞后2年"] = rd_lag.loc[frame["股票代码"]].values
        frame["研发强度_TTM滞后3年"] = rd_lag12.loc[frame["股票代码"]].values
        frame["毛利率_TTM当期"] = gm_t.loc[frame["股票代码"]].values
        frame["毛利率_TTM去年同期"] = gm_lag4.loc[frame["股票代码"]].values
        rows.append(frame)

    columns = [
        "股票代码", "截面季度", "报告期", "可用日期", "三级行业名称",
        "研发强度_TTM滞后2年", "研发强度_TTM滞后3年", "研发强度同比上升",
        "毛利率_TTM当期", "毛利率_TTM去年同期", "毛利率同比上升",
        "方向哑变量", "研发强度百分位", "毛利率百分位",
        "基础因子", "中性化因子", "最终因子值",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.concat(rows, ignore_index=True)
    result["可用日期"] = pd.to_datetime(result["可用日期"]).dt.strftime("%Y-%m-%d")
    return result[columns].sort_values(["报告期", "股票代码"]).reset_index(drop=True)


def main():
    for path in (FINANCIAL_FILE, UNADJUSTED_QUOTE_FILE, INDUSTRY_HISTORY_FILE):
        if not path.is_file():
            log.error("输入文件不存在: %s", path)
            return 1

    global industry_history
    industry_history = load_industry_history()

    financials = pd.read_csv(
        FINANCIAL_FILE, dtype={"股票代码": str}, encoding="utf-8-sig"
    )
    quotes = pd.read_csv(
        UNADJUSTED_QUOTE_FILE, usecols=["股票代码", "日期"],
        dtype={"股票代码": str}, encoding="utf-8-sig",
    )

    factor = build_factor(financials, quotes)
    if factor.empty:
        log.error("样本区间内没有可计算的研发-毛利转化因子")
        return 1

    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    log.info("因子名称：研发-毛利转化方向哑变量（研发强度_t-8 同比上升 且 毛利率_t 同比上升）")
    log.info("输出文件：%s", OUTPUT_RAW)
    log.info(
        "有效记录：%s；股票数：%s；截面季度：%s ~ %s；可用日期：%s ~ %s",
        len(factor),
        factor["股票代码"].nunique(),
        factor["截面季度"].min(),
        factor["截面季度"].max(),
        factor["可用日期"].min(),
        factor["可用日期"].max(),
    )
    log.info(
        "方向哑变量=1 的占比：%.2f%%（信号触发 %s / %s）",
        factor["方向哑变量"].mean() * 100,
        int(factor["方向哑变量"].sum()),
        len(factor),
    )
    log.info("基础连续因子描述统计：\n%s", factor["最终因子值"].describe().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

