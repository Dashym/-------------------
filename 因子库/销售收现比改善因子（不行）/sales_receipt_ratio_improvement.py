# -*- coding: utf-8 -*-
"""
销售收现比改善因子（TTM 同比改善）构建脚本
==========================================
现金流质量类因子：衡量企业"销售回款能力"相对去年同期的改善程度。

因子定义：
    销售收现比        = 销售收现TTM / 营业收入TTM
                        （销售收现 = 现金流量表"销售商品、提供劳务收到的现金"）
    销售收现比改善因子 = 当期销售收现比 - 去年同期销售收现比
    因子为正 → 回款质量同比改善（方向：值越大越好）

数据口径（与 07_财报爬取.py 一致）：
    1. 东财"按报告期"接口返回的是**年初至期末累计值**，先按"同年内差分"还原为单季度值；
    2. 单季度值在**连续季度网格**上滚动求和得到 TTM（缺季度 → NaN，不做插值）；
    3. 去年同期 = 季度网格 shift(4)（网格连续，等价于自然年同季度）；
    4. 可用日期 = 因子涉及的 8 个季度（当期 TTM 4 个 + 去年同期 TTM 4 个）中**最晚的公告日**，
       保证不使用尚未披露的数据。

截面口径（防未来函数）：
    对每个"月末交易日"截面日 t，取该日之前（含当日）已公告的最新因子值；
    截面日上限 2024-11-01（因子形成日约束，严禁穿越）。

输出：
    销售收现比改善因子_raw.csv   —— 月末截面因子值（IC / 分层 / 回归三大检验的输入）
    销售收现比改善因子_明细.csv  —— 季度级构建过程明细（审计用）

运行环境：aksahre_demo/.venv（pandas / numpy）
    python sales_receipt_ratio_improvement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "aksahre_demo"

CRAWL_FILE = BASE_DIR / "raw_data" / "汽车零部件_销售收现与营业收入_原始.csv"
POOL_FILE = BASE_DIR / "汽车零部件_申万二级成分股.csv"
ADJUSTED_QUOTE_FILE = (
    BASE_DIR / "clean_temp" / "A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv"
)

FACTOR_FILE = SCRIPT_DIR / "销售收现比改善因子_raw.csv"
DETAIL_FILE = SCRIPT_DIR / "销售收现比改善因子_明细.csv"

FACTOR_COLUMN = "销售收现比改善因子"
# 因子形成日上限（严禁穿越）
MAX_CROSS_SECTION_DATE = pd.Timestamp("2024-11-01")
# 截面起始日
MIN_CROSS_SECTION_DATE = pd.Timestamp("2019-01-01")

CUMULATIVE_COLUMNS = ["营业总收入_累计", "销售收现_累计"]


def progress(message):
    """按仓库脚本惯例：即时打印并 flush。"""
    print(message, flush=True)


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


def load_crawl_cache(path=CRAWL_FILE):
    """读取爬虫缓存（东财按报告期接口的累计值）。"""
    data = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    data["股票代码"] = normalize_codes(data["股票代码"])
    # format="mixed"：爬虫缓存中同时存在 "2017-03-31" 与 "2017-03-31 00:00:00" 两种写法，
    # 若让 pandas 自行推断格式，会把带时间的行整体静默转成 NaT（随后被 dropna 删除，股票数被腰斩）。
    data["报告期"] = pd.to_datetime(data["报告期"], errors="coerce", format="mixed").dt.normalize()
    data["首次公告日期"] = pd.to_datetime(
        data["首次公告日期"], errors="coerce", format="mixed"
    ).dt.normalize()
    for column in CUMULATIVE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["股票代码", "报告期"])
    data = data.drop_duplicates(["股票代码", "报告期"], keep="last")
    data["季度序号"] = data["报告期"].dt.year * 4 + data["报告期"].dt.quarter
    return data.sort_values(["股票代码", "季度序号"], kind="mergesort").reset_index(drop=True)


def to_contiguous_grid(data):
    """长表 → 连续季度网格宽表（index=季度序号，columns=股票代码），缺失季度留 NaN。"""
    quarters = np.arange(int(data["季度序号"].min()), int(data["季度序号"].max()) + 1)
    codes = sorted(data["股票代码"].unique())
    grid = {}
    for column in CUMULATIVE_COLUMNS + ["首次公告日期"]:
        wide = data.pivot(index="季度序号", columns="股票代码", values=column)
        grid[column] = wide.reindex(index=quarters, columns=codes)
    return grid


def to_single_quarter(cumulative):
    """累计值 → 单季度值：Q1 取累计值本身，Q2/Q3/Q4 与上一季度（网格上一行）差分。

    网格连续，因此 shift(1) 即上一季度；上一季度缺失时结果为 NaN（不做插值）。
    """
    first_quarter = (cumulative.index % 4) == 1
    single = cumulative - cumulative.shift(1)
    single.loc[first_quarter] = cumulative.loc[first_quarter]
    return single


def rolling_max_window(frame, window=4):
    """沿季度网格滚动窗口（当前季 + 前 3 季）取最大值，仅对公告日期使用。"""
    values = frame.to_numpy(dtype="datetime64[ns]")
    integers = values.astype("int64")
    numeric = pd.DataFrame(
        integers.astype("float64"), index=frame.index, columns=frame.columns
    )
    missing = integers == np.iinfo("int64").min
    numeric = numeric.mask(missing)
    return numeric.rolling(window, min_periods=window).max()


def float_ns_to_datetime(values):
    """float64（ns，NaN 表示缺失）→ datetime64[ns]（NaT 表示缺失）。"""
    values = np.asarray(values, dtype="float64")
    converted = np.full(values.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = ~np.isnan(values)
    converted[valid] = values[valid].astype("int64").astype("datetime64[ns]")
    return converted


def wide_frames_to_long(frames):
    """{列名: 宽表(index=季度序号, columns=股票代码)} → 长表。"""
    pieces = []
    for name, frame in frames.items():
        piece = (
            frame.rename_axis(index="季度序号", columns="股票代码")
            .reset_index()
            .melt(id_vars="季度序号", var_name="股票代码", value_name=name)
            .set_index(["季度序号", "股票代码"])
        )
        pieces.append(piece)
    return pd.concat(pieces, axis=1).reset_index()


def build_ttm(data, trading_days=None):
    """构建季度级 TTM、销售收现比与同比改善，返回长表。

    传入交易日历时，可用日期取公告日之后的第一个交易日（避免使用当日收盘后才披露的财报）。
    """
    grid = to_contiguous_grid(data)
    single_revenue = to_single_quarter(grid["营业总收入_累计"])
    single_received = to_single_quarter(grid["销售收现_累计"])
    ttm_revenue = single_revenue.rolling(4, min_periods=4).sum()
    ttm_received = single_received.rolling(4, min_periods=4).sum()

    positive_revenue = ttm_revenue.where(ttm_revenue > 0)
    ratio = ttm_received.where(positive_revenue.notna()) / positive_revenue
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    prior_ratio = ratio.shift(4)
    factor = ratio - prior_ratio

    # 可用日期 = 本期 TTM 窗口（4 季）与去年同期 TTM 窗口（shift(4)）中最晚的公告日
    notice_float = rolling_max_window(grid["首次公告日期"])
    current_window = notice_float
    prior_window = notice_float.shift(4)
    available = pd.DataFrame(
        float_ns_to_datetime(
            np.fmax(current_window.to_numpy(dtype="float64"),
                    prior_window.to_numpy(dtype="float64"))
        ),
        index=current_window.index,
        columns=current_window.columns,
    )
    if trading_days is not None:
        available = pd.DataFrame(
            next_trading_day(available.to_numpy(dtype="datetime64[ns]"), trading_days),
            index=available.index,
            columns=available.columns,
        )

    detail = wide_frames_to_long(
        {
            "营业收入TTM": ttm_revenue,
            "销售收现TTM": ttm_received,
            "销售收现比": ratio,
            "去年同期销售收现比": prior_ratio,
            FACTOR_COLUMN: factor,
            "首次公告日期": grid["首次公告日期"],
            "可用日期": available,
        }
    )
    detail["报告期"] = pd.to_datetime(detail["季度序号"].map(quarter_to_period_end))
    detail = detail.drop(columns=["季度序号"])
    return detail.replace([np.inf, -np.inf], np.nan)


def load_pool(path=POOL_FILE):
    """股票池（申万二级汽车零部件）+ 上市日期（用于动态剔除未上市个股）。"""
    pool = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    pool["股票代码"] = normalize_codes(pool["股票代码"])
    pool["listing_date"] = pd.to_datetime(pool["listing_date"], errors="coerce").dt.normalize()
    pool = pool.dropna(subset=["股票代码"]).drop_duplicates("股票代码", keep="last")
    return pool.rename(columns={"listing_date": "上市日"})


def load_trading_calendar(path=ADJUSTED_QUOTE_FILE):
    """全量交易日历（升序 numpy 数组），用于把公告日顺延到其后第一个交易日。"""
    quotes = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig", usecols=["日期"])
    calendar = pd.to_datetime(quotes["日期"], errors="coerce").dropna().unique()
    return np.sort(calendar.astype("datetime64[ns]"))


def next_trading_day(notice_dates, trading_days):
    """公告日（NaT 表示缺失）→ 严格晚于公告日的第一个交易日；NaT 保持缺失。"""
    values = np.asarray(notice_dates, dtype="datetime64[ns]")
    shape = values.shape
    flat = values.reshape(-1).astype("datetime64[D]")
    days = np.asarray(trading_days, dtype="datetime64[D]")
    result = np.full(flat.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = ~np.isnat(flat)
    if valid.any():
        positions = np.searchsorted(days, flat[valid], side="right")
        inside = positions < len(days)
        target = np.nonzero(valid)[0][inside]
        result[target] = days[positions[inside]].astype("datetime64[ns]")
    return result.reshape(shape)


def load_cross_section_dates(path=ADJUSTED_QUOTE_FILE):
    """月末交易日序列（截面日），区间 [2019-01-01, 2024-11-01]。"""
    calendar = pd.DataFrame({"日期": load_trading_calendar(path)})
    calendar["月份"] = calendar["日期"].dt.to_period("M")
    month_end = calendar.groupby("月份")["日期"].max().sort_values()
    month_end = month_end.loc[
        (month_end >= MIN_CROSS_SECTION_DATE) & (month_end <= MAX_CROSS_SECTION_DATE)
    ]
    return list(month_end)


def build_month_end_panel(detail, pool, cross_section_dates):
    """按月末截面日取"已公告的最新因子值"，生成截面面板。"""
    usable = detail.dropna(subset=[FACTOR_COLUMN, "可用日期"])
    usable = usable.sort_values(["股票代码", "报告期"], kind="mergesort")
    frames = []
    for as_of in cross_section_dates:
        available = usable.loc[usable["可用日期"] <= as_of]
        if available.empty:
            continue
        latest = available.drop_duplicates("股票代码", keep="last").copy()
        latest = latest.merge(pool, on="股票代码", how="inner", validate="many_to_one")
        latest = latest.loc[latest["上市日"] <= as_of]
        latest["可用日期"] = as_of
        latest["截面月份"] = as_of.to_period("M").strftime("%Y-%m")
        frames.append(latest)
    if not frames:
        raise ValueError("没有任何月末截面可用，请检查数据区间与公告日期")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["可用日期", "股票代码"], kind="mergesort").reset_index(drop=True)


def _format_date_columns(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result


def to_cross_section_file(panel):
    """截面因子文件（三大检验的输入）。"""
    columns = [
        "股票代码",
        "可用日期",
        "报告期",
        "首次公告日期",
        "营业收入TTM",
        "销售收现TTM",
        "销售收现比",
        "去年同期销售收现比",
        FACTOR_COLUMN,
        "上市日",
        "截面月份",
    ]
    return _format_date_columns(panel.loc[:, columns], ["可用日期", "报告期", "首次公告日期", "上市日"])


def to_detail_file(detail):
    """季度级构建明细（审计用）。"""
    columns = [
        "股票代码",
        "报告期",
        "首次公告日期",
        "可用日期",
        "营业收入TTM",
        "销售收现TTM",
        "销售收现比",
        "去年同期销售收现比",
        FACTOR_COLUMN,
    ]
    result = _format_date_columns(detail.loc[:, columns], ["报告期", "首次公告日期", "可用日期"])
    return result.sort_values(["股票代码", "报告期"], kind="mergesort").reset_index(drop=True)



def quarter_to_period_end(quarter_number):
    """季度序号（年*4+季）→ 报告期（季末日期）。"""
    year, quarter = divmod(int(quarter_number) - 1, 4)
    return pd.Timestamp(year=year, month=(quarter + 1) * 3, day=1) + pd.offsets.MonthEnd(0)


def describe_panel(panel, detail, pool, cross_section_dates):
    """构建过程摘要（打印用）。"""
    lines = [
        "=" * 72,
        "销售收现比改善因子构建完成",
        "=" * 72,
        "",
        "口径：",
        "  销售收现比 = 销售收现TTM / 营业收入TTM，TTM 由连续 4 个季度单季值滚动求和",
        "  因子值     = 当期销售收现比 - 去年同期销售收现比（正值 = 回款质量同比改善）",
        "  可用日期   = 因子涉及 8 个季度公告日中的最晚值，顺延至其后第一个交易日（不使用未来财报）",
        f"  截面日     = 月末交易日，区间上限 {MAX_CROSS_SECTION_DATE:%Y-%m-%d}（因子形成日约束）",
        "",
        "数据概况：",
        f"  股票池（申万二级汽车零部件，含动态上市/纳入过滤）：{len(pool)} 只",
        f"  截面日数量：{len(cross_section_dates)} 期",
        f"  季度级因子记录：{len(detail):,} 条，报告期 "
        f"{detail['报告期'].min()} ~ {detail['报告期'].max()}",
        f"  月末截面观测：{len(panel):,} 条，截面 {panel['可用日期'].min():%Y-%m-%d} ~ "
        f"{panel['可用日期'].max():%Y-%m-%d}",
    ]
    per_section = panel.groupby("可用日期")["股票代码"].count()
    lines.extend(
        [
            f"  平均股票数/截面：{per_section.mean():.1f}，范围 {per_section.min()} ~ {per_section.max()}",
            "",
            "因子分布（月末截面原始值）：",
            f"  均值 {panel[FACTOR_COLUMN].mean():.6f}，标准差 {panel[FACTOR_COLUMN].std(ddof=1):.6f}",
            f"  最小值 {panel[FACTOR_COLUMN].min():.6f}，最大值 {panel[FACTOR_COLUMN].max():.6f}",
            f"  正值占比 {(panel[FACTOR_COLUMN] > 0).mean():.2%}",
            "",
            f"已输出 → {FACTOR_FILE.name}（月末截面因子）",
            f"已输出 → {DETAIL_FILE.name}（季度级明细）",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def main():
    for input_file in (CRAWL_FILE, POOL_FILE, ADJUSTED_QUOTE_FILE):
        if not input_file.is_file():
            progress(f"[错误] 输入文件不存在：{input_file}")
            return 1

    progress("[1/6] 读取爬虫缓存（销售收现 / 营业总收入 累计值）...")
    crawl = load_crawl_cache()
    progress(
        f"  {crawl['股票代码'].nunique()} 只股票，{len(crawl):,} 条，报告期 "
        f"{crawl['报告期'].min():%Y-%m-%d} ~ {crawl['报告期'].max():%Y-%m-%d}"
    )

    progress("[2/6] 读取交易日历（公告日顺延至次一交易日）...")
    trading_days = load_trading_calendar()
    progress(
        f"  交易日历 {len(trading_days)} 个交易日，"
        f"{pd.Timestamp(trading_days[0]):%Y-%m-%d} ~ {pd.Timestamp(trading_days[-1]):%Y-%m-%d}"
    )

    progress("[3/6] 累计值 → 单季度值 → TTM → 销售收现比同比改善 ...")
    detail = build_ttm(crawl, trading_days)
    valid = detail.dropna(subset=[FACTOR_COLUMN])
    progress(f"  季度级因子有效记录 {len(valid):,} 条")

    progress("[4/6] 读取股票池与月末截面日 ...")
    pool = load_pool()
    cross_section_dates = load_cross_section_dates()
    progress(f"  股票池 {len(pool)} 只，截面日 {len(cross_section_dates)} 期")

    progress("[5/6] 生成月末截面面板（严格使用已公告数据）...")
    panel = build_month_end_panel(detail, pool, cross_section_dates)

    progress("[6/6] 写出因子文件 ...")
    to_cross_section_file(panel).to_csv(FACTOR_FILE, index=False, encoding="utf-8-sig")
    to_detail_file(detail).to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")
    progress(describe_panel(panel, detail, pool, cross_section_dates))
    return 0


if __name__ == "__main__":
    sys.exit(main())

