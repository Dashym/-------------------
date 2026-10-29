"""
因子质量前置体检模块
====================
对已完成预处理（去极值+MAD、Z-score标准化）的因子进行4项检验，
在计算IC之前判断因子是否可用。

检验项目:
  1. 缺失率检查
  2. 截面分布检查
  3. 因子自相关与衰减
  4. 分组区分度（基于原始ln流通市值）

用法:
  python factor_quality_check.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------
FACTOR_COL = 'ln流通市值_cleaned'
RAW_COL = 'ln流通市值'
DATE_LEVEL = '可用日期'
CODE_LEVEL = '股票代码'
CSV_REL_PATH = Path(__file__).resolve().parent.parent / '因子库' / 'scale_factor_cleaned_v2.csv'


def load_data(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """
    读取因子CSV并构建双层索引 (可用日期, 股票代码)。

    参数
    ----
    filepath : str or Path, optional
        CSV文件路径，默认自动查找因子库目录下的 scale_factor_cleaned_v2.csv

    返回
    ----
    pd.DataFrame : 双层索引 (可用日期, 股票代码)，列包含 ln流通市值, ln流通市值_cleaned
    """
    if filepath is None:
        filepath = CSV_REL_PATH
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'因子文件不存在: {path}')

    df = pd.read_csv(path, index_col=[0, 1], dtype={CODE_LEVEL: str})
    df.index.names = [DATE_LEVEL, CODE_LEVEL]
    df = df.sort_index()
    return df


# ===================================================================
# 检验1: 缺失率检查
# ===================================================================

def check_missing_rate(
    df: pd.DataFrame,
    factor_col: str = FACTOR_COL,
) -> dict:
    """
    检验1: 缺失率检查。

    计算每期因子缺失率及全样本平均缺失率。
    平均缺失率 > 15% → WARN

    返回
    ----
    dict : 包含 status, per_date_missing, mean_missing, max_missing, reason
    """
    per_date_missing = df.groupby(level=DATE_LEVEL)[factor_col].apply(
        lambda x: x.isna().mean()
    )
    mean_missing = per_date_missing.mean()
    max_missing = per_date_missing.max()
    n_dates = len(per_date_missing)

    reason = f'平均缺失率: {mean_missing:.4%}, 最大单期缺失率: {max_missing:.4%}, 总截面数: {n_dates}'
    if mean_missing > 0.15:
        status = 'WARN'
    else:
        status = 'PASS'

    return {
        'status': status,
        'per_date_missing': per_date_missing,
        'mean_missing': mean_missing,
        'max_missing': max_missing,
        'n_dates': n_dates,
        'reason': reason,
    }


# ===================================================================
# 检验2: 截面分布检查
# ===================================================================

def check_cross_sectional_distribution(
    df: pd.DataFrame,
    factor_col: str = FACTOR_COL,
) -> dict:
    """
    检验2: 截面分布检查。

    每期计算因子值的均值、标准差、偏度、峰度。
    偏度绝对值 > 2 → WARN

    返回
    ----
    dict : 包含 status, cs_stats (DataFrame), reason
    """
    records: list[dict] = []

    for dt, grp in df.groupby(level=DATE_LEVEL):
        vals = grp[factor_col].dropna()
        n = len(vals)
        if n < 8:
            continue
        records.append({
            'date': dt,
            'n': n,
            'mean': float(vals.mean()),
            'std': float(vals.std(ddof=0)),
            'skew': float(stats.skew(vals)),
            'kurtosis': float(stats.kurtosis(vals)),
        })

    cs_df = pd.DataFrame(records).set_index('date')

    if cs_df.empty:
        return {
            'status': 'FAIL',
            'cs_stats': cs_df,
            'reason': '无足够截面数据计算分布统计',
        }

    mean_range = (cs_df['mean'].min(), cs_df['mean'].max())
    std_range = (cs_df['std'].min(), cs_df['std'].max())
    mean_skew = cs_df['skew'].mean()
    mean_kurt = cs_df['kurtosis'].mean()

    reason = (
        f'均值范围: [{mean_range[0]:.4f}, {mean_range[1]:.4f}], '
        f'标准差范围: [{std_range[0]:.4f}, {std_range[1]:.4f}], '
        f'平均偏度: {mean_skew:.4f}, '
        f'平均峰度: {mean_kurt:.4f}'
    )

    if abs(mean_skew) > 2:
        status = 'WARN'
    else:
        status = 'PASS'

    return {
        'status': status,
        'cs_stats': cs_df,
        'mean_range': mean_range,
        'std_range': std_range,
        'mean_skew': mean_skew,
        'mean_kurt': mean_kurt,
        'reason': reason,
    }


# ===================================================================
# 检验3: 因子自相关与衰减
# ===================================================================

def check_autocorrelation(
    df: pd.DataFrame,
    factor_col: str = FACTOR_COL,
) -> dict:
    """
    检验3: 因子自相关与衰减。

    每只股票计算因子值在滞后1/5/10/20期的自相关系数，
    汇总全样本平均值。
    滞后1期均值 < 0.6 → WARN

    返回
    ----
    dict : 包含 status, ac_means, half_life, n_stocks, reason
    """
    lags = [1, 5, 10, 20]
    all_ac: dict[int, list[float]] = {lag: [] for lag in lags}

    for code, grp in df.groupby(level=CODE_LEVEL):
        grp = grp.sort_index()
        series = grp[factor_col].dropna()
        if len(series) < max(lags) + 10:
            continue
        for lag in lags:
            if len(series) <= lag:
                continue
            ac = series.autocorr(lag=lag)
            if not np.isnan(ac):
                all_ac[lag].append(ac)

    ac_means = {lag: float(np.mean(vals)) if vals else np.nan for lag, vals in all_ac.items()}
    n_stocks = len(all_ac[1])
    lag1_mean = ac_means.get(1, np.nan)

    part_details = ', '.join(f'lag{lag}={ac_means[lag]:.4f}' for lag in lags)
    reason = f'全样本平均自相关: {part_details}, 有效股票数: {n_stocks}'

    if not np.isnan(lag1_mean) and lag1_mean < 0.6:
        status = 'WARN'
    elif np.isnan(lag1_mean):
        status = 'FAIL'
    else:
        status = 'PASS'

    half_life: Optional[float] = None
    if lag1_mean is not None and not np.isnan(lag1_mean) and 0 < lag1_mean < 1:
        half_life = float(np.log(0.5) / np.log(lag1_mean))
        reason += f', 半衰期: {half_life:.2f}天'

    return {
        'status': status,
        'ac_means': ac_means,
        'lag1_mean': lag1_mean,
        'half_life': half_life,
        'n_stocks': n_stocks,
        'reason': reason,
    }


# ===================================================================
# 检验4: 分组区分度（基于原始ln流通市值）
# ===================================================================

def check_group_discrimination(
    df: pd.DataFrame,
    factor_col: str = FACTOR_COL,
    raw_col: str = RAW_COL,
    n_groups: int = 10,
) -> dict:
    """
    检验4: 分组区分度。

    每期按 cleaned 因子值分10组，取最大组（G10）和最小组（G1）
    的原始 ln流通市值 均值，计算比值 ratio = mean(G10) / mean(G1)。
    全样本平均比值 < 1.5 → WARN

    返回
    ----
    dict : 包含 status, per_date_ratio, mean_ratio, n_valid_dates, reason
    """
    if raw_col not in df.columns:
        return {
            'status': 'FAIL',
            'reason': f'原始因子列 "{raw_col}" 不存在于数据中',
        }

    ratios: list[float] = []

    for dt, grp in df.groupby(level=DATE_LEVEL):
        valid = grp[[factor_col, raw_col]].dropna()
        n = len(valid)
        if n < n_groups:
            continue

        try:
            labels = pd.qcut(valid[factor_col], q=n_groups, labels=False, duplicates='drop')
        except (ValueError, IndexError):
            continue

        g_means = valid.groupby(labels)[raw_col].mean()
        g_idx = g_means.index.tolist()
        if 1 + n_groups not in [len(set(g_idx)), len(g_idx)]:
            unique_groups = len(set(g_idx))
            if unique_groups < 4:
                continue

        g_min = g_means.min()
        g_max = g_means.max()
        if g_min is None or g_max is None or g_min <= 0:
            continue

        ratios.append(float(g_max / g_min))

    if not ratios:
        return {
            'status': 'FAIL',
            'reason': '无足够截面数据计算分组区分度',
        }

    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    n_valid = len(ratios)

    reason = (
        f'最大组/最小组原始ln流通市值比值均值: {mean_ratio:.4f} '
        f'(std={std_ratio:.4f}), 有效截面数: {n_valid}'
    )

    if mean_ratio < 1.5:
        status = 'WARN'
    else:
        status = 'PASS'

    return {
        'status': status,
        'per_date_ratio': pd.Series(ratios),
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'n_valid_dates': n_valid,
        'reason': reason,
    }


# ===================================================================
# 汇总报告
# ===================================================================

def generate_report(df: pd.DataFrame) -> dict:
    """
    运行全部4项检验并打印格式化体检报告。

    返回
    ----
    dict : {
        'overall': 'PASS' / 'WARN' / 'FAIL',
        'checks': {检查名: {status, reason, ...}},
    }
    """
    checks = {
        '缺失率检查': check_missing_rate(df),
        '截面分布检查': check_cross_sectional_distribution(df),
        '因子自相关与衰减': check_autocorrelation(df),
        '分组区分度': check_group_discrimination(df),
    }

    severity = {'PASS': 3, 'WARN': 2, 'FAIL': 1}
    overall_sev = min(severity[info['status']] for info in checks.values())
    overall = {v: k for k, v in severity.items()}[overall_sev]

    lines: list[str] = []
    lines.append('=' * 62)
    lines.append('  因子质量前置体检报告')
    lines.append('=' * 62)
    lines.append(f'  因子列: {FACTOR_COL}')
    lines.append(f'  数据量: {len(df):,} 行, '
                 f'{df.index.get_level_values(DATE_LEVEL).nunique()} 个截面, '
                 f'{df.index.get_level_values(CODE_LEVEL).nunique()} 只股票')
    lines.append('-' * 62)

    for idx, (name, info) in enumerate(checks.items(), 1):
        status = info['status']
        flag = {'PASS': '[PASS]', 'WARN': '[WARN]', 'FAIL': '[FAIL]'}[status]
        lines.append(f'  {idx}. {flag} {name}')
        lines.append(f'     {info["reason"]}')

    lines.append('-' * 62)
    if overall == 'PASS':
        lines.append(f'  综合判断: [PASS] 因子质量良好，可进入IC检验')
    elif overall == 'WARN':
        lines.append(f'  综合判断: [WARN] 因子存在质量隐患，建议修复后重试')
    else:
        lines.append(f'  综合判断: [FAIL] 因子质量不达标，必须修复后重试')
    lines.append('=' * 62)

    report_text = '\n'.join(lines)
    print(report_text)

    return {
        'overall': overall,
        'checks': checks,
    }


# ===================================================================
# 入口
# ===================================================================

if __name__ == '__main__':
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f'[ERROR] {e}')
        print(f'请确认文件路径: {CSV_REL_PATH}')
        raise SystemExit(1)

    print(f'读取因子数据: {df.shape[0]:,} 行, 列: {list(df.columns)}')
    print(f'日期范围: {df.index.get_level_values(DATE_LEVEL).min()} ~ '
          f'{df.index.get_level_values(DATE_LEVEL).max()}\n')

    generate_report(df)