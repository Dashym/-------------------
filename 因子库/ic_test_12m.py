# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np
import pandas as pd

FACTOR_FILE = Path(sys.argv[1])
FACTOR_COLUMN = sys.argv[2]
OUTPUT_FILE = Path(sys.argv[3])
REPORT_TITLE = sys.argv[4]
DATE_COLUMN = sys.argv[5] if len(sys.argv) > 5 else '可用日期'
SKIP_MCAP = len(sys.argv) > 6 and sys.argv[6] == 'skip_mcap'

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / 'aksahre_demo'
UNADJUSTED_QUOTE_FILE = BASE_DIR / 'A股日线_20190101-20260331_不复权.csv'
ADJUSTED_QUOTE_FILE = BASE_DIR / 'clean_temp' / 'A股日线_20190101-20260331_后复权_去重排序_剔除异常_格式统一.csv'
IN_SAMPLE_END = pd.Period('2024-03', freq='M')
MIN_CROSS_SECTION_SIZE = 10
FORWARD_HORIZON_MONTHS = 12
MAD_MULTIPLIER = 5.0
MAD_NORMAL_SCALE = 1.4826


def normalize_codes(values):
    return (values.astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
            .str.replace(r'\.(SZ|SH|BJ)$', '', case=False, regex=True)
            .str.replace(r'[^0-9]', '', regex=True)
            .str.zfill(6))


def monthly_market_data(quotes, need_mcap):
    cols = ['股票代码', '日期', '收盘'] + (['成交量', '换手率'] if need_mcap else [])
    data = quotes.loc[:, cols].copy()
    data['股票代码'] = normalize_codes(data['股票代码'])
    data['日期'] = pd.to_datetime(data['日期'], errors='raise').dt.normalize()
    for c in cols[2:]:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(subset=['股票代码', '日期', '收盘'])
    data = data.loc[data['收盘'] > 0].copy()
    data['月份'] = data['日期'].dt.to_period('M')
    if need_mcap:
        vt = data.loc[data['换手率'] > 0, '换手率']
        scale = 10000 if vt.max() > 1.5 else 100
        shares = pd.Series(np.nan, index=data.index, dtype=float)
        mask = data['换手率'] > 0
        shares[mask] = data.loc[mask, '成交量'] / data.loc[mask, '换手率'] * scale
        mv = data['收盘'] * shares
        data['ln流通市值'] = np.log(mv.where(mv > 0))
        data.loc[~np.isfinite(data['ln流通市值']), 'ln流通市值'] = np.nan
    return (data.sort_values(['股票代码', '日期'])
            .groupby(['股票代码', '月份'], as_index=False).tail(1).copy())


def newey_west_se(values, max_lags=11):
    v = np.asarray(values, dtype=float)
    n = len(v)
    r = v - v.mean()
    lrv = float(np.dot(r, r) / n)
    for lag in range(1, max_lags + 1):
        w = 1 - lag / (max_lags + 1)
        lrv += 2 * w * float(np.dot(r[lag:], r[:-lag]) / n)
    return float(np.sqrt(max(lrv / n * n / (n - 1), 0.0)))


def statistics(values):
    v = values.dropna()
    if v.empty:
        return dict(mean=np.nan, std=np.nan, ir=np.nan, pos=np.nan, t=np.nan)
    mean = v.mean()
    std = v.std(ddof=1) if len(v) > 1 else np.nan
    se = newey_west_se(v, 11) if len(v) > 11 else np.nan
    return dict(mean=mean, std=std, ir=mean / std if np.isfinite(std) and std > 0 else np.nan,
                pos=(v > 0).mean(), t=mean / se if np.isfinite(se) and se > 0 else np.nan)


def main():
    factor = pd.read_csv(FACTOR_FILE, dtype={'股票代码': str}, encoding='utf-8-sig')
    unadj = pd.read_csv(UNADJUSTED_QUOTE_FILE, dtype={'股票代码': str}, encoding='utf-8-sig')
    adj = pd.read_csv(ADJUSTED_QUOTE_FILE, dtype={'股票代码': str}, encoding='utf-8-sig')

    signals = factor.loc[:, ['股票代码', DATE_COLUMN, FACTOR_COLUMN]].copy()
    signals = signals.rename(columns={DATE_COLUMN: '可用日期'})
    signals['股票代码'] = normalize_codes(signals['股票代码'])
    signals['可用日期'] = pd.to_datetime(signals['可用日期'], errors='raise').dt.normalize()
    signals[FACTOR_COLUMN] = pd.to_numeric(signals[FACTOR_COLUMN], errors='coerce')
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(subset=['股票代码', '可用日期', FACTOR_COLUMN])
    signals = signals.sort_values(['股票代码', '可用日期']).drop_duplicates(['股票代码', '可用日期'], keep='last')

    market = monthly_market_data(unadj, need_mcap=True)
    adjusted = monthly_market_data(adj, need_mcap=False)
    market = market.rename(columns={'日期': '市值日期', 'ln流通市值': 'mcap'})
    adj_prices = adjusted.pivot(index='月份', columns='股票代码', values='收盘')
    all_months = pd.period_range(adj_prices.index.min(), adj_prices.index.max(), freq='M')
    adj_prices = adj_prices.reindex(all_months)
    fwd = adj_prices.shift(-12).div(adj_prices).sub(1)
    fwd.index.name = '月份'
    fwd = (fwd.rename_axis(columns='股票代码').reset_index()
           .melt(id_vars='月份', var_name='股票代码', value_name='持有期收益')
           .dropna(subset=['持有期收益']))

    results = []
    for month in sorted(market['月份'].unique()):
        if month > IN_SAMPLE_END:
            continue
        month_end = market.loc[market['月份'] == month, '市值日期'].max()
        as_of = signals.loc[signals['可用日期'] <= month_end]
        if as_of.empty:
            continue
        as_of = as_of.sort_values(['股票代码', '可用日期']).drop_duplicates('股票代码', keep='last')
        cm = market.loc[market['月份'] == month]
        cr = fwd.loc[fwd['月份'] == month]
        cs = (as_of.merge(cm[['股票代码', '月份', 'mcap']], on='股票代码', how='inner')
              .merge(cr[['股票代码', '月份', '持有期收益']], on=['股票代码', '月份'], how='inner'))
        cs = cs.replace([np.inf, -np.inf], np.nan).dropna(subset=[FACTOR_COLUMN, 'mcap', '持有期收益'])
        if len(cs) < MIN_CROSS_SECTION_SIZE:
            continue
        values = cs[FACTOR_COLUMN]
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0 and np.isfinite(mad):
            values = values.clip(lower=median - MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad,
                                 upper=median + MAD_MULTIPLIER * MAD_NORMAL_SCALE * mad)
        if SKIP_MCAP:
            std = values.std(ddof=0)
            if std == 0 or not np.isfinite(std):
                continue
            standardized = (values - values.mean()) / std
        else:
            design = np.column_stack([np.ones(len(cs)), cs['mcap'].to_numpy(dtype=float)])
            coef, _, _, _ = np.linalg.lstsq(design, values.to_numpy(dtype=float), rcond=None)
            residuals = values.to_numpy(dtype=float) - design @ coef
            rs = residuals.std(ddof=0)
            if rs == 0 or not np.isfinite(rs):
                continue
            standardized = (residuals - residuals.mean()) / rs
        returns = cs['持有期收益'].to_numpy(dtype=float)
        rank_ic = pd.Series(standardized).rank(method='average').corr(pd.Series(returns).rank(method='average'))
        pearson_ic = pd.Series(standardized).corr(pd.Series(returns))
        results.append(dict(截面月份=str(month), 截面日期=month_end.strftime('%Y-%m-%d'),
                            预测月份=str(month + FORWARD_HORIZON_MONTHS), 股票数=len(cs),
                            Rank_IC=rank_ic, Pearson_IC=pearson_ic))

    df = pd.DataFrame(results, columns=['截面月份', '截面日期', '预测月份', '股票数', 'Rank_IC', 'Pearson_IC'])
    if df.empty:
        print('无有效截面')
        return 1
    rk = statistics(df['Rank_IC'])
    pc = statistics(df['Pearson_IC'])
    cum = df['Rank_IC'].cumsum()
    mcap_note = '（规模因子：跳过市值中性化）' if SKIP_MCAP else ''
    lines = ['=' * 72, REPORT_TITLE, '=' * 72, '', '检测口径：',
             '  月末截面使用截至该月最后交易日已公告的最新因子值（不使用未来财报）',
             '  对月末截面重新执行 MAD(5倍) 去极值、对数流通市值回归中性化及 Z-score' + mcap_note,
             '  收益：未来 12 个月累计后复权收益 = t+12 月末后复权收盘 / t 月末后复权收盘 - 1',
             '  Rank IC：Spearman 秩相关；Pearson IC：线性相关',
             f'  最少有效股票数/截面：{MIN_CROSS_SECTION_SIZE}',
             '  IR：月度 IC 均值 / 月度 IC 样本标准差（ddof=1）',
             '  t-stat 经 Newey-West HAC（Bartlett 核，滞后 11）调整', '', '数据概况：',
             f'  有效月度截面：{len(df)} 期；因子月份 {df["截面月份"].iloc[0]} ~ {df["截面月份"].iloc[-1]}',
             f'  平均有效股票数：{df["股票数"].mean():.1f}，范围 {df["股票数"].min()} ~ {df["股票数"].max()}', '',
             f"{'指标':<16}{'Rank IC':>14}{'Pearson IC':>16}", '-' * 46,
             f"{'IC 均值':<16}{rk['mean']:>14.4f}{pc['mean']:>16.4f}",
             f"{'IC 标准差':<16}{rk['std']:>14.4f}{pc['std']:>16.4f}",
             f"{'ICIR':<16}{rk['ir']:>14.4f}{pc['ir']:>16.4f}",
             f"{'IC > 0 占比':<16}{rk['pos']:>13.2%}{pc['pos']:>15.2%}",
             f"{'HAC t-stat':<16}{rk['t']:>14.4f}{pc['t']:>16.4f}", '',
             f'累计 Rank IC：{cum.iloc[0]:.4f} -> {cum.iloc[-1]:.4f}',
             f"综合提示：样本内 Rank IC {'为正' if rk['mean'] > 0 else '为负'}，均值 {rk['mean']:.4f}，ICIR {rk['ir']:.4f}；请结合样本长度及后续分层回测判断，不代表未来表现。", '',
             '逐月结果：', f"{'因子月份':<12}{'预测月份':<12}{'样本数':>8}{'Rank IC':>12}{'Pearson IC':>14}", '-' * 58]
    lines.extend(f"{r['截面月份']:<12}{r['预测月份']:<12}{r['股票数']:>8d}{r['Rank_IC']:>12.4f}{r['Pearson_IC']:>14.4f}" for _, r in df.iterrows())
    lines.extend(['', '=' * 72, ''])
    text = '\n'.join(lines)
    OUTPUT_FILE.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
