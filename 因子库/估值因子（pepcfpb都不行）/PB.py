# 创建虚拟环境并安装依赖的命令：
#   python -m venv factor_env && factor_env\Scripts\activate && pip install pandas numpy
#   (当前可用工作区 .venv: pip install pandas numpy)

import pandas as pd
import numpy as np
import re
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / 'aksahre_demo'
FACTOR_DIR = SCRIPT_DIR

FIN_REPORT = BASE_DIR / 'processed_data' / '财报_清洗完成.csv'
QUOTE_FILE = BASE_DIR / 'A股日线_20190101-20260331_不复权.csv'
OUTPUT_RAW = FACTOR_DIR / 'PB_raw.csv'
OUTPUT_CLEANED = FACTOR_DIR / 'PB_cleaned.csv'


def format_code(code):
    s = str(code).strip()
    s = re.sub(r'\.(SZ|SH|BJ|sz|sh|bj)$', '', s)
    s = re.sub(r'[^0-9]', '', s)
    return s.zfill(6)


def safe_read_csv(filepath, **kwargs):
    try:
        df = pd.read_csv(filepath, **kwargs)
        log.info(f'读取成功: {filepath.name} ({len(df)} 行)')
        return df
    except FileNotFoundError:
        log.error(f'文件不存在: {filepath}')
        sys.exit(1)
    except Exception as e:
        log.error(f'读取失败 {filepath.name}: {e}')
        sys.exit(1)


print('=' * 60)
print('汽车零部件行业 - PB 市净率 (不复权原始指标)')
print('=' * 60)

fin = safe_read_csv(FIN_REPORT, dtype={'股票代码': str})
quote = safe_read_csv(QUOTE_FILE, dtype={'股票代码': str})

fin['股票代码'] = fin['股票代码'].apply(format_code)
quote['股票代码'] = quote['股票代码'].apply(format_code)

fin['首次公告日期'] = pd.to_datetime(fin['首次公告日期'])
fin['报告期'] = pd.to_datetime(fin['报告期'])
quote['日期'] = pd.to_datetime(quote['日期'])

print(f'\n[1] 财报数据: {fin.shape[0]} 行, {fin["股票代码"].nunique()} 只股票')
print(f'    字段: 股票代码, 报告期, 首次公告日期, 归属于母公司股东权益合计')

print(f'\n[2] 行情数据: {quote.shape[0]} 行, {quote["股票代码"].nunique()} 只股票')
print(f'    日期范围: {quote["日期"].min().date()} ~ {quote["日期"].max().date()}')
print(f'    换手率 max: {quote["换手率"].max():.2f} → 百分比形式 → 流通股本 = 成交量/换手率*10000')

print(f'\n[3] 行业筛选')
# TODO: 待补充行业字段，当前数据源本身已限定汽车零部件股票池
print('    [待补充] 无行业字段 → 跳过筛选')

print(f'\n[4] 可用日期 (公告日后第一个严格大于的交易日)')
trading_days = sorted(quote['日期'].unique())
td_df = pd.DataFrame({'交易日': trading_days})

ann = fin[['股票代码', '报告期', '首次公告日期']].sort_values('首次公告日期').copy()
ann = pd.merge_asof(
    ann, td_df.sort_values('交易日'),
    left_on='首次公告日期', right_on='交易日',
    direction='forward', allow_exact_matches=False,
)
ann = ann.rename(columns={'交易日': '可用日期'})
ann = ann.dropna(subset=['可用日期'])
print(f'    有效记录: {len(ann)}')

q_key = quote[['股票代码', '日期', '收盘', '换手率', '成交量']].rename(columns={'日期': '可用日期'})
merged = ann.merge(q_key, on=['股票代码', '可用日期'], how='left').dropna(subset=['收盘', '换手率'])

merged['流通股本'] = merged['成交量'] / merged['换手率'] * 10000
merged['流通市值'] = merged['收盘'] * merged['流通股本']

fin_key = fin[['股票代码', '报告期', '首次公告日期', '归属于母公司股东权益合计']].copy()
fin_key['首次公告日期'] = pd.to_datetime(fin_key['首次公告日期'])
merged['首次公告日期_dt'] = pd.to_datetime(merged['首次公告日期'])
merged = merged.drop(columns=['首次公告日期'])
merged = merged.merge(fin_key, on=['股票代码', '报告期'], suffixes=('', '_fin'))
merged = merged.drop(columns=['首次公告日期_fin'], errors='ignore')

merged['PB'] = np.where(
    merged['归属于母公司股东权益合计'] > 0,
    merged['流通市值'] / merged['归属于母公司股东权益合计'],
    np.nan,
)

print(f'\n[5] PB = 流通市值 / 归属母公司股东权益 (净资产≤0 → NaN)')
print(f'    合并后记录: {len(merged)}')
print(f'    有效PB记录: {merged["PB"].notna().sum()}')
print(f'    净资产≤0剔除: {merged["PB"].isna().sum()}')

merged['首次公告日期'] = pd.to_datetime(merged['首次公告日期_dt']).dt.strftime('%Y-%m-%d')
merged['可用日期'] = merged['可用日期'].dt.strftime('%Y-%m-%d')
merged['报告期'] = merged['报告期'].dt.strftime('%Y-%m-%d')

result = merged[['股票代码', '报告期', '首次公告日期', '可用日期',
                 '收盘', '流通市值', '归属于母公司股东权益合计', 'PB']].sort_values(
    ['股票代码', '可用日期']).reset_index(drop=True)

print(f'\n[6] PB 描述性统计:')
pb_valid = result['PB'].dropna()
print(pb_valid.describe().round(4).to_string())
print(f'\n    PB范围: [{pb_valid.min():.4f}, {pb_valid.max():.4f}]')
print(f'    PB中位数: {pb_valid.median():.4f}')

print(f'\n[7] 空值检查:')
print(f'    PB空值: {result["PB"].isna().sum()} / {len(result)}')

OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_RAW, index=False, encoding='utf-8-sig')
print(f'\n[8] 已输出: {OUTPUT_RAW}')
print(f'    数据量: {len(result)} 行, {len(result.columns)} 列')
print(f'    前5行预览:')
print(result.head(5).to_string())
# ===================== 截面清洗: 去重 → MAD → Z-score =====================

print('\n' + '-' * 60)
print('截面清洗: lnPB → 去重 → MAD(5x) → 中性化(跳过) → Z-score')
print('-' * 60)

factor = result[['股票代码', '可用日期', 'PB']].copy()
factor = factor.dropna(subset=['PB']).copy()
n0 = len(factor)
factor['lnPB'] = np.log(factor['PB'])

n_dup = factor.duplicated(subset=['股票代码', '可用日期'], keep=False).sum()
print(f'\n[9] 去重 (股票代码+可用日期, 保留lnPB较大):')
print(f'    去重前: {n0} 行, 重复涉及: {n_dup} 行')
factor = factor.sort_values(['股票代码', '可用日期', 'lnPB'], ascending=[True, True, False])
factor = factor.drop_duplicates(subset=['股票代码', '可用日期'], keep='first')
factor = factor.sort_values(['可用日期', '股票代码']).reset_index(drop=True)
print(f'    去重后: {len(factor)} 行')

print(f'\n[10] MAD 去极值 (5x, 截面独立)')
n_clipped = 0

def _winsor(x):
    global n_clipped
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x
    lo, hi = med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad
    c = x.clip(lower=lo, upper=hi)
    n_clipped += int(((x - c).abs() > 1e-10).sum())
    return c

bw = factor['lnPB'].copy()
factor['lnPB_winsor'] = factor.groupby('可用日期')['lnPB'].transform(_winsor)
print(f'    截断: {n_clipped} / {len(factor)} ({n_clipped/len(factor)*100:.2f}%)')
print(pd.DataFrame({'before': bw.agg(['min','max','mean','std']),
                     'after': factor['lnPB_winsor'].agg(['min','max','mean','std'])}).round(4).to_string())

print(f'\n[11] 市值中性化 → 跳过 (无独立市值因子)')
factor['lnPB_neutral'] = factor['lnPB_winsor'].copy()

print(f'\n[12] Z-score (截面独立, std=0→填0)')
zc = 0

def _z(x):
    global zc
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        zc += 1
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / s

factor['lnPB_cleaned'] = factor.groupby('可用日期')['lnPB_neutral'].transform(_z)
print(f'    mean={factor["lnPB_cleaned"].mean():.6f}, std={factor["lnPB_cleaned"].std():.6f}, 单截面填0: {zc}')

counts = factor.groupby('可用日期')['lnPB'].count()
single = counts[counts <= 1].index
factor = factor[~factor['可用日期'].isin(single)].copy()
print(f'    剔除单股票截面: {len(single)} 日期, 保留 {len(factor)} 行')

factor_out = (factor[['可用日期', '股票代码', 'PB', 'lnPB_cleaned']]
              .set_index(['可用日期', '股票代码']).sort_index())
factor_out.to_csv(OUTPUT_CLEANED, encoding='utf-8-sig')
print(f'\n[13] 已输出: {OUTPUT_CLEANED} ({len(factor_out)} 行)')
print(f'    lnPB_cleaned: mean={factor_out["lnPB_cleaned"].mean():.4f}, std={factor_out["lnPB_cleaned"].std():.4f}')
print(f'    前3行:')
print(factor_out.head(3).to_string())

print('\n' + '=' * 60)
print('PB 原始指标 + 清洗完成')
print('=' * 60)