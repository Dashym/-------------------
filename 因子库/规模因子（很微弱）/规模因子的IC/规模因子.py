# 创建虚拟环境并安装依赖的命令：
#   python -m venv factor_env && factor_env\Scripts\activate && pip install pandas numpy scipy statsmodels
#   (Windows: factor_env\Scripts\activate ; Linux/Mac: source factor_env/bin/activate)

import pandas as pd
import numpy as np
import re
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = SCRIPT_DIR.parent.parent.parent
BASE_DIR = DESKTOP_DIR / 'aksahre_demo'
FACTOR_DIR = SCRIPT_DIR

FIN_REPORT = BASE_DIR / 'processed_data' / '财报_清洗完成.csv'
QUOTE_FILE = BASE_DIR / 'A股日线_20190101-20260331_不复权.csv'
OUTPUT = FACTOR_DIR / 'scale_factor.csv'
OUTPUT_CLEANED = FACTOR_DIR / 'scale_factor_cleaned_v2.csv'

SAMPLE_START = pd.Timestamp('2019-01-01')
SAMPLE_END   = pd.Timestamp('2024-12-31')


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
print('汽车零部件行业 - 流通市值规模因子 (v3)')
print('=' * 60)

fin = safe_read_csv(FIN_REPORT, dtype={'股票代码': str})
quote = safe_read_csv(QUOTE_FILE, dtype={'股票代码': str})

fin['股票代码'] = fin['股票代码'].apply(format_code)
quote['股票代码'] = quote['股票代码'].apply(format_code)

fin['首次公告日期'] = pd.to_datetime(fin['首次公告日期'])
fin['报告期'] = pd.to_datetime(fin['报告期'])
quote['日期'] = pd.to_datetime(quote['日期'])

print(f'\n[1] 读取财报数据: {fin.shape[0]} 行, {fin["股票代码"].nunique()} 只股票')
print(f'    公告日期范围: {fin["首次公告日期"].min().date()} ~ {fin["首次公告日期"].max().date()}')

print(f'\n[2] 读取行情数据: {quote.shape[0]} 行, {quote["股票代码"].nunique()} 只股票')
print(f'    行情日期范围: {quote["日期"].min().date()} ~ {quote["日期"].max().date()}')

turnover_range = quote['换手率'].describe()
print(f'\n[2.1] 换手率单位检测')
print(f'    换手率范围: [{turnover_range["min"]:.4f}, {turnover_range["max"]:.4f}]')
print(f'    中位数: {turnover_range["50%"]:.4f}, 均值: {turnover_range["mean"]:.4f}')

if turnover_range['max'] > 1.5:
    TURNOVER_IS_PCT = True
    print('    判断: 百分比形式 (0~100) → 流通股本公式 = 成交量 / 换手率 * 10000')
else:
    TURNOVER_IS_PCT = False
    print('    判断: 小数形式 (0~1) → 流通股本公式 = 成交量 / 换手率 * 100')

# --- 行业筛选 ---
print(f'\n[2.2] 行业筛选')
industry_col = None
for col in fin.columns:
    if '行业' in col or '申万' in col or '中信' in col:
        industry_col = col
        break

if industry_col:
    target = '汽车零部件'
    mask = fin[industry_col].astype(str).str.contains(target, na=False)
    n_before_ind = len(fin)
    fin = fin[mask].copy()
    log.info(f'按 {industry_col} 筛选"{target}": {n_before_ind} → {len(fin)} 行')
    print(f'    筛选项: {industry_col} 包含"{target}"')
    print(f'    筛选前: {n_before_ind} 行, 筛选后: {len(fin)} 行')
else:
    # TODO: 待补充行业字段到财报数据源，当前数据源本身已限定汽车零部件股票池
    print('    [待补充] 财报无行业字段 → 跳过筛选 (数据源本身已限定汽车零部件)')

print(f'\n[2.3] 样本区间截断: {SAMPLE_START.date()} ~ {SAMPLE_END.date()}')
quote = quote[(quote['日期'] >= SAMPLE_START) & (quote['日期'] <= SAMPLE_END)]
print(f'    行情截断后: {quote.shape[0]} 行')
print(f'    行情日期范围: {quote["日期"].min().date()} ~ {quote["日期"].max().date()}')

print(f'\n[2.4] 时间对齐: 因子生效日期 = 公告日后第一个严格大于的交易日')
trading_days = sorted(quote['日期'].unique())
td_df = pd.DataFrame({'交易日': trading_days})

ann = fin[['股票代码', '报告期', '首次公告日期']].sort_values('首次公告日期').copy()

ann = pd.merge_asof(
    ann,
    td_df.sort_values('交易日'),
    left_on='首次公告日期',
    right_on='交易日',
    direction='forward',
    allow_exact_matches=False,
)

ann = ann.rename(columns={'交易日': '可用日期'})

before_filter = ann.shape[0]
ann = ann[(ann['可用日期'] >= SAMPLE_START) & (ann['可用日期'] <= SAMPLE_END)].copy()
print(f'    样本期过滤后: {before_filter} → {ann.shape[0]} 行')

before_nat = ann.shape[0]
ann = ann[ann['可用日期'].notna()].copy()
after_nat = ann.shape[0]
print(f'\n[3] 计算可用日期')
print(f'    超出行情范围: {before_nat - after_nat} 行, 已剔除')
print(f'    有效记录: {after_nat} / {before_nat}')
print(f'    可用日期范围: {ann["可用日期"].min().date()} ~ {ann["可用日期"].max().date()}')

q_key = quote[['股票代码', '日期', '收盘', '换手率', '成交量']].rename(columns={'日期': '可用日期'})

merged = ann.merge(q_key, on=['股票代码', '可用日期'], how='left')

before_drop = merged.shape[0]
merged = merged.dropna(subset=['收盘', '换手率'])
after_drop = merged.shape[0]
print(f'\n[4] 合并行情数据')
print(f'    合并后: {before_drop}, 有效: {after_drop}')

if TURNOVER_IS_PCT:
    merged['流通股本'] = merged['成交量'] / merged['换手率'] * 10000
else:
    merged['流通股本'] = merged['成交量'] / merged['换手率'] * 100

merged['流通市值'] = merged['收盘'] * merged['流通股本']

merged['ln流通市值'] = np.where(
    merged['流通市值'] > 0,
    np.log(merged['流通市值']),
    np.nan,
)

merged['首次公告日期'] = merged['首次公告日期'].dt.strftime('%Y-%m-%d')
merged['可用日期'] = merged['可用日期'].dt.strftime('%Y-%m-%d')
merged['报告期'] = merged['报告期'].dt.strftime('%Y-%m-%d')

result = merged[['股票代码', '报告期', '首次公告日期', '可用日期', '收盘',
                 '换手率', '成交量', '流通股本', '流通市值', 'ln流通市值']].sort_values(
    ['股票代码', '可用日期']).reset_index(drop=True)

print(f'\n[5] 数据统计')
stats = result[['收盘', '换手率', '成交量', '流通股本', '流通市值', 'ln流通市值']].describe().round(4)
print(stats.to_string())

print(f'\n[6] 空值检查')
null_counts = result.isnull().sum()
null_info = pd.DataFrame({'空值数': null_counts, '空值占比(%)': (null_counts / len(result) * 100).round(2)})
if null_counts.sum() == 0:
    print('    无空值 OK')
else:
    print(null_info[null_info['空值数'] > 0].to_string())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f'\n[7] 已输出基础因子: {OUTPUT}')
print(f'    数据量: {result.shape[0]} 行, {result.shape[1]} 列')


# ===================== 截面预处理: 去重 → MAD → 中性化 → Z-score =====================

print('\n' + '-' * 60)
print('截面预处理: 去重 → MAD(5x) → 中性化(跳过) → Z-score')
print('-' * 60)

factor = result[['股票代码', '可用日期', 'ln流通市值']].copy()
factor = factor.dropna(subset=['ln流通市值']).copy()

n_before_dedup = len(factor)
n_dup = factor.duplicated(subset=['股票代码', '可用日期'], keep=False).sum()
print(f'\n[8] 去重 (股票代码 + 可用日期)')
print(f'    去重前: {n_before_dedup} 行, 重复涉及: {n_dup} 行 ({int(n_dup/2)} 对)')

factor = factor.sort_values(['股票代码', '可用日期', 'ln流通市值'], ascending=[True, True, False])
factor = factor.drop_duplicates(subset=['股票代码', '可用日期'], keep='first')
factor = factor.sort_values(['可用日期', '股票代码']).reset_index(drop=True)
print(f'    去重后: {len(factor)} 行')

print(f'\n[9] MAD 去极值 (5倍中位数绝对偏差, 截面独立)')
n_clipped_total = 0
clip_detail = []

def _winsorize_log(x):
    global n_clipped_total
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x
    lo = med - 5 * 1.4826 * mad
    hi = med + 5 * 1.4826 * mad
    clipped = x.clip(lower=lo, upper=hi)
    n = int(((x - clipped).abs() > 1e-10).sum())
    n_clipped_total += n
    if n > 0:
        clip_detail.append((str(x.name), len(x), n))
    return clipped

before_w = factor['ln流通市值'].copy()
factor['ln流通市值_winsor'] = factor.groupby('可用日期')['ln流通市值'].transform(_winsorize_log)

cmp = pd.DataFrame({
    'before': before_w.agg(['min', 'max', 'mean', 'std']),
    'after': factor['ln流通市值_winsor'].agg(['min', 'max', 'mean', 'std']),
})
print(cmp.round(4).to_string())
print(f'    截断值: {n_clipped_total} / {len(factor)} ({n_clipped_total/len(factor)*100:.2f}%)')
if clip_detail:
    cdf = pd.DataFrame(clip_detail, columns=['截面', '股票数', '截断数'])
    print(f'    涉及截面: {len(cdf)} / {factor["可用日期"].nunique()}')
    print('    前5:')
    print(cdf.head(5).to_string(index=False))

print(f'\n[10] 市值中性化')
print('    说明: 当前仅构建规模因子本身, 无独立市值数据可用于回归')
print('    说明: (中性化需另一个因子如 Beta/EP 对规模做截面回归取残差)')
print('    → 跳过, 使用去极值后 ln流通市值 进入标准化')
factor['ln流通市值_neutral'] = factor['ln流通市值_winsor'].copy()

print(f'\n[11] Z-score 标准化 (截面独立, std=0 截面填 0)')
zero_count = 0

def _zscore_safe(x):
    global zero_count
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        zero_count += 1
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / s

factor['ln流通市值_cleaned'] = factor.groupby('可用日期')['ln流通市值_neutral'].transform(_zscore_safe)

print(f'    全局 mean={factor["ln流通市值_cleaned"].mean():.6f}, std={factor["ln流通市值_cleaned"].std():.6f}')
cs = factor.groupby('可用日期')['ln流通市值_cleaned'].agg(['mean', 'std', 'count'])
print(f'    截面数: {len(cs)}, 单股截面填0: {zero_count}')
print(f'    截面均值范围: [{cs["mean"].min():.6f}, {cs["mean"].max():.6f}]')
print(f'    截面 std 范围: [{cs["std"].min():.6f}, {cs["std"].max():.6f}]')
print(f'    ln流通市值_cleaned 空值: {factor["ln流通市值_cleaned"].isna().sum()} / {len(factor)}')

counts = factor.groupby('可用日期')['ln流通市值'].count()
single_dates = counts[counts <= 1].index
factor = factor[~factor['可用日期'].isin(single_dates)].copy()

print(f'\n[12.5] 剔除单股票截面: {len(single_dates)} 个日期, {len(factor)} 行保留')

# --- 双层索引输出 ---
factor_out = (factor[['可用日期', '股票代码', 'ln流通市值', 'ln流通市值_cleaned']]
              .set_index(['可用日期', '股票代码'])
              .sort_index())

factor_out.to_csv(OUTPUT_CLEANED, encoding='utf-8-sig')
print(f'\n[12] 已输出清洗因子 (双层索引): {OUTPUT_CLEANED}')
print(f'    数据量: {factor_out.shape[0]:,} 行, 索引: {list(factor_out.index.names)}')

print(f'\n[13] 最终数据校验')
print(f'    基础因子: scale_factor.csv ({len(result)} 行)')
print(f'    清洗因子: {OUTPUT_CLEANED.name} ({len(factor_out)} 行, 去重-{n_before_dedup - len(factor)})')
print(f'    ln流通市值_cleaned 描述:')
print(factor_out['ln流通市值_cleaned'].describe().round(4).to_string())

print('\n预览 (按索引前5):')
print(factor_out.head(5).to_string())

print('\n' + '=' * 60)
print('因子构建完成')
print('=' * 60)