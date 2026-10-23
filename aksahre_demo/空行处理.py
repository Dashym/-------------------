from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
CLEAN_DIR = BASE_DIR / "clean_temp"
CSV_FILES = [
	"A股日线_20190101-20260331_前复权.csv",
	"A股日线_20190101-20260331_后复权.csv",
	"A股日线_20190101-20260331_不复权.csv",
]
REQUIRED_COLUMNS = {"股票代码", "成交量", "换手率"}


def find_input_file(filename: str) -> Path:
	"""优先读取 raw_data；兼容原始文件仍放在项目根目录的情况。"""
	for directory in (RAW_DIR, BASE_DIR):
		candidate = directory / filename
		if candidate.is_file():
			return candidate
	raise FileNotFoundError(f"找不到输入文件：{RAW_DIR / filename} 或 {BASE_DIR / filename}")


def clean_file(filename: str) -> None:
	input_path = find_input_file(filename)
	frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype={"股票代码": str})
	missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
	if missing_columns:
		raise ValueError(f"{input_path.name} 缺少列：{', '.join(sorted(missing_columns))}")

	# 空白字符串先转为 NaN，其他无法解析的换手率也保留为缺失值。
	frame["换手率"] = frame["换手率"].replace(r"^\s*$", np.nan, regex=True)
	frame["成交量"] = pd.to_numeric(frame["成交量"], errors="coerce")
	frame["换手率"] = pd.to_numeric(frame["换手率"], errors="coerce")

	zero_volume = frame["成交量"].eq(0)
	frame.loc[zero_volume, "换手率"] = 0

	# 成交量大于 0 且换手率缺失，视为接口缺字段，保留 NaN 并统计。
	normal_trade_missing = frame["成交量"].gt(0) & frame["换手率"].isna()
	missing_count = int(normal_trade_missing.sum())
	zero_volume_count = int(zero_volume.sum())

	output_path = CLEAN_DIR / filename
	frame.to_csv(output_path, index=False, encoding="utf-8-sig")
	print(
		f"{filename}: 行数={len(frame)}，成交量为 0 并填充换手率={zero_volume_count}，"
		f"正常交易日换手率 NaN={missing_count}，输出={output_path}"
	)


def main() -> None:
	# 创建清洗目录；原始文件只读，所有结果只写入 clean_temp。
	CLEAN_DIR.mkdir(parents=True, exist_ok=True)
	for filename in CSV_FILES:
		clean_file(filename)


if __name__ == "__main__":
	main()
