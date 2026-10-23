from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "clean_temp"
CSV_FILES = [
	"A股日线_20190101-20260331_不复权.csv",
	"A股日线_20190101-20260331_前复权.csv",
	"A股日线_20190101-20260331_后复权.csv",
]
DATE_COLUMN_CANDIDATES = ("交易日期", "日期", "date", "Date")


def find_date_column(frame: pd.DataFrame, filename: str) -> str:
	for column in DATE_COLUMN_CANDIDATES:
		if column in frame.columns:
			return column
	raise ValueError(
		f"{filename} 中找不到日期列，支持列名：{', '.join(DATE_COLUMN_CANDIDATES)}"
	)


def deduplicate_and_sort(filename: str) -> None:
	input_path = CLEAN_DIR / filename
	if not input_path.is_file():
		raise FileNotFoundError(f"找不到输入文件：{input_path}")

	frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype={"股票代码": "string"})
	if "股票代码" not in frame.columns:
		raise ValueError(f"{filename} 中找不到股票代码列")
	date_column = find_date_column(frame, filename)
	original_rows = len(frame)

	# 统一日期类型，并把纯空白日期视为缺失值。
	frame[date_column] = frame[date_column].replace(r"^\s*$", np.nan, regex=True)
	frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")

	# 字符串读取可保留前导零；清理可能出现的 .0 后缀并补足六位。
	codes = frame["股票代码"].astype("string").str.strip()
	codes = codes.str.replace(r"\.0+$", "", regex=True)
	frame["股票代码"] = codes.str.zfill(6)

	# 联合主键重复时保留文件中最后出现的一条，再按代码和日期升序排列。
	frame = frame.drop_duplicates(
		subset=["股票代码", date_column], keep="last"
	)
	frame = frame.sort_values(
		by=["股票代码", date_column],
		ascending=[True, True],
		kind="mergesort",
		na_position="last",
	)
	removed_rows = original_rows - len(frame)

	output_path = CLEAN_DIR / f"{input_path.stem}_去重排序.csv"
	frame.to_csv(output_path, index=False, encoding="utf-8-sig")
	print(
		f"{filename}: 原始行数={original_rows}，去重删除={removed_rows}，"
		f"保留行数={len(frame)}，输出={output_path.name}"
	)


def main() -> None:
	for filename in CSV_FILES:
		deduplicate_and_sort(filename)


if __name__ == "__main__":
	main()
