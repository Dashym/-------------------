from pathlib import Path

import akshare as ak
import baostock as bs
import pandas as pd


INDUSTRY_NAME = "汽车零部件"
OUTPUT_FILE = Path(__file__).with_name("汽车零部件_申万二级成分股.csv")
BEFORE_CUTOFF_FILE = Path(__file__).with_name(
	"汽车零部件_剔除ST_上市日期截至2024-12-31.csv"
)
AFTER_CUTOFF_FILE = Path(__file__).with_name(
	"汽车零部件_剔除ST_上市日期2025-01-01及以后.csv"
)
ST_PERIOD_START = "2019-01-01"
ST_PERIOD_END = "2024-12-31"


def normalize_stock_codes(codes: pd.Series) -> pd.Series:
	return codes.astype(str).str.extract(r"(\d{6})", expand=False)


def had_st_during_period(stock_code: str) -> bool:
	market = "sh" if stock_code.startswith("6") else "sz"
	history = bs.query_history_k_data_plus(
		f"{market}.{stock_code}",
		"date,isST",
		start_date=ST_PERIOD_START,
		end_date=ST_PERIOD_END,
		frequency="d",
		adjustflag="3",
	)
	if history.error_code != "0":
		raise RuntimeError(
			f"查询 {stock_code} 历史 ST 状态失败：{history.error_msg}"
		)

	row_count = 0
	had_st = False
	while history.next():
		row_count += 1
		status = history.get_row_data()[1]
		if status not in {"0", "1"}:
			raise RuntimeError(f"{stock_code} 的历史 ST 状态数据异常：{status!r}")
		had_st = had_st or status == "1"
	if row_count == 0:
		raise RuntimeError(f"{stock_code} 在目标期间没有历史数据，无法判断 ST 状态")
	return had_st


def main() -> None:
	industries = ak.sw_index_second_info()
	matched_industries = industries.loc[
		industries["行业名称"].astype(str).str.strip() == INDUSTRY_NAME
	]
	if matched_industries.empty:
		raise RuntimeError(f"申万二级行业列表中未找到：{INDUSTRY_NAME}")

	industry_code = str(matched_industries.iloc[0]["行业代码"]).split(".")[0]
	constituents = ak.index_component_sw(symbol=industry_code)
	members = constituents[["证券代码", "证券名称"]].copy()
	members.columns = ["股票代码", "股票名称"]
	members["股票代码"] = normalize_stock_codes(members["股票代码"])
	members = members.dropna(subset=["股票代码"]).drop_duplicates("股票代码")

	listing_sources = [
		(ak.stock_info_sh_name_code(symbol="主板A股"), "证券代码", "上市日期"),
		(ak.stock_info_sh_name_code(symbol="科创板"), "证券代码", "上市日期"),
		(ak.stock_info_sz_name_code(symbol="A股列表"), "A股代码", "A股上市日期"),
		(ak.stock_info_bj_name_code(), "证券代码", "上市日期"),
	]
	listing_frames = []
	for frame, code_column, date_column in listing_sources:
		listing = frame[[code_column, date_column]].copy()
		listing.columns = ["股票代码", "listing_date"]
		listing["股票代码"] = normalize_stock_codes(listing["股票代码"])
		listing_frames.append(listing)

	listings = pd.concat(listing_frames, ignore_index=True)
	listings = listings.dropna(subset=["股票代码"]).drop_duplicates(
		"股票代码", keep="last"
	)
	listings["listing_date"] = pd.to_datetime(
		listings["listing_date"], errors="coerce"
	).dt.strftime("%Y-%m-%d")

	result = members.merge(listings, on="股票代码", how="left", validate="one_to_one")
	result = result[["股票代码", "股票名称", "listing_date"]].sort_values("股票代码")
	cutoff_date = pd.Timestamp("2024-12-31")
	listing_dates = pd.to_datetime(result["listing_date"], errors="coerce")
	if listing_dates.isna().any():
		raise RuntimeError("存在缺失的上市日期，无法正确拆分 CSV")

	historical_candidates = result.loc[listing_dates <= cutoff_date]
	login = bs.login()
	if login.error_code != "0":
		raise RuntimeError(f"Baostock 登录失败：{login.error_msg}")

	st_codes = set()
	try:
		for index, stock_code in enumerate(historical_candidates["股票代码"], start=1):
			if had_st_during_period(stock_code):
				st_codes.add(stock_code)
			if index % 20 == 0 or index == len(historical_candidates):
				print(f"历史 ST 状态检查进度：{index}/{len(historical_candidates)}")
	finally:
		bs.logout()

	current_st_mask = result["股票名称"].astype(str).str.match(
		r"^\s*(?:\*?ST|S\*ST)", case=False, na=False
	)
	current_st_codes = set(result.loc[current_st_mask, "股票代码"])
	excluded_codes = st_codes | current_st_codes
	result = result.loc[~result["股票代码"].isin(excluded_codes)].copy()
	listing_dates = pd.to_datetime(result["listing_date"])
	before_cutoff = result.loc[listing_dates <= cutoff_date]
	after_cutoff = result.loc[listing_dates > cutoff_date]

	result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
	before_cutoff.to_csv(BEFORE_CUTOFF_FILE, index=False, encoding="utf-8-sig")
	after_cutoff.to_csv(AFTER_CUTOFF_FILE, index=False, encoding="utf-8-sig")

	missing_dates = int(result["listing_date"].isna().sum())
	print(
		f"已导出 2019-2024 年期间未被 ST 的全量表 {len(result)} 家"
		f"（历史 ST 排除 {len(st_codes)} 家，当前 ST 排除 {len(current_st_codes)} 家）："
		f"{OUTPUT_FILE}"
	)
	print(f"截至 2024-12-31：{len(before_cutoff)} 家：{BEFORE_CUTOFF_FILE}")
	print(f"2025-01-01 及以后：{len(after_cutoff)} 家：{AFTER_CUTOFF_FILE}")
	if missing_dates:
		print(f"警告：{missing_dates} 家企业未能匹配到上市日期，请检查数据源。")


if __name__ == "__main__":
	main()


stock_codes = [
	"000030", "000338", "000559", "000570", "000581", "000589", "000599", "000678", "000700", "000887",
	"000901", "001260", "001278", "001282", "001311", "001319", "001380", "002048", "002085", "002101",
	"002126", "002239", "002265", "002283", "002284", "002328", "002355", "002363", "002406", "002434",
	"002448", "002454", "002472", "002488", "002510", "002516", "002536", "002537", "002553", "002590",
	"002662", "002664", "002703", "002708", "002715", "002725", "002765", "002806", "002813", "002863",
	"002870", "002906", "002921", "002937", "002965", "002976", "002984", "002997", "300100", "300176",
	"300258", "300304", "300375", "300428", "300473", "300507", "300547", "300580", "300585", "300611",
	"300643", "300652", "300680", "300681", "300694", "300695", "300707", "300733", "300745", "300816",
	"300863", "300893", "300912", "300926", "300928", "300969", "300978", "300998", "301000", "301005",
	"301007", "301020", "301022", "301072", "301119", "301133", "301170", "301181", "301186", "301192",
	"301221", "301225", "301229", "301233", "301298", "301310", "301397", "301398", "301488", "301499",
	"301529", "301539", "301550", "301552", "301607", "301613", "301631", "600081", "600148", "600151",
	"600178", "600218", "600469", "600480", "600523", "600609", "600626", "600660", "600699", "600741",
	"600742", "600841", "600933", "600960", "601058", "601163", "601279", "601500", "601689", "601799",
	"601966", "603006", "603009", "603013", "603035", "603037", "603040", "603048", "603085", "603089",
	"603107", "603109", "603119", "603121", "603158", "603161", "603166", "603178", "603179", "603190",
	"603197", "603201", "603211", "603239", "603266", "603286", "603305", "603306", "603319", "603335",
	"603348", "603358", "603390", "603586", "603596", "603655", "603701", "603730", "603758", "603767",
	"603768", "603786", "603788", "603809", "603917", "603926", "603949", "603950", "603960", "603982",
	"603997", "605005", "605018", "605068", "605088", "605128", "605133", "605151", "605228", "605255",
	"605319", "605333", "688021", "688280", "688533", "688612", "688667", "688737",
]