from __future__ import annotations

import json
import logging
import pickle
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import akshare as ak
import baostock as bs
import pandas as pd
import requests

from 股票池 import stock_codes


BASE_DIR = Path(__file__).resolve().parent
START_DATE = "20190101"
END_DATE = "20260331"
DATE_START_ISO = "2019-01-01"
DATE_END_ISO = "2026-03-31"
PROBE_STOCK_CODE = "600000"
ADJUSTMENTS = {
	"不复权": "",
	"前复权": "qfq",
	"后复权": "hfq",
}
OUTPUT_FILES = {
	"不复权": BASE_DIR / f"A股日线_{START_DATE}-{END_DATE}_不复权.csv",
	"前复权": BASE_DIR / f"A股日线_{START_DATE}-{END_DATE}_前复权.csv",
	"后复权": BASE_DIR / f"A股日线_{START_DATE}-{END_DATE}_后复权.csv",
}
CACHE_DIR = BASE_DIR / ".行情缓存"
CHECKPOINT_FILE = CACHE_DIR / "checkpoint.json"
EVENTS_FILE = BASE_DIR / "行情失败与跳过.jsonl"
LOG_FILE = BASE_DIR / "行情运行.log"
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
REQUEST_INTERVAL_SECONDS = 0.15
REQUEST_TIMEOUT_SECONDS = 15
OUTPUT_COLUMNS = [
	"股票代码",
	"日期",
	"开盘",
	"最高",
	"最低",
	"收盘",
	"成交量",
	"成交额",
	"换手率",
]


class NoDataError(Exception):
	def __init__(self, message: str, source: str = "未知"):
		self.source = source
		super().__init__(message)


class SourceFailure(Exception):
	def __init__(self, category: str, message: str):
		self.category = category
		self.message = message
		super().__init__(message)


@dataclass
class Provider:
	name: str
	fetch: Callable[[str, str, str, str], pd.DataFrame]


def setup_logger() -> logging.Logger:
	logger = logging.getLogger("stock_daily")
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		formatter = logging.Formatter(
			"%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
		)
		file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
		file_handler.setFormatter(formatter)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		logger.addHandler(stream_handler)
	return logger


def classify_exception(error: Exception) -> str:
	message = str(error).lower()
	blocked_markers = (
		"429",
		"403",
		"418",
		"too many requests",
		"rate limit",
		"frequency",
		"forbidden",
		"blocked",
		"封禁",
		"限流",
		"请求频繁",
		"访问频繁",
	)
	if any(marker in message for marker in blocked_markers):
		return "blocked"
	if isinstance(error, requests.exceptions.HTTPError):
		status = getattr(getattr(error, "response", None), "status_code", None)
		if status == 408 or (status is not None and status >= 500):
			return "network"
		return "source_error"
	if isinstance(
		error,
		(
			requests.exceptions.Timeout,
			requests.exceptions.ConnectionError,
			requests.exceptions.ChunkedEncodingError,
			requests.exceptions.ContentDecodingError,
			socket.timeout,
			TimeoutError,
			ConnectionError,
			OSError,
		),
	):
		return "network"
	if isinstance(error, requests.exceptions.RequestException) and any(
		marker in message
		for marker in (
			"remote end closed",
			"connection aborted",
			"connection reset",
			"read timed out",
			"max retries exceeded",
			"chunkedencodingerror",
			"protocolerror",
		)
	):
		return "network"
	return "source_error"


def normalize_frame(
	frame: pd.DataFrame,
	stock_code: str,
	source: str,
) -> pd.DataFrame:
	if frame is None or frame.empty:
		raise NoDataError(f"{source} 返回空数据", source)

	frame = frame.rename(
		columns={
			"date": "日期",
			"trade_date": "日期",
			"open": "开盘",
			"high": "最高",
			"low": "最低",
			"close": "收盘",
			"vol": "成交量",
			"volume": "成交量",
			"amount": "成交额",
			"turn": "换手率",
			"turnover": "换手率",
			"turnover_rate": "换手率",
		}
	).copy()
	missing = [column for column in OUTPUT_COLUMNS[1:] if column not in frame.columns]
	if missing:
		raise ValueError(f"{source} 缺少字段：{', '.join(missing)}")

	frame = frame[OUTPUT_COLUMNS[1:]]
	frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
	for column in OUTPUT_COLUMNS[2:]:
		frame[column] = pd.to_numeric(frame[column], errors="coerce")
	frame = frame.dropna(subset=["日期", "开盘", "最高", "最低", "收盘"])
	start = pd.Timestamp(DATE_START_ISO)
	end = pd.Timestamp(DATE_END_ISO)
	frame = frame.loc[frame["日期"].between(start, end)].copy()
	if frame.empty:
		raise NoDataError(f"{source} 在目标日期范围没有数据", source)

	frame["日期"] = frame["日期"].dt.strftime("%Y-%m-%d")
	frame["股票代码"] = stock_code
	return frame[OUTPUT_COLUMNS].drop_duplicates(["股票代码", "日期"])


def fetch_akshare(
	stock_code: str,
	adjustment: str,
	start_date: str = DATE_START_ISO,
	end_date: str = DATE_END_ISO,
) -> pd.DataFrame:
	frame = ak.stock_zh_a_hist(
		symbol=stock_code,
		period="daily",
		start_date=start_date.replace("-", ""),
		end_date=end_date.replace("-", ""),
		adjust=adjustment,
		timeout=REQUEST_TIMEOUT_SECONDS,
	)
	return normalize_frame(frame, stock_code, "AkShare-东方财富")


_BAOSTOCK_LOGGED_IN = False


def ensure_baostock_login() -> None:
	global _BAOSTOCK_LOGGED_IN
	if _BAOSTOCK_LOGGED_IN:
		return
	login = bs.login()
	if login.error_code != "0":
		raise RuntimeError(f"Baostock 登录失败：{login.error_msg}")
	_BAOSTOCK_LOGGED_IN = True


def fetch_baostock(
	stock_code: str,
	adjustment: str,
	start_date: str = DATE_START_ISO,
	end_date: str = DATE_END_ISO,
) -> pd.DataFrame:
	ensure_baostock_login()
	market = "sh" if stock_code.startswith("6") else "sz"
	adjust_flag = {"": "3", "qfq": "1", "hfq": "2"}[adjustment]
	result = bs.query_history_k_data_plus(
		f"{market}.{stock_code}",
		"date,open,high,low,close,volume,amount,turn",
		start_date=start_date,
		end_date=end_date,
		frequency="d",
		adjustflag=adjust_flag,
	)
	if result.error_code != "0":
		raise RuntimeError(f"Baostock {result.error_code}: {result.error_msg}")

	rows = []
	while result.next():
		rows.append(result.get_row_data())
	if not rows:
		return pd.DataFrame()
	frame = pd.DataFrame(rows, columns=result.fields)
	frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") / 100
	return normalize_frame(frame, stock_code, "Baostock")


def fetch_tencent(
	stock_code: str,
	adjustment: str,
	start_date: str = DATE_START_ISO,
	end_date: str = DATE_END_ISO,
) -> pd.DataFrame:
	market = "sh" if stock_code.startswith("6") else "sz"
	frame = ak.stock_zh_a_hist_tx(
		symbol=f"{market}{stock_code}",
		start_date=start_date.replace("-", ""),
		end_date=end_date.replace("-", ""),
		adjust=adjustment,
		timeout=REQUEST_TIMEOUT_SECONDS,
	)
	if frame is not None and not frame.empty:
		frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce") * 100
	return normalize_frame(frame, stock_code, "腾讯")


def make_providers() -> list[Provider]:
	return [
		Provider("AkShare-东方财富", fetch_akshare),
		Provider("Baostock", fetch_baostock),
		Provider("腾讯", fetch_tencent),
	]


def fetch_with_retry(
	provider: Provider,
	stock_code: str,
	adjustment: str,
	logger: logging.Logger,
	start_date: str = DATE_START_ISO,
	end_date: str = DATE_END_ISO,
) -> pd.DataFrame:
	for attempt in range(1, MAX_ATTEMPTS + 1):
		try:
			frame = provider.fetch(stock_code, adjustment, start_date, end_date)
			if frame is None or frame.empty:
				raise NoDataError(f"{provider.name} 返回空数据", provider.name)
			return frame
		except NoDataError:
			raise
		except Exception as error:
			category = classify_exception(error)
			if category == "network" and attempt < MAX_ATTEMPTS:
				delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), 8)
				logger.warning(
					"网络抖动：源=%s 股票=%s 复权=%s 第%d/%d次失败，%.1f秒后重试：%s",
					provider.name,
					stock_code,
				adjustment or "none",
				attempt,
				MAX_ATTEMPTS,
				delay,
				error,
			)
				time.sleep(delay)
				continue
			if category == "blocked":
				logger.error(
					"限流/封禁：源=%s 股票=%s 复权=%s，立即切换后备源：%s",
					provider.name,
				stock_code,
				adjustment or "none",
				error,
			)
			elif category == "network":
				logger.error(
					"网络重试达到上限：源=%s 股票=%s 复权=%s：%s",
					provider.name,
				stock_code,
				adjustment or "none",
				error,
			)
			else:
				logger.error(
					"数据源异常：源=%s 股票=%s 复权=%s：%s",
					provider.name,
				stock_code,
				adjustment or "none",
				error,
			)
			raise SourceFailure(category, str(error)) from error
		time.sleep(REQUEST_INTERVAL_SECONDS)
	raise SourceFailure("network", "重试次数耗尽")


def probe_provider(
	providers: list[Provider],
	stock_code: str,
	logger: logging.Logger,
) -> int:
	for index, provider in enumerate(providers):
		logger.info("探测行情源：优先级=%d 源=%s", index + 1, provider.name)
		try:
			fetch_with_retry(
				provider,
				stock_code,
				"",
				logger,
				start_date="2019-01-04",
				end_date="2019-01-04",
			)
			time.sleep(REQUEST_INTERVAL_SECONDS)
			logger.info("已选定行情源：%s；本次运行保持固定，除故障外不切回", provider.name)
			return index
		except NoDataError as error:
			logger.warning("探测源返回空数据，尝试下一源：%s：%s", provider.name, error)
		except SourceFailure as error:
			logger.warning("探测源不可用，尝试下一源：%s：%s", provider.name, error.message)
	logger.error("启动探测未找到可用线上源；仅使用本地缓存，缺失数据记为失败")
	return len(providers)


def cache_path(stock_code: str, adjustment: str) -> Path:
	adj_name = adjustment or "none"
	return CACHE_DIR / f"{stock_code}_{adj_name}_{START_DATE}_{END_DATE}.pkl"


def read_cache(stock_code: str, adjustment: str) -> pd.DataFrame | None:
	path = cache_path(stock_code, adjustment)
	if not path.exists():
		return None
	try:
		with path.open("rb") as file:
			frame = pickle.load(file)
		if not isinstance(frame, pd.DataFrame) or not set(OUTPUT_COLUMNS).issubset(
			frame.columns
		):
			return None
		return frame[OUTPUT_COLUMNS]
	except Exception:
		return None


def write_cache(stock_code: str, adjustment: str, frame: pd.DataFrame) -> None:
	path = cache_path(stock_code, adjustment)
	temporary_path = path.with_suffix(".tmp")
	with temporary_path.open("wb") as file:
		pickle.dump(frame, file, protocol=pickle.HIGHEST_PROTOCOL)
	temporary_path.replace(path)


def write_event(
	status: str,
	stock_code: str,
	adjustment: str,
	reason: str,
	source: str,
) -> None:
	event = {
		"状态": status,
		"代码": stock_code,
		"复权": adjustment,
		"原因": reason,
		"数据源": source,
		"时间": datetime.now().isoformat(timespec="seconds"),
	}
	with EVENTS_FILE.open("a", encoding="utf-8") as file:
		file.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_checkpoint(checkpoint: dict, stock_code: str, status: str) -> None:
	checkpoint[stock_code] = {
		"状态": status,
		"更新时间": datetime.now().isoformat(timespec="seconds"),
	}
	temporary_path = CHECKPOINT_FILE.with_suffix(".tmp")
	with temporary_path.open("w", encoding="utf-8") as file:
		json.dump(checkpoint, file, ensure_ascii=False, indent=2)
	temporary_path.replace(CHECKPOINT_FILE)


def load_checkpoint() -> dict:
	try:
		with CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
			return json.load(file)
	except (OSError, json.JSONDecodeError):
		return {}


def fetch_with_fallback(
	stock_code: str,
	adjustment: str,
	providers: list[Provider],
	start_index: int,
	logger: logging.Logger,
) -> tuple[pd.DataFrame, int, str]:
	for index in range(start_index, len(providers)):
		provider = providers[index]
		try:
			frame = fetch_with_retry(provider, stock_code, adjustment, logger)
			if index != start_index:
				logger.warning("行情源已单向切换为：%s", provider.name)
			time.sleep(REQUEST_INTERVAL_SECONDS)
			return frame, index, provider.name
		except NoDataError:
			raise
		except SourceFailure as error:
			if index + 1 < len(providers):
				logger.warning(
					"切换行情源：%s -> %s（原因：%s）",
					provider.name,
				providers[index + 1].name,
				"限流/封禁" if error.category == "blocked" else error.message,
				)
			continue
			raise
	raise SourceFailure("source_error", "没有可用线上数据源")


def main() -> None:
	logger = setup_logger()
	socket.setdefaulttimeout(REQUEST_TIMEOUT_SECONDS)
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	checkpoint = load_checkpoint()
	providers = make_providers()
	if not providers:
		logger.error("没有已配置的数据源")
		return

	all_cache_complete = all(
		read_cache(stock_code, adjustment) is not None
		for stock_code in stock_codes
		for adjustment in ADJUSTMENTS.values()
	)
	if all_cache_complete:
		active_source_index = len(providers)
		logger.info("所有股票的三种复权缓存均存在，跳过线上探测")
	else:
		active_source_index = probe_provider(providers, PROBE_STOCK_CODE, logger)

	collected = {name: [] for name in ADJUSTMENTS}
	success_count = 0
	failure_count = 0
	skipped_count = 0
	baostock_was_used = False

	try:
		for stock_index, stock_code in enumerate(stock_codes, start=1):
			stock_failed = False
			stock_skipped = False
			for adjustment_name, adjustment_value in ADJUSTMENTS.items():
				cached = read_cache(stock_code, adjustment_value)
				if cached is not None:
					collected[adjustment_name].append(cached)
					logger.info(
						"缓存命中，未请求网络：代码=%s 复权=%s",
						stock_code,
						adjustment_name,
					)
					continue

				try:
					frame, active_source_index, source_name = fetch_with_fallback(
						stock_code,
						adjustment_value,
						providers,
						active_source_index,
						logger,
					)
				except NoDataError as error:
					stock_skipped = True
					write_event(
						"跳过",
						stock_code,
						adjustment_name,
						str(error),
						error.source,
					)
					logger.warning(
						"无数据，按要求跳过：代码=%s 复权=%s 原因=%s",
						stock_code,
						adjustment_name,
						error,
					)
					continue
				except SourceFailure as error:
					cached = read_cache(stock_code, adjustment_value)
					if cached is not None:
						frame = cached
						collected[adjustment_name].append(frame)
						logger.warning(
							"线上源全失败，使用本地缓存兜底：代码=%s 复权=%s",
							stock_code,
							adjustment_name,
						)
						continue
					stock_failed = True
					write_event(
						"失败",
						stock_code,
						adjustment_name,
						error.message,
						"所有线上源",
					)
					logger.error(
						"所有线上源失败：代码=%s 复权=%s 原因=%s",
						stock_code,
						adjustment_name,
						error.message,
					)
					continue

				collected[adjustment_name].append(frame)
				write_cache(stock_code, adjustment_value, frame)
				if source_name == "Baostock":
					baostock_was_used = True
				save_checkpoint(checkpoint, stock_code, "处理中")

			if stock_failed:
				failure_count += 1
				final_status = "失败"
			elif stock_skipped:
				skipped_count += 1
				final_status = "跳过"
			else:
				success_count += 1
				final_status = "成功"
			save_checkpoint(checkpoint, stock_code, final_status)
			if stock_index % 10 == 0 or stock_index == len(stock_codes):
				logger.info(
					"批量进度：%d/%d 成功=%d 失败=%d 跳过=%d",
					stock_index,
				len(stock_codes),
				success_count,
				failure_count,
				skipped_count,
				)
	finally:
		if _BAOSTOCK_LOGGED_IN:
			bs.logout()

	for adjustment_name, output_path in OUTPUT_FILES.items():
		frames = collected[adjustment_name]
		if frames:
			output = pd.concat(frames, ignore_index=True)
			output = output[OUTPUT_COLUMNS].sort_values(["股票代码", "日期"])
		else:
			output = pd.DataFrame(columns=OUTPUT_COLUMNS)
		output.to_csv(output_path, index=False, encoding="utf-8-sig")
		logger.info("CSV 已写出：%s（%d 行）", output_path.name, len(output))

	logger.info(
		"任务汇总：股票总数=%d 成功=%d 失败=%d 跳过=%d；复权模式=%d；备用源Baostock=%s",
		len(stock_codes),
		success_count,
		failure_count,
		skipped_count,
		len(ADJUSTMENTS),
		"已使用" if baostock_was_used else "未使用",
	)
	print(
		f"抓取完成：总数={len(stock_codes)}，成功={success_count}，"
		f"失败={failure_count}，跳过={skipped_count}。"
	)
	print("结果文件：")
	for path in OUTPUT_FILES.values():
		print(f"- {path.name}")
	print(f"日志：{LOG_FILE.name}；失败/跳过明细：{EVENTS_FILE.name}")


if __name__ == "__main__":
	main()
