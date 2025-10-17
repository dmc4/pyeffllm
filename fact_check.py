import argparse
import asyncio
import os
import pickle
import time

import aiohttp
import requests
from dotenv import load_dotenv
from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from src.coop import AsyncHandler, async_, await_
from src.eff import Handler, Operation, coroutine_decorator
from src.llm import (
	AsyncChatHandler,
	AsyncReplayLLMHandler,
	ChatHandler,
	LLMHandler,
	ReplayLLMHandler,
	TraceLLMHandler,
	addmsg,
	complete,
)
from src.util import Timer, awaitable_args_decorator

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "yunwu/gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.3
MAX_TOKENS = 4000

WIKI_API_URL = "http://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = (
	"Mozilla/5.0 (Linux; Linux i561 x86_64; en-US) Gecko/20100101 Firefox/70.0"
)


def wiki_search_params(query):
	return {
		"format": "json",
		"action": "query",
		"list": "search",
		"srsearch": query,
		"srlimit": 1,
	}


def wiki_search_process(raw_result):
	search_results = [
		(_["title"], str(_["pageid"])) for _ in raw_result["query"]["search"]
	]
	return search_results


def wiki_summary_params(title):
	return {
		"format": "json",
		"action": "query",
		"prop": "extracts",
		"explaintext": "",
		"titles": title,
	}


def wiki_summary_process(pageid, raw_result):
	summary = raw_result["query"]["pages"][pageid]["extract"]
	return summary


wiki_search = Operation()
wiki_summary = Operation()


class WikiHandler(Handler):
	def __init__(self):
		super().__init__()
		self.register(wiki_search, self.wiki_search)
		self.register(wiki_summary, self.wiki_summary)

	def wiki_search(self, query):
		raw_result = requests.get(
			WIKI_API_URL,
			params=wiki_search_params(query),
			headers={"User-Agent": WIKI_USER_AGENT},
		).json()
		return wiki_search_process(raw_result)

	def wiki_summary(self, title, pageid):
		raw_result = requests.get(
			WIKI_API_URL,
			params=wiki_summary_params(title),
			headers={"User-Agent": WIKI_USER_AGENT},
		).json()
		return wiki_summary_process(pageid, raw_result)


class AsyncWikiHandler(Handler):
	def __init__(self):
		super().__init__()
		self.register(wiki_search, self.wiki_search)
		self.register(wiki_summary, self.wiki_summary)

	def wiki_search(self, query):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(query):
			async with aiohttp.ClientSession(
				trust_env=True, headers={"User-Agent": WIKI_USER_AGENT}
			) as session:
				async with session.get(
					WIKI_API_URL, params=wiki_search_params(query)
				) as resp:
					raw_result = await resp.json()
					return wiki_search_process(raw_result)

		return AsyncHandler.wrap_future_object(
			async_(aux(query)), "__len__", "__getitem__"
		)

	def wiki_summary(self, title, pageid):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(title, pageid):
			async with aiohttp.ClientSession(
				trust_env=True, headers={"User-Agent": WIKI_USER_AGENT}
			) as session:
				async with session.get(
					WIKI_API_URL, params=wiki_summary_params(title)
				) as resp:
					raw_result = await resp.json()
					return wiki_summary_process(pageid, raw_result)

		return AsyncHandler.wrap_future_object(async_(aux(title, pageid)), "__str__")


class TraceWikiHandler(Handler):
	def __init__(self, trace):
		super().__init__()
		self.trace = trace
		self.register(wiki_search, self.wiki_search)
		self.register(wiki_summary, self.wiki_summary)

	def wiki_search(self, query):
		start_time = time.perf_counter()
		response = wiki_search(query)
		response_time = time.perf_counter() - start_time
		self.trace[f"wiki_search_{query}"].append((response, response_time))
		return response

	def wiki_summary(self, title, pageid):
		start_time = time.perf_counter()
		response = wiki_summary(title, pageid)
		response_time = time.perf_counter() - start_time
		self.trace[f"wiki_summary_{title}_{pageid}"].append((response, response_time))
		return response


class ReplayWikiHandler(Handler):
	def __init__(self, trace):
		super().__init__()
		self.trace = trace
		self.register(wiki_search, self.wiki_search)
		self.register(wiki_summary, self.wiki_summary)

	def wiki_search(self, query):
		response, response_time = self.trace[f"wiki_search_{query}"].popleft()
		time.sleep(response_time)
		return response

	def wiki_summary(self, title, pageid):
		response, response_time = self.trace[f"wiki_summary_{title}_{pageid}"].popleft()
		time.sleep(response_time)
		return response


class AsyncReplayWikiHandler(Handler):
	def __init__(self, trace):
		super().__init__()
		self.trace = trace
		self.register(wiki_search, self.wiki_search)
		self.register(wiki_summary, self.wiki_summary)

	def wiki_search(self, query):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(query):
			response, response_time = self.trace[f"wiki_search_{query}"].popleft()
			await asyncio.sleep(response_time)
			return response

		return AsyncHandler.wrap_future_object(
			async_(aux(query)), "__len__", "__getitem__"
		)

	def wiki_summary(self, title, pageid):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(title, pageid):
			response, response_time = self.trace[
				f"wiki_summary_{title}_{pageid}"
			].popleft()
			await asyncio.sleep(response_time)
			return response

		return AsyncHandler.wrap_future_object(async_(aux(title, pageid)), "__str__")


check_fact = Operation()


class GetTopicInfoFromWikipedia(BaseModel):
	topic: str = Field(
		...,
		description="A topic or Wikipedia entry, e.g., Topology, Mercury (planet), London, Vincent van Gogh, etc.",
	)


class FactCheckHandler(Handler):
	def __init__(self):
		super().__init__()
		self.register(check_fact, self.check_fact)

	def check_fact(self, fact, tool, idx=0):
		return (
			complete(
				f"fact_{idx}",
				f"Is it true that {fact}?",
				dict(tools=[tool]),
			)
			.choices[0]
			.message
		)


class AsyncFactCheckHandler(Handler):
	def __init__(self):
		super().__init__()
		self.register(check_fact, self.check_fact)

	def check_fact(self, fact, tool, idx=0):
		return (
			await_(
				complete(
					f"fact_{idx}",
					f"Is it true that {fact}?",
					dict(tools=[tool]),
				)
			)
			.choices[0]
			.message
		)


def main(fact, verbose=False):
	message = check_fact(fact, pydantic_function_tool(GetTopicInfoFromWikipedia))
	idx = 0
	while message.tool_calls is not None:
		if verbose:
			print(f"[INFO] number of tool calls: {len(message.tool_calls)}")
		search_results = []
		for tool_call in message.tool_calls:
			tool_call_id = tool_call.id
			topic = GetTopicInfoFromWikipedia.model_validate_json(
				tool_call.function.arguments
			).topic
			search_results.append((tool_call.id, wiki_search(topic)))
		summaries = []
		for tool_call_id, search_result in search_results:
			if len(search_result) > 0:
				title, pageid = search_result[0]
				summaries.append((tool_call_id, wiki_summary(title, pageid)))
			else:
				summaries.append((tool_call_id, "Nothing found."))
		for tool_call_id, summary in summaries:
			addmsg("tool", str(summary), dict(tool_call_id=tool_call_id))
		idx += 1
		message = check_fact(
			fact, pydantic_function_tool(GetTopicInfoFromWikipedia), idx
		)
	print(message.content)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument(
		"input", type=str, help='example: "Andrew Yao won Turing Award in 2000"'
	)
	args = args.parse_args()

	fact = args.input
	fname = f"fact_check_{'-'.join(fact.split())}.trace"

	if not os.path.exists(fname):
		with (
			LLMHandler(base_url=BASE_URL, api_key=API_KEY),
			TraceLLMHandler() as trace,
			ChatHandler(
				model=MODEL,
				temperature=TEMPERATURE,
				max_tokens=MAX_TOKENS,
			),
			WikiHandler(),
			TraceWikiHandler(trace),
			FactCheckHandler(),
		):
			main(fact, verbose=True)
		with open(fname, "wb") as f:
			pickle.dump(trace, f)

	with open(fname, "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_sync:
		with (
			ReplayLLMHandler(trace),
			ChatHandler(),
			ReplayWikiHandler(trace),
			FactCheckHandler(),
		):
			main(fact)
	print(f"sync time: {t_sync.time}")

	with open(fname, "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_async:
		with (
			AsyncHandler(),
			AsyncReplayLLMHandler(trace),
			AsyncChatHandler(),
			AsyncReplayWikiHandler(trace),
			AsyncFactCheckHandler(),
		):
			main(fact)
	print(f"async time: {t_async.time}")

	print(f"speedup: {t_sync.time / t_async.time}x")
