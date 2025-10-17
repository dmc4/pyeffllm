import argparse
import os
import pickle

from dotenv import load_dotenv
from pydantic import BaseModel

from src.coop import AsyncHandler, async_
from src.eff import Handler, coroutine_decorator
from src.llm import (
	AsyncOneRoundChatHandler,
	AsyncReplayLLMHandler,
	LLMHandler,
	OneRoundChatHandler,
	ReplayLLMHandler,
	TraceLLMHandler,
	parse,
)
from tree_of_thoughts import beam_search, expand, get_init, score
from src.util import Timer, awaitable_args_decorator

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "ali/qwen-turbo-latest"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

PROMPT_PROPOSE = """Use numbers and basic arithmetic operations (+ - * /) to construct next steps. You are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""
PROMPT_COT = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
"""
PROMPT_SCORE = """Evaluate if given numbers can reach 24 (sure/likely/impossible).
Input: 10 14
Thoughts:
10 + 14 = 24
Judge:
sure
Input: 11 12
Thoughts:
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
Judge:
impossible
Input: 4 4 10
Thoughts:
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
Judge:
sure
Input: 4 9 11
Thoughts:
9 + 11 + 4 = 20 + 4 = 24
Judge:
sure
Input: 5 7 8
Thoughts:
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
Judge:
likely
Input: 5 6 6
Thoughts:
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
Judge:
likely
Input: 10 10 11
Thoughts:
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
Judge:
impossible
Input: 1 3 3
Thoughts:
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
Judge:
impossible
Input: {input}
"""
PROMPT_SCORE_LAST_STEP = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge:
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge:
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge:
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge:
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge:
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge:
impossible
Input: {input}
Answer: {answer}
Judge:
"""


class Proposal(BaseModel):
	expression: str
	left_numbers: str


class Proposals(BaseModel):
	content: list[Proposal]


class Answer(BaseModel):
	expression: str


class Value(BaseModel):
	thoughts: list[str]
	judge: str


def str_of_state(y):
	return "_".join([_.model_dump_json() for _ in y])


def propose_prompt_wrap(x, y=[]):
	current_numbers = x if len(y) == 0 else y[-1].left_numbers
	if current_numbers == "24":
		prompt = (
			PROMPT_COT.format(input=x)
			+ "Steps:\n"
			+ "\n".join([_.expression + " (left: " + _.left_numbers + ")" for _ in y])
		)
		return prompt, Answer
	else:
		prompt = PROMPT_PROPOSE.format(input=current_numbers)
		return prompt, Proposals


def score_prompt_wrap(x, y):
	if isinstance(y[-1], Answer):
		return PROMPT_SCORE_LAST_STEP.format(input=x, answer=y[-1].expression)
	else:
		return PROMPT_SCORE.format(input=y[-1].left_numbers)


def score_outputs_unwrap(x, y, vs):
	if len(y) == 4 and not isinstance(y[-1], Answer):
		return 0
	value_map = {"impossible": 0.001, "likely": 1, "sure": 20}
	value = 0.0
	for v in vs:
		value += value_map[v.judge] if v.judge in value_map else 0
	return value


class Game24Handler(Handler):
	"""
	handles: get_init, expand, score

	forward: parse
	"""

	def __init__(self, x):
		super().__init__()
		self.x = x
		self.register(get_init, self.get_init)
		self.register(expand, self.expand)
		self.register(score, self.score)

	def get_init(self):
		return []

	def expand(self, y):
		prompt, fmt = propose_prompt_wrap(self.x, y)
		response = (
			parse(f"expand_{str_of_state(y)}", prompt, fmt).choices[0].message.parsed
		)
		if isinstance(response, Answer):
			return [y + [response]]
		else:
			return [y + [_] for _ in response.content]

	def score(self, y, n_eval):
		prompt = score_prompt_wrap(self.x, y)
		responses = []
		for _ in range(n_eval):
			responses.append(
				parse(f"score_{str_of_state(y)}", prompt, Value)
				.choices[0]
				.message.parsed
			)
		return (y, score_outputs_unwrap(self.x, y, responses))


class AsyncGame24Handler(Handler):
	"""
	handles: get_init, expand, score

	forward: parse, async, await
	"""

	def __init__(self, x):
		super().__init__()
		self.x = x
		self.register(get_init, self.get_init)
		self.register(expand, self.expand)
		self.register(score, self.score)

	def get_init(self):
		return []

	def expand(self, y):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(y):
			prompt, schema = propose_prompt_wrap(self.x, y)
			response = (
				(await parse(f"expand_{str_of_state(y)}", prompt, schema))
				.choices[0]
				.message.parsed
			)
			if isinstance(response, Answer):
				return [y + [response]]
			else:
				return [y + [_] for _ in response.content]

		return AsyncHandler.wrap_future_object(async_(aux(y)), "__iter__")

	def score(self, y, n_eval):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(y, n_eval):
			prompt = score_prompt_wrap(self.x, y)
			p = []
			for _ in range(n_eval):
				p.append(parse(f"score_{str_of_state(y)}", prompt, Value))
			responses = [(await p[i]).choices[0].message.parsed for i in range(n_eval)]
			return (y, score_outputs_unwrap(self.x, y, responses))

		return AsyncHandler.wrap_future_object(async_(aux(y, n_eval)), "__getitem__")


def main(verbose=False):
	frontier = beam_search(n_steps=4, n_select=5, n_eval=3, verbose=verbose)
	for state in frontier[:1]:
		if isinstance(state[-1], Answer):
			print(state[-1].expression)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("input", type=str, help='example: "4 9 10 13"')
	args = args.parse_args()

	x = args.input
	fname = f"game24_{'-'.join(x.split())}.trace"

	if not os.path.exists(fname):
		with (
			LLMHandler(base_url=BASE_URL, api_key=API_KEY),
			TraceLLMHandler() as trace,
			OneRoundChatHandler(
				model=MODEL,
				temperature=TEMPERATURE,
				max_tokens=MAX_TOKENS,
			),
			Game24Handler(x),
		):
			main(verbose=True)
		with open(fname, "wb") as f:
			pickle.dump(trace, f)

	with open(fname, "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_sync:
		with ReplayLLMHandler(trace), OneRoundChatHandler(), Game24Handler(x):
			main()
	print(f"sync time: {t_sync.time}")

	with open(fname, "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_async:
		with (
			AsyncHandler(),
			AsyncReplayLLMHandler(trace),
			AsyncOneRoundChatHandler(),
			AsyncGame24Handler(x),
		):
			main()
	print(f"async time: {t_async.time}")

	print(f"speedup: {t_sync.time / t_async.time}x")
