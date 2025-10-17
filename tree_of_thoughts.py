from itertools import chain
from typing import Callable, TypeVar

from src.eff import Operation

_T = TypeVar("T")

get_init: Callable[[], _T] = Operation()
expand = Operation()
score: Callable[[_T, int], tuple[_T, float]] = Operation()


def top_k(scored: list[tuple[_T, float]], n_select: int) -> list[_T]:
	return list(
		map(lambda x: x[0], sorted(scored, key=lambda x: x[1], reverse=True)[:n_select])
	)


def beam_search(n_steps: int, n_select: int, n_eval: int, verbose=False) -> list[_T]:
	frontier = [get_init()]
	for _ in range(n_steps):
		if verbose:
			print(f"[INFO] step: {_}")
		expanded = []
		for state in frontier:
			expanded.append(expand(state))
		cands = list(chain(*expanded))
		if verbose:
			print(f"[INFO] get {len(cands)} candidates")
		scored = []
		for cand in cands:
			scored.append(score(cand, n_eval))
		frontier = top_k(scored, n_select)
		if verbose:
			print(f"[INFO] best candidate: {frontier[:1]}")
	return frontier
