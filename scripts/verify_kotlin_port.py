"""Cross-check the Kotlin solver logic against the OR-Tools implementation.

The Kotlin in android/ cannot be compiled here, so this transliterates it back
into Python -- same bitmasks, same MRV ordering, same combination and cartesian
enumeration -- and runs both against the real puzzle grids in data/wicht_sudoku.

It cannot catch Kotlin syntax errors.  It does catch the errors that actually
worry me in a port of this kind: an off-by-one in the combination generator, a
mask restored wrongly on backtrack, uniqueness answered differently from the
CP-SAT formulation.

Usage:
    uv run python scripts/verify_kotlin_port.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sudoku_solver.sudoku_solver import SudokuSolver as OrToolsSolver

N, CELLS, ALL = 9, 81, 0x3FE


def box_of(i: int) -> int:
    return (i // 27) * 3 + (i % N) // 3


# ── transliteration of SudokuSolver.kt ───────────────────────────────────────

def kt_is_valid(grid: list[int]) -> bool:
    row = [0] * N; col = [0] * N; box = [0] * N
    for i, v in enumerate(grid):
        if v == 0:
            continue
        if v < 1 or v > 9:
            return False
        bit = 1 << v
        r, c, b = i // N, i % N, box_of(i)
        if row[r] & bit or col[c] & bit or box[b] & bit:
            return False
        row[r] |= bit; col[c] |= bit; box[b] |= bit
    return True


class KtSearch:
    def __init__(self, grid: list[int], limit: int):
        self.grid, self.limit = grid, limit
        self.row = [0] * N; self.col = [0] * N; self.box = [0] * N
        self.count = 0
        self.first: list[int] | None = None

    def run(self) -> None:
        for i, v in enumerate(self.grid):
            if v:
                bit = 1 << v
                self.row[i // N] |= bit; self.col[i % N] |= bit; self.box[box_of(i)] |= bit
        self.step()

    def step(self) -> bool:
        best_i, best_mask, best_n = -1, 0, 10
        for i in range(CELLS):
            if self.grid[i]:
                continue
            avail = ALL & ~(self.row[i // N] | self.col[i % N] | self.box[box_of(i)])
            n = bin(avail).count("1")
            if n == 0:
                return False
            if n < best_n:
                best_n, best_mask, best_i = n, avail, i
                if n == 1:
                    break

        if best_i == -1:
            self.count += 1
            if self.first is None:
                self.first = list(self.grid)
            return self.count >= self.limit

        r, c, b = best_i // N, best_i % N, box_of(best_i)
        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            digit = bit.bit_length() - 1
            self.grid[best_i] = digit
            self.row[r] |= bit; self.col[c] |= bit; self.box[b] |= bit
            stop = self.step()
            self.grid[best_i] = 0
            self.row[r] &= ~bit; self.col[c] &= ~bit; self.box[b] &= ~bit
            if stop:
                return True
        return False


def kt_solve_or_null(grid: list[int]) -> list[int] | None:
    if not kt_is_valid(grid):
        return None
    s = KtSearch(list(grid), 1)
    s.run()
    return s.first


def kt_count(grid: list[int], limit: int = 2) -> int:
    if not kt_is_valid(grid):
        return 0
    s = KtSearch(list(grid), limit)
    s.run()
    return s.count


# ── transliteration of ConstraintRecovery.kt helpers ─────────────────────────

def kt_combinations(items: list[int], k: int):
    if k <= 0 or k > len(items):
        return
    idx = list(range(k))
    while True:
        yield [items[j] for j in idx]
        i = k - 1
        while i >= 0 and idx[i] == i + len(items) - k:
            i -= 1
        if i < 0:
            return
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1


def kt_cartesian(lists: list[list[int]]):
    if not lists or any(len(x) == 0 for x in lists):
        return
    pos = [0] * len(lists)
    while True:
        yield [lists[i][pos[i]] for i in range(len(lists))]
        i = len(lists) - 1
        while i >= 0 and pos[i] == len(lists[i]) - 1:
            pos[i] = 0
            i -= 1
        if i < 0:
            return
        pos[i] += 1


# ── checks ───────────────────────────────────────────────────────────────────

def read_dat(path: Path) -> np.ndarray:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()][-9:]
    return np.array([[int(v) for v in r.split()] for r in rows], int)


def check_helpers() -> bool:
    ok = True
    for n in range(0, 8):
        items = list(range(n))
        for k in range(0, 5):
            mine = [tuple(x) for x in kt_combinations(items, k)]
            ref = list(itertools.combinations(items, k)) if 0 < k <= n else []
            if mine != ref:
                print(f"  FAIL combinations(n={n}, k={k}): {mine} != {ref}")
                ok = False
    cases = [[[1, 2], [3], [4, 5, 6]], [[7]], [[1, 2], []], []]
    for lists in cases:
        mine = [tuple(x) for x in kt_cartesian(lists)]
        ref = ([] if not lists or any(not x for x in lists)
               else [tuple(x) for x in itertools.product(*lists)])
        if mine != ref:
            print(f"  FAIL cartesian({lists}): {mine} != {ref}")
            ok = False
    print(f"  combinations + cartesian: {'match itertools' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    print("Helper enumeration")
    ok = check_helpers()

    dats = sorted(ROOT.glob("data/wicht_sudoku/*/*.dat"))
    if not dats:
        raise SystemExit("No .dat puzzles found under data/wicht_sudoku")
    grids = [read_dat(p) for p in dats]
    print(f"\nSolver agreement over {len(grids)} real puzzles")

    ref_solver = OrToolsSolver()
    solved_both = disagree = ref_only = kt_only = 0
    uniq_agree = uniq_disagree = 0
    t_kt = t_or = 0.0

    for g in grids:
        flat = [int(x) for x in g.flatten()]

        t = time.perf_counter()
        mine = kt_solve_or_null(flat)
        t_kt += time.perf_counter() - t

        t = time.perf_counter()
        try:
            ref, _ = ref_solver.solve(g.copy())
            ref = [int(x) for x in ref.flatten()]
        except RuntimeError:
            ref = None
        t_or += time.perf_counter() - t

        if mine is None and ref is None:
            continue
        if mine is None:
            ref_only += 1
            continue
        if ref is None:
            kt_only += 1
            continue

        solved_both += 1
        # Both must be *valid* completions agreeing with the clues; a puzzle with
        # several answers may legitimately produce two different ones.
        if not kt_is_valid(mine) or any(
            c != 0 and c != m for c, m in zip(flat, mine)
        ) or 0 in mine:
            print("  FAIL kotlin produced an invalid completion")
            ok = False
        if mine != ref:
            disagree += 1

        kt_multi = kt_count(flat, 2) >= 2
        or_multi = ref_solver.has_other_solution(
            g.copy(), np.array(ref, dtype=np.uint8).reshape(9, 9)
        )
        if kt_multi == or_multi:
            uniq_agree += 1
        else:
            uniq_disagree += 1
            print(f"  FAIL uniqueness disagreement (kotlin={kt_multi} ortools={or_multi})")
            ok = False

    print(f"  solved by both      : {solved_both}")
    print(f"  only OR-Tools solved: {ref_only}")
    print(f"  only Kotlin solved  : {kt_only}")
    print(f"  different (but both valid) completions: {disagree}")
    print(f"  uniqueness agree/disagree: {uniq_agree}/{uniq_disagree}")
    print(f"\n  time: kotlin-logic {t_kt:.2f}s   OR-Tools {t_or:.2f}s "
          f"({t_or / max(t_kt, 1e-9):.1f}x)")

    if ref_only or kt_only:
        ok = False
    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
