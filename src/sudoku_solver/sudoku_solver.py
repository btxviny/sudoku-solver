"""Solve sudoku puzzles with Google OR-Tools CP-SAT."""

from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model


class SudokuSolver:
    """Solve 9×9 sudoku grids using constraint programming."""

    def solve(self, grid: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve a 9×9 sudoku grid.

        Args:
            grid: 9×9 array with values 0–9 (0 = empty).

        Returns:
            ``(solved_grid, solve_time_seconds)``

        Raises:
            ValueError: If the grid is not 9×9.
            RuntimeError: If clues conflict or no solution exists.
        """
        if grid.shape != (9, 9):
            raise ValueError(f"Expected 9x9 grid, got {grid.shape}")
        if not self.is_valid(grid):
            raise RuntimeError(
                "Puzzle clues conflict (duplicate in a row, column, or box)."
            )

        model = cp_model.CpModel()
        cells: dict[tuple[int, int], cp_model.IntVar | int] = {}

        for i in range(9):
            for j in range(9):
                value = int(grid[i, j])
                if value != 0:
                    cells[i, j] = value
                else:
                    cells[i, j] = model.NewIntVar(1, 9, f"x[{i},{j}]")

        for i in range(9):
            model.AddAllDifferent([cells[i, j] for j in range(9)])
        for j in range(9):
            model.AddAllDifferent([cells[i, j] for i in range(9)])
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                model.AddAllDifferent(
                    [cells[r + i, c + j] for i in range(3) for j in range(3)]
                )

        solver = cp_model.CpSolver()
        t0 = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - t0

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                "No valid solution found. The puzzle may be unsolvable."
            )

        result = np.zeros((9, 9), dtype=np.uint8)
        for i in range(9):
            for j in range(9):
                cell = cells[i, j]
                result[i, j] = int(cell if isinstance(cell, int) else solver.Value(cell))
        return result, elapsed

    def has_other_solution(self, grid: np.ndarray, solution: np.ndarray) -> bool:
        """True if `grid` admits a solution other than `solution`.

        A sudoku read from a photo is only genuinely "solved" if its clues
        determine one answer.  When digit recognition misses enough clues the
        remaining puzzle is under-determined, and the solver returns one of many
        valid completions — a confident wrong answer.  This re-solves with the
        first solution forbidden: if a second exists, the reading was too
        incomplete to trust.
        """
        model = cp_model.CpModel()
        cells: dict[tuple[int, int], cp_model.IntVar | int] = {}
        free: list[tuple[cp_model.IntVar, int]] = []

        for i in range(9):
            for j in range(9):
                value = int(grid[i, j])
                if value != 0:
                    cells[i, j] = value
                else:
                    var = model.NewIntVar(1, 9, f"x[{i},{j}]")
                    cells[i, j] = var
                    free.append((var, int(solution[i, j])))

        if not free:
            return False

        for i in range(9):
            model.AddAllDifferent([cells[i, j] for j in range(9)])
        for j in range(9):
            model.AddAllDifferent([cells[i, j] for i in range(9)])
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                model.AddAllDifferent(
                    [cells[r + i, c + j] for i in range(3) for j in range(3)]
                )

        # Forbid the known solution: at least one free cell must differ.
        differs = []
        for var, val in free:
            b = model.NewBoolVar("")
            model.Add(var != val).OnlyEnforceIf(b)
            model.Add(var == val).OnlyEnforceIf(b.Not())
            differs.append(b)
        model.AddBoolOr(differs)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        return solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    @staticmethod
    def is_valid(grid: np.ndarray) -> bool:
        """Return True if a partial or complete grid has no duplicate clues."""
        for i in range(9):
            row = grid[i][grid[i] != 0]
            if len(row) != len(set(row.tolist())):
                return False
            col = grid[:, i][grid[:, i] != 0]
            if len(col) != len(set(col.tolist())):
                return False
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = grid[r : r + 3, c : c + 3].flatten()
                nz = box[box != 0]
                if len(nz) != len(set(nz.tolist())):
                    return False
        return True

    @staticmethod
    def print_grid(grid: np.ndarray | None, title: str = "Grid") -> None:
        if grid is None:
            print(f"\n{title}: (none)")
            return
        print(f"\n{title}:")
        print("+" + "-" * 21 + "+")
        for i in range(9):
            row = "| "
            for j in range(9):
                row += f"{int(grid[i, j]):d} " if grid[i, j] else "  "
                if j in (2, 5):
                    row += "| "
            print(row + "|")
            if i in (2, 5):
                print("+" + "-" * 21 + "+")
        print("+" + "-" * 21 + "+")
