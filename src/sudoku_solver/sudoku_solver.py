import numpy as np
from datetime import datetime
from ortools.sat.python import cp_model


class SudokuSolver:
    """Solve sudoku puzzles using Google OR-Tools CP-SAT solver."""

    def solve(self, grid: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve a 9x9 sudoku grid.

        Args:
            grid: 9x9 array with values 0-9 (0 = empty).

        Returns:
            (solved_grid, solve_time_seconds)

        Raises:
            ValueError: If grid is not 9x9.
            RuntimeError: If no solution exists.
        """
        if grid.shape != (9, 9):
            raise ValueError(f"Expected 9x9 grid, got {grid.shape}")

        model = cp_model.CpModel()
        x = {}

        for i in range(9):
            for j in range(9):
                if grid[i, j] != 0:
                    x[i, j] = grid[i, j]
                else:
                    x[i, j] = model.NewIntVar(1, 9, f"x[{i},{j}]")

        for i in range(9):
            model.AddAllDifferent([x[i, j] for j in range(9)])
        for j in range(9):
            model.AddAllDifferent([x[i, j] for i in range(9)])
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                model.AddAllDifferent([x[r + i, c + j] for i in range(3) for j in range(3)])

        solver = cp_model.CpSolver()
        t0 = datetime.now()
        status = solver.Solve(model)
        elapsed = (datetime.now() - t0).total_seconds()

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("No valid solution found. The puzzle may be unsolvable.")

        result = np.zeros((9, 9), dtype=np.uint8)
        for i in range(9):
            for j in range(9):
                result[i, j] = int(solver.Value(x[i, j]))
        return result, elapsed

    @staticmethod
    def is_valid(grid: np.ndarray) -> bool:
        """Check if a partial or complete grid has no conflicts."""
        for i in range(9):
            row = grid[i][grid[i] != 0]
            if len(row) != len(set(row)):
                return False
        for j in range(9):
            col = grid[:, j][grid[:, j] != 0]
            if len(col) != len(set(col)):
                return False
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = grid[r:r + 3, c:c + 3].flatten()
                nz = box[box != 0]
                if len(nz) != len(set(nz)):
                    return False
        return True

    @staticmethod
    def print_grid(grid: np.ndarray, title: str = "Grid") -> None:
        print(f"\n{title}:")
        print("+" + "-" * 21 + "+")
        for i in range(9):
            row = "| "
            for j in range(9):
                row += f"{grid[i, j]:d} " if grid[i, j] else "  "
                if j in (2, 5):
                    row += "| "
            print(row + "|")
            if i in (2, 5):
                print("+" + "-" * 21 + "+")
        print("+" + "-" * 21 + "+")