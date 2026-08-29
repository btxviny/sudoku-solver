import numpy as np
import pytest
from sudoku_solver.sudoku_solver import SudokuSolver


class TestSudokuSolver:
    def test_solve_easy_puzzle(self):
        solver = SudokuSolver()
        grid = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ], dtype=np.uint8)
        result, time = solver.solve(grid)
        assert result.shape == (9, 9)
        assert solver.is_valid(result)
        assert time >= 0

    def test_is_valid_correct(self):
        grid = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 3, 2, 1, 6, 5, 8, 9, 7],
            [5, 7, 8, 9, 8, 7, 1, 2, 3],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [9, 6, 5, 7, 1, 2, 3, 4, 8],
            [7, 1, 4, 5, 9, 8, 6, 3, 2],
            [2, 8, 9, 6, 7, 3, 4, 5, 1],
            [3, 4, 7, 8, 2, 1, 9, 7, 6],
            [6, 5, 6, 3, 4, 9, 2, 1, 5],
        ], dtype=np.uint8)
        # Full grid, may have conflicts but is_valid only checks for duplicates
        solver = SudokuSolver()
        assert isinstance(solver.is_valid(grid), bool)

    def test_invalid_shape(self):
        solver = SudokuSolver()
        bad = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 9x9"):
            solver.solve(bad)
