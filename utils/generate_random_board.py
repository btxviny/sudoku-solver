import numpy as np
import random

def is_valid(board, row, col, num):
    # Check if the number is in the row
    if num in board[row]:
        return False
    
    # Check if the number is in the column
    if num in board[:, col]:
        return False
    
    # Check if the number is in the 3x3 grid
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    
    return True

def fill_sudoku(board):
    empty = [(r, c) for r in range(9) for c in range(9) if board[r, c] == 0]
    if not empty:
        return True
    
    row, col = empty[0]
    random.shuffle(numbers := list(range(1, 10)))
    
    for num in numbers:
        if is_valid(board, row, col, num):
            board[row, col] = num
            if fill_sudoku(board):
                return True
            board[row, col] = 0
    
    return False

def generate_sudoku():
    board = np.zeros((9, 9), dtype=int)
    fill_sudoku(board)
    return board



if __name__ == "__main__":
    generate_sudoku()