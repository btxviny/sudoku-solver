import numpy as np
from datetime import datetime
from ortools.sat.python import cp_model


def solve_with_cp(grid: np.ndarray) -> (np.matrix, float):
    '''
    Solve Sudoku instance (np.matrix) with CP modeling. 
    Returns a tuple with the resulting matrix and the 
    execution time in seconds.'''
    assert grid.shape == (9,9)
    
    grid_size = 9
    region_size = 3 
    model = cp_model.CpModel()

    #reate and initialize variables.
    x = {}
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] != 0:
                x[i, j] = grid[i, j] # Initial values (values already defined on the puzzle).
            else:
                x[i, j] = model.NewIntVar(1, grid_size, 'x[{},{}]'.format(i,j) ) # Values to be found (variyng from 1 to 9).

    #constraints.
    #Row constraint.
    for i in range(grid_size):
        model.AddAllDifferent([x[i, j] for j in range(grid_size)])

    #Column constraint.
    for j in range(grid_size):
        model.AddAllDifferent([x[i, j] for i in range(grid_size)])

    #Box constraint.
    for row_idx in range(0, grid_size, region_size):
        for col_idx in range(0, grid_size, region_size):
            model.AddAllDifferent(
                [x[row_idx + i, j] for j in range(col_idx, (col_idx + region_size)) for i in range(region_size)]
            )
    
    solver = cp_model.CpSolver()
    start = datetime.now()
    status = solver.Solve(model)
    exec_time = datetime.now() - start
    result = np.zeros((grid_size, grid_size), dtype=np.uint8)
    #Get values defined by the solver
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i in range(grid_size):
            for j in range(grid_size):
                result[i,j] = int(solver.Value(x[i,j]))
    else:
        raise Exception('Unfeasible Sudoku')
    return result, exec_time.total_seconds()
