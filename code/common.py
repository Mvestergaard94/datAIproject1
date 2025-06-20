import numpy as np

def read_instance(fname: str):
    """
    Reads an instance file for Problem 1.

    Returns
    -------
    S : int
        Board size
    board : np.ndarray  (shape S×S)
        Cell values:
            0 → fixed 0
            1 → fixed 1
            2 → free decision variable
    """
    with open(fname) as f:
        S = int(f.readline().strip())
        board = [list(map(int, f.readline().split())) for _ in range(S)]
    return S, np.array(board, dtype=int)

def write_solution(fname: str, arr: np.ndarray):
    """
    Saves an S×S 0/1 numpy array in the contest format.
    """
    np.savetxt(fname, arr, fmt="%d", delimiter=" ")

