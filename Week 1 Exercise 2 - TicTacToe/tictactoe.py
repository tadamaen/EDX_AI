"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns the starting state of the board as a 3x3 grid.
    The grid is initialized with all cells being EMPTY.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns the player who has the next turn on the board.
    X starts first, then players alternate turns.

    If X has more moves than O, it's O's turn, otherwise it's X's turn.
    """
    # Count the occurrences of X and O on the board
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    # If there are more X's, it's O's turn, otherwise it's X's turn
    if x_count > o_count:
        return O
    else:
        return X

def actions(board):
    """
    Returns a set of all possible actions (i, j) that can be taken on the board.
    An action is a valid move where the cell is EMPTY.

    Each action is represented as a tuple (i, j) where i is the row and j is the column.
    """
    actions_set = set()

    # Iterate over the board and add valid actions (empty spaces) to the set
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions_set.add((i, j))

    return actions_set

def result(board, action):
    """
    Returns the board that results from making a move (i, j) on the current board.

    The function assumes that the action is valid. If the action is invalid, an exception is raised.
    A new board is created (deep copy of the original) to avoid modifying the original.
    The current player (X or O) makes the move on the new board.
    """
    i, j = action

    # Check if the action is out of bounds
    if i < 0 or i >= 3 or j < 0 or j >= 3:
        raise ValueError("Action is out of bounds.")

    # If the cell is not empty, raise an error (invalid move)
    if board[i][j] != EMPTY:
        raise ValueError(f"Cell ({i}, {j}) is already occupied.")

    # Create a new board (deep copy of the original)
    new_board = [row[:] for row in board]

    # Determine the current player
    current_player = player(board)

    # Make the move on the new board
    new_board[i][j] = current_player
    return new_board

def winner(board):
    """
    Returns the winner of the game if there is one.
    A winner is determined by checking all rows, columns, and diagonals.
    If there is no winner, it returns None.
    """
    # Check each row and column for a winning combination
    for i in range(3):
        # Check row
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        # Check column
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # Check the diagonals for a winning combination
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None

def terminal(board):
    """
    Returns True if the game is over, either because a player has won or the board is full (tie).
    If the game is still in progress, returns False.
    """
    # Check if there is a winner or if all cells are filled (indicating a tie)
    return winner(board) is not None or all(board[i][j] != EMPTY for i in range(3) for j in range(3))

def utility(board):
    """
    Returns the utility value of the board if the game is over.
    If X wins, return 1; if O wins, return -1; if it's a tie, return 0.
    """
    if winner(board) == X:
        return 1  # X wins
    elif winner(board) == O:
        return -1  # O wins
    return 0  # Tie

def minimax(board):
    """
    Returns the optimal action for the current player (X or O) on the board.
    The function uses Minimax to evaluate the best possible move.
    It recursively explores the game tree and chooses the move that maximizes the score
    for the current player (X) and minimizes the score for the opponent (O).
    """
    if terminal(board):
        return None  # No move possible if the game is over

    current_player = player(board)

    if current_player == X:
        # Maximizing for X
        best_value = -math.inf
        best_move = None
        # Evaluate each possible move for X
        for action in actions(board):
            new_board = result(board, action)
            value = min_value(new_board)  # Evaluate from O's perspective
            if value > best_value:
                best_value = value
                best_move = action
        return best_move
    else:
        # Minimizing for O
        best_value = math.inf
        best_move = None
        # Evaluate each possible move for O
        for action in actions(board):
            new_board = result(board, action)
            value = max_value(new_board)  # Evaluate from X's perspective
            if value < best_value:
                best_value = value
                best_move = action
        return best_move

def max_value(board):
    """
    Returns the maximum value for the current board (for X).
    It recursively evaluates all possible moves, maximizing the value for X.
    """
    if terminal(board):
        return utility(board)  # Return the utility if the game is over

    value = -math.inf
    # Explore all actions for X and take the maximum value
    for action in actions(board):
        new_board = result(board, action)
        value = max(value, min_value(new_board))  # Explore O's response
    return value

def min_value(board):
    """
    Returns the minimum value for the current board (for O).
    It recursively evaluates all possible moves, minimizing the value for O.
    """
    if terminal(board):
        return utility(board)  # Return the utility if the game is over

    value = math.inf
    # Explore all actions for O and take the minimum value
    for action in actions(board):
        new_board = result(board, action)
        value = min(value, max_value(new_board))  # Explore X's response
    return value
