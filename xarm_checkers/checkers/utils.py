from checkers.game import Game
from typing import List
import numpy as np

def render_game(checkers: Game) -> str:
    """
    Render the current state of the checkers game.

    :param checkers: the game of checkers
    :type checkers: Game
    :return: a string representation of this game of checkers.
    :rtype: str
    """

    # Render the board
    board_pieces = ["  " for _ in range(32)]

    for piece in checkers.board.pieces:
        if piece.captured:
            continue
        king = "K" if piece.king else "P"
        player = str(piece.player)
        identifier = f"{king}{player}"
        board_pieces[piece.position-1] = identifier

    # use a numpy array of strings so we can quickly handle the formatting
    board_groups = []
    left_adjust = True
    for i, x in enumerate(board_pieces):

        if left_adjust:
            board_groups.append([" ", x])
        else:
            board_groups.append([x, " "])

        if (i+1) % 4 == 0:
            left_adjust = not left_adjust

    board = np.array(board_groups, dtype=object)
    board = board.flatten().reshape(8, 8)

    player_display = ""
    if checkers.is_over():
        player_display = f"Winner: {checkers.get_winner()}"
    else:
        player_display = f"Current turn: Player {checkers.whose_turn()}"

    return f"{str(board)}\n{player_display}"

def game_as_numpy(checkers: Game, as_board: bool = False) -> np.ndarray:
    """
    Convert the checkers game as a numpy array

    :param checkers: the game of checkers
    :type checkers: Game
    :param as_board: if true, the array will be an 8x8 representation of the board,
        otherwise the numpy array will be a 32-element vector representing the state of
        each position (index 0 -> top left position, index 31 -> bottom right, analogous to
        the checkers game implementation)
    :type as_board: Optional[bool]
    :return: a int8 numpy array representing the game, with the following representations for players:
        - player 1 is represented by a 1 if the piece is normal, 2 if it is a king
        - player 2 is represented by a -1 if the piece is normal, -2 if it is a king
        - an empty space is represented by a 0
    :rtype: np.ndarray
    """
    board_pieces = [0 for _ in range(32)]

    for piece in checkers.board.pieces:
        if piece.captured:
            continue
        king = 2 if piece.king else 1
        player = 1 if piece.player == 1 else -1
        identifier = king * player
        board_pieces[piece.position-1] = identifier

    if as_board:

        # use a numpy array of strings so we can quickly handle the formatting
        board_groups = []
        left_adjust = True
        for i, x in enumerate(board_pieces):

            if left_adjust:
                board_groups.append([0, x])
            else:
                board_groups.append([x, 0])

            if (i+1) % 4 == 0:
                left_adjust = not left_adjust

        board = np.array(board_groups, dtype=np.int8)
        board = board.flatten().reshape(8, 8)

        return board
    else:

        return np.array(board_pieces, dtype=np.int8)


def parse_input(input_string: str) -> List[int]:
    """
    Parse the input string and get
        * The starting position
        * The ending position

    :param input_string: a string expecting the above format with a space between each entry. Examples:
        '1 5' -> move or jump from position 1 to 5
    :type input_string: str
    :return: a list of ints, the first describing the starting position and the second describing the ending
        position
    :rtype: List[int]
    """

    input_pieces = input_string.split()
    moves = [int(input_pieces[0]), int(input_pieces[1])]
    return moves