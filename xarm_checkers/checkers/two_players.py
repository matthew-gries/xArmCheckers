from checkers.game import Game
from typing import List
import numpy as np

DEBUG = True

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
    moves = [input_pieces[0], input_pieces[1]]
    return moves

if __name__ == "__main__":
    checkers = Game()
    while not checkers.is_over():
        # Render board and optional debug info
        print(render_game(checkers))
        if DEBUG:
            print(checkers.get_possible_moves())
        
        # Attempt to parse the move string, go back to loop start if needed
        move = None
        input_string = input()
        try:
            move = parse_input(input_string)
        except Exception as e:
            print("Could not parse input!")
            if DEBUG:
                print(e)
            continue
        
        # Check if move is valid, if not go back to loop start
        if move not in checkers.get_possible_moves():
            print("Invalid move requested!")
            continue

        # Perform the requested move
        checkers.move(move)

    # Print final state of game
    print(render_game(checkers))
