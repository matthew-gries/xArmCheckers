from checkers.game import Game

from xarm_checkers.checkers.utils import (render_game, parse_input)

DEBUG = True

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
