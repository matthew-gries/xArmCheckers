from pathlib import Path
import numpy as np
import pickle
import sys

from checkers.game import Game
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch

from xarm_checkers.mcts.mcts_checkers import CheckersGameState
from xarm_checkers.checkers.utils import (render_game, parse_input)

DEBUG = True
PLAYER_ONE = 1
PLAYER_TWO = 2
NUMBER_SIMULATIONS = 500

def _handle_player_one(checkers: Game):
        # Attempt to parse the move string, go back to loop start if needed
        move = None
        input_string = input()
        try:
            move = parse_input(input_string)
        except Exception as e:
            print("Could not parse input!")
            if DEBUG:
                print(e)
            return
        
        # Check if move is valid, if not go back to loop start
        if move not in checkers.get_possible_moves():
            print("Invalid move requested by player 1!")
            return

        # Perform the requested move
        checkers.move(move)


if __name__ == "__main__":
    checkers = Game()
    # set up initial MCTS

    mcts = None
    if len(sys.argv) == 1:
        checkers = Game()
        # set up initial MCTS
        mcts_state = CheckersGameState(checkers=checkers)
        mcts_root = TwoPlayersGameMonteCarloTreeSearchNode(state=mcts_state)
        mcts = MonteCarloTreeSearch(mcts_root)
        mcts.best_action(simulations_number=NUMBER_SIMULATIONS)
    elif len(sys.argv) == 2:
        path = Path(sys.argv[1])
        with open(str(path), 'rb') as f:
            mcts = pickle.load(f)
    else:
        print("Usage: python mcts_game.py <optional mcts pickle file>")
        exit(1)

    player = PLAYER_ONE
    while not checkers.is_over():
        # Render board and optional debug info
        print(render_game(checkers))

        if DEBUG:
            print(checkers.get_possible_moves())

        if player == PLAYER_ONE:
            _handle_player_one(checkers)
            player = checkers.whose_turn()
        else:
            # need to check the player 2 node that has the current state of the board
            node_with_current_state = None
            # this is kind of a hack but we can compare the string inputs pretty easily
            current_state = render_game(checkers=checkers)
            for node in mcts_node.children:
                assert node.state.next_to_move == -1
                state = render_game(checkers=node.state.checkers)
                if np.array_equal(current_state, state):
                    node_with_current_state = node
                    break
            # TODO what do we do with node_with_current_state is None? Maybe try rerunning
            # the tree search from this node
            if node_with_current_state is None:
                if DEBUG:
                    print("Couldn't find the current state from the previous node!")
                # Take the state (currently this player's turn) and run a new instance of MCTS, 
                current_mcts_state = CheckersGameState(checkers=checkers)
                current_mcts_root = TwoPlayersGameMonteCarloTreeSearchNode(state=current_mcts_state)
                mcts = MonteCarloTreeSearch(current_mcts_root)
                mcts.best_action(simulations_number=NUMBER_SIMULATIONS)
                node_with_current_state = current_mcts_root
            next_node = None
            try:
                next_node = node_with_current_state.best_child()
            except Exception as e:
                if DEBUG:
                    print(f"Caught exception: {e}")
                    print("Couldn't find a next state to go to!")
                mcts = MonteCarloTreeSearch(node_with_current_state)
                mcts.best_action(simulations_number=NUMBER_SIMULATIONS)
                next_node = node_with_current_state.best_child()
            checkers = next_node.state.checkers
            mcts_node = next_node
            player = checkers.whose_turn()

    # Print final state of game
    print(render_game(checkers))
