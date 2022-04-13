from xarm_checkers.learning.checkers import CheckersGameState
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from checkers.game import Game
from pathlib import Path

import pickle
import sys

NUMBER_SIMULATIONS = 100000

def pickle_mcts(path: Path):

    checkers = Game()
    # set up initial MCTS
    mcts_state = CheckersGameState(checkers=checkers)
    mcts_root = TwoPlayersGameMonteCarloTreeSearchNode(state=mcts_state)
    mcts = MonteCarloTreeSearch(mcts_root)
    mcts.best_action(simulations_number=NUMBER_SIMULATIONS)

    print("Writing MCTS object to pickle file...")
    with open(str(path), 'wb') as f:
        pickle.dump(mcts, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pickle_mcts.py <path>")
        exit(1)

    pickle_mcts(Path(sys.argv[1]))
