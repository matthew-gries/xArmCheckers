from typing import Tuple, List
import numpy as np
import torch

from xarm_checkers.alphago_zero.NeuralNet import NeuralNet
from xarm_checkers.learning.checkers import CheckersGame

class CheckersNNWrapper(NeuralNet):

    def __init__(self, game: CheckersGame):
        self.game = game

    def canonical_board_into_nn_rep(board: CheckersGame) -> torch.Tensor:
        """
        Convert the canonical representation of the board into a game state representation that the neural network can use.
        Uses an approach similar to the game of Go, but each positional layer is doubled such that there is a positional layer
        for normal pieces and a positional layer for kinged pieces

        The top most layers will represent the positions of player 1's pieces
        """
        nn_rep = torch.zeros()

    def train(self, examples: List[Tuple[CheckersGame, np.ndarray, float]]):
        # Convert boards into neural network representations

