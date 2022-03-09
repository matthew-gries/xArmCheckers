from mctspy.games.common import TwoPlayersAbstractGameState, AbstractGameAction
from checkers.game import Game
from copy import deepcopy
from typing import List

class CheckersMove(AbstractGameAction):

    def __init__(self, move: List[int]):
        self.move = move

    def __repr__(self):
        return f"[{self.move[0]}, {self.move[1]}]"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, CheckersMove) and (
            self.move[0] == o.move[0]
            and self.move[1] == o.move[1]
        )


class CheckersGameState(TwoPlayersAbstractGameState):

    def __init__(self, checkers: Game = None):
        self.checkers = Game() if checkers is None else checkers

    def game_result(self):
        if self.checkers.get_winner() == 1:
            return 1
        elif self.checkers.get_winner() == 2:
            return -1
        else:
            return None

    def is_game_over(self):
        return self.checkers.is_over()

    def move(self, action):
        if move not in self.get_legal_actions():
            raise ValueError(f"{action} not in {self.get_legal_actions}")
        move = action.move
        # make a deep copy of the game to assign to the next state
        next_game = deepcopy(self.checkers)
        next_game.move(move)
        return CheckersGameState(checkers=next_game)

    def get_legal_actions(self):
        moves = self.checkers.get_possible_moves()
        return [CheckersMove(x) for x in moves]
