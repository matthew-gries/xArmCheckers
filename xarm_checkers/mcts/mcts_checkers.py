from mctspy.games.common import TwoPlayersAbstractGameState, AbstractGameAction
from checkers.game import Game
from copy import deepcopy
from typing import List

class CheckersMove(AbstractGameAction):

    def __init__(self, move: List[int], player: int):
        """
        Representation of a checkers move

        :param move: action to take by the given player, where the first element
            is the space to move from and the second element is the space to move
            to
        :type move: List[int]
        :param player: the player who is taking this move, -1 if player 2 else player 1
        :type player: int
        """
        self.move = move
        self.player = player

    def __repr__(self):
        return f"Move: [{self.move[0]}, {self.move[1]}] Player: {self.player}"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, CheckersMove) and (
            self.move[0] == o.move[0]
            and self.move[1] == o.move[1]
            and self.player == o.player
        )


class CheckersGameState(TwoPlayersAbstractGameState):

    def __init__(self, checkers: Game = None, next_to_move: int = None):

        self.checkers = Game() if checkers is None else checkers
        self.next_to_move = 1 if next_to_move is None else next_to_move 

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
        if action not in self.get_legal_actions():
            raise ValueError(f"{action} not in {self.get_legal_actions()}")
        move = action.move
        # make a deep copy of the game to assign to the next state TODO verify this does what we want
        next_game = deepcopy(self.checkers)
        next_game.move(move)
        # next player should be what the next game says the player is, because the player doesn't switch
        # if we still need to do jumps
        next_player = 1 if next_game.whose_turn() == 1 else -1
        return CheckersGameState(checkers=next_game, next_to_move=next_player)

    def get_legal_actions(self):
        moves = self.checkers.get_possible_moves()
        return [CheckersMove(x, self.next_to_move) for x in moves]
