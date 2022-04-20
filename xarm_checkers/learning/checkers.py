import checkers.game as checkers_game
import numpy as np
import copy
from typing import List, Tuple
import logging

from xarm_checkers.alphago_zero.Game import Game
from xarm_checkers.checkers.utils import render_game

class CheckersWrapper(checkers_game.Game):
    """
    Wrapper around checkers game implementation that lets us get the canonical form of a game by
    seting whether to invert the player or not.
    """

    def __init__(self):
        super(CheckersWrapper, self).__init__()
        self.invert = False

    def whose_turn(self):
        whose_turn = super().whose_turn()
        if whose_turn is not None and self.invert:
            return (1 if whose_turn == 2 else 2)
        return whose_turn

    def get_winner(self):
        if self.whose_turn() == 1 and not self.board.count_movable_player_pieces(1):
            return 1 if self.invert else 2 # swapped
        elif self.whose_turn() == 2 and not self.board.count_movable_player_pieces(2):
            return 2 if self.invert else 1 # swapped
        else:
            return None

# Set the type for a game state (or 'board')
# This is meant to store the current state (position 0) plus the 7 previous histories of the game state
CheckersGameState = List[CheckersWrapper]

class CheckersGame(Game):

    PLAYER1 = 1
    PLAYER2 = -1

    def __init__(self):
        super().__init__()
        self._move_map = {i: {j: [] for j in range(4)} for i in range(1, 33)}
        self._fill_move_map()

    def _fill_move_map(self):
        """
        Fill the move map that can be used to get potential moves from a given position
        """

        for i in range(1, 33):
            for j in range(4):
                # fill lower left
                if j == 0:
                    # nothing lower than these spots
                    if i in range(29, 33):
                        continue
                    # nothing further left
                    elif i in [5, 13, 21]:
                        continue
                    # just one piece
                    elif i in [1, 9, 17, 25, 26, 27, 28]:
                        self._move_map[i][j] = [i+4]
                    # all other pieces
                    elif i in [6, 7, 8, 14, 15, 16, 22, 23, 24]:
                        self._move_map[i][j] = [i+3, i+7]
                    else:
                        self._move_map[i][j] = [i+4, i+7]
                # fill lower right
                elif j == 1:
                    # nothing lower than these spots
                    if i in range(29, 33):
                        continue
                    # nothing further right
                    elif i in [4, 12, 20]:
                        continue
                    # just one piece
                    elif i in [8, 16, 24]:
                        self._move_map[i][j] = [i+4]
                    elif i in [25, 26, 27]:
                        self._move_map[i][j] = [i+5]
                    # all other pieces
                    elif i in [1, 2, 3, 9, 10, 11, 17, 18, 19]:
                        self._move_map[i][j] = [i+5, i+9]
                    else:
                        self._move_map[i][j] = [i+4, i+9]
                # fill upper left
                if j == 2:
                    # nothing higher than these spots
                    if i in range(1, 5):
                        continue
                    # nothing further left
                    elif i in [5, 13, 21, 29]:
                        continue
                    # just one piece
                    elif i in [6, 7, 8]:
                        self._move_map[i][j] = [i-5]
                    elif i in [9, 17, 25]:
                        self._move_map[i][j] = [i-4]
                    # all other pieces
                    elif i in [10, 11, 12, 18, 19, 20, 26, 27, 28]:
                        self._move_map[i][j] = [i-4, i-9]
                    else:
                        self._move_map[i][j] = [i-5, i-9]
                # fill upper right
                if j == 3:
                    # nothing higher than these spots
                    if i in range(1, 5):
                        continue
                    # nothing further right
                    elif i in [12, 20, 28]:
                        continue
                    # just one piece
                    elif i in [5, 6, 7, 8, 16, 24, 32]:
                        self._move_map[i][j] = [i-4]
                    # all other pieces
                    elif i in [9, 10, 11, 17, 18, 19, 25, 26, 27]:
                        self._move_map[i][j] = [i-3, i-7]
                    else:
                        self._move_map[i][j] = [i-4, i-7]

    @staticmethod
    def _get_max_subsequent_jumps_count(game: CheckersWrapper, current_player: int, move: Tuple[int, int]) -> Tuple[int, CheckersWrapper]:
        game.move(move)
        if game.whose_turn() != current_player:
            return 1, game
        else:
            moves = game.get_possible_moves()
            res = [CheckersGame._get_max_subsequent_jumps_count(copy.deepcopy(game), current_player, m) for m in moves]
            counts = [r[0] for r in res]
            idx = np.argmax(counts)
            return 1 + res[idx][0], res[idx][1]

    def get_action_from_action_space_idx(self, board: CheckersWrapper, from_pos: int, idx: int) -> List[int]:
        """
        Get the action from the given position in the current state to the position that is in the
        direction specified by the index (see the action space for what this means)
        If a move cannot be made, returns None
        """

        potential_positions = self._move_map[from_pos][idx]
        current_legal_moves = board.get_possible_moves()

        # potential positions considers scenarios for moves and jumps, will be either one of the other
        # in the legal moveset
        for to_pos in potential_positions:
            if [from_pos, to_pos] in current_legal_moves:
                return [from_pos, to_pos]

        return None

    def getInitBoard(self) -> CheckersGameState:
        """THIS IS NOT THE NN REPRESENTATION, this is the logical implementation that stores all our state
        """
        return [CheckersWrapper()]

    def getBoardSize(self):
        return (8,8)

    def getActionSize(self):
        return 4 * 32

    def get_random_action(self, board: CheckersWrapper) -> Tuple[int, int]:
        """
        Get a random legal action from the board, and return the action in tuple form
        """
        actions = board.get_possible_moves()
        if len(actions) == 0:
            # print(board.get_winner())
            return None
        idx = np.random.randint(len(actions))
        return actions[idx]

    def getNextState(self, board: CheckersGameState, player: int, action: int) -> Tuple[CheckersGameState, int]:

        if action < 0:
            logging.warn(f"Invalid action {action} given!")
            return board, player
        if action > 127:
            logging.warn(f"Invalid action {action} given!")
            return board, player
        
        # make a copy of the current board to mutate
        new_board = copy.deepcopy(board[0])
        # make a copy of the whole list to add the mutated board to and return
        new_board_list = copy.deepcopy(board)
        current_player = new_board.whose_turn()

        # Whose turn it is between implementation and MCTS should always be in lockstep
        assert ((player == self.PLAYER1 and current_player == 1) or (player == self.PLAYER2 and current_player == 2))

        from_pos = (action // 4) + 1
        direction = action % 4

        action_tuple = self.get_action_from_action_space_idx(new_board, from_pos, direction)

        # if the action is not legal, pick a random legal action
        if action_tuple is None:
            logging.warn("No lgeal actions could be found from this state! Picking a random action...")
            action_tuple = self.get_random_action(new_board)

        if action_tuple is None:
            logging.warn("No random legal actions to choose from, something went wrong in this branch...")
            return board, player

        # Make the move
        new_board.move(action_tuple)

        # If we still have more jumps, recursively find the longest jump sequence
        if new_board.whose_turn() == current_player:
            game = new_board
            moves = game.get_possible_moves()
            res = [self._get_max_subsequent_jumps_count(copy.deepcopy(game), current_player, m) for m in moves]
            counts = [r[0] for r in res]
            idx = np.argmax(counts)
            new_board = res[idx][1]

        next_player = (self.PLAYER1 if new_board.whose_turn() == 1 else self.PLAYER2)

        # make sure we have the next player
        assert next_player == -player

        if len(board) < 8:
            new_board_list.insert(0, new_board)
        else:
            new_board_list = new_board_list[:7]
            new_board_list.insert(0, new_board)

        return new_board_list, next_player

    def getValidMoves(self, board: CheckersGameState, player: int) -> np.ndarray:
        # Don't actually need player field but its good to check anyway
        # assert (player == self.PLAYER1 and board[0].whose_turn() == 1) or (player == self.PLAYER1 and board[0].whose_turn() == 2)
        moves = np.zeros((self.getActionSize(),))
        legal_actions = board[0].get_possible_moves()
        for from_pos, to_pos in legal_actions:
            potential_directions = self._move_map[from_pos]
            direction = 0
            # Try to find the direction to move in
            for j in range(4):
                if to_pos in potential_directions[j]:
                    direction = j
                    break
            numeric_action_encoding = ((from_pos-1) * 4) + direction
            moves[numeric_action_encoding] = 1

        return moves

    def getGameEnded(self, board: CheckersWrapper, player: int) -> int:
        # Don't actually need player field but its good to check anyway
        if not board[0].is_over():
            return 0
        winner = board[0].get_winner()
        if winner is None:
            return 0.001
        else:
            winner = (self.PLAYER1 if board[0].get_winner() == 1 else self.PLAYER2)
            return winner

    def getCanonicalForm(self, board: CheckersWrapper, player: int) -> CheckersWrapper:
        """
        Canonical form is from the perspective of player 1, so we leave the board alone if the player is player 1,
        report the opposite player if player 2
        """

        new_boards = copy.deepcopy(board)
        # if we currently have P2's state, swap whose turn we say it is for the current state and its histories
        # (pretend we started with P2)
        # if player == self.PLAYER2:
        #     for new_board in new_boards:
        #         new_board.invert = True 
        # else:
        #     for new_board in new_boards:
        #         new_board.invert = False               
        if player == self.PLAYER2:
            for new_board in new_boards:
                new_board.invert = not new_board.invert                          

        return new_boards

    def getSymmetries(self, board, pi):
        # TODO try to actually find symmetries, not sure if this is a thing that can be done with checkers
        return [(board, pi)]

    def stringRepresentation(self, board):
        return "\n".join(render_game(b) for b in board)
