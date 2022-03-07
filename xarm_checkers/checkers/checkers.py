from typing import List, Optional
import numpy as np

class Checkers:
    """
    Implementation of a standard game of checkers, using an 8x8 board with 12 pieces
    per player. Rules can be found here: https://www.fgbradleys.com/rules/Checkers.pdf
    """

    PLAYER_ONE = 0
    PLAYER_TWO = 1

    P1_KING = -2
    P1_NORMAL = -1
    EMPTY = 0
    P2_NORMAL = 1
    P2_KING = 2

    def __init__(self, board_dim: int = 8) -> None:
        """
        Create a new instance of a game of checkers.

        :param board_dim: size of the board in a single dimension, making the board size
            (board_dim, board_dim), must be >= 6 and an even number. Defaults to 8
        :type board_dim: int, optional
        """

        if board_dim < 6 or board_dim % 2 != 0:
            raise ValueError(f"Board dimension must be >= 6 and even, got {board_dim}")

        self.board_dim = board_dim
        self.board = np.full((board_dim, board_dim), self.EMPTY, dtype=np.int8)
        self.current_player = self.PLAYER_ONE

        self.reset()

    def __str__(self) -> str:

        board = self.board.copy().astype(object)
        board[board == self.EMPTY] = "  "
        board[board == self.P1_NORMAL] = "P1"
        board[board == self.P1_KING] = "K1"
        board[board == self.P2_NORMAL] = "P2"
        board[board == self.P2_KING] = "K2"
        current_player = "Player 1" if (self.current_player == self.PLAYER_ONE) else "Player 2"

        return f"{str(board)}\nCurrent player: {current_player}"

    def reset(self) -> None:
        """
        Reset this game to the initial state of checkers. Player 1 starts and is positioned at the top of
        the board, while player 2 is placed at the bottom.
        """

        self.current_player = self.PLAYER_ONE

        '''
        Board is implemented as follows:
            * a -2 denotes a player 1 king piece
            * a -1 denotes a player 1 normal piece
            * a 0 denotes an empty square
            * a 1 denotes a player 2 normal piece
            * a 2 denotes a player 2 king piece
        '''

        self.board.fill(0)

        rows_per_player = self.board_dim // 2 - 1
        pieces_per_row = self.board_dim // 2

        # Fill out player 1's pieces
        for i in range(0, rows_per_player):
            # If i is even we place pieces shifted one space to the right
            if i % 2 == 0:
                for j in range(0, pieces_per_row*2, 2):
                    self.board[i][j+1] = self.P1_NORMAL
            # Otherwise we start on the left side of the board
            else:
                for j in range(0, pieces_per_row*2, 2):
                    self.board[i][j] = self.P1_NORMAL

        # Fill out player 2's pieces
        for i in range(self.board_dim-1, self.board_dim-rows_per_player-1, -1):
            # If i is even we place pieces shifted one space to the right
            if i % 2 == 0:
                for j in range(0, pieces_per_row*2, 2):
                    self.board[i][j+1] = self.P2_NORMAL
            # Otherwise we start on the left side of the board
            else:
                for j in range(0, pieces_per_row*2, 2):
                    self.board[i][j] = self.P2_NORMAL
    
    def winner(self) -> Optional[int]:
        """
        Get the winner of this game, if there is one.

        :return: Checkers.PLAYER_ONE or Checkers.PLAYER_TWO, whichever has won this game of,
            checkers, otherwise None
        :rtype: Optional[int]
        """

        player_one_piece_count = np.count_nonzero(self.board == self.P1_NORMAL) + np.count_nonzero(self.board == self.P1_KING)
        if player_one_piece_count == 0:
            return self.PLAYER_TWO
        
        player_two_piece_count = np.count_nonzero(self.board == self.P2_NORMAL) + np.count_nonzero(self.board == self.P2_KING)
        if player_two_piece_count == 0:
            return self.PLAYER_ONE

        return None

    def is_game_over(self) -> bool:
        """
        Check if the game is over

        :return: True if the game is over, False otherwise
        :rtype: bool
        """

        return self.winner() is not None