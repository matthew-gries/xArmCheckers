from typing import List, Optional, Tuple
from collections import deque
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

        # score as defined here is how many of the enemy pieces has been collected
        self.p1_score = 0
        self.p2_score = 0

        self.reset()

    def __str__(self) -> str:
        """
        Make a string representation of this game

        :return: this game represented as a string
        :rtype: str
        """

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
        self.p1_score = 0
        self.p2_score = 0

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

    def switch_turns(self):
        """
        Switch whose turn it is
        """
        if self.current_player == Checkers.PLAYER_ONE:
            self.current_player = Checkers.PLAYER_TWO
        else:
            self.current_player = Checkers.PLAYER_ONE

    def get_legal_moves(self, start: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Get the set of legal moves from this state. A move is defined represented as one or move
        positions to move to, in sequence.

        :return: a list of list of tuples, where each list of tuples is a sequence of moves (with one or more moves per
            sequence)
        :rtype: List[List[Tuple[int, int]]]
        """
        is_king = self.board[start[0]][start[1]] == (Checkers.P1_KING if (self.current_player == Checkers.PLAYER_ONE) else Checkers.P2_KING)
        sequences = self.find_jumps(start=start, is_king=is_king)

        # since we must take the longest jump, check if there is a jump and get the longest sequences
        if len(sequences) != 0:
            max_sequence_length = max(len(x) for x in sequences)
            sequences = [s for s in sequences if len(s) == max_sequence_length]
            return sequences

        # if there are no jumps to take, check diagonals
        sequences = []
        i, j = start

        valid_diagonals = []
        if self.current_player == Checkers.PLAYER_ONE:
            if i+1 in range(0, self.board_dim) and j-1 in range(0, self.board_dim):
                valid_diagonals.append((i+1, j-1))
            if i+1 in range(0, self.board_dim) and j+1 in range(0, self.board_dim):
                valid_diagonals.append((i+1, j+1))
            if is_king and i-1 in range(0, self.board_dim) and j-1 in range(0, self.board_dim):
                valid_diagonals.append((i-1, j-1))
            if is_king and i-1 in range(0, self.board_dim) and j+1 in range(0, self.board_dim):
                valid_diagonals.append((i-1, j+1))
        else:
            if is_king and i+1 in range(0, self.board_dim) and j-1 in range(0, self.board_dim):
                valid_diagonals.append((i+1, j-1))
            if is_king and i+1 in range(0, self.board_dim) and j+1 in range(0, self.board_dim):
                valid_diagonals.append((i+1, j+1))
            if i-1 in range(0, self.board_dim) and j-1 in range(0, self.board_dim):
                valid_diagonals.append((i-1, j-1))
            if i-1 in range(0, self.board_dim) and j+1 in range(0, self.board_dim):
                valid_diagonals.append((i-1, j+1))

        for diag in valid_diagonals:
            if self.board[diag[0]][diag[1]] == Checkers.EMPTY:
                sequences.append([diag])

        return sequences
                
    # TODO move shouldnt really enforce any rules
    def move(self, move_from: Tuple[int, int], move_to: Tuple[int, int]) -> bool:
        """
        Perform the given action, if possible. Does not enforce movement rules and
        allows for one jump at a time

        :param move_from: tuple describing the (row, col) to move from
        :type move_from: Tuple[int, int]
        :param move_to: tuple describing the (row, col) to move to
        :type move_to: Tuple[int, int]
        :return: True if the move was able to be performed, False if not
        :rtype: bool
        """

        # Check that the indicies are valid
        if (
            move_from[0] not in range(0, self.board_dim)
            or move_from[1] not in range(0, self.board_dim)
            or move_to[0] not in range(0, self.board_dim)
            or move_to[1] not in range(0, self.board_dim)
        ):
            return False
        
        # First check if the piece we are looking at is valid, given the current player
        piece = self.board[move_from[0]][move_from[1]]
        
        if self.current_player == self.PLAYER_ONE and not (piece == self.P1_KING or piece == self.P1_NORMAL):
            return False
        if self.current_player == self.PLAYER_TWO and not (piece == self.P2_KING or piece == self.P2_NORMAL):
            return False

        if self.board[move_to[0]][move_to[1]] != self.EMPTY:
            return False

        # Make the move, depending on if the piece was a king or a normal piece
        if piece == self.P1_KING or piece == self.P2_KING:
            return self._move_king(move_from, move_to)

        return self._move_normal(move_from, move_to)

    def _move_king(self, move_from: Tuple[int, int], move_to: Tuple[int, int]) -> bool:
        """
        Move a king piece, if possible

        :param move_from: tuple describing the (row, col) to move from
        :type move_from: Tuple[int, int]
        :param move_to: tuple describing the (row, col) to move to
        :type move_to: Tuple[int, int]
        :return: True if the move was able to be performed, False if not
        :rtype: bool
        """
        from_i, from_j = move_from
        to_i, to_j = move_to
        # if the different in rows is greater than one, assume we are trying to
        # jump
        if abs(from_i - to_i) > 1:
            jump_sequences = self.find_jumps(start=move_from, is_king=True)
            # filter these jump sequences such that its only those of length one
            jump_sequences = [js for js in jump_sequences if len(js) == 1]
            
            if jumps is None:
                return False
            
            if jumps[-1] != move_to:
                return False

            self._execute_jumps(start=move_from, jumps=jumps)
            return True

        # otherwise we just do a single space move
        else:
            # make sure it is a diagonal move
            if abs(from_j - to_j) != 1:
                return False
            self.board[from_i][from_j] = Checkers.EMPTY
            self._set_new_piece(move_to=move_to, already_king=True)
            return True

    def _move_normal(self, move_from: Tuple[int, int], move_to: Tuple[int, int]) -> bool:
        """
        Move a normal piece, if possible

        :param move_from: tuple describing the (row, col) to move from
        :type move_from: Tuple[int, int]
        :param move_to: tuple describing the (row, col) to move to
        :type move_to: Tuple[int, int]
        :return: True if the move was able to be performed, False if not
        :rtype: bool
        """
        from_i, from_j = move_from
        to_i, to_j = move_to
        jump_sequences = self.find_jumps(start=move_from, is_king=False)
         
        # depending on if player 1 or player 2, check if the direction is ok
        if self.current_player == Checkers.PLAYER_ONE:
            if to_i <= from_i:
                return False
        else:
            if to_i >= from_i:
                return False

        # if the different in rows is greater than one, assume we are trying to
        # jump
        if abs(from_i - to_i) > 1:
            jumps = None
            num_jumps = -1
            for jump_sequence in jump_sequences:
                if len(jump_sequence) > num_jumps:
                    jumps = jump_sequence
                    num_jumps = len(jump_sequence)
            
            if jumps is None:
                return False
            
            if jumps[-1] != move_to:
                return False

            self._execute_jumps(start=move_from, jumps=jumps)
            return True

        # otherwise we just do a single space move
        else:
            # if there is a jump we must take it, so return False
            if len(jump_sequences) != 0:
                return False
            # make sure it is a diagonal move
            if abs(from_j - to_j) != 1:
                return False
            self.board[from_i][from_j] = Checkers.EMPTY
            self._set_new_piece(move_to=move_to)
            return True
            
    def _set_new_piece(self, move_to: Tuple[int, int], already_king: bool = False) -> None:

        to_i, to_j = move_to

        # check for making a new king
        make_king = (
            (self.current_player == Checkers.PLAYER_ONE and to_i == self.board_dim-1)
            or (self.current_player == Checkers.PLAYER_TWO and to_i == 0)
        ) and not already_king
        
        # take a piece away from the other player
        if make_king:
            if self.current_player == Checkers.PLAYER_ONE:
                self.p2_score -= 1
            else:
                self.p1_score -= 1

        new_piece = None
        if self.current_player == Checkers.PLAYER_ONE and (make_king or already_king):
            new_piece = Checkers.P1_KING
        elif self.current_player == Checkers.PLAYER_TWO and (make_king or already_king):
            new_piece = Checkers.P2_KING
        elif self.current_player == Checkers.PLAYER_ONE and not (make_king or already_king):
            new_piece = Checkers.P1_NORMAL
        else:
            new_piece = Checkers.P2_NORMAL
        
        self.board[to_i][to_j] = new_piece

    def find_jumps(self, start: Tuple[int, int], is_king: bool = False) -> List[List[Tuple[int, int]]]:
        """
        Find all jumps possible from a normal piece from the given starting position

        :param start: the starting position of the piece
        :type start: Tuple[int, int]
        :param is_king: True if the piece we are moving is a king, False otherwise. Defaults to False
        :type is_king: bool
        :return: a list of list of tuples representing sequences of (row, col) positions where the piece can jump to,
            in order
        :rtype: List[List[Tuple[int, int]]]
        """
        jump_sequences = self._find_jumps_helper(start=start, is_king=is_king)
        return [list(s) for s in jump_sequences]

    def _find_jumps_helper(self, start: Tuple[int, int], is_king: bool) -> List[deque]:
        """
        Helper for finding normal jumps

        :param start: the starting position of the piece
        :type start: Tuple[int, int]
        :param is_king: True if the piece we are moving is a king, False otherwise
        :type is_king: bool
        :return: a list of deques representing sequences of (row, col) positions where the piece can jump to,
            in order
        :rtype: List[deque]
        """

        def jump_sequence_helper(player_normal, player_king, i1: int, i2: int, j1: int, j2: int, jumps: list):
            # Helper for recursively getting jumps. i and j1 would be the position of the enemy piece,
            # while i2 and j2 would be the free piece diagonal to the enemy (and as such the position
            # we want to jump to)

            # j2 should be j1 + or - 1, depending on the direction, i2 should be i1 + or - depending on the direction

            # We should make the jump if the following conditions are satisfied:
            #   - j1 is in bounds
            #   - j2 is in bounds
            #   - the piece diagonal to use is an enemy
            #   - the piece diagonal to the enemy should be empty
            # We already do a bounds check for i1 and i2
            if (
                j1 in range(0, self.board_dim)
                and j2 in range(0, self.board_dim)
                and (self.board[i1][j1] == player_normal or self.board[i1][j1] == player_king)
                and self.board[i2][j2] == self.EMPTY
            ):
                jump_sequences = self._find_jumps_helper(start=(i2, j2), is_king=is_king)
                if len(jump_sequences) == 0:
                    jump_sequences.append(deque([(i2, j2)]))
                else:
                    for jump_sequence in jump_sequences:
                        jump_sequence.appendleft((i2, j2))
                
                for jump_sequence in jump_sequences:
                    jumps.append(jump_sequence)

        # Since we are dealing with the normal piece we only need to move forward
        jumps = []
        i, j = start

        if self.current_player == self.PLAYER_ONE:
            # if we are at the bottom two rows we cannot jump
            if i < self.board_dim - 2:
                # check if there is an enemy piece diagonal to this piece and if there is an empty
                # square we can jump to
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i+1, i+2, j-1, j-2, jumps)
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i+1, i+2, j+1, j+2, jumps)
            # if we are a king check the other direction
            if is_king and i > 1:
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j-1, j-2, jumps)
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j+1, j+2, jumps)

        else:
            # if we are at the top two rows we cannot jump
            if i > 1:
                # check if there is an enemy piece diagonal to this piece and if there is an empty
                # square we can jump to
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j-1, j-2, jumps)
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j+1, j+2, jumps)
            # if we are a king check the other direction
            if is_king and i < self.board_dim - 2:
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i+1, i+2, j-1, j-2, jumps)
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i+1, i+2, j+1, j+2, jumps)

        return jumps

    def _execute_jumps(self, start: Tuple[int, int], jumps: List[Tuple[int, int]]) -> None:
        """
        Execute the jumps in the given series of jumps

        :param start: the position to start from
        :type start: Tuple[int, int]
        :param jumps: the sequence of jumps to take, starting with the first jump and ending
            with the position the piece should stop at
        :type jumps: List[Tuple[int, int]]
        """

        self.board[start[0]][start[1]] = Checkers.EMPTY
        current_pos = start
        for next_pos in jumps:
            enemy_pos = (abs(current_pos[0] + next_pos[0]) // 2, abs(current_pos[1] + next_pos[1]) // 2)
            enemy_piece = self.board[enemy_pos[0]][enemy_pos[1]]
            self.board[enemy_pos[0]][enemy_pos[1]] = self.EMPTY
            if self.current_player == self.PLAYER_ONE:
                self.p1_score += (1 if enemy_piece == Checkers.P2_NORMAL else 2)
            else:
                self.p2_score += (1 if enemy_piece == Checkers.P1_NORMAL else 2)
            current_pos = next_pos

        last_spot = jumps[-1]
        self._set_new_piece(move_to=last_spot)