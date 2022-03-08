from tabnanny import check
from typing import List, Optional, Tuple, Set
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

    def copy(self) -> "Checkers":
        """
        Copy this checkers game to a new object

        :return: the new checkers object
        :rtype: Checkers
        """
        checkers = Checkers(board_dim=self.board_dim)
        checkers.board = self.board.copy()
        checkers.current_player = self.current_player
        checkers.p1_score = self.p1_score
        checkers.p2_score = self.p2_score
        return checkers

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

        self.board.fill(self.EMPTY)

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

        current_player = self.current_player

        p1_pieces_with_moves = 0
        p2_pieces_with_moves = 0

        for i in range(0, self.board.shape[0]):
            for j in range(0, self.board.shape[1]):
                sqr = self.board[i][j]
                if sqr == Checkers.EMPTY:
                    continue
                elif sqr == Checkers.P1_NORMAL or sqr == Checkers.P1_KING:
                    self.current_player = Checkers.PLAYER_ONE
                else:
                    self.current_player = Checkers.PLAYER_TWO
                moves = self.get_legal_moves((i, j))
                if len(moves) != 0:
                    if self.current_player == Checkers.PLAYER_ONE:
                        p1_pieces_with_moves += 1
                    else:
                        p2_pieces_with_moves += 1


        self.current_player = current_player

        winning_player = None
        if p1_pieces_with_moves == 0:
            winning_player = Checkers.PLAYER_TWO
        elif p2_pieces_with_moves == 0:
            winning_player = Checkers.PLAYER_ONE

        return winning_player

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

    def get_legal_moves(self, start: Tuple[int, int], ignore_turn: bool = False) -> List[List[Tuple[int, int]]]:
        """
        Get the set of legal moves from this state. A move is defined represented as one or move
        positions to move to, in sequence. Note that if at least one sequence of jumps is available,
        then jumps are the only moves that are legal (i.e. jumps are enforced)

        :param start: the position of the piece to get moves from
        :type start: Tuple[int, int]
        :param ignore_turn: if True, get moves regardless if it is this players turn or not, otherwise if it is
            not the players turn return an empty list. Defaults to False
        :type ignore_turn: Optional[bool]
        :return: a list of list of tuples, where each list of tuples is a sequence of moves (with one or more moves per
            sequence)
        :rtype: List[List[Tuple[int, int]]]
        """
        
        piece = self.board[start[0]][start[1]]
        if piece == Checkers.EMPTY:
            return []
        
        if not ignore_turn and (
            self.current_player == Checkers.PLAYER_ONE and not (piece == Checkers.P1_KING or piece == Checkers.P1_NORMAL)
            or self.current_player == Checkers.PLAYER_TWO and not (piece == Checkers.P2_KING or piece == Checkers.P2_NORMAL)
        ):
            return []
        
        is_king = piece == (Checkers.P1_KING if (self.current_player == Checkers.PLAYER_ONE) else Checkers.P2_KING)
        sequences = self.find_jumps(start=start, is_king=is_king)

        if len(sequences) != 0:
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
                
    def move(self, move_from: Tuple[int, int], move_to: Tuple[int, int]) -> bool:
        """
        Move a piece from one position to another, if possible. Does not enforce movement
        rules or whose turn it is but does enforce the following:
            * There must be a piece at `move_from`
            * `move_to` must be empty
            * Both `move_to` and `move_from` must be points within the board
        
        This function also does not update normal pieces to kings, nor does it modify the number
        of pieces on the board or the score (i.e. does not do jumps, just physically moves pieces).

        The board is only mutated when the result of this function is True

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

        if piece == self.EMPTY:
            return False

        # Check where we are moving is an empty space
        if self.board[move_to[0]][move_to[1]] != self.EMPTY:
            return False

        # Finally move the piece
        self.board[move_to[0]][move_to[1]] = piece
        self.board[move_from[0]][move_from[1]] = self.EMPTY
        return True

    def execute_turn(self, piece: Tuple[int, int], moves: List[Tuple[int, int]], move_type: str = "one") -> bool:
        """
        Execute a turn. This function will perform the given move or jump, depending on what is
        specified. This will modify the board, update the score, add new kings and switch players.

        :param piece: the piece to move, represented by its row, col position
        :param moves: one or move positions to move. If move type is 'one', this performs a single diagonal move and
            only one position is expected. If 'jump', the piece jumps to one or more positions and removes enemy pieces. This
            input is expected to be a result of `get_legal_moves`.
        :type moves: List[Tuple[int, int]]
        :param move_type: the type of move to make, if 'one' this is a single diagonal move, if 'jump'
            this performs one or more jumps. Defaults to "one"
        :type move_type: str, optional
        :return: True if the move could be peformed successfully, False otherwise. If False, the state of the game
            will not be changed.
        :rtype: bool
        """

        legal_moves = self.get_legal_moves(start=piece)
        if moves not in legal_moves:
            return False

        if move_type == 'one':
            success = self.move(move_from=piece, move_to=moves[0])
            # only move one space
            if abs(piece[0] - moves[0][0]) != 1 or abs(piece[1] - moves[0][1]) != 1:
                return False
            if not success:
                return False
            self._add_king_if_needed()
            self.switch_turns()
            return True
        elif move_type == 'jump':
            if not self._check_jumps(start=piece, jumps=moves):
                return False
            self._execute_jumps(start=piece, jumps=moves)
            self._add_king_if_needed()
            self.switch_turns()
            return True
        else:
            return False

    def _add_king_if_needed(self):
        """
        Scan the edges of the map and add kings depeding on the player, if possible. Should only
        be called after a new call to move or after executing jumps
        """
        if self.current_player == Checkers.PLAYER_ONE:
            bottom_row = self.board[self.board_dim-1]
            for i in range(0, bottom_row.size):
                if bottom_row[i] == Checkers.P1_NORMAL and self.p2_score > 0:
                    bottom_row[i] = Checkers.P1_KING
                    self.p2_score -= 1
                    break
        else:
            top_row = self.board[0]
            for i in range(0, top_row.size):
                if top_row[i] == Checkers.P2_NORMAL and self.p1_score > 0:
                    top_row[i] = Checkers.P2_KING
                    self.p1_score -= 1
                    break

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

        jump_sequences = self._find_jumps_helper(start=start, seen=set(), is_king=is_king)
        return [list(s) for s in jump_sequences]

    def _find_jumps_helper(self, start: Tuple[int, int], seen: Set[Tuple[int, int]], is_king: bool) -> List[deque]:
        """
        Helper for finding normal jumps

        :param start: the starting position of the piece
        :type start: Tuple[int, int]
        :param seen: the positions we have already jumped to
        :type seen: Set[Tuple[int, int]]
        :param is_king: True if the piece we are moving is a king, False otherwise
        :type is_king: bool
        :return: a list of deques representing sequences of (row, col) positions where the piece can jump to,
            in order
        :rtype: List[deque]
        """

        def jump_sequence_helper(player_normal, player_king, i1: int, i2: int, j1: int, j2: int, jumps: list, seen_in_path: set):
            # Helper for recursively getting jumps. i and j1 would be the position of the enemy piece,
            # while i2 and j2 would be the free piece diagonal to the enemy (and as such the position
            # we want to jump to)

            # j2 should be j1 + or - 1, depending on the direction, i2 should be i1 + or - depending on the direction

            # We should make the jump if the following conditions are satisfied:
            #   - j1 is in bounds
            #   - j2 is in bounds
            #   - the piece diagonal to use is an enemy
            #   - the piece diagonal to the enemy should be empty OR we we already saw this cell (AKA its empty now)
            # We already do a bounds check for i1 and i2

            if (i2, j2) in seen_in_path:
                return

            if (
                j1 in range(0, self.board_dim)
                and j2 in range(0, self.board_dim)
                and (self.board[i1][j1] == player_normal or self.board[i1][j1] == player_king)
                and (self.board[i2][j2] == self.EMPTY)
            ):
                seen_in_path.add((i2, j2))
                # Copy this game's board and current player, then make the move that removes the piece
                checkers = self.copy()
                checkers.move(move_from=start, move_to=(i2, j2))
                checkers.board[i1][j1] = Checkers.EMPTY
                jump_sequences = checkers._find_jumps_helper(start=(i2, j2), is_king=is_king, seen=seen_in_path)
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
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i+1, i+2, j-1, j-2, jumps, seen_in_path=set(seen))
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i+1, i+2, j+1, j+2, jumps, seen_in_path=set(seen))
            # if we are a king check the other direction
            if is_king and i > 1:
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i-1, i-2, j-1, j-2, jumps, seen_in_path=set(seen))
                jump_sequence_helper(self.P2_NORMAL, self.P2_KING, i-1, i-2, j+1, j+2, jumps, seen_in_path=set(seen))

        else:
            # if we are at the top two rows we cannot jump
            if i > 1:
                # check if there is an enemy piece diagonal to this piece and if there is an empty
                # square we can jump to
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j-1, j-2, jumps, seen_in_path=set(seen))
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i-1, i-2, j+1, j+2, jumps, seen_in_path=set(seen))
            # if we are a king check the other direction
            if is_king and i < self.board_dim - 2:
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i+1, i+2, j-1, j-2, jumps, seen_in_path=set(seen))
                jump_sequence_helper(self.P1_NORMAL, self.P1_KING, i+1, i+2, j+1, j+2, jumps, seen_in_path=set(seen))

        return jumps

    def _check_jumps(self, start: Tuple[int, int], jumps: List[Tuple[int, int]]) -> bool:
        """
        Check that the given sequence of jumps is valid

        :param start: see _execute_jumps
        :type start: Tuple[int, int]
        :param jumps: see _execute_jumps
        :type jumps: List[Tuple[int, int]]
        :return: True if the given sequence of jumps can be taken, False otherwise
        :rtype: bool
        """
        piece = self.board[start[0]][start[1]]
        if (self.current_player == Checkers.PLAYER_ONE and not (piece == Checkers.P1_KING or piece == Checkers.P1_NORMAL)
            or self.current_player == Checkers.PLAYER_TWO and not (piece == Checkers.P2_KING or piece == Checkers.P2_NORMAL)):
            return False
        current_pos = start
        for next_pos in jumps:
            enemy_pos = (abs(current_pos[0] + next_pos[0]) // 2, abs(current_pos[1] + next_pos[1]) // 2)
            enemy_piece = self.board[enemy_pos[0]][enemy_pos[1]]
            if (self.current_player == Checkers.PLAYER_ONE and not (enemy_piece == Checkers.P2_KING or enemy_piece == Checkers.P2_NORMAL)
                or self.current_player == Checkers.PLAYER_TWO and not (enemy_piece == Checkers.P1_KING or enemy_piece == Checkers.P1_NORMAL)):
                return False
            if self.board[next_pos[0]][next_pos[1]] != Checkers.EMPTY:
                return False

        last_spot = jumps[-1]
        if self.board[last_spot[0]][last_spot[1]] != Checkers.EMPTY:
            return False
        else:
            return True

    def _execute_jumps(self, start: Tuple[int, int], jumps: List[Tuple[int, int]]) -> None:
        """
        Execute the jumps in the given series of jumps

        :param start: the position to start from
        :type start: Tuple[int, int]
        :param jumps: the sequence of jumps to take, starting with the first jump and ending
            with the position the piece should stop at
        :type jumps: List[Tuple[int, int]]
        """

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
        self.move(move_from=start, move_to=last_spot)