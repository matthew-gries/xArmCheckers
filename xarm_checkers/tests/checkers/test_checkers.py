from tabnanny import check
from tkinter.tix import CheckList
from xarm_checkers.checkers.checkers import Checkers
import numpy as np

p1_norm = Checkers.P1_NORMAL
p1_king = Checkers.P1_KING
empty = Checkers.EMPTY
p2_norm = Checkers.P2_NORMAL
p2_king = Checkers.P2_KING


def test_checkers_init():

    # Test a normal 8x8 board
    checkers = Checkers()

    assert checkers.board_dim == 8
    assert checkers.current_player == Checkers.PLAYER_ONE
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert np.array_equiv(checkers.board, np.array([
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm, empty],
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
        [empty, p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
    ], dtype=np.int8))

    # Test 6x6 board
    checkers6 = Checkers(board_dim=6)

    assert checkers6.board_dim == 6
    assert checkers6.current_player == Checkers.PLAYER_ONE
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert np.array_equiv(checkers6.board, np.array([
        [empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [p1_norm, empty, p1_norm, empty, p1_norm, empty],
        [empty, empty, empty, empty, empty, empty],
        [empty, empty, empty, empty, empty, empty],
        [empty, p2_norm, empty, p2_norm, empty, p2_norm],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty],
    ], dtype=np.int8))

    # Test failing conditions
    try:
        Checkers(board_dim=7)
        assert False
    except ValueError:
        pass

    try:
        Checkers(board_dim=4)
        assert False
    except ValueError:
        pass


def test_reset():

    checkers = Checkers()

    assert checkers.current_player == Checkers.PLAYER_ONE
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert np.array_equiv(checkers.board, np.array([
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm, empty],
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
        [empty, p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
    ], dtype=np.int8))

    # simulate a move
    checkers.board[2][1] = Checkers.EMPTY
    checkers.board[3][2] = Checkers.P1_NORMAL
    checkers.p1_score == 999
    checkers.p2_score == 999
    checkers.current_player = Checkers.PLAYER_TWO

    checkers.reset()

    assert checkers.current_player == Checkers.PLAYER_ONE
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert np.array_equiv(checkers.board, np.array([
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm, empty],
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
        [empty, p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
    ], dtype=np.int8))


def test_winner():

    checkers = Checkers()

    assert checkers.winner() is None

    # replace all P1 pieces with empty
    board = checkers.board
    board[board == Checkers.P1_NORMAL] = Checkers.EMPTY
    assert checkers.winner() == Checkers.PLAYER_TWO

    checkers.reset()
    assert checkers.winner() is None

    # replace all P2 pieces with empty
    board = checkers.board
    board[board == Checkers.P2_NORMAL] = Checkers.EMPTY
    assert checkers.winner() == Checkers.PLAYER_ONE


def test_is_game_over():
    
    checkers = Checkers()

    assert not checkers.is_game_over()

    # replace all P1 pieces with empty
    board = checkers.board
    board[board == Checkers.P1_NORMAL] = Checkers.EMPTY
    assert checkers.is_game_over()

    checkers.reset()
    assert not checkers.is_game_over()

    # replace all P2 pieces with empty
    board = checkers.board
    board[board == Checkers.P2_NORMAL] = Checkers.EMPTY
    assert checkers.is_game_over()


def test_switch_turns():

    checkers = Checkers()
    assert checkers.current_player == Checkers.PLAYER_ONE
    checkers.switch_turns()
    assert checkers.current_player == Checkers.PLAYER_TWO
    checkers.switch_turns()
    assert checkers.current_player == Checkers.PLAYER_ONE


def test_simple_normal_move():
    # test that simple moves work

    checkers = Checkers()

    # make a single move
    assert checkers.move((2, 1), (3, 0))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][0] == Checkers.P1_NORMAL
    checkers.switch_turns()
    assert checkers.move((5, 0), (4, 1))
    assert checkers.board[5][0] == Checkers.EMPTY
    assert checkers.board[4][1] == Checkers.P2_NORMAL

    checkers.reset()

    # make a single move in the opposite left/right direction
    assert checkers.move((2, 1), (3, 2))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.P1_NORMAL
    checkers.switch_turns()
    assert checkers.move((5, 6), (4, 5))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.P2_NORMAL

    checkers.reset()

    # make two single moves
    assert checkers.move((2, 1), (3, 2))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.P1_NORMAL
    checkers.switch_turns()
    assert checkers.move((5, 6), (4, 5))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.P2_NORMAL
    checkers.switch_turns()
    assert checkers.move((3, 2), (4, 1))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.EMPTY
    assert checkers.board[4][1] == Checkers.P1_NORMAL
    checkers.switch_turns()
    assert checkers.move((4, 5), (3, 6))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.EMPTY
    assert checkers.board[3][6] == Checkers.P2_NORMAL

    checkers.reset()

    # try to make some invalid moves
    assert not checkers.move((2, 1), (2, 2)) # to the right
    assert not checkers.move((2, 1), (2, 0)) # to the left
    assert not checkers.move((2, 1), (1, 1)) # up
    assert not checkers.move((2, 1), (3, 1)) # down
    assert not checkers.move((2, 1), (4, 3)) # two spaces diagonally

    # check that the board is not mutated when failing
    assert np.array_equiv(checkers.board, np.array([
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm, empty],
        [empty, p1_norm, empty, p1_norm, empty, p1_norm, empty, p1_norm],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [empty, empty, empty, empty, empty, empty, empty, empty],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
        [empty, p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm],
        [p2_norm, empty, p2_norm, empty, p2_norm, empty, p2_norm, empty],
    ], dtype=np.int8))

    # check that trying to move to a spot with an enemy piece fails
    checkers.reset()
    assert checkers.move((2, 1), (3, 2))
    assert checkers.move((3, 2), (4, 3))
    assert not checkers.move((4, 3), (5, 4))
