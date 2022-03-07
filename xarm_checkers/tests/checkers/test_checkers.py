from tabnanny import check
from xarm_checkers.checkers.checkers import Checkers
import numpy as np

p1_norm = Checkers.P1_NORMAL
empty = Checkers.EMPTY
p2_norm = Checkers.P2_NORMAL


def test_checkers_init():

    # Test a normal 8x8 board
    checkers = Checkers()

    assert checkers.board_dim == 8
    assert checkers.current_player == Checkers.PLAYER_ONE
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
    checkers.current_player = Checkers.PLAYER_TWO

    checkers.reset()

    assert checkers.current_player == Checkers.PLAYER_ONE
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