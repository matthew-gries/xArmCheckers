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

    checkers.reset()

    # box in player one and confirm that player two wins
    checkers.board.fill(empty)
    checkers.board[0][1] = p1_norm
    checkers.board[1][0] = p2_norm
    checkers.board[1][2] = p2_norm
    checkers.board[2][3] = p2_norm

    assert checkers.winner() == Checkers.PLAYER_TWO
    checkers.switch_turns()
    assert checkers.winner() == Checkers.PLAYER_TWO


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


def test_legal_moves_normal_pieces_no_jumps():

    checkers = Checkers()
    # check that looking at an empty piece gives us no moves
    assert checkers.get_legal_moves((2, 0)) == []
    assert checkers.get_legal_moves((2, 0), ignore_turn=True) == []
    # check that looking at a boxed in piece gives no moves
    assert checkers.get_legal_moves((1, 0)) == []
    assert checkers.get_legal_moves((7, 0), ignore_turn=True) == []

    # can move diagonally to the left or right, no jumps
    moves_from_2_1 = checkers.get_legal_moves((2, 1))
    assert len(moves_from_2_1) == 2
    assert [(3, 0)] in moves_from_2_1
    assert [(3, 2)] in moves_from_2_1

    # can move diagonally only to the left
    moves_from_2_7 = checkers.get_legal_moves((2, 7))
    assert len(moves_from_2_7) == 1
    assert [(3, 6)] in moves_from_2_7

    # do the same for the other side, but first see that we don't
    # ignore if not the current player
    assert checkers.get_legal_moves((5, 2)) == []
    # Check that ignoring also gives [] because we are still player one
    assert checkers.get_legal_moves((5, 2), ignore_turn=True) == []

    # check that there are two diagonal moves
    checkers.switch_turns()
    moves_from_5_2 = checkers.get_legal_moves((5,2))
    assert len(moves_from_5_2) == 2
    assert [(4, 1)] in moves_from_5_2
    assert [(4, 3)] in moves_from_5_2

    # check that there is one diagonal move
    moves_from_5_0 = checkers.get_legal_moves((5,0))
    assert len(moves_from_5_0) == 1
    assert [(4, 1)] in moves_from_5_0

    checkers.reset()

    # make a contrived example that shows that the piece will only move forward
    checkers.board[2][1] = empty
    checkers.board[3][2] = p1_norm
    moves_from_3_2 = checkers.get_legal_moves((3, 2))
    assert len(moves_from_3_2) == 2
    assert [(4, 1)] in moves_from_3_2
    assert [(4, 3)] in moves_from_3_2

    # do the same for player 2
    checkers.reset()
    checkers.switch_turns()
    checkers.board[5][2] = empty
    checkers.board[4][3] = p2_norm
    moves_from_4_3 = checkers.get_legal_moves((4, 3))
    assert len(moves_from_4_3) == 2
    assert [(3, 2)] in moves_from_4_3
    assert [(3, 4)] in moves_from_4_3


def test_legal_moves_king_pieces_no_jumps():

    # make a contrived example with only one king piece that can move in any diagonal
    # direction, do that same with both players
    checkers = Checkers()
    checkers.board.fill(empty)
    checkers.board[2][1] = p1_king
    checkers.board[5][4] = p2_king

    p1_moves = checkers.get_legal_moves((2, 1))
    assert len(p1_moves) == 4
    assert [(1, 0)] in p1_moves
    assert [(1, 2)] in p1_moves
    assert [(3, 0)] in p1_moves
    assert [(3, 2)] in p1_moves

    assert checkers.get_legal_moves((5, 4)) == []
    checkers.switch_turns()
    p2_moves = checkers.get_legal_moves((5, 4))
    assert len(p2_moves) == 4
    assert [(4, 3)] in p2_moves
    assert [(4, 5)] in p2_moves
    assert [(6, 3)] in p2_moves
    assert [(6, 5)] in p2_moves

    # make a contrived example that blocks two diagonals
    checkers.reset()
    checkers.board.fill(empty)
    checkers.board[4][3] = p1_king
    checkers.board[3][2] = p2_norm
    checkers.board[2][1] = p2_norm
    checkers.board[5][4] = p2_king
    checkers.board[6][5] = p2_king

    p1_moves = checkers.get_legal_moves((4, 3))
    assert len(p1_moves) == 2
    assert [(3, 4)] in p1_moves
    assert [(5, 2)] in p1_moves


def test_legal_moves_normal_piece_jumps():

    checkers = Checkers()
    # make a contrived example where there is exactly one forward jump
    checkers.board.fill(empty)
    checkers.board[2][1] = p1_norm
    checkers.board[3][2] = p2_norm

    jump1_2_1 = checkers.get_legal_moves((2, 1))
    assert len(jump1_2_1) == 1
    assert [(4, 3)] in jump1_2_1

    checkers.reset()

    checkers.board.fill(empty)
    checkers.board[2][1] = p1_norm
    checkers.board[3][2] = p2_norm
    checkers.switch_turns()

    jump1_3_2 = checkers.get_legal_moves((3, 2))
    assert len(jump1_3_2) == 1
    assert [(1, 0)] in jump1_3_2

    checkers.reset()
    # make an example where there are two jumps, (since we can only go
    # forward these jumps are separate), one jump is one piece the other is two
    checkers.board.fill(empty)
    checkers.board[2][3] = p1_norm
    checkers.board[3][4] = p2_norm
    checkers.board[5][6] = p2_king
    checkers.board[3][2] = p2_norm
    jump2_2_3 = checkers.get_legal_moves((2, 3))
    assert len(jump2_2_3) == 2
    assert [(4, 5), (6, 7)] in jump2_2_3
    assert [(4, 1)] in jump2_2_3


def test_legal_moves_king_piece_jumps():

    checkers = Checkers()
    # make a contrived example where there is exactly one jump in each diagonal direction
    checkers.board.fill(empty)
    checkers.board[4][3] = p1_king
    checkers.board[3][2] = p2_norm
    checkers.board[3][4] = p2_norm
    checkers.board[5][2] = p2_king
    checkers.board[5][4] = p2_king

    jump4 = checkers.get_legal_moves((4, 3))
    assert len(jump4) == 4
    assert [(2, 1)] in jump4
    assert [(2, 5)] in jump4
    assert [(6, 1)] in jump4
    assert [(6, 5)] in jump4

    # make a contrived example where player two makes a sequence of jumps, in differnt directions
    # there are two answers to this depending on if you start to the left or the right
    checkers.reset()
    checkers.board.fill(empty)
    checkers.switch_turns()
    checkers.board[5][4] = p2_king
    checkers.board[4][3] = p1_norm
    checkers.board[4][5] = p1_norm
    checkers.board[2][3] = p1_norm
    checkers.board[2][5] = p1_norm

    jump2 = checkers.get_legal_moves((5, 4))
    assert len(jump2) == 2
    assert [(3, 2), (1, 4), (3, 6), (5, 4)] in jump2
    assert [(3, 6), (1, 4), (3, 2), (5, 4)] in jump2


def test_move():
    # test that simple moves work

    checkers = Checkers()

    # make a single move
    assert checkers.move((2, 1), (3, 0))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][0] == Checkers.P1_NORMAL
    checkers.reset()
    assert checkers.move((5, 0), (4, 1))
    assert checkers.board[5][0] == Checkers.EMPTY
    assert checkers.board[4][1] == Checkers.P2_NORMAL

    checkers.reset()

    # make a single move in the opposite left/right direction
    assert checkers.move((2, 1), (3, 2))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.P1_NORMAL
    checkers.reset()
    assert checkers.move((5, 6), (4, 5))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.P2_NORMAL

    checkers.reset()

    # make two single moves
    assert checkers.move((2, 1), (3, 2))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.P1_NORMAL
    assert checkers.move((5, 6), (4, 5))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.P2_NORMAL
    assert checkers.move((3, 2), (4, 1))
    assert checkers.board[2][1] == Checkers.EMPTY
    assert checkers.board[3][2] == Checkers.EMPTY
    assert checkers.board[4][1] == Checkers.P1_NORMAL
    assert checkers.move((4, 5), (3, 6))
    assert checkers.board[5][6] == Checkers.EMPTY
    assert checkers.board[4][5] == Checkers.EMPTY
    assert checkers.board[3][6] == Checkers.P2_NORMAL

    checkers.reset()

    # try to make some moves that go against movement rules but don't violate the assumptions
    # of this function
    assert checkers.move((2, 1), (2, 2)) # to the right
    checkers.reset()
    assert checkers.move((2, 1), (2, 0)) # to the left
    checkers.reset()
    assert checkers.move((2, 1), (1, 1)) # up
    checkers.reset()
    assert checkers.move((2, 1), (3, 1)) # down
    checkers.reset()
    assert checkers.move((2, 1), (4, 3)) # two spaces diagonally
    checkers.reset()

    # check that trying to move to a spot with an enemy piece fails
    checkers.reset()
    assert checkers.move((2, 1), (3, 2))
    assert checkers.move((3, 2), (4, 3))
    assert not checkers.move((4, 3), (5, 4))

    checkers.reset()
    assert not checkers.move((4, 3), (5, 4))
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

def test_execute_turn():

    checkers = Checkers()

    # Do a single move for p1 and p2
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert checkers.current_player == Checkers.PLAYER_ONE

    assert checkers.execute_turn(piece=(2, 1), moves=[(3, 2)])
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert checkers.current_player == Checkers.PLAYER_TWO

    assert checkers.execute_turn(piece=(5, 4), moves=[(4, 3)])
    assert checkers.p1_score == 0
    assert checkers.p2_score == 0
    assert checkers.current_player == Checkers.PLAYER_ONE

    # Check we can make the jump as player 1, and check we cant do a normal move
    assert not checkers.execute_turn(piece=(3, 2), moves=[(4, 1)])
    assert checkers.execute_turn(piece=(3, 2), moves=[(5,4)], move_type='jump')
    assert checkers.p1_score == 1
    assert checkers.p2_score == 0
    assert checkers.current_player == Checkers.PLAYER_TWO

