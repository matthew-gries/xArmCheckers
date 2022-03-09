from xarm_checkers.checkers.checkers import Checkers

def parse_input(s):
    pieces = s.split()

if __name__ == "__main__":
    checkers = Checkers()
    while not checkers.is_game_over():
        print(str(checkers))
        input_string = input()
