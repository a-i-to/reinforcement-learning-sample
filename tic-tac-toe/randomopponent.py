"""
Random Opponent
"""
import game


class RandomOpponent:

    def __init__(self, random_state):
        self.random_state = random_state

    def move(self, state):
        """
        choose a next action randomly.
        """
        action = self.get_last_cell(state)
        if action >= 0:
            return action
        candidates = [i for i in range(game.NUM_CELLS)
                      if (3 << i * 2) & state == 0]
        return self.random_state.choice(candidates)

    def get_last_cell(self, state):
        for i in range(game.NUM_CELLS_IN_ROW):
            row_state = (state & (63 << i * 6)) >> i * 6
            if row_state == 10:
                return 2 + i * 3
            if row_state == 34:
                return 1 + i * 3
            if row_state == 40:
                return i * 3
            col_state = (state & (12483 << i * 2)) >> i * 2
            if col_state == 130:
                return 6 + i
            if col_state == 8194:
                return 3 + i
            if col_state == 8320:
                return i
        diag_state = state & 197379
        if diag_state == 514:
            return 8
        if diag_state == 131074:
            return 4
        if diag_state == 131584:
            return 0
        diag_state2 = state & 13104
        if diag_state2 == 544:
            return 6
        if diag_state2 == 8224:
            return 4
        if diag_state2 == 8704:
            return 2

        for i in range(game.NUM_CELLS_IN_ROW):
            row_state = (state & (63 << i * 6)) >> i * 6
            if row_state == 5:
                return 2 + i * 3
            if row_state == 17:
                return 1 + i * 3
            if row_state == 20:
                return i * 3
            col_state = (state & (12483 << i * 2)) >> i * 2
            if col_state == 65:
                return 6 + i
            if col_state == 4097:
                return 3 + i
            if col_state == 4160:
                return i
        diag_state = state & 197379
        if diag_state == 257:
            return 8
        if diag_state == 65537:
            return 4
        if diag_state == 65792:
            return 0
        diag_state2 = state & 13104
        if diag_state2 == 272:
            return 6
        if diag_state2 == 4112:
            return 4
        if diag_state2 == 4352:
            return 2
        return -1
