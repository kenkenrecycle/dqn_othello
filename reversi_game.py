import numpy as np
from math import floor

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class ReversiGame:
    BLACK = 1
    WHITE = -1
    EMPTY = 0
    DRAW = 0
    PASS_MOVE = (-1, -1)

    def __init__(self, size=8):
        size = int(size)
        if size % 2 != 0:
            size += 1
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)

        mid_left = self.size // 2 - 1
        mid_right = self.size // 2

        self.board[mid_left, mid_left] = self.BLACK
        self.board[mid_right, mid_right] = self.BLACK
        self.board[mid_left, mid_right] = self.WHITE
        self.board[mid_right, mid_left] = self.WHITE

        self.current_player = self.BLACK

    def clone(self):
        game = ReversiGame(self.size)
        game.board = self.board.copy()
        game.current_player = self.current_player
        return game

    def flatten(self):
        return self.board.flatten().astype(np.float32)

    def canonical_state(self, color):
        return self.flatten() * color

    def is_on_board(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def get_flips(self, color, row, col):
        if not self.is_on_board(row, col):
            return []
        if self.board[row, col] != self.EMPTY:
            return []

        other = -color
        flips = []

        for dr, dc in DIRECTIONS:
            r = row + dr
            c = col + dc
            line = []

            while self.is_on_board(r, c) and self.board[r, c] == other:
                line.append((r, c))
                r += dr
                c += dc

            if line and self.is_on_board(r, c) and self.board[r, c] == color:
                flips.extend(line)

        return flips

    def is_valid_move(self, color, row, col):
        return len(self.get_flips(color, row, col)) > 0

    def legal_moves(self, color):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(color, row, col):
                    moves.append((row, col))
        return moves

    def legal_action_indices(self, color):
        return [row * self.size + col for row, col in self.legal_moves(color)]

    def has_any_move(self, color):
        return len(self.legal_moves(color)) > 0

    def apply_move(self, color, row, col):
        if (row, col) == self.PASS_MOVE:
            self.current_player = -color
            return True

        flips = self.get_flips(color, row, col)
        if not flips:
            return False

        self.board[row, col] = color
        for r, c in flips:
            self.board[r, c] = color

        self.current_player = -color
        return True

    def auto_pass(self):
        if self.has_any_move(self.current_player):
            return False

        if self.has_any_move(-self.current_player):
            self.current_player = -self.current_player
            return True

        return False

    def is_game_over(self):
        return (not self.has_any_move(self.BLACK)) and (not self.has_any_move(self.WHITE))

    def score(self):
        black = int(np.sum(self.board == self.BLACK))
        white = int(np.sum(self.board == self.WHITE))
        return black, white

    def winner(self):
        black, white = self.score()
        if black > white:
            return self.BLACK
        if white > black:
            return self.WHITE
        return self.DRAW

    def print_board(self):
        symbols = {
            self.BLACK: "B",
            self.WHITE: "W",
            self.EMPTY: ".",
        }
        print("  " + " ".join(str(i) for i in range(self.size)))
        for r in range(self.size):
            print(str(r) + " " + " ".join(symbols[int(v)] for v in self.board[r]))
            
    def apply_move(self, color, row, col):
        """這個函數將執行玩家選擇的步驟，並更新棋盤"""
        if (row, col) == self.PASS_MOVE:
            self.current_player = -color
            #print("Passing turn to the other player.")  # 增加輸出，顯示跳過回合
            return True

        flips = self.get_flips(color, row, col)
        if not flips:
            #print(f"Invalid move attempted at {(row, col)}")  # 增加輸出，顯示無效的步驟
            return False

        self.board[row, col] = color
        for r, c in flips:
            self.board[r, c] = color

        self.current_player = -color
        #print(f"Move applied: {color} at {(row, col)}")  # 增加輸出，顯示有效步驟
        #self.print_board()  # 打印更新後的棋盤
        return True

    def auto_pass(self):
        """這個函數檢查當前玩家是否可以進行合法步驟"""
        if self.has_any_move(self.current_player):
            return False

        if self.has_any_move(-self.current_player):
            self.current_player = -self.current_player
            #print(f"Auto-passing: {self.current_player} player's turn.")  # 增加輸出，顯示跳過回合
            return True

        return False