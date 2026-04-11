import tkinter as tk
from reversi_game import ReversiGame


class OthelloApp:
    def __init__(self, master):
        self.master = master
        self.board = ReversiGame()
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        self.create_board()

    def create_board(self):
        for row in range(self.board.size):
            for col in range(self.board.size):
                x1 = col * 50
                y1 = row * 50
                x2 = x1 + 50
                y2 = y1 + 50
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="green")

    def start_game(self):
        self.board.reset()

    def update_board(self):
        self.canvas.delete("all")
        self.create_board()
        for row in range(self.board.size):
            for col in range(self.board.size):
                tile = self.board.board[row, col]
                if tile == self.board.BLACK:
                    color = "black"
                elif tile == self.board.WHITE:
                    color = "white"
                else:
                    continue
                x1 = col * 50 + 25
                y1 = row * 50 + 25
                self.canvas.create_oval(x1 - 20, y1 - 20, x1 + 20, y1 + 20, fill=color)