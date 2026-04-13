import tkinter as tk
from tkinter import messagebox
import torch
import os

from reversi_game import ReversiGame
from reinforce_tf import SharedDQNAgent


class OthelloApp:
    def __init__(self, master, model_path="shared_othello_dqn.pth"):
        self.master = master
        self.master.title("Othello DQN App")

        self.board = ReversiGame()
        self.agent = SharedDQNAgent(self.board, explore=0.0)

        self.model_path = model_path
        self.model_loaded = False
        self.ai_pending = False

        # 預設：人類執黑，AI 執白
        self.side_var = tk.StringVar(value="黑棋")
        self.show_hints_var = tk.BooleanVar(value=True)

        self.cell_size = 70
        self.board_px = self.board.size * self.cell_size

        self._build_ui()
        self._load_model()
        self.new_game()

    # =========================
    # UI 建構
    # =========================
    def _build_ui(self):
        root_frame = tk.Frame(self.master)
        root_frame.pack(padx=12, pady=12)

        left_frame = tk.Frame(root_frame)
        left_frame.pack(side=tk.LEFT, padx=(0, 12))

        right_frame = tk.Frame(root_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(
            left_frame,
            width=self.board_px,
            height=self.board_px,
            bg="#1f7a1f",
            highlightthickness=1,
            highlightbackground="black",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        ctrl_frame = tk.Frame(right_frame)
        ctrl_frame.pack(anchor="nw", fill=tk.X)

        tk.Label(ctrl_frame, text="人類執子").pack(anchor="w")
        tk.OptionMenu(ctrl_frame, self.side_var, "黑棋", "白棋").pack(fill=tk.X)

        tk.Checkbutton(
            ctrl_frame,
            text="顯示推薦落子",
            variable=self.show_hints_var,
            command=self.refresh_view
        ).pack(anchor="w", pady=(8, 0))

        tk.Button(
            ctrl_frame,
            text="重新開始",
            command=self.new_game
        ).pack(fill=tk.X, pady=(10, 0))

        tk.Button(
            ctrl_frame,
            text="讓 AI 走一步",
            command=self.force_ai_move
        ).pack(fill=tk.X, pady=(6, 0))

        self.model_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.score_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.hint_var = tk.StringVar()

        tk.Label(right_frame, textvariable=self.model_var, justify="left", fg="blue").pack(anchor="w", pady=(14, 4))
        tk.Label(right_frame, textvariable=self.turn_var, justify="left").pack(anchor="w", pady=2)
        tk.Label(right_frame, textvariable=self.score_var, justify="left").pack(anchor="w", pady=2)
        tk.Label(right_frame, textvariable=self.status_var, justify="left", wraplength=280).pack(anchor="w", pady=(10, 6))
        tk.Label(right_frame, text="推薦前三手：", justify="left").pack(anchor="w", pady=(10, 2))
        tk.Label(right_frame, textvariable=self.hint_var, justify="left", wraplength=280).pack(anchor="w")

    # =========================
    # 模型
    # =========================
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.agent.load_weights(self.model_path)
                self.model_loaded = True
                self.model_var.set(f"模型：已載入 {self.model_path}")
            except Exception as e:
                self.model_loaded = False
                self.model_var.set(f"模型載入失敗：{e}")
        else:
            self.model_loaded = False
            self.model_var.set(f"模型：找不到 {self.model_path}，目前會用未訓練網路/隨機合法步")

    def get_human_color(self):
        return self.board.BLACK if self.side_var.get() == "黑棋" else self.board.WHITE

    def get_ai_color(self):
        return -self.get_human_color()

    def color_name(self, color):
        return "黑棋" if color == self.board.BLACK else "白棋"

    def move_to_text(self, move):
        row, col = move
        col_char = chr(ord("A") + col)
        return f"{col_char}{row + 1}"

    def get_q_ranking(self, color):
        legal_indices = self.board.legal_action_indices(color)
        if not legal_indices:
            return []

        state = self.board.canonical_state(color)
        state_tensor = self.agent.state_to_tensor(state)

        self.agent.model.eval()
        with torch.no_grad():
            q_values = self.agent.model(state_tensor).squeeze(0).cpu().numpy()

        ranking = []
        for idx in legal_indices:
            row = idx // self.board.size
            col = idx % self.board.size
            ranking.append(((row, col), float(q_values[idx])))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def choose_ai_action(self, color):
        legal_moves = self.board.legal_moves(color)
        if not legal_moves:
            return self.board.PASS_MOVE, None

        if self.model_loaded:
            action, qv = self.agent.choose_action(color, explore=False)
            if action in legal_moves:
                return action, qv

        # fallback：若模型沒載入或輸出不合法，隨便挑一個合法步
        return legal_moves[0], None

    # =========================
    # 遊戲流程
    # =========================
    def new_game(self):
        self.board.reset()
        self.ai_pending = False
        self.status_var.set("新對局開始。")
        self.refresh_view()

        if self.board.current_player == self.get_ai_color():
            self.schedule_ai_move()

    def handle_pass_if_needed(self):
        if self.board.is_game_over():
            return

        current = self.board.current_player
        if self.board.has_any_move(current):
            return

        passed_name = self.color_name(current)
        changed = self.board.auto_pass()

        if changed:
            self.status_var.set(f"{passed_name} 無合法步，系統自動跳過。")
        else:
            self.status_var.set("雙方都無合法步，對局結束。")

    def end_game_message(self):
        black, white = self.board.score()
        winner = self.board.winner()

        if winner == self.board.DRAW:
            msg = f"對局結束：平手\n黑棋 {black} : 白棋 {white}"
        elif winner == self.board.BLACK:
            msg = f"對局結束：黑棋獲勝\n黑棋 {black} : 白棋 {white}"
        else:
            msg = f"對局結束：白棋獲勝\n黑棋 {black} : 白棋 {white}"

        self.status_var.set(msg)
        messagebox.showinfo("Game Over", msg)

    def after_any_move(self):
        self.handle_pass_if_needed()
        self.refresh_view()

        if self.board.is_game_over():
            self.end_game_message()
            return

        if self.board.current_player == self.get_ai_color():
            self.schedule_ai_move()

    def schedule_ai_move(self):
        if self.ai_pending:
            return
        self.ai_pending = True
        self.master.after(400, self.ai_move)

    def ai_move(self):
        self.ai_pending = False

        if self.board.is_game_over():
            return
        if self.board.current_player != self.get_ai_color():
            return

        color = self.board.current_player
        action, qv = self.choose_ai_action(color)

        if action == self.board.PASS_MOVE:
            self.handle_pass_if_needed()
            self.refresh_view()
            return

        ok = self.board.apply_move(color, action[0], action[1])
        if ok:
            if qv is None:
                self.status_var.set(f"AI（{self.color_name(color)}）下在 {self.move_to_text(action)}")
            else:
                self.status_var.set(
                    f"AI（{self.color_name(color)}）下在 {self.move_to_text(action)}，Q={qv:.4f}"
                )
            self.after_any_move()

    def force_ai_move(self):
        if self.board.is_game_over():
            return
        if self.board.current_player != self.get_ai_color():
            self.status_var.set("現在不是 AI 的回合。")
            self.refresh_view()
            return
        self.ai_move()

    # =========================
    # 點擊事件
    # =========================
    def on_canvas_click(self, event):
        if self.board.is_game_over():
            return

        human_color = self.get_human_color()
        if self.board.current_player != human_color:
            self.status_var.set("現在不是你的回合。")
            self.refresh_view()
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if not self.board.is_on_board(row, col):
            return

        if not self.board.is_valid_move(human_color, row, col):
            self.status_var.set(f"{self.move_to_text((row, col))} 不是合法步。")
            self.refresh_view()
            return

        ok = self.board.apply_move(human_color, row, col)
        if ok:
            self.status_var.set(f"你下在 {self.move_to_text((row, col))}")
            self.after_any_move()

    # =========================
    # 畫面更新
    # =========================
    def refresh_view(self):
        self.draw_board()
        self.update_info_panel()

    def update_info_panel(self):
        black, white = self.board.score()
        current = self.board.current_player

        self.turn_var.set(f"目前輪到：{self.color_name(current)}")
        self.score_var.set(f"比分：黑棋 {black}  /  白棋 {white}")

        ranking = self.get_q_ranking(current) if self.show_hints_var.get() else []
        if ranking:
            lines = []
            for i, (move, qv) in enumerate(ranking[:3], start=1):
                lines.append(f"{i}. {self.move_to_text(move)}    Q = {qv:.4f}")
            self.hint_var.set("\n".join(lines))
        else:
            if self.board.is_game_over():
                self.hint_var.set("對局已結束。")
            elif self.board.has_any_move(current):
                self.hint_var.set("目前未顯示推薦。")
            else:
                self.hint_var.set("目前沒有合法步。")

    def draw_board(self):
        self.canvas.delete("all")

        # 棋盤格線
        for row in range(self.board.size):
            for col in range(self.board.size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="black",
                    fill="#1f7a1f"
                )

        # 標出合法步與最佳推薦
        current = self.board.current_player
        legal_moves = self.board.legal_moves(current)
        ranking = self.get_q_ranking(current) if self.show_hints_var.get() else []
        best_move = ranking[0][0] if ranking else None

        for row, col in legal_moves:
            cx = col * self.cell_size + self.cell_size // 2
            cy = row * self.cell_size + self.cell_size // 2

            # 一般合法步：黃點
            self.canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5,
                fill="yellow", outline=""
            )

            # 最佳推薦：紅框
            if best_move is not None and (row, col) == best_move:
                pad = 6
                x1 = col * self.cell_size + pad
                y1 = row * self.cell_size + pad
                x2 = (col + 1) * self.cell_size - pad
                y2 = (row + 1) * self.cell_size - pad
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="red",
                    width=3
                )

        # 畫棋子
        for row in range(self.board.size):
            for col in range(self.board.size):
                tile = self.board.board[row, col]
                if tile == self.board.EMPTY:
                    continue

                x1 = col * self.cell_size + 8
                y1 = row * self.cell_size + 8
                x2 = (col + 1) * self.cell_size - 8
                y2 = (row + 1) * self.cell_size - 8

                if tile == self.board.BLACK:
                    fill = "black"
                    outline = "white"
                else:
                    fill = "white"
                    outline = "black"

                self.canvas.create_oval(x1, y1, x2, y2, fill=fill, outline=outline, width=2)

        # 座標文字
        for c in range(self.board.size):
            self.canvas.create_text(
                c * self.cell_size + self.cell_size // 2,
                12,
                text=chr(ord("A") + c),
                fill="white",
                font=("Arial", 10, "bold")
            )
        for r in range(self.board.size):
            self.canvas.create_text(
                12,
                r * self.cell_size + self.cell_size // 2,
                text=str(r + 1),
                fill="white",
                font=("Arial", 10, "bold")
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloApp(root, model_path="shared_othello_dqn.pth")
    root.mainloop()