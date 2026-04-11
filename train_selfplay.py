import random
import matplotlib.pyplot as plt
from reversi_game import ReversiGame
from reinforce_tf import SharedDQNAgent


def play_one_game(board, agent, train=True):
    board.reset()

    while not board.is_game_over():
        color = board.current_player

        # 目前輪到 color，從 color 的視角看棋盤
        state = board.canonical_state(color)

        action, _ = agent.choose_action(color, explore=train)

        if action == board.PASS_MOVE:
            board.auto_pass()
            next_state = board.canonical_state(board.current_player)
            done = board.is_game_over()
            next_valid_actions = board.legal_action_indices(board.current_player) if not done else []

            if train:
                agent.remember(state, action, 0.0, next_state, done, next_valid_actions)
                agent.train()
            continue

        ok = board.apply_move(color, action[0], action[1])

        if not ok:
            # 保底機制，避免非法步卡住
            legal = board.legal_moves(color)
            if not legal:
                board.auto_pass()
                next_state = board.canonical_state(board.current_player)
                done = board.is_game_over()
                next_valid_actions = board.legal_action_indices(board.current_player) if not done else []

                if train:
                    agent.remember(state, board.PASS_MOVE, 0.0, next_state, done, next_valid_actions)
                    agent.train()
                continue
            else:
                fallback = random.choice(legal)
                board.apply_move(color, fallback[0], fallback[1])
                action = fallback

        done = board.is_game_over()

        if done:
            winner = board.winner()
            if winner == color:
                reward = 1.0
            elif winner == board.DRAW:
                reward = 0.0
            else:
                reward = -1.0

            # 終局時 next_state 不再使用，但先放同維度陣列
            next_state = state.copy()
            next_valid_actions = []
        else:
            reward = 0.0
            # 下完之後換對手，所以這裡是對手視角的 canonical state
            next_state = board.canonical_state(board.current_player)
            next_valid_actions = board.legal_action_indices(board.current_player)

        if train:
            agent.remember(state, action, reward, next_state, done, next_valid_actions)
            agent.train()

    return board.winner()


def evaluate_vs_random(agent, n_games=20):
    eval_board = ReversiGame()

    # 很重要：先保存原本訓練棋盤，評估後再切回去
    old_board = agent.board
    old_explore = agent.explore

    agent.board = eval_board
    agent.explore = 0.0

    wins = 0
    draws = 0
    losses = 0

    for g in range(n_games):
        eval_board.reset()

        # 一半黑棋、一半白棋，避免只測單邊
        agent_color = eval_board.BLACK if g % 2 == 0 else eval_board.WHITE

        while not eval_board.is_game_over():
            color = eval_board.current_player
            legal_moves = eval_board.legal_moves(color)

            if not legal_moves:
                eval_board.auto_pass()
                continue

            if color == agent_color:
                action, _ = agent.choose_action(color, explore=False)
                if action not in legal_moves:
                    action = random.choice(legal_moves)
            else:
                action = random.choice(legal_moves)

            eval_board.apply_move(color, action[0], action[1])

        winner = eval_board.winner()
        if winner == agent_color:
            wins += 1
        elif winner == eval_board.DRAW:
            draws += 1
        else:
            losses += 1

    # 切回原本訓練棋盤
    agent.board = old_board
    agent.explore = old_explore

    return wins / n_games, draws / n_games, losses / n_games


def plot_eval_progress(win_rates, eval_every):
    plt.figure(figsize=(8, 6))
    xs = [eval_every * (i + 1) for i in range(len(win_rates))]
    plt.plot(xs, win_rates, marker='o', label='Shared DQN Win Rate vs Random')
    plt.xlabel("Games")
    plt.ylabel("Win Rate")
    plt.title("Shared-Network Othello Self-Play Progress")
    plt.legend()
    plt.tight_layout()
    plt.show()


def self_play_train(iterations=5000, eval_every=100):
    board = ReversiGame()
    agent = SharedDQNAgent(board)

    eval_win_rates = []

    for game in range(iterations):
        play_one_game(board, agent, train=True)

        if (game + 1) % eval_every == 0:
            win_rate, draw_rate, loss_rate = evaluate_vs_random(agent, n_games=100)
            eval_win_rates.append(win_rate)

            print(
                f"After {game + 1} games: "
                f"win={win_rate:.2f}, draw={draw_rate:.2f}, loss={loss_rate:.2f}, "
                f"eps={agent.explore:.3f}"
            )

    agent.save_weights("shared_othello_dqn")
    plot_eval_progress(eval_win_rates, eval_every)

    return agent

def evaluate_vs_random(agent, n_games=100):
    eval_board = ReversiGame()

    old_board = agent.board
    old_explore = agent.explore

    agent.board = eval_board
    agent.explore = 0.0

    wins = 0
    draws = 0
    losses = 0

    for g in range(n_games):
        eval_board.reset()

        # 一半執黑，一半執白
        agent_color = eval_board.BLACK if g % 2 == 0 else eval_board.WHITE

        while not eval_board.is_game_over():
            color = eval_board.current_player
            legal_moves = eval_board.legal_moves(color)

            if not legal_moves:
                eval_board.auto_pass()
                continue

            if color == agent_color:
                action, _ = agent.choose_action(color, explore=False)
                if action not in legal_moves:
                    action = random.choice(legal_moves)
            else:
                action = random.choice(legal_moves)

            eval_board.apply_move(color, action[0], action[1])

        winner = eval_board.winner()
        if winner == agent_color:
            wins += 1
        elif winner == eval_board.DRAW:
            draws += 1
        else:
            losses += 1

    agent.board = old_board
    agent.explore = old_explore

    return wins / n_games, draws / n_games, losses / n_games


if __name__ == "__main__":
    self_play_train(iterations=10000, eval_every=500)
    
    board = ReversiGame()
    agent = SharedDQNAgent(board)
    agent.load_weights("shared_othello_dqn.pth")
    evaluate_vs_random(agent, n_games=1000)