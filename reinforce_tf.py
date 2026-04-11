import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_EVERY = 200
ACTIONS_SIZE = 64


class Experience:
    def __init__(self, state, action, reward, next_state, done, next_valid_actions):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.next_valid_actions = next_valid_actions


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),   # [B,1,64] -> [B,32,64]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # [B,64,64]
            nn.ReLU(),
            nn.Flatten(),                                 # [B,4096]
            nn.Linear(64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, ACTIONS_SIZE)
        )

    def forward(self, x):
        return self.net(x)


class SharedDQNAgent:
    def __init__(self, board, lr=1e-3, explore=1.0, explore_min=0.05, explore_decay=0.9995):
        self.board = board
        self.explore = explore
        self.explore_min = explore_min
        self.explore_decay = explore_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNet().to(self.device)
        self.target_model = QNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_memory = []
        self.train_steps = 0
        self.loss_list = []

    def state_to_tensor(self, state_batch):
        arr = np.array(state_batch, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, 1, 64)
        elif arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], 1, 64)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def choose_action(self, color, explore=True):
        valid_actions = self.board.legal_action_indices(color)

        if not valid_actions:
            return self.board.PASS_MOVE, 0.0

        # 關鍵：canonical state，讓同一個網路學黑白雙方
        state = self.board.canonical_state(color)
        state_tensor = self.state_to_tensor(state)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()

        if explore and random.random() < self.explore:
            act = random.choice(valid_actions)
        else:
            act = max(valid_actions, key=lambda a: q_values[a])

        row = act // self.board.size
        col = act % self.board.size
        return (row, col), float(q_values[act])

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        if action == self.board.PASS_MOVE:
            action_idx = -1
        else:
            action_idx = action[0] * self.board.size + action[1]

        self.replay_memory.append(
            Experience(
                state.copy(),
                action_idx,
                float(reward),
                next_state.copy(),
                bool(done),
                list(next_valid_actions),
            )
        )

        if len(self.replay_memory) > MEMORY_SIZE:
            self.replay_memory.pop(0)

    def train(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        states = np.array([exp.state for exp in batch], dtype=np.float32)
        actions = [exp.action for exp in batch]
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        next_states = np.array([exp.next_state for exp in batch], dtype=np.float32)

        states_t = self.state_to_tensor(states)
        next_states_t = self.state_to_tensor(next_states)

        q_values = self.model(states_t)

        q_selected = []
        for i, a in enumerate(actions):
            if a == -1:
                q_selected.append(torch.tensor(0.0, device=self.device))
            else:
                q_selected.append(q_values[i, a])
        q_selected = torch.stack(q_selected)

        with torch.no_grad():
            q_next_all = self.target_model(next_states_t)
            q_targets = []

            for i, exp in enumerate(batch):
                if exp.done or len(exp.next_valid_actions) == 0:
                    q_targets.append(rewards[i])
                else:
                    # 注意：下一手是對手走，所以是零和關係
                    next_best = torch.max(q_next_all[i, exp.next_valid_actions])
                    q_targets.append(rewards[i] - GAMMA * next_best)

            q_targets = torch.stack(q_targets)

        loss = self.loss_fn(q_selected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_list.append(float(loss.item()))
        self.train_steps += 1

        if self.train_steps % TARGET_UPDATE_EVERY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.explore > self.explore_min:
            self.explore = max(self.explore_min, self.explore * self.explore_decay)

    def save_weights(self, filename):
        if not filename.endswith(".pth"):
            filename += ".pth"
        torch.save(self.model.state_dict(), filename)

    def load_weights(self, filename):
        if not filename.endswith(".pth"):
            filename += ".pth"
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())