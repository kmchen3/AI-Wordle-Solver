import json
import random
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import math

# --- utilities ---

def load_words(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        wd = json.load(f)
    return sorted([w for w in wd if len(w) == 5])


def get_feedback(guess, target):
    feedback = [''] * 5
    tchars = list(target)
    # greens
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 'g'
            tchars[i] = None
    # yellows and blanks
    for i, g in enumerate(guess):
        if feedback[i] == '':
            if g in tchars:
                feedback[i] = 'y'
                tchars[tchars.index(g)] = None
            else:
                feedback[i] = 'b'
    return ''.join(feedback)


def filter_words(words, guess, fb):
    return [w for w in words if get_feedback(guess, w) == fb]


def compute_state(possible_words):
    # 130-dim positional letter frequency
    pos_counts = torch.zeros(5, 26)
    for w in possible_words:
        for i, ch in enumerate(w):
            pos_counts[i, ord(ch) - 97] += 1
    pos_counts /= len(possible_words)
    return pos_counts.view(-1)

# --- model definitions ---

class PolicyNet(nn.Module):
    def __init__(self, state_dim, vocab_size, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, state, mask):
        # state: (B, state_dim), mask: (B, vocab_size)
        logits = self.net(state)  # (B, V)
        logits[~mask] = -1e9
        return Categorical(logits=logits)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # state: (B, state_dim)
        return self.net(state).squeeze(-1)  # (B,)


# --- training & evaluation ---

def train_actor_critic(all_words, episodes=2000, lr=1e-3, gamma=0.99, print_interval=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = len(all_words)
    state_dim = 5 * 26

    actor = PolicyNet(state_dim, V).to(device)
    critic = CriticNet(state_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    word2idx = {w: i for i, w in enumerate(all_words)}

    running_avg = 0.0
    solved_count = 0
    sum_guesses = 0

    for ep in range(1, episodes + 1):
        target = random.choice(all_words)
        possible = all_words.copy()
        log_probs, values, rewards = [], [], []

        # rollout
        for step in range(6):
            state = compute_state(possible).to(device)
            mask = torch.zeros(V, dtype=torch.bool, device=device)
            for w in possible:
                mask[word2idx[w]] = True

            dist = actor(state.unsqueeze(0), mask.unsqueeze(0))
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device))
            value = critic(state.unsqueeze(0))

            guess = all_words[action]
            fb = get_feedback(guess, target)

            ######### REWARD #########
            green = fb.count('g')
            yellow = fb.count('y')

            # Info gain
            before = len(possible)
            possible = filter_words(possible, guess, fb)
            after = len(possible)
            eliminated = before - after
            info_reward = (eliminated / before) ** 2 if before > 0 else 0.0

            # Solved reward
            solved_bonus = 0
            if fb == "ggggg":
                solved_bonus = 2.0 + ((6 - step) ** 2) / 18.0  # stronger boost for faster solve

            r = 0.3 * green + 0.1 * yellow + 2.0 * info_reward + solved_bonus

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(r)

            if fb == "ggggg":
                break

        # update metrics
        if fb == "ggggg":
            solved_count += 1
            sum_guesses += len(rewards)

        pct_solved = solved_count / ep
        avg_guesses = sum_guesses / solved_count if solved_count > 0 else 0.0

        # compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        values = torch.stack(values)
        advantages = returns - values.detach()

        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running avg reward
        running_avg = 0.99 * running_avg + 0.01 * sum(rewards)

        if ep % print_interval == 0:
            print(
                f"Episode {ep}/{episodes}, loss = {loss.item():.4f}, "
                f"avg reward ≈ {running_avg:.3f}, "
                f"solve rate = {pct_solved:.2%}, "
                f"avg guesses = {avg_guesses:.2f}"
            )

    return actor, critic


def evaluate_policy(actor, all_words, word2idx):
    V = len(all_words)
    tries = Counter()
    for target in tqdm(all_words, desc="Eval"):
        possible = all_words.copy()
        for t in range(1, 7):
            state = compute_state(possible)
            mask = torch.zeros(V, dtype=torch.bool)
            for w in possible:
                mask[word2idx[w]] = True
            dist = actor(state.unsqueeze(0), mask.unsqueeze(0))
            action = dist.probs.argmax().item()
            guess = all_words[action]
            fb = get_feedback(guess, target)
            if fb == "ggggg":
                tries[str(t)] += 1
                break
            possible = filter_words(possible, guess, fb)
        else:
            tries['fail'] += 1

    total = sum(tries.values())
    for k in sorted(tries):
        print(f"Solved in {k}: {tries[k]} ({tries[k]/total:.2%})")


if __name__ == "__main__":
    words = load_words("valid_wordle_words.json")
    actor_model, critic_model = train_actor_critic(words)
    word2idx = {w: i for i, w in enumerate(words)}
    evaluate_policy(actor_model, words, word2idx)
