import json
import random
from collections import Counter
from tqdm import tqdm
import numpy as np
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


def compute_state(possible_words, max_log, vocab_size):
    n = len(possible_words)
    # 1. Positional letter frequency (5x26)
    pos_counts = torch.zeros(5, 26)
    # 2. Global letter frequency (26)
    global_counts = torch.zeros(26)
    # 3. Letter presence in word (26)
    presence = torch.zeros(26)
    # 4. Set size features
    log_n = torch.tensor([math.log(n + 1) / max_log])
    rel_n = torch.tensor([n / vocab_size])
    
    for w in possible_words:
        for i, ch in enumerate(w):
            idx = ord(ch) - 97
            pos_counts[i, idx] += 1
            global_counts[idx] += 1
        # Track unique letters in word
        for ch in set(w):
            presence[ord(ch) - 97] += 1
    
    # Normalize
    pos_counts /= n
    global_counts /= (n * 5)
    presence /= n
    
    return torch.cat([pos_counts.view(-1), global_counts, presence, log_n, rel_n])


def score_entropy(guess: str, possible: list[str]) -> float:
    total = len(possible)
    if total == 0:
        return 0.0
    counts = Counter(get_feedback(guess, w) for w in possible)
    return -sum((cnt / total) * math.log2(cnt / total + 1e-10)
                for cnt in counts.values() if cnt > 0)

# --- model definitions ---

class PolicyNet(nn.Module):
    def __init__(self, state_dim, vocab_size, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, state, mask):
        logits = self.net(state)
        logits[~mask] = -1e9
        return Categorical(logits=logits)


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# --- training & evaluation ---

def train_actor_critic(all_words, episodes=2000, lr=1e-3, gamma=0.99, print_interval=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = len(all_words)
    max_log = math.log(V + 1)
    state_dim = 5*26 + 26 + 26 + 1 + 1  # positional + global + presence + log_size + relative_size

    actor = PolicyNet(state_dim, V).to(device)
    critic = CriticNet(state_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1000, factor=0.5)

    word2idx = {w: i for i, w in enumerate(all_words)}

    best_solve_rate = 0.0
    solved_count = 0
    sum_guesses = 0
    solve_rates = []

    for ep in range(1, episodes + 1):
        target = random.choice(all_words)
        possible = all_words.copy()
        log_probs, values, rewards = [], [], []
        entropy_list = []

        for step in range(6):
            state = compute_state(possible, max_log, V).to(device)
            mask = torch.zeros(V, dtype=torch.bool, device=device)
            for w in possible:
                mask[word2idx[w]] = True

            dist = actor(state.unsqueeze(0), mask.unsqueeze(0))
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device))
            value = critic(state.unsqueeze(0))
            entropy = dist.entropy()
            entropy_list.append(entropy)

            guess = all_words[action]
            fb = get_feedback(guess, target)
            
            # Calculate information gain
            entropy_before = score_entropy(guess, possible)
            possible_next = filter_words(possible, guess, fb)
            entropy_after = score_entropy(guess, possible_next) if possible_next else 0
            info_gain = entropy_before - entropy_after

            # Strategic reward structure
            if fb == "ggggg":
                # Higher reward for fewer steps
                r = 10.0 * (1 - step/6)
            elif len(possible_next) == 0:
                # Penalize dead ends
                r = -5.0
            else:
                # Reward information gain and solution progress
                r = 2.0 * info_gain + 0.5 * (len(possible) - len(possible_next)) / len(possible)
                
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(r)
            
            # Update possible words
            possible = possible_next
            
            if fb == "ggggg" or not possible:
                break

        # Update metrics
        if rewards and fb == "ggggg":
            solved_count += 1
            sum_guesses += len(rewards)
        pct_solved = solved_count / ep
        solve_rates.append(pct_solved)
        avg_guesses = sum_guesses / solved_count if solved_count > 0 else 0.0

        # Compute returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)
        values = torch.stack(values)
        advantages = returns - values.detach()

        # Loss calculation with entropy bonus
        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = torch.stack(entropy_list).mean() * 0.1  # Encourage exploration
        loss = actor_loss + critic_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        
        # Update learning rate based on performance
        if ep % 1000 == 0:
            scheduler.step(pct_solved)
            if pct_solved > best_solve_rate:
                best_solve_rate = pct_solved
                # Save best model
                torch.save(actor.state_dict(), f"best_actor_{pct_solved:.2f}.pth")

        if ep % print_interval == 0:
            print(f"Episode {ep}/{episodes}, Loss: {loss.item():.4f}, "
                  f"Solve Rate: {pct_solved:.2%}, "
                  f"Avg Guesses: {avg_guesses:.2f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    return actor, critic, solve_rates


def evaluate_policy(actor, all_words, word2idx, max_log):
    device = next(actor.parameters()).device
    V = len(all_words)
    tries = Counter()
    fails = []
    
    for target in tqdm(all_words, desc="Evaluating"):
        possible = all_words.copy()
        solved = False
        
        for step in range(6):
            state = compute_state(possible, max_log, V).to(device)
            mask = torch.zeros(V, dtype=torch.bool, device=device)
            for w in possible:
                mask[word2idx[w]] = True
            
            with torch.no_grad():
                dist = actor(state.unsqueeze(0), mask.unsqueeze(0))
                action = dist.probs.argmax().item()
            
            guess = all_words[action]
            fb = get_feedback(guess, target)
            
            if fb == "ggggg":
                tries[step+1] += 1
                solved = True
                break
                
            possible = filter_words(possible, guess, fb)
            if not possible:
                break
        
        if not solved:
            tries['fail'] += 1
            fails.append(target)

    total = sum(tries.values())
    print("\n=== Evaluation Results ===")
    for k in [1,2,3,4,5,6,'fail']:
        count = tries.get(k, 0)
        print(f"Solved in {k}: {count} ({count/total:.2%})")
    
    solve_rate = 1 - (tries.get('fail', 0) / total)
    print(f"\nSolve Rate: {solve_rate:.2%}")
    
    if fails:
        print(f"\nFailed words ({len(fails)}):")
        print(", ".join(fails[:min(20, len(fails))]) + ("..." if len(fails) > 20 else ""))
    
    return solve_rate


if __name__ == "__main__":
    # Load and prepare words
    words = load_words("valid_wordle_words.json")
    print(f"Loaded {len(words)} words")
    
    # Train the model
    actor, critic, solve_rates = train_actor_critic(words)
    
    # Evaluate final performance
    word2idx = {w: i for i, w in enumerate(words)}
    max_log = math.log(len(words) + 1)
    solve_rate = evaluate_policy(actor, words, word2idx, max_log)
    
    # Save final model if it achieves 100% solve rate
    if solve_rate >= 0.999:
        torch.save(actor.state_dict(), "wordle_solver_100pct.pth")
        print("100% solve rate achieved! Model saved.")