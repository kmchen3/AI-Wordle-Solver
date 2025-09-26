import json
import math
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# -----------------------------------------------------------------------------
# ——— Wordle utilities (shared by all workers) ——————————————————————————
# -----------------------------------------------------------------------------

def load_words(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    return sorted([w for w in words if len(w) == 5])


@lru_cache(maxsize=None)
def feedback_cached(guess: str, target: str) -> str:
    fb = [''] * 5
    tchars = list(target)
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            fb[i] = 'g'
            tchars[i] = None
    for i, g in enumerate(guess):
        if fb[i] == '':
            if g in tchars:
                fb[i] = 'y'
                tchars[tchars.index(g)] = None
            else:
                fb[i] = 'b'
    return ''.join(fb)


def score_entropy(guess: str, possible: list[str]) -> float:
    total = len(possible)
    counts = Counter(feedback_cached(guess, w) for w in possible) # counts each word that would give the same feedback
    return -sum((cnt/total) * math.log2(cnt/total) # entropy equation
                for cnt in counts.values() if cnt > 0)


def pick_best_guess(possible: list[str], all_guesses: list[str]) -> str:
    best, best_score = None, -1.0
    possible_set = set(possible)
    for g in all_guesses:
        h = score_entropy(g, possible)
        if h > best_score + 1e-9:
            best_score, best = h, g
        elif abs(h - best_score) < 1e-9 and best not in possible_set and g in possible_set:
            best = g
    return best


# -----------------------------------------------------------------------------
# ——— Solver & worker setup ——————————————————————————————
# -----------------------------------------------------------------------------

SOLUTIONS: list[str]
GUESSES:   list[str]

def init_worker(solutions: list[str], guesses: list[str]):
    global SOLUTIONS, GUESSES
    SOLUTIONS = solutions
    GUESSES   = guesses


def solve_for_secret(secret: str, starter: str = "slate") -> tuple[str,int]:
    possible = SOLUTIONS.copy()
    guess    = starter
    for turn in range(1, 7):
        fb = feedback_cached(guess, secret)
        if fb == "ggggg":
            return secret, turn
        possible = [w for w in possible if feedback_cached(guess, w) == fb] # if its like slime, and slate, cuz sl--e IF those 3 were green
        pool = GUESSES if len(possible) > 2 else possible 
        guess = pick_best_guess(possible, pool)
    return secret, 0


# -----------------------------------------------------------------------------
# ——— Main: parallel solve with live tqdm feedback ——————————————————————
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    solutions = load_words("prev_wordle_words.json")
    guesses   = solutions

    total     = len(solutions)
    solved    = 0
    sum_turns = 0
    failures  = []

    with Pool(processes=cpu_count(),
              initializer=init_worker,
              initargs=(solutions, guesses)) as pool:

        # get a tqdm instance
        pbar = tqdm(pool.imap_unordered(solve_for_secret, solutions),
                    total=total,
                    desc="Solving",
                    unit="word")

        for idx, (secret, turns) in enumerate(pbar, start=1):
            if turns:
                solved    += 1
                sum_turns += turns
            else:
                failures.append(secret)

            avg_turns    = sum_turns / solved if solved else 0
            success_rate = solved / idx

            # update the bar's postfix with running stats
            pbar.set_postfix({
                "succ_rate": f"{success_rate:.2%}",
                "avg_turns": f"{avg_turns:.2f}"
            })

    print(f"\nSolved   : {solved}/{total} words  ({solved/total*100:.2f}%)")
    print(f"Avg turns: {avg_turns:.2f}  (only counting solved)")
    if failures:
        print(f"Failures ({len(failures)}): {failures[:10]}")
