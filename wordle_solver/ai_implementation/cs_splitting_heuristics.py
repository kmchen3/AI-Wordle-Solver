import json
from collections import Counter
from tqdm import tqdm

DICT_PATH = "prev_wordle_words.json"
START_WORD = "slate"

def load_words(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        words_dict = json.load(f)
    return sorted([w for w in words_dict if len(w) == 5])

def get_feedback(guess, target):
    """generates guess feedback (returns a string of 5 letters (e.g. ggyby))"""
    feedback = [''] * 5
    target_chars = list(target)
    # greens
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 'g'
            target_chars[i] = None
    # yellows and blanks
    for i, g in enumerate(guess):
        if feedback[i] == '':
            if g in target_chars:
                feedback[i] = 'y'
                target_chars[target_chars.index(g)] = None
            else:
                feedback[i] = 'b'
    return ''.join(feedback)

def filter_words(words, guess, feedback):
    """Return list of words that match the feedback constraints."""
    filtered = []
    for word in words:
        if get_feedback(guess, word) == feedback:
            filtered.append(word)
    return filtered

def score_words(words):
    """Score words based on frequency of letters across candidates."""
    letter_counts = Counter("".join(words))
    def score(word):
        return sum(letter_counts[c] for c in set(word))
    return sorted(words, key=score, reverse=True)

def solve(target, all_words, start_word=START_WORD, max_tries=6, return_history=False):
    possible_words = list(all_words)
    guess = start_word
    tries = 0
    history = []  # stores (guess, feedback)

    while tries < max_tries:
        tries += 1
        feedback = get_feedback(guess, target)
        history.append((guess, feedback))
        if feedback == "ggggg":
            if return_history:
                return history
            else:
                return tries
        possible_words = filter_words(possible_words, guess, feedback)
        if not possible_words:
            if return_history:
                return history
            else:
                return -1
        # choose word with best letter frequency score
        scored = score_words(possible_words)
        guess = scored[0]
    if return_history:
        return history
    else:
        return -1 

# --- simulation ---
if __name__ == "__main__":
    all_words = load_words(DICT_PATH)
    tries_counter = Counter()

    for target in tqdm(all_words, desc="Heuristic Solving"):
        result = solve(target, all_words, start_word=START_WORD)
        if result == -1:
            tries_counter["fail"] += 1
        else:
            tries_counter[result] += 1

    total = sum(tries_counter.values())
    print(f"\nSimulation complete: {total} words.")
    for i in range(1, 7):
        print(f"  Solved in {i} try{'es' if i > 1 else ''}: {tries_counter.get(i, 0)}")
    print(f"  Failed to solve in 6 tries: {tries_counter.get('fail', 0)}")
    solved_total = sum(tries_counter[i] for i in range(1, 7))
    avg_tries = sum(i * tries_counter[i] for i in range(1, 7)) / solved_total
    print(f"Average tries (excluding failures): {avg_tries:.2f}")

