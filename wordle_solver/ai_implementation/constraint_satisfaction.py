import json
from collections import Counter
from tqdm import tqdm

DICT_PATH = "words_dictionary.json"
START_WORD = "slate"

def load_words(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        words_dict = json.load(f)
    return sorted([w for w in words_dict if len(w) == 5])

def get_feedback(guess, target): # how similar guess was to target word and color indication
    feedback = [''] * 5
    target_chars = list(target)
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            feedback[i] = 'g'
            target_chars[i] = None
    for i, g in enumerate(guess):
        if feedback[i] == '':
            if g in target_chars:
                feedback[i] = 'y'
                target_chars[target_chars.index(g)] = None
            else:
                feedback[i] = 'b'
    return ''.join(feedback)

def add_constraints(guess, feedback, green, yellow, gray): 
    for i, (g_char, f_char) in enumerate(zip(guess, feedback)):
        if f_char == 'g':
            green[i] = g_char
        elif f_char == 'y':
            yellow.setdefault(g_char, set()).add(i)
        elif f_char == 'b':
            gray.add(g_char)
    for letter in list(gray):
        if letter in green or letter in yellow:
            gray.remove(letter)

def filter_possibilities(words, green, yellow, gray):
    filtered = []
    for w in words:
        # Green check
        if any(green[i] and w[i] != green[i] for i in range(5)):
            continue
        # Yellow check
        yellow_fail = False
        for letter, bad_pos in yellow.items(): # if yellow not in word and yellow in same position
            if letter not in w or any(w[i] == letter for i in bad_pos):
                yellow_fail = True
                break
        if yellow_fail:
            continue
        # Gray check
        if any(c in w for c in gray):
            continue
        filtered.append(w)
    return filtered

# discrete random variable
# calculate entropy of distrubtion
# 3^5 = 5 letters, g b y
# vector (243) 
# 2000 words in dict, pick word, how good is it, take vector, go through, set aside word
# go through
# 100/2000, pick word with highest entropy

def solve(target, all_words, start_word=START_WORD, max_tries=6, return_history=False):
    possible_words = list(all_words)
    green = [None] * 5
    yellow = {}
    gray = set()
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
        add_constraints(guess, feedback, green, yellow, gray)
        possible_words = filter_possibilities(possible_words, green, yellow, gray)
        if not possible_words:
            if return_history:
                return history
            else:
                return -1
        guess = sorted(possible_words)[0]
    if return_history:
        return history
    else:
        return -1

# --- simulation ---
if __name__ == "__main__":
    all_words = load_words(DICT_PATH)
    tries_counter = Counter()

    for target in tqdm(all_words, desc="Constraint Solving"):
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

