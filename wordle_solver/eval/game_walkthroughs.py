import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ai_implementation"))

from constraint_satisfaction import solve as constraint_solve, load_words
from cs_splitting_heuristics import solve as heuristic_solve

DICT_PATH = "words_dictionary.json"
START_WORD = "slate"

# Test list
TARGET_WORDS = [
    "crane",
    "flint",
    "zesty",
    "abbey",
    "eerie"
]

def run_walkthroughs(word_list, all_words):
    for target in word_list:
        for solver_name, solver_fn in [("constraint", constraint_solve), ("heuristic", heuristic_solve)]:
            history = solver_fn(target, all_words, start_word=START_WORD, return_history=True)

            # Check if solved
            solved_turn = next((i + 1 for i, (_, feedback) in enumerate(history) if feedback == "ggggg"), None)
            if solved_turn:
                result_line = f"Solved in {solved_turn}"
            else:
                result_line = "Failed to solve"

            print(f"\n Target word: {target} | Solver: {solver_name} | {result_line}")
            for turn, (guess, feedback) in enumerate(history, start=1):
                print(f"  {turn}. {guess} → {feedback}")

if __name__ == "__main__":
    all_words = load_words(DICT_PATH)
    run_walkthroughs(TARGET_WORDS, all_words)
