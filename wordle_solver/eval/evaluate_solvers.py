import sys
import os
import csv
from collections import Counter
from tqdm import tqdm

# Ensure the ai_implementation directory is in the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ai_implementation"))

from constraint_satisfaction import solve as constraint_solve, load_words
from cs_splitting_heuristics import solve as heuristic_solve

DICT_PATH = os.path.join(os.path.dirname(__file__), "..", "words_dictionary.json")
MAX_TRIES = 6

def run_solver(solver_fn, all_words):
    tries_counter = Counter()
    for target in tqdm(all_words, desc=f"Running {solver_fn.__name__}"):
        result = solver_fn(target, all_words)
        if result == -1:
            tries_counter["fail"] += 1
        else:
            tries_counter[result] += 1
    return tries_counter

def save_results_to_csv(filename, tries_counter):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Tries", "Count"])
        for i in range(1, MAX_TRIES + 1):
            writer.writerow([i, tries_counter.get(i, 0)])
        writer.writerow(["fail", tries_counter.get("fail", 0)])

# Computes the average number of tries only for successful attempts
# (i.e., those that did not fail)
def compute_average_tries(tries_counter):
    num_solved = sum(tries_counter[i] for i in range(1, MAX_TRIES + 1))
    if num_solved == 0:
        return float("inf")
    total_tries = sum(i * tries_counter[i] for i in range(1, MAX_TRIES + 1))
    return total_tries / num_solved

if __name__ == "__main__":
    all_words = load_words(DICT_PATH)
    all_words_cs = all_words
    all_words_heur = all_words

    cs_results = run_solver(constraint_solve, all_words_cs)
    heur_results = run_solver(heuristic_solve, all_words_heur)

    # Save results in same directory as this script
    base_dir = os.path.dirname(__file__)
    cs_results_path = os.path.join(base_dir, "constraint_results.csv")
    heur_results_path = os.path.join(base_dir, "heuristic_results.csv")
    save_results_to_csv(cs_results_path, cs_results)
    save_results_to_csv(heur_results_path, heur_results)

    print("\n--- Evaluation Summary ---")
    print("Constraint Satisfaction:")
    print(f"  Avg Tries: {compute_average_tries(cs_results):.2f}")
    print(f"  Failures: {cs_results.get('fail', 0)}")
    print("Heuristic:")
    print(f"  Avg Tries: {compute_average_tries(heur_results):.2f}")
    print(f"  Failures: {heur_results.get('fail', 0)}")
