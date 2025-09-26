import pandas as pd
import matplotlib.pyplot as plt
import os

# This function plots the distribution of the number of tries it took to guess the word
def plot_distribution(csv_path, title, output_filename):
    df = pd.read_csv(csv_path)
    df["Tries"] = df["Tries"].astype(str)

    plt.figure(figsize=(8, 5))
    plt.bar(df["Tries"], df["Count"])
    plt.xlabel("Number of Tries")
    plt.ylabel("Number of Words Solved")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    constraint_path_results = os.path.join(base_dir, "constraint_results.csv")
    heuristic_path_results = os.path.join(base_dir, "heuristic_results.csv")

    constraint_path_figure = os.path.join(base_dir, "constraint_plot.png")
    heuristic_path_figure = os.path.join(base_dir, "heuristic_plot.png")

    plot_distribution(constraint_path_results, "Constraint Satisfaction Guess Distribution", constraint_path_figure)
    plot_distribution(heuristic_path_results, "Heuristic Solver Guess Distribution", heuristic_path_figure)
