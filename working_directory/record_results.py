import subprocess
import re
import csv

# Embedding types to test
EMBEDDINGS = ["word2vec", "glove", "fasttext", "multi"]

def run_experiment(embedding_type):
    """
    Runs train.py with a specified embedding type and captures the console output.
    Returns a dictionary of parsed metrics like Accuracy and F1 Score.
    """
    print(f"\nRunning train.py with --embedding_type {embedding_type} ...")

    # Run train.py as a subprocess
    process = subprocess.Popen(
        ["python", "train.py", "--embedding_type", embedding_type, "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate()

    # Initialize placeholders for metrics
    accuracy = None
    f1_score_val = None

    # Parse the output line by line to find the lines containing metrics
    for line in out.splitlines():
        # Example line we might see: "Accuracy: 0.9234"
        # Updated block (handles lowercase and different formatting)
        if re.search(r"accuracy[:\s]", line, re.IGNORECASE):
            match = re.search(r"accuracy[:\s]*([\d\.]+)", line, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))


        # Example line we might see: "F1 Score: 0.9213"
        if "F1 Score:" in line:
            match = re.search(r"F1 Score:\s*([\d\.]+)", line)
            if match:
                f1_score_val = float(match.group(1))

    # Return the captured metrics
    return {
        "embedding": embedding_type,
        "accuracy": accuracy,
        "f1_score": f1_score_val,
        "raw_output": out,
        "raw_error": err
    }

def main():
    # List to store results from each run
    results = []

    # Run the experiment for each embedding type
    for emb_type in EMBEDDINGS:
        metrics = run_experiment(emb_type)
        results.append(metrics)

    # Write results to a CSV file
    csv_filename = "results_summary.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Embedding Type", "Accuracy", "F1 Score"])
        for row in results:
            writer.writerow([row["embedding"], row["accuracy"], row["f1_score"]])

    print(f"\nAll experiments finished! Results saved to {csv_filename}")

    # (Optional) Print summary in console
    print("\n--- Summary of Runs ---")
    for r in results:
        print(f"{r['embedding']}: Accuracy={r['accuracy']}, F1={r['f1_score']}")

if __name__ == "__main__":
    main()