import subprocess
import re
import csv

# Embedding types to test
EMBEDDINGS = ["word2vec", "glove", "fasttext", "multi"]

def run_contrastive_experiment(embedding_type):
    """
    Runs contrastive_train.py for a given embedding type and extracts accuracy metrics.
    """
    print(f"\nRunning contrastive_train.py with --embedding_type {embedding_type} ...")

    # Run the contrastive training script as subprocess
    process = subprocess.Popen(
        ["python", "contrastive_train.py", "--embedding_type", embedding_type, "--debug", "--num_epochs", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate()

    # Initialize metric holders
    train_acc = None
    val_acc = None
    f1_score_val = None


    # Parse output
    for line in out.splitlines():
        match = re.search(r"final training accuracy\s*:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            train_acc = float(match.group(1))

        match = re.search(r"final validation accuracy\s*:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            val_acc = float(match.group(1))

        match = re.search(r"final f1 score\s*:\s*([\d\.]+)", line, re.IGNORECASE)
        if match:
            f1_score_val = float(match.group(1))


    return {
        "embedding": embedding_type,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "f1_score": f1_score_val,
        "raw_output": out,
        "raw_error": err
    }


def main():
    results = []

    for emb_type in EMBEDDINGS:
        result = run_contrastive_experiment(emb_type)
        results.append(result)

    # Save results
    csv_file = "contrastive_results.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Embedding Type", "Train Accuracy", "Validation Accuracy", "F1 Score"])
        for row in results:
            writer.writerow([row["embedding"], row["train_accuracy"], row["val_accuracy"], row["f1_score"]])

    print(f"\n✅ All contrastive experiments done! Results saved to {csv_file}")
    print("\n--- Summary ---")
    for r in results:
        print(f"{r['embedding']}: Train Acc={r['train_accuracy']}, Val Acc={r['val_accuracy']}")

if __name__ == "__main__":
    main()
