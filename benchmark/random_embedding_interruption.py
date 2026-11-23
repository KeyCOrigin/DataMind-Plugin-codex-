"""Run the offline build experiment with a 20% random embedding interruption.

Each completed batch with work remaining in the same file is eligible.
Only the first build attempt draws faults. Recovery workers run with no draws.
"""
import sys

from benchmark.robustness import main


if __name__ == "__main__":
    main([
        "build", "--embedding-batch-size", "1", *sys.argv[1:],
        "--offline", "--count-embedding-tokens",
        "--crash-stage", "embedding-mid-file", "--embedding-fault-probability", "0.2",
    ])
