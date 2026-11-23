"""Interrupt the named document at the batch boundary nearest its midpoint."""

import sys

from benchmark.robustness import main


if __name__ == "__main__":
    main([
        "build", "--embedding-batch-size", "16", *sys.argv[1:],
        "--offline", "--count-embedding-tokens", "--embedding-midpoints",
        "--crash-stage", "embedding-mid-file",
    ])
