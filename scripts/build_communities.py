from __future__ import annotations

import argparse
from pathlib import Path

from src.graph import load_or_build_communities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline SCB spectral communities.")
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--communities", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ids = load_or_build_communities(args.adjacency, args.output, args.communities, args.seed)
    print(f"saved {len(ids)} community IDs to {Path(args.output)}")
