#!/usr/bin/env python3
"""Main entry point for the Message Notification Router."""

from pathlib import Path
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent))

from agent.router import DATASET_PATH, load_context, load_messages, route_messages, write_predictions


def main() -> None:
    dataset_path = Path(DATASET_PATH)
    context = load_context(dataset_path)
    messages = load_messages(dataset_path / "messages.csv")
    predictions = route_messages(messages, context)
    write_predictions(dataset_path / "output.csv", predictions)

    counts = Counter(row["action"] for row in predictions)
    total = len(predictions)
    print(f"Processed {total} messages")
    for action in ("notify", "digest", "mute"):
        count = counts.get(action, 0)
        share = (count / total * 100.0) if total else 0.0
        print(f"{action}: {count} ({share:.1f}%)")


if __name__ == "__main__":
    main()