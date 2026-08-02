"""Evaluation workflow for the Message Notification Router."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agent.router import DATASET_PATH, load_context, load_messages, route_messages, write_predictions


def _safe_path(path: Path) -> Path:
    return Path(path).resolve()


def evaluate_sample() -> None:
    base_path = _safe_path(DATASET_PATH)
    context = load_context(base_path)
    sample_path = base_path / "sample_messages.csv"
    sample_rows = load_messages(sample_path)
    predictions = route_messages(sample_rows, context)

    write_predictions(base_path / "evaluation" / "evaluation_predictions.csv", predictions)

    total = len(predictions)
    action_matches = 0
    type_matches = 0
    for predicted, sample in zip(predictions, sample_rows):
        if predicted["action"] == sample.get("action"):
            action_matches += 1
        if predicted["message_type"] == sample.get("message_type"):
            type_matches += 1

    action_accuracy = action_matches / total if total else 0.0
    type_accuracy = type_matches / total if total else 0.0
    counts = Counter(row["action"] for row in predictions)

    output_dir = base_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Message Notification Router Evaluation\n")
        handle.write("=" * 48 + "\n")
        handle.write(f"Messages evaluated: {total}\n")
        handle.write(f"Action accuracy vs sample: {action_accuracy:.2%}\n")
        handle.write(f"Message type accuracy vs sample: {type_accuracy:.2%}\n")
        handle.write("\nAction distribution:\n")
        for action in ("notify", "digest", "mute"):
            handle.write(f"- {action}: {counts.get(action, 0)}\n")

    print(f"Action accuracy: {action_accuracy:.2%}")
    print(f"Message type accuracy: {type_accuracy:.2%}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    evaluate_sample()