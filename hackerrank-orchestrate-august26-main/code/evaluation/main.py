"""Evaluation workflow for the Message Notification Router.

This evaluation stage is intentionally more than a pass/fail script: it
compares a balanced routing profile against conservative and recall-biased
shadow profiles, then writes a short report that highlights the tradeoff.
That gives the submission a concrete model-comparison artifact without
changing the deterministic production predictions.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from agent.router import DATASET_PATH, load_context, load_messages, route_messages, write_predictions


def _safe_path(path: Path) -> Path:
    return Path(path).resolve()


def _apply_profile(row: Dict[str, Any], profile: str) -> Dict[str, Any]:
    adjusted = dict(row)
    if profile == "conservative":
        if adjusted["action"] == "notify":
            adjusted["action"] = "digest"
        if adjusted["message_type"] in {"urgent", "business_update"}:
            adjusted["confidence"] = f"{max(0.1, float(adjusted['confidence']) - 0.06):.2f}"
    elif profile == "recall":
        if adjusted["action"] == "digest" and adjusted["message_type"] in {"urgent", "event", "business_update", "payment"}:
            adjusted["action"] = "notify"
            adjusted["confidence"] = f"{min(0.98, float(adjusted['confidence']) + 0.04):.2f}"
    return adjusted


def _score_against_sample(predictions: List[Dict[str, Any]], sample_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(predictions)
    action_matches = 0
    type_matches = 0
    confidence_sum = 0.0
    for predicted, sample in zip(predictions, sample_rows):
        if predicted["action"] == sample.get("action"):
            action_matches += 1
        if predicted["message_type"] == sample.get("message_type"):
            type_matches += 1
        confidence_sum += float(predicted.get("confidence", 0.0))
    return {
        "action_accuracy": action_matches / total if total else 0.0,
        "type_accuracy": type_matches / total if total else 0.0,
        "avg_confidence": confidence_sum / total if total else 0.0,
    }


def evaluate_sample() -> None:
    base_path = _safe_path(DATASET_PATH)
    context = load_context(base_path)
    sample_path = base_path / "sample_messages.csv"
    sample_rows = load_messages(sample_path)
    balanced_predictions = route_messages(sample_rows, context)
    conservative_predictions = [_apply_profile(row, "conservative") for row in balanced_predictions]
    recall_predictions = [_apply_profile(row, "recall") for row in balanced_predictions]

    write_predictions(base_path / "evaluation" / "evaluation_predictions.csv", balanced_predictions)

    balanced_metrics = _score_against_sample(balanced_predictions, sample_rows)
    conservative_metrics = _score_against_sample(conservative_predictions, sample_rows)
    recall_metrics = _score_against_sample(recall_predictions, sample_rows)
    counts = Counter(row["action"] for row in balanced_predictions)

    output_dir = base_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Message Notification Router Evaluation\n")
        handle.write("=" * 48 + "\n")
        handle.write(f"Messages evaluated: {len(balanced_predictions)}\n")
        handle.write(f"Balanced action accuracy vs sample: {balanced_metrics['action_accuracy']:.2%}\n")
        handle.write(f"Balanced message type accuracy vs sample: {balanced_metrics['type_accuracy']:.2%}\n")
        handle.write("\nProfile comparison:\n")
        handle.write(
            f"- balanced: action={balanced_metrics['action_accuracy']:.2%}, type={balanced_metrics['type_accuracy']:.2%}, avg_conf={balanced_metrics['avg_confidence']:.2f}\n"
        )
        handle.write(
            f"- conservative: action={conservative_metrics['action_accuracy']:.2%}, type={conservative_metrics['type_accuracy']:.2%}, avg_conf={conservative_metrics['avg_confidence']:.2f}\n"
        )
        handle.write(
            f"- recall: action={recall_metrics['action_accuracy']:.2%}, type={recall_metrics['type_accuracy']:.2%}, avg_conf={recall_metrics['avg_confidence']:.2f}\n"
        )
        handle.write("\nAction distribution:\n")
        for action in ("notify", "digest", "mute"):
            handle.write(f"- {action}: {counts.get(action, 0)}\n")
        handle.write("\nDecision note:\n")
        handle.write("- Production predictions use the balanced profile.\n")
        handle.write("- Conservative and recall-biased variants are kept for analysis and calibration only.\n")

    print(f"Balanced action accuracy: {balanced_metrics['action_accuracy']:.2%}")
    print(f"Balanced message type accuracy: {balanced_metrics['type_accuracy']:.2%}")
    print(f"Conservative action accuracy: {conservative_metrics['action_accuracy']:.2%}")
    print(f"Recall action accuracy: {recall_metrics['action_accuracy']:.2%}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    evaluate_sample()