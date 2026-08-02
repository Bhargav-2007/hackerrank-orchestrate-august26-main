"""
Evaluation module for the Message Notification Router.
Compares different strategies and outputs metrics.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# Add the code directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from agent.enricher import enrich_message
from agent.features import extract_features
from agent.classifier import classify_message
from agent.evidence import get_evidence_and_confidence
from utils.logging import setup_logging
from utils.config import DATASET_PATH

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def load_csv_as_dict(file_path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    """Load a CSV file into a dictionary keyed by the specified field."""
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get(key_field)
                if key:
                    data[key] = row
        logger.info(f"Loaded {len(data)} records from {file_path} keyed by {key_field}")
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
    return data

def load_messages(file_path: Path) -> List[Dict[str, Any]]:
    """Load messages from CSV file."""
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                messages.append(row)
        logger.info(f"Loaded {len(messages)} messages from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load messages from {file_path}: {e}")
        sys.exit(1)
    return messages

def process_message(
    message: Dict[str, Any],
    users: dict,
    groups: dict,
    group_members: dict,
    businesses: dict,
    user_business_history: dict,
    images: dict,
    voice_notes: dict,
    message_history: dict
) -> Dict[str, Any]:
    """
    Process a single message through the pipeline.
    
    Returns a dictionary with the results.
    """
    try:
        # Step 1: Enrich message with context
        enriched = enrich_message(message, users, groups, group_members, businesses, user_business_history, images, voice_notes)

        # Step 2: Extract features
        features = extract_features(enriched)

        # Step 3: Classify message (get action and rule-based confidence)
        action, rule_confidence = classify_message(features)

        # Step 4: Get evidence and adjust confidence
        evidence_ids, final_confidence = get_evidence_and_confidence(
            message, enriched, features, action, rule_confidence, message_history
        )

        return {
            "message_id": message["message_id"],
            "action": action,
            "confidence": f"{final_confidence:.2f}",
            "evidence_message_ids": " ".join(evidence_ids) if evidence_ids else "none"
        }
    except Exception as e:
        logger.error(f"Error processing message {message.get('message_id', 'unknown')}: {e}")
        return {
            "message_id": message.get("message_id", "unknown"),
            "action": "mute",
            "confidence": "0.0",
            "evidence_message_ids": "none"
        }

def run_strategy(
    messages: List[Dict[str, Any]],
    ref_data: Dict[str, Dict[str, Any]],
    strategy_name: str
) -> List[Dict[str, Any]]:
    """
    Process all messages with a given strategy.
    For now, we only have one strategy (the default classifier).
    In the future, we could pass different classifier parameters.
    """
    results = []
    for msg in messages:
        result = process_message(msg, **ref_data)
        result["strategy"] = strategy_name
        results.append(result)
    return results

def compare_strategies(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare two sets of results and compute agreement metrics.
    """
    if len(results_a) != len(results_b):
        raise ValueError("Result sets must have the same length")
    
    total = len(results_a)
    action_agreement = 0
    confidence_diff_sum = 0.0
    
    for res_a, res_b in zip(results_a, results_b):
        if res_a["message_id"] != res_b["message_id"]:
            raise ValueError("Message IDs must match in order")
        
        if res_a["action"] == res_b["action"]:
            action_agreement += 1
        
        conf_a = float(res_a["confidence"])
        conf_b = float(res_b["confidence"])
        confidence_diff_sum += abs(conf_a - conf_b)
    
    action_agreement_rate = action_agreement / total if total > 0 else 0.0
    avg_confidence_diff = confidence_diff_sum / total if total > 0 else 0.0
    
    return {
        "total_messages": total,
        "action_agreement_count": action_agreement,
        "action_agreement_rate": action_agreement_rate,
        "average_confidence_difference": avg_confidence_diff
    }

def main():
    """Main evaluation function."""
    logger.info("Starting evaluation of Message Notification Router")
    
    # Define paths
    base_path = Path(DATASET_PATH)
    sample_messages_path = base_path / "sample_messages.csv"
    
    # Load reference data (same for all strategies)
    users = load_csv_as_dict(base_path / "users.csv", "user_id")
    groups = load_csv_as_dict(base_path / "groups.csv", "group_id")
    group_members = load_csv_as_dict(base_path / "group_members.csv", "group_id")
    businesses = load_csv_as_dict(base_path / "business_accounts.csv", "business_id")
    user_business_history = load_csv_as_dict(base_path / "user_business_history.csv", "user_id")
    message_history = load_csv_as_dict(base_path / "message_history.csv", "message_id")
    images = load_csv_as_dict(base_path / "dataset" / "images.csv", "image_id") if (base_path / "dataset" / "images.csv").exists() else {}
    voice_notes = load_csv_as_dict(base_path / "dataset" / "voice_notes.csv", "voice_note_id") if (base_path / "dataset" / "voice_notes.csv").exists() else {}
    
    ref_data = {
        "users": users,
        "groups": groups,
        "group_members": group_members,
        "businesses": businesses,
        "user_business_history": user_business_history,
        "images": images,
        "voice_notes": voice_notes,
        "message_history": message_history
    }
    
    # Load sample messages
    messages = load_messages(sample_messages_path)
    
    # For now, we only have one strategy (the default classifier).
    # To compare two strategies, we would need to modify the classifier's behavior.
    # Since we don't have ground truth, we'll run the same strategy twice and compare
    # (which will give perfect agreement) - this is just a placeholder.
    # In a real scenario, we would have two different classifiers (e.g., different thresholds).
    
    logger.info("Running Strategy A (default)")
    results_a = run_strategy(messages, ref_data, "Strategy_A")
    
    logger.info("Running Strategy B (default) - for comparison, same as A")
    results_b = run_strategy(messages, ref_data, "Strategy_B")
    
    # Compare the two strategies
    comparison = compare_strategies(results_a, results_b)
    
    # Output comparison results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Total messages: {comparison['total_messages']}")
    logger.info(f"  Action agreement: {comparison['action_agreement_count']}/{comparison['total_messages']} ({comparison['action_agreement_rate']:.2%})")
    logger.info(f"  Average confidence difference: {comparison['average_confidence_difference']:.3f}")
    
    # Save detailed results to CSV for inspection
    output_dir = base_path / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    detailed_path = output_dir / "evaluation_detailed.csv"
    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["message_id", "strategy", "action", "confidence", "evidence_message_ids"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_a + results_b:
            writer.writerow(row)
    logger.info(f"Detailed results saved to {detailed_path}")
    
    # Save summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Message Notification Router Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total messages evaluated: {comparison['total_messages']}\n")
        f.write(f"Action agreement between strategies: {comparison['action_agreement_count']}/{comparison['total_messages']} ({comparison['action_agreement_rate']:.2%})\n")
        f.write(f"Average confidence difference: {comparison['average_confidence_difference']:.3f}\n")
        f.write("\nNote: This evaluation uses the same strategy for both runs.\n")
        f.write("To compare different strategies, modify the classifier parameters.\n")
    logger.info(f"Summary saved to {summary_path}")
    
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()