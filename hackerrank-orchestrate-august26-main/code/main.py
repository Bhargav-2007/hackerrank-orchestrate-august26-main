#!/usr/bin/env python3
"""
Main entry point for the Message Notification Router.
Orchestrates the pipeline: load data, enrich, extract features, classify, and output.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the code directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from agent.enricher import enrich_message
from agent.features import extract_features
from agent.classifier import classify_message
from agent.evidence import get_evidence_and_confidence
from utils.logging import setup_logging
from utils.config import DATASET_PATH

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

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

def main():
    """Main pipeline."""
    logger.info("Starting Message Notification Router")

    # Define paths
    base_path = Path(DATASET_PATH)
    messages_path = base_path / "messages.csv"
    # Fallback to sample messages if messages.csv doesn't exist
    if not messages_path.exists():
        messages_path = base_path / "sample_messages.csv"
        logger.warning(f"messages.csv not found, using {messages_path}")

    # Load all reference data
    users = load_csv_as_dict(base_path / "users.csv", "user_id")
    groups = load_csv_as_dict(base_path / "groups.csv", "group_id")
    group_members = load_csv_as_dict(base_path / "group_members.csv", "group_id")  # Note: group_members.csv has group_id and user_id
    businesses = load_csv_as_dict(base_path / "business_accounts.csv", "business_id")
    user_business_history = load_csv_as_dict(base_path / "user_business_history.csv", "user_id")  # Simplified key
    message_history = load_csv_as_dict(base_path / "message_history.csv", "message_id")  # Simplified
    images = load_csv_as_dict(base_path / "dataset" / "images.csv", "image_id") if (base_path / "dataset" / "images.csv").exists() else {}
    voice_notes = load_csv_as_dict(base_path / "dataset" / "voice_notes.csv", "voice_note_id") if (base_path / "dataset" / "voice_notes.csv").exists() else {}

    # Load messages to process
    messages = load_messages(messages_path)

    # Prepare output
    output_path = base_path / "output.csv"
    output_fields = ["message_id", "action", "confidence", "evidence_message_ids"]

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields)
        writer.writeheader()

        for msg in messages:
            try:
                # Step 1: Enrich message with context
                enriched = enrich_message(msg, users, groups, group_members, businesses, user_business_history, images, voice_notes)

                # Step 2: Extract features
                features = extract_features(enriched)

                # Step 3: Classify message (get action and rule-based confidence)
                action, rule_confidence = classify_message(features)

                # Step 4: Get evidence and adjust confidence
                evidence_ids, final_confidence = get_evidence_and_confidence(
                    msg, enriched, features, action, rule_confidence, message_history
                )

                # Prepare output row
                output_row = {
                    "message_id": msg["message_id"],
                    "action": action,
                    "confidence": f"{final_confidence:.2f}",
                    "evidence_message_ids": " ".join(evidence_ids) if evidence_ids else "none"
                }
                writer.writerow(output_row)
                logger.debug(f"Processed message {msg['message_id']}: {output_row}")

            except Exception as e:
                logger.error(f"Error processing message {msg.get('message_id', 'unknown')}: {e}")
                # Write a default mute action to avoid breaking the batch
                writer.writerow({
                    "message_id": msg.get("message_id", "unknown"),
                    "action": "mute",
                    "confidence": "0.0",
                    "evidence_message_ids": "none"
                })

    logger.info(f"Output written to {output_path}")

if __name__ == "__main__":
    main()