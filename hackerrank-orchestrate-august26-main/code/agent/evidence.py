"""
Evidence module for the Message Notification Router.
Handles retrieval of historical messages as evidence and adjusts confidence based on historical patterns.
"""

import csv
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def get_evidence_and_confidence(
    original_message: Dict[str, Any],
    enriched_message: Dict[str, Any],
    features: Dict[str, Any],
    action: str,
    rule_confidence: float,
    message_history: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], float]:
    """
    Retrieve evidence message IDs from history and adjust confidence based on historical patterns.
    
    Args:
        original_message: The original message dictionary
        enriched_message: The enriched message dictionary
        features: Extracted features
        action: The action decided by the classifier
        rule_confidence: Confidence from the rule-based classifier
        message_history: Dictionary of message history keyed by message_id
    
    Returns:
        Tuple of (evidence_message_ids, adjusted_confidence)
    """
    try:
        # We'll look for historical messages that are similar to the current message
        # Similarity criteria:
        #   - Same sender_id
        #   - Same group_id (if applicable)
        #   - Same message_type (text, image, voice_note)
        #   - Similar content (we'll use a simple keyword match for now)
        
        sender_id = original_message.get('sender_id')
        group_id = original_message.get('group_id')
        message_type = original_message.get('message_type', 'text')
        
        # Get the text content for keyword set of the current message for keyword matching
        current_text = (original_message.get('text', '') or original_message.get('message', '')).lower()
        # Extract a few keywords (we'll use a simple split and take first 3 words)
        current_keywords = set(current_text.split()[:3]) if current_text else set()
        
        evidence_ids = []
        match_score = 0.0
        matches_found = 0
        
        # We'll look through the message history (up to a limit to avoid too much computation)
        # We want to find up to 2 messages that match our criteria
        for msg_id, hist_msg in message_history.items():
            if len(evidence_ids) >= 2:
                break
                
            # Check sender
            if hist_msg.get('sender_id') != sender_id:
                continue
                
            # Check group
            hist_group_id = hist_msg.get('group_id')
            if group_id:  # Current message is in a group
                if hist_group_id != group_id:
                    continue
            else:  # Current message is direct message
                if hist_group_id:  # Historical message is in a group
                    continue
                    
            # Check message type (we need to infer from historical message)
            hist_media_type = hist_msg.get('media_type')
            hist_msg_type = 'text'
            if hist_media_type == 'image':
                hist_msg_type = 'image'
            elif hist_media_type == 'voice_note':
                hist_msg_type = 'voice_note'
                
            if hist_msg_type != message_type:
                continue
                
            # Optional: check keyword overlap (if we have text)
            hist_text = (hist_msg.get('text', '') or hist_msg.get('message', '')).lower()
            hist_keywords = set(hist_text.split()[:3]) if hist_text else set()
            
            # Calculate simple keyword overlap (Jaccard similarity)
            if current_keywords and hist_keywords:
                intersection = len(current_keywords & hist_keywords)
                union = len(current_keywords | hist_keywords)
                if union > 0:
                    keyword_similarity = intersection / union
                else:
                    keyword_similarity = 0.0
            else:
                keyword_similarity = 0.5  # Neutral if no text
                
            # If we have at least some similarity, consider it a match
            if keyword_similarity > 0.3 or (not current_text and not hist_text):
                evidence_ids.append(msg_id)
                match_score += keyword_similarity
                matches_found += 1
        
        # Adjust confidence based on historical evidence
        adjusted_confidence = _adjust_confidence(
            rule_confidence, 
            matches_found, 
            action,
            enriched_message
        )
        
        # If no similar messages found, evidence is "none"
        if not evidence_ids:
            evidence_ids = ["none"]
        
        logger.debug(
            f"Message {original_message.get('message_id')}: "
            f"found {matches_found} similar messages, "
            f"confidence adjusted from {rule_confidence:.2f} to {adjusted_confidence:.2f}"
        )
        
        return evidence_ids, adjusted_confidence
        
    except Exception as e:
        logger.error(f"Error in evidence lookup for message {original_message.get('message_id')}: {e}")
        # Return default on error
        return ["none"], max(0.1, rule_confidence - 0.1)

def _adjust_confidence(
    base_confidence: float, 
    evidence_count: int, 
    action: str,
    enriched_message: Dict[str, Any]
) -> float:
    """
    Adjust confidence based on historical evidence count and other factors.
    
    Args:
        base_confidence: Confidence from rule-based classifier
        evidence_count: Number of similar historical messages found (0, 1, or 2+)
        action: The predicted action
        enriched_message: The enriched message for additional context
    
    Returns:
        Adjusted confidence between 0.0 and 1.0
    """
    # Start with base confidence
    confidence = base_confidence
    
    # Adjust based on evidence count
    if evidence_count >= 2:
        # Strong historical pattern - increase confidence
        confidence = min(0.95, confidence + 0.1)
    elif evidence_count == 1:
        # Some historical pattern - slight increase
        confidence = min(0.95, confidence + 0.05)
    else:  # evidence_count == 0
        # No historical pattern - decrease confidence slightly
        confidence = max(0.1, confidence - 0.05)
    
    # Additional adjustments based on message characteristics
    # For example, if we have high spam score but found similar messages that were not spam,
    # we might want to adjust. But we don't have historical actions, so we skip.
    
    # Ensure confidence stays in bounds
    return max(0.0, min(1.0, confidence))

def load_message_history(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load message history from CSV file into a dictionary.
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        Dictionary mapping message_id to message record
    """
    history = {}
    history_path = dataset_path / "message_history.csv"
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg_id = row.get('message_id')
                if msg_id:
                    history[msg_id] = row
        logger.info(f"Loaded {len(history)} messages from history")
    except Exception as e:
        logger.error(f"Failed to load message history: {e}")
    
    return history