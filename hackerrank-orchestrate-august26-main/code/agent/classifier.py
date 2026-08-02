"""
Classifier module for the Message Notification Router.
Implements the rule-based decision engine that maps features to actions.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def classify_message(features: Dict[str, Any]) -> Tuple[str, float]:
    """
    Classify a message into one of three actions: notify, digest, mute.
    Returns the action and a rule-based confidence score.
    
    Args:
        features: Dictionary of extracted features
    
    Returns:
        Tuple of (action, confidence) where action is one of 'notify', 'digest', 'mute'
    """
    try:
        # Extract key features
        sender_trust = features.get('sender_trust_score', 0.5)
        group_relevance = features.get('group_relevance', 0.5)
        time_sensitivity = features.get('time_sensitivity', 0.5)
        spam_score = features.get('spam_score', 0.5)
        media_risk = features.get('media_risk_score', 0.5)
        urgency_keywords = features.get('urgency_keywords_present', 0.0)
        is_forwarded = features.get('is_forwarded', 0.0)
        has_media = features.get('has_media', 0.0)
        sender_is_business = features.get('sender_is_business', 0.0)
        business_verified = features.get('business_verified', 0.0)
        
        # Initialize scores for each action
        notify_score = 0.0
        digest_score = 0.0
        mute_score = 0.0
        
        # --- Rules for NOTIFY ---
        # High urgency keywords
        if urgency_keywords > 0.5:
            notify_score += 0.3
        
        # Direct message (group_relevance low means direct message? Actually, we set group_relevance high for relevant groups)
        # Let's redefine: group_relevance is high for relevant groups, low for irrelevant groups or direct messages?
        # In our feature extraction, group_relevance is 0.8 for direct messages (no group) and varies for groups.
        # So high group_relevance means either direct message or relevant group.
        # We'll consider direct messages as high priority for notify if from trusted sender.
        if group_relevance > 0.7 and sender_trust > 0.7:
            notify_score += 0.2
        
        # Time sensitive and trusted sender
        if time_sensitivity > 0.7 and sender_trust > 0.6:
            notify_score += 0.2
        
        # Business verified and relevant (e.g., transaction-related)
        if sender_is_business > 0.5 and business_verified > 0.5 and group_relevance > 0.5:
            notify_score += 0.2
        
        # --- Rules for DIGEST ---
        # Useful but not urgent: moderate trust, moderate relevance, low spam
        if sender_trust > 0.4 and group_relevance > 0.4 and spam_score < 0.4:
            digest_score += 0.2
        
        # Business message that is verified but not urgent
        if sender_is_business > 0.5 and business_verified > 0.5 and urgency_keywords < 0.5 and time_sensitivity < 0.5:
            digest_score += 0.2
        
        # Media content that is not risky
        if has_media > 0.5 and media_risk < 0.3 and spam_score < 0.3:
            digest_score += 0.1
        
        # --- Rules for MUTE ---
        # High spam score
        if spam_score > 0.6:
            mute_score += 0.3
        
        # High media risk
        if media_risk > 0.5:
            mute_score += 0.2
        
        # Low trust and low relevance (likely spam or irrelevant)
        if sender_trust < 0.3 and group_relevance < 0.3:
            mute_score += 0.2
        
        # Forwarded message from unknown sender (often spam)
        if is_forwarded > 0.5 and sender_trust < 0.5:
            mute_score += 0.2
        
        # Repeated forwarded content (we don't have forward count, but we can use is_forwarded as proxy)
        # Already covered above.
        
        # Adjust scores based on conflicting signals
        # If high urgency and high spam, it might be a scam emergency (e.g., "URGENT: YOUR ACCOUNT IS COMPROMISED")
        if urgency_keywords > 0.5 and spam_score > 0.5:
            # Boost mute, reduce notify
            mute_score += 0.2
            notify_score = max(0.0, notify_score - 0.1)
        
        # Normalize scores to get probabilities (but we don't need to normalize strictly, we just pick the max)
        # However, we want to compute confidence as the difference between the top score and the next, or as a softmax.
        # For simplicity, we'll compute confidence as the softmax of the scores.
        
        scores = {
            'notify': notify_score,
            'digest': digest_score,
            'mute': mute_score
        }
        
        # If all scores are zero, default to digest with low confidence
        if sum(scores.values()) == 0:
            return 'digest', 0.1
        
        # Apply softmax to get probabilities
        import math
        sum_exp = sum(math.exp(s) for s in scores.values())
        probs = {action: math.exp(score) / sum_exp for action, score in scores.items()}
        
        # Choose the action with the highest probability
        action = max(probs, key=probs.get)
        confidence = probs[action]
        
        # Ensure confidence is at least 0.1 and at most 0.95 (we'll adjust with evidence later)
        confidence = max(0.1, min(0.95, confidence))
        
        logger.debug(f"Feature scores: notify={notify_score:.2f}, digest={digest_score:.2f}, mute={mute_score:.2f} -> {action} ({confidence:.2f})")
        
        return action, confidence
        
    except Exception as e:
        logger.error(f"Error in classify_message: {e}")
        # Fallback to digest with low confidence
        return 'digest', 0.1