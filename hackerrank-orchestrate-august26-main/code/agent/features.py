"""
Feature extraction module for the Message Notification Router.
Extracts numerical and categorical features from enriched messages for classification.
"""

import logging
import re
from typing import Dict, Any, List
from datetime import datetime, time

logger = logging.getLogger(__name__)

# Urgency keywords that might indicate a need for immediate notification
URGENCY_KEYWORDS = {
    'urgent', 'emergency', 'help', 'asap', 'immediately', 'right now', 
    'important', 'critical', 'break', 'breakdown', 'accident', 'police',
    'ambulance', 'fire', 'danger', 'unsafe', 'attack', 'threat'
}

# Spam/scam indicators
SPAM_KEYWORDS = {
    'free', 'win', 'winner', 'congratulations', 'selected', 'chosen',
    'claim', 'prize', 'award', 'bonus', 'cash', 'money', 'dollar', '

def extract_features(enriched_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from an enriched message.
    
    Args:
        enriched_message: Message dictionary after enrichment
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    try:
        # 1. Sender trust score (0.0 to 1.0)
        features['sender_trust_score'] = _calculate_sender_trust(enriched_message)
        
        # 2. Group relevance (0.0 to 1.0)
        features['group_relevance'] = _calculate_group_relevance(enriched_message)
        
        # 3. Time sensitivity (0.0 to 1.0)
        features['time_sensitivity'] = _calculate_time_sensitivity(enriched_message)
        
        # 4. Content spam/scam score (0.0 to 1.0, higher = more spam_score = _calculate_content_spam_score(enriched_message))
        
        # 5. Media features['urgency_score'] = _more likely spam)
        features['spam_score'] = _calculate_content_spam_score(enriched_message)
        
        # 5. Media risk score (0.0 to 1.0)
        features['media_risk_score'] = _calculate_media_risk(enriched_message)
        
        # 6. Urgency keyword presence (boolean)
        features['urgency_keywords_present'] = _check_urgency_keywords(enriched_message)
        
        # 7. Is forwarded (boolean, if we have that data)
        features['is_forwarded'] = _is_forwarded(enriched_message)
        
        # 8. Message length (normalized)
        features['message_length_norm'] = _normalize_message_length(enriched_message)
        
        # 9. Has media (boolean)
        features['has_media'] = bool(enriched_message.get('media'))
        
        # 10. Media type (categorical: none, image, voice_note, unknown)
        media_type = enriched_message.get('media_type', 'none')
        features['media_type_image'] = 1.0 if media_type == 'image' else 0.0
        features['media_type_voice'] = 1.0 if media_type == 'voice_note' else 0.0
        
        # 11. Sender is business (boolean)
        features['sender_is_business'] = 1.0 if enriched_message.get('business') else 0.0
        
        # 12. Business verified (if applicable)
        business = enriched_message.get('business', {})
        features['business_verified'] = 1.0 if business.get('is_verified', '').lower() == 'true' else 0.0
        
        # 13. User's historical action preference (if available)
        # We'll compute this in the classifier based on history, but we can add a feature
        # for now, we'll set to 0.5 (neutral)
        features['user_prefers_notify'] = 0.5  # Placeholder
        
        logger.debug(f"Extracted features for message {enriched_message.get('message_id')}: {features}")
        
    except Exception as e:
        logger.error(f"Error extracting features for message {enriched_message.get('message_id')}: {e}")
        # Return a default feature set to avoid breaking the pipeline
        features = {
            'sender_trust_score': 0.5,
            'group_relevance': 0.5,
            'time_sensitivity': 0.5,
            'spam_score': 0.5,
            'media_risk_score': 0.5,
            'urgency_keywords_present': 0.0,
            'is_forwarded': 0.0,
            'message_length_norm': 0.5,
            'has_media': 0.0,
            'media_type_image': 0.0,
            'media_type_voice': 0.0,
            'sender_is_business': 0.0,
            'business_verified': 0.0,
            'user_prefers_notify': 0.5
        }
    
    return features

def _calculate_sender_trust(enriched: Dict[str, Any]) -> float:
    """Calculate trust score for the sender based on historical interactions and verification."""
    sender = enriched.get('sender', {})
    # If sender is a business, use business verification and interaction history
    if enriched.get('sender_is_business'):
        business = enriched.get('business', {})
        # Start with verification status
        trust = 0.7 if business.get('is_verified', '').lower() == 'true' else 0.4
        # Adjust by interaction count (more interactions = more trust)
        interaction_count = int(business.get('interaction_count', 0))
        if interaction_count > 50:
            trust += 0.2
        elif interaction_count > 10:
            trust += 0.1
        # Cap at 1.0
        return min(trust, 1.0)
    else:
        # For personal users, we could look at message history, but for simplicity
        # we'll use a default medium trust. In a real system, we'd use historical
        # interaction data from message_history.csv.
        return 0.6  # Default trust for personal contacts

def _calculate_group_relevance(enriched: Dict[str, Any]) -> float:
    """Calculate how relevant the group is to the user."""
    group = enriched.get('group', {})
    member_info = enriched.get('member_info', {})
    
    if not group:
        # Direct message (no group) - high relevance by default
        return 0.8
    
    # Check if user is admin in the group (admins might want more notifications)
    admin_status = member_info.get('admin_status', '').lower()
    if admin_status == 'true' or admin_status == 'yes':
        return 0.9
    
    # Check group type or name for relevance (simplified)
    group_type = group.get('group_type', '').lower()
    if 'family' in group_type or 'close' in group_type:
        return 0.8
    elif 'work' in group_type or 'office' in group_type:
        return 0.7
    elif 'school' in group_type or 'college' in group_type:
        return 0.6
    else:
        return 0.5  # Default for unknown groups

def _calculate_time_sensitivity(enriched: Dict[str, Any]) -> float:
    """Calculate time sensitivity based on message timestamp and content."""
    timestamp_dt = enriched.get('timestamp_dt')
    if not timestamp_dt:
        return 0.5  # Default if we can't parse time
    
    # Check if it's during business hours (9 AM - 6 PM) on weekdays
    # This is a simplification; in reality, urgency might be higher outside business hours for emergencies
    hour = timestamp_dt.hour
    weekday = timestamp_dt.weekday()  # Monday is 0, Sunday is 6
    
    # Base score
    sensitivity = 0.5
    
    # Adjust for time of day: early morning or late night might be more urgent for personal messages
    if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
        sensitivity += 0.2
    elif 9 <= hour <= 18:  # Business hours
        sensitivity -= 0.1  # Less urgent during business hours for non-work messages
    
    # Adjust for weekend: might be more urgent for family messages
    if weekday >= 5:  # Saturday or Sunday
        sensitivity += 0.1
    
    
    return min(max(sensitivity, 0.0), 1.0)

def _calculate_content_spam_score(enriched: Dict[str, Any]) -> float:
    """Calculate spam/scam score based on message content."""
    # Get message text - we need to know where the text is stored
    # Assuming the enriched message has a 'text' field or similar
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        # If no text, check media type - voice notes and images might have text via OCR/ASR
        # but we don't have that, so we'll return a low score
        return 0.1
    
    text_lower = text.lower()
    
    # Count spam keyword matches
    spam_matches = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
    # Count urgent keyword matches (might reduce spam score if it's a legitimate urgent message)
    urgent_matches = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    
    # Base score from spam matches
    score = min(spam_matches * 0.1, 0.8)  # Cap spam contribution at 0.8
    
    # Reduce score if urgent keywords are present (could be legitimate emergency)
    if urgent_matches > 0:
        score = max(score - urgent_matches * 0.1, 0.0)
    
    # Check for suspicious patterns
    # Multiple exclamation marks or all caps
    if text.count('!!!') > 0 or text.count('!!') > 2:
        score += 0.1
    if text.isupper() and len(text) > 10:
        score += 0.1
    
    # Check for suspicious URLs or phone numbers
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
        score += 0.1
    if re.search(r'\\b\\d{10,}\\b', text):  # Long numbers could be phone numbers or OTPs
        score += 0.05
    
    return min(score, 1.0)

def _calculate_media_risk(enriched: Dict[str, Any]) -> float:
    """Calculate risk score for media content."""
    media_type = enriched.get('media_type')
    if not media_type:
        return 0.0
    
    # Voice notes and images might be higher risk for spam/scam
    # In a real system, we'd run OCR/ASR and then check the content
    # For now, we'll assign a base risk
    if media_type == 'voice_note':
        return 0.3  # Voice notes can be spam but often personal
    elif media_type == 'image':
        return 0.4  # Images can be spam posters or scams
    else:
        return 0.0

def _check_urgency_keywords(enriched: Dict[str, Any]) -> float:
    """Check if urgency keywords are present in the message."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        return 0.0
    
    text_lower = text.lower()
    for keyword in URGENCY_KEYWORDS:
        if keyword in text_lower:
            return 1.0
    return 0.0

def _is_forwarded(enriched: Dict[str, Any]) -> float:
    """Check if the message is forwarded."""
    # Assuming there's a 'forwarded' field in the original message
    forwarded = enriched.get('forwarded', '') or enriched.get('is_forwarded', '')
    if isinstance(forwarded, str):
        return 1.0 if forwarded.lower() in ('true', 'yes', '1') else 0.0
    elif isinstance(forwarded, bool):
        return 1.0 if forwarded else 0.0
    else:
        # Try to infer from content: common forwarded markers
        text = enriched.get('text', '') or enriched.get('message', '') or ''
        if text.startswith('FW:') or text.startswith('Fwd:') or 'forwarded' in text.lower():
            return 1.0
        return 0.0

def _normalize_message_length(enriched: Dict[str, Any]) -> float:
    """Normalize message length to 0-1 range."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    length = len(text)
    # Assume max length of 500 characters for normalization
    return min(length / 500.0, 1.0), '₹', 'rs',
    'offer', 'deal', 'discount', 'sale', 'limited time', 'act now', 'don\\'t miss',
    'click here', 'link', 'website', 'visit', 'http', 'www', '.com', '.in',
    'otp', 'password', 'pin', 'ssn', 'bank account', 'credit card', 'debit card',
    'verify', 'confirm', 'update', 'suspend', 'blocked', 'unusual activity',
    'login', 'sign in', 'account', 'paypal', 'amazon', 'flipkart', 'google',
    'apple', 'microsoft', 'invoice', 'bill', 'payment', 'overdue', 'due date'
}

# Words that might indicate useful but non-urgent information (digest)
DIGEST_KEYWORDS = {
    'meeting', 'schedule', 'appointment', 'reminder', 'deadline', 'due',
    'report', 'update', 'news', 'information', 'details', 'schedule',
    'event', 'webinar', 'training', 'session', 'holiday', 'leave',
    'policy', 'procedure', 'guideline', 'announcement', 'notice',
    'congratulations', 'happy birthday', 'anniversary', 'festival'
}

def extract_features(enriched_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from an enriched message.
    
    Args:
        enriched_message: Message dictionary after enrichment
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    try:
        # 1. Sender trust score (0.0 to 1.0)
        features['sender_trust_score'] = _calculate_sender_trust(enriched_message)
        
        # 2. Group relevance (0.0 to 1.0)
        features['group_relevance'] = _calculate_group_relevance(enriched_message)
        
        # 3. Time sensitivity (0.0 to 1.0)
        features['time_sensitivity'] = _calculate_time_sensitivity(enriched_message)
        
        # 4. Content spam/scam score (0.0 to 1.0, higher = more spam_score = _calculate_content_spam_score(enriched_message))
        
        # 5. Media features['urgency_score'] = _more likely spam)
        features['spam_score'] = _calculate_content_spam_score(enriched_message)
        
        # 5. Media risk score (0.0 to 1.0)
        features['media_risk_score'] = _calculate_media_risk(enriched_message)
        
        # 6. Urgency keyword presence (boolean)
        features['urgency_keywords_present'] = _check_urgency_keywords(enriched_message)
        
        # 7. Is forwarded (boolean, if we have that data)
        features['is_forwarded'] = _is_forwarded(enriched_message)
        
        # 8. Message length (normalized)
        features['message_length_norm'] = _normalize_message_length(enriched_message)
        
        # 9. Has media (boolean)
        features['has_media'] = bool(enriched_message.get('media'))
        
        # 10. Media type (categorical: none, image, voice_note, unknown)
        media_type = enriched_message.get('media_type', 'none')
        features['media_type_image'] = 1.0 if media_type == 'image' else 0.0
        features['media_type_voice'] = 1.0 if media_type == 'voice_note' else 0.0
        
        # 11. Sender is business (boolean)
        features['sender_is_business'] = 1.0 if enriched_message.get('business') else 0.0
        
        # 12. Business verified (if applicable)
        business = enriched_message.get('business', {})
        features['business_verified'] = 1.0 if business.get('is_verified', '').lower() == 'true' else 0.0
        
        # 13. User's historical action preference (if available)
        # We'll compute this in the classifier based on history, but we can add a feature
        # for now, we'll set to 0.5 (neutral)
        features['user_prefers_notify'] = 0.5  # Placeholder
        
        logger.debug(f"Extracted features for message {enriched_message.get('message_id')}: {features}")
        
    except Exception as e:
        logger.error(f"Error extracting features for message {enriched_message.get('message_id')}: {e}")
        # Return a default feature set to avoid breaking the pipeline
        features = {
            'sender_trust_score': 0.5,
            'group_relevance': 0.5,
            'time_sensitivity': 0.5,
            'spam_score': 0.5,
            'media_risk_score': 0.5,
            'urgency_keywords_present': 0.0,
            'is_forwarded': 0.0,
            'message_length_norm': 0.5,
            'has_media': 0.0,
            'media_type_image': 0.0,
            'media_type_voice': 0.0,
            'sender_is_business': 0.0,
            'business_verified': 0.0,
            'user_prefers_notify': 0.5
        }
    
    return features

def _calculate_sender_trust(enriched: Dict[str, Any]) -> float:
    """Calculate trust score for the sender based on historical interactions and verification."""
    sender = enriched.get('sender', {})
    # If sender is a business, use business verification and interaction history
    if enriched.get('sender_is_business'):
        business = enriched.get('business', {})
        # Start with verification status
        trust = 0.7 if business.get('is_verified', '').lower() == 'true' else 0.4
        # Adjust by interaction count (more interactions = more trust)
        interaction_count = int(business.get('interaction_count', 0))
        if interaction_count > 50:
            trust += 0.2
        elif interaction_count > 10:
            trust += 0.1
        # Cap at 1.0
        return min(trust, 1.0)
    else:
        # For personal users, we could look at message history, but for simplicity
        # we'll use a default medium trust. In a real system, we'd use historical
        # interaction data from message_history.csv.
        return 0.6  # Default trust for personal contacts

def _calculate_group_relevance(enriched: Dict[str, Any]) -> float:
    """Calculate how relevant the group is to the user."""
    group = enriched.get('group', {})
    member_info = enriched.get('member_info', {})
    
    if not group:
        # Direct message (no group) - high relevance by default
        return 0.8
    
    # Check if user is admin in the group (admins might want more notifications)
    admin_status = member_info.get('admin_status', '').lower()
    if admin_status == 'true' or admin_status == 'yes':
        return 0.9
    
    # Check group type or name for relevance (simplified)
    group_type = group.get('group_type', '').lower()
    if 'family' in group_type or 'close' in group_type:
        return 0.8
    elif 'work' in group_type or 'office' in group_type:
        return 0.7
    elif 'school' in group_type or 'college' in group_type:
        return 0.6
    else:
        return 0.5  # Default for unknown groups

def _calculate_time_sensitivity(enriched: Dict[str, Any]) -> float:
    """Calculate time sensitivity based on message timestamp and content."""
    timestamp_dt = enriched.get('timestamp_dt')
    if not timestamp_dt:
        return 0.5  # Default if we can't parse time
    
    # Check if it's during business hours (9 AM - 6 PM) on weekdays
    # This is a simplification; in reality, urgency might be higher outside business hours for emergencies
    hour = timestamp_dt.hour
    weekday = timestamp_dt.weekday()  # Monday is 0, Sunday is 6
    
    # Base score
    sensitivity = 0.5
    
    # Adjust for time of day: early morning or late night might be more urgent for personal messages
    if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
        sensitivity += 0.2
    elif 9 <= hour <= 18:  # Business hours
        sensitivity -= 0.1  # Less urgent during business hours for non-work messages
    
    # Adjust for weekend: might be more urgent for family messages
    if weekday >= 5:  # Saturday or Sunday
        sensitivity += 0.1
    
    
    return min(max(sensitivity, 0.0), 1.0)

def _calculate_content_spam_score(enriched: Dict[str, Any]) -> float:
    """Calculate spam/scam score based on message content."""
    # Get message text - we need to know where the text is stored
    # Assuming the enriched message has a 'text' field or similar
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        # If no text, check media type - voice notes and images might have text via OCR/ASR
        # but we don't have that, so we'll return a low score
        return 0.1
    
    text_lower = text.lower()
    
    # Count spam keyword matches
    spam_matches = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
    # Count urgent keyword matches (might reduce spam score if it's a legitimate urgent message)
    urgent_matches = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    
    # Base score from spam matches
    score = min(spam_matches * 0.1, 0.8)  # Cap spam contribution at 0.8
    
    # Reduce score if urgent keywords are present (could be legitimate emergency)
    if urgent_matches > 0:
        score = max(score - urgent_matches * 0.1, 0.0)
    
    # Check for suspicious patterns
    # Multiple exclamation marks or all caps
    if text.count('!!!') > 0 or text.count('!!') > 2:
        score += 0.1
    if text.isupper() and len(text) > 10:
        score += 0.1
    
    # Check for suspicious URLs or phone numbers
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
        score += 0.1
    if re.search(r'\\b\\d{10,}\\b', text):  # Long numbers could be phone numbers or OTPs
        score += 0.05
    
    return min(score, 1.0)

def _calculate_media_risk(enriched: Dict[str, Any]) -> float:
    """Calculate risk score for media content."""
    media_type = enriched.get('media_type')
    if not media_type:
        return 0.0
    
    # Voice notes and images might be higher risk for spam/scam
    # In a real system, we'd run OCR/ASR and then check the content
    # For now, we'll assign a base risk
    if media_type == 'voice_note':
        return 0.3  # Voice notes can be spam but often personal
    elif media_type == 'image':
        return 0.4  # Images can be spam posters or scams
    else:
        return 0.0

def _check_urgency_keywords(enriched: Dict[str, Any]) -> float:
    """Check if urgency keywords are present in the message."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        return 0.0
    
    text_lower = text.lower()
    for keyword in URGENCY_KEYWORDS:
        if keyword in text_lower:
            return 1.0
    return 0.0

def _is_forwarded(enriched: Dict[str, Any]) -> float:
    """Check if the message is forwarded."""
    # Assuming there's a 'forwarded' field in the original message
    forwarded = enriched.get('forwarded', '') or enriched.get('is_forwarded', '')
    if isinstance(forwarded, str):
        return 1.0 if forwarded.lower() in ('true', 'yes', '1') else 0.0
    elif isinstance(forwarded, bool):
        return 1.0 if forwarded else 0.0
    else:
        # Try to infer from content: common forwarded markers
        text = enriched.get('text', '') or enriched.get('message', '') or ''
        if text.startswith('FW:') or text.startswith('Fwd:') or 'forwarded' in text.lower():
            return 1.0
        return 0.0

def _normalize_message_length(enriched: Dict[str, Any]) -> float:
    """Normalize message length to 0-1 range."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    length = len(text)
    # Assume max length of 500 characters for normalization
    return min(length / 500.0, 1.0), '₹', 'rs',
    'offer', 'deal', 'discount', 'sale', 'limited time', 'act now', 'don\\'t miss',
    'click here', 'link', 'website', 'visit', 'http', 'www', '.com', '.in',
    'otp', 'password', 'pin', 'ssn', 'bank account', 'credit card', 'debit card',
    'verify', 'confirm', 'update', 'suspend', 'blocked', 'unusual activity',
    'login', 'sign in', 'account', 'paypal', 'amazon', 'flipkart', 'google',
    'apple', 'microsoft', 'invoice', 'bill', 'payment', 'overdue', 'due date'
}

def extract_features(enriched_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from an enriched message.
    
    Args:
        enriched_message: Message dictionary after enrichment
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    try:
        # 1. Sender trust score (0.0 to 1.0)
        features['sender_trust_score'] = _calculate_sender_trust(enriched_message)
        
        # 2. Group relevance (0.0 to 1.0)
        features['group_relevance'] = _calculate_group_relevance(enriched_message)
        
        # 3. Time sensitivity (0.0 to 1.0)
        features['time_sensitivity'] = _calculate_time_sensitivity(enriched_message)
        
        # 4. Content spam/scam score (0.0 to 1.0, higher = more spam_score = _calculate_content_spam_score(enriched_message))
        
        # 5. Media features['urgency_score'] = _more likely spam)
        features['spam_score'] = _calculate_content_spam_score(enriched_message)
        
        # 5. Media risk score (0.0 to 1.0)
        features['media_risk_score'] = _calculate_media_risk(enriched_message)
        
        # 6. Urgency keyword presence (boolean)
        features['urgency_keywords_present'] = _check_urgency_keywords(enriched_message)
        
        # 7. Is forwarded (boolean, if we have that data)
        features['is_forwarded'] = _is_forwarded(enriched_message)
        
        # 8. Message length (normalized)
        features['message_length_norm'] = _normalize_message_length(enriched_message)
        
        # 9. Has media (boolean)
        features['has_media'] = bool(enriched_message.get('media'))
        
        # 10. Media type (categorical: none, image, voice_note, unknown)
        media_type = enriched_message.get('media_type', 'none')
        features['media_type_image'] = 1.0 if media_type == 'image' else 0.0
        features['media_type_voice'] = 1.0 if media_type == 'voice_note' else 0.0
        
        # 11. Sender is business (boolean)
        features['sender_is_business'] = 1.0 if enriched_message.get('business') else 0.0
        
        # 12. Business verified (if applicable)
        business = enriched_message.get('business', {})
        features['business_verified'] = 1.0 if business.get('is_verified', '').lower() == 'true' else 0.0
        
        # 13. User's historical action preference (if available)
        # We'll compute this in the classifier based on history, but we can add a feature
        # for now, we'll set to 0.5 (neutral)
        features['user_prefers_notify'] = 0.5  # Placeholder
        
        logger.debug(f"Extracted features for message {enriched_message.get('message_id')}: {features}")
        
    except Exception as e:
        logger.error(f"Error extracting features for message {enriched_message.get('message_id')}: {e}")
        # Return a default feature set to avoid breaking the pipeline
        features = {
            'sender_trust_score': 0.5,
            'group_relevance': 0.5,
            'time_sensitivity': 0.5,
            'spam_score': 0.5,
            'media_risk_score': 0.5,
            'urgency_keywords_present': 0.0,
            'is_forwarded': 0.0,
            'message_length_norm': 0.5,
            'has_media': 0.0,
            'media_type_image': 0.0,
            'media_type_voice': 0.0,
            'sender_is_business': 0.0,
            'business_verified': 0.0,
            'user_prefers_notify': 0.5
        }
    
    return features

def _calculate_sender_trust(enriched: Dict[str, Any]) -> float:
    """Calculate trust score for the sender based on historical interactions and verification."""
    sender = enriched.get('sender', {})
    # If sender is a business, use business verification and interaction history
    if enriched.get('sender_is_business'):
        business = enriched.get('business', {})
        # Start with verification status
        trust = 0.7 if business.get('is_verified', '').lower() == 'true' else 0.4
        # Adjust by interaction count (more interactions = more trust)
        interaction_count = int(business.get('interaction_count', 0))
        if interaction_count > 50:
            trust += 0.2
        elif interaction_count > 10:
            trust += 0.1
        # Cap at 1.0
        return min(trust, 1.0)
    else:
        # For personal users, we could look at message history, but for simplicity
        # we'll use a default medium trust. In a real system, we'd use historical
        # interaction data from message_history.csv.
        return 0.6  # Default trust for personal contacts

def _calculate_group_relevance(enriched: Dict[str, Any]) -> float:
    """Calculate how relevant the group is to the user."""
    group = enriched.get('group', {})
    member_info = enriched.get('member_info', {})
    
    if not group:
        # Direct message (no group) - high relevance by default
        return 0.8
    
    # Check if user is admin in the group (admins might want more notifications)
    admin_status = member_info.get('admin_status', '').lower()
    if admin_status == 'true' or admin_status == 'yes':
        return 0.9
    
    # Check group type or name for relevance (simplified)
    group_type = group.get('group_type', '').lower()
    if 'family' in group_type or 'close' in group_type:
        return 0.8
    elif 'work' in group_type or 'office' in group_type:
        return 0.7
    elif 'school' in group_type or 'college' in group_type:
        return 0.6
    else:
        return 0.5  # Default for unknown groups

def _calculate_time_sensitivity(enriched: Dict[str, Any]) -> float:
    """Calculate time sensitivity based on message timestamp and content."""
    timestamp_dt = enriched.get('timestamp_dt')
    if not timestamp_dt:
        return 0.5  # Default if we can't parse time
    
    # Check if it's during business hours (9 AM - 6 PM) on weekdays
    # This is a simplification; in reality, urgency might be higher outside business hours for emergencies
    hour = timestamp_dt.hour
    weekday = timestamp_dt.weekday()  # Monday is 0, Sunday is 6
    
    # Base score
    sensitivity = 0.5
    
    # Adjust for time of day: early morning or late night might be more urgent for personal messages
    if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
        sensitivity += 0.2
    elif 9 <= hour <= 18:  # Business hours
        sensitivity -= 0.1  # Less urgent during business hours for non-work messages
    
    # Adjust for weekend: might be more urgent for family messages
    if weekday >= 5:  # Saturday or Sunday
        sensitivity += 0.1
    
    
    return min(max(sensitivity, 0.0), 1.0)

def _calculate_content_spam_score(enriched: Dict[str, Any]) -> float:
    """Calculate spam/scam score based on message content."""
    # Get message text - we need to know where the text is stored
    # Assuming the enriched message has a 'text' field or similar
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        # If no text, check media type - voice notes and images might have text via OCR/ASR
        # but we don't have that, so we'll return a low score
        return 0.1
    
    text_lower = text.lower()
    
    # Count spam keyword matches
    spam_matches = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
    # Count urgent keyword matches (might reduce spam score if it's a legitimate urgent message)
    urgent_matches = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    
    # Base score from spam matches
    score = min(spam_matches * 0.1, 0.8)  # Cap spam contribution at 0.8
    
    # Reduce score if urgent keywords are present (could be legitimate emergency)
    if urgent_matches > 0:
        score = max(score - urgent_matches * 0.1, 0.0)
    
    # Check for suspicious patterns
    # Multiple exclamation marks or all caps
    if text.count('!!!') > 0 or text.count('!!') > 2:
        score += 0.1
    if text.isupper() and len(text) > 10:
        score += 0.1
    
    # Check for suspicious URLs or phone numbers
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
        score += 0.1
    if re.search(r'\\b\\d{10,}\\b', text):  # Long numbers could be phone numbers or OTPs
        score += 0.05
    
    return min(score, 1.0)

def _calculate_media_risk(enriched: Dict[str, Any]) -> float:
    """Calculate risk score for media content."""
    media_type = enriched.get('media_type')
    if not media_type:
        return 0.0
    
    # Voice notes and images might be higher risk for spam/scam
    # In a real system, we'd run OCR/ASR and then check the content
    # For now, we'll assign a base risk
    if media_type == 'voice_note':
        return 0.3  # Voice notes can be spam but often personal
    elif media_type == 'image':
        return 0.4  # Images can be spam posters or scams
    else:
        return 0.0

def _check_urgency_keywords(enriched: Dict[str, Any]) -> float:
    """Check if urgency keywords are present in the message."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        return 0.0
    
    text_lower = text.lower()
    for keyword in URGENCY_KEYWORDS:
        if keyword in text_lower:
            return 1.0
    return 0.0

def _is_forwarded(enriched: Dict[str, Any]) -> float:
    """Check if the message is forwarded."""
    # Assuming there's a 'forwarded' field in the original message
    forwarded = enriched.get('forwarded', '') or enriched.get('is_forwarded', '')
    if isinstance(forwarded, str):
        return 1.0 if forwarded.lower() in ('true', 'yes', '1') else 0.0
    elif isinstance(forwarded, bool):
        return 1.0 if forwarded else 0.0
    else:
        # Try to infer from content: common forwarded markers
        text = enriched.get('text', '') or enriched.get('message', '') or ''
        if text.startswith('FW:') or text.startswith('Fwd:') or 'forwarded' in text.lower():
            return 1.0
        return 0.0

def _normalize_message_length(enriched: Dict[str, Any]) -> float:
    """Normalize message length to 0-1 range."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    length = len(text)
    # Assume max length of 500 characters for normalization
    return min(length / 500.0, 1.0), '₹', 'rs',
    'offer', 'deal', 'discount', 'sale', 'limited time', 'act now', 'don\\'t miss',
    'click here', 'link', 'website', 'visit', 'http', 'www', '.com', '.in',
    'otp', 'password', 'pin', 'ssn', 'bank account', 'credit card', 'debit card',
    'verify', 'confirm', 'update', 'suspend', 'blocked', 'unusual activity',
    'login', 'sign in', 'account', 'paypal', 'amazon', 'flipkart', 'google',
    'apple', 'microsoft', 'invoice', 'bill', 'payment', 'overdue', 'due date'
}

# Words that might indicate useful but non-urgent information (digest)
DIGEST_KEYWORDS = {
    'meeting', 'schedule', 'appointment', 'reminder', 'deadline', 'due',
    'report', 'update', 'news', 'information', 'details', 'schedule',
    'event', 'webinar', 'training', 'session', 'holiday', 'leave',
    'policy', 'procedure', 'guideline', 'announcement', 'notice',
    'congratulations', 'happy birthday', 'anniversary', 'festival'
}

def extract_features(enriched_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from an enriched message.
    
    Args:
        enriched_message: Message dictionary after enrichment
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    try:
        # 1. Sender trust score (0.0 to 1.0)
        features['sender_trust_score'] = _calculate_sender_trust(enriched_message)
        
        # 2. Group relevance (0.0 to 1.0)
        features['group_relevance'] = _calculate_group_relevance(enriched_message)
        
        # 3. Time sensitivity (0.0 to 1.0)
        features['time_sensitivity'] = _calculate_time_sensitivity(enriched_message)
        
        # 4. Content spam/scam score (0.0 to 1.0, higher = more spam_score = _calculate_content_spam_score(enriched_message))
        
        # 5. Media features['urgency_score'] = _more likely spam)
        features['spam_score'] = _calculate_content_spam_score(enriched_message)
        
        # 5. Media risk score (0.0 to 1.0)
        features['media_risk_score'] = _calculate_media_risk(enriched_message)
        
        # 6. Urgency keyword presence (boolean)
        features['urgency_keywords_present'] = _check_urgency_keywords(enriched_message)
        
        # 7. Is forwarded (boolean, if we have that data)
        features['is_forwarded'] = _is_forwarded(enriched_message)
        
        # 8. Message length (normalized)
        features['message_length_norm'] = _normalize_message_length(enriched_message)
        
        # 9. Has media (boolean)
        features['has_media'] = bool(enriched_message.get('media'))
        
        # 10. Media type (categorical: none, image, voice_note, unknown)
        media_type = enriched_message.get('media_type', 'none')
        features['media_type_image'] = 1.0 if media_type == 'image' else 0.0
        features['media_type_voice'] = 1.0 if media_type == 'voice_note' else 0.0
        
        # 11. Sender is business (boolean)
        features['sender_is_business'] = 1.0 if enriched_message.get('business') else 0.0
        
        # 12. Business verified (if applicable)
        business = enriched_message.get('business', {})
        features['business_verified'] = 1.0 if business.get('is_verified', '').lower() == 'true' else 0.0
        
        # 13. User's historical action preference (if available)
        # We'll compute this in the classifier based on history, but we can add a feature
        # for now, we'll set to 0.5 (neutral)
        features['user_prefers_notify'] = 0.5  # Placeholder
        
        logger.debug(f"Extracted features for message {enriched_message.get('message_id')}: {features}")
        
    except Exception as e:
        logger.error(f"Error extracting features for message {enriched_message.get('message_id')}: {e}")
        # Return a default feature set to avoid breaking the pipeline
        features = {
            'sender_trust_score': 0.5,
            'group_relevance': 0.5,
            'time_sensitivity': 0.5,
            'spam_score': 0.5,
            'media_risk_score': 0.5,
            'urgency_keywords_present': 0.0,
            'is_forwarded': 0.0,
            'message_length_norm': 0.5,
            'has_media': 0.0,
            'media_type_image': 0.0,
            'media_type_voice': 0.0,
            'sender_is_business': 0.0,
            'business_verified': 0.0,
            'user_prefers_notify': 0.5
        }
    
    return features

def _calculate_sender_trust(enriched: Dict[str, Any]) -> float:
    """Calculate trust score for the sender based on historical interactions and verification."""
    sender = enriched.get('sender', {})
    # If sender is a business, use business verification and interaction history
    if enriched.get('sender_is_business'):
        business = enriched.get('business', {})
        # Start with verification status
        trust = 0.7 if business.get('is_verified', '').lower() == 'true' else 0.4
        # Adjust by interaction count (more interactions = more trust)
        interaction_count = int(business.get('interaction_count', 0))
        if interaction_count > 50:
            trust += 0.2
        elif interaction_count > 10:
            trust += 0.1
        # Cap at 1.0
        return min(trust, 1.0)
    else:
        # For personal users, we could look at message history, but for simplicity
        # we'll use a default medium trust. In a real system, we'd use historical
        # interaction data from message_history.csv.
        return 0.6  # Default trust for personal contacts

def _calculate_group_relevance(enriched: Dict[str, Any]) -> float:
    """Calculate how relevant the group is to the user."""
    group = enriched.get('group', {})
    member_info = enriched.get('member_info', {})
    
    if not group:
        # Direct message (no group) - high relevance by default
        return 0.8
    
    # Check if user is admin in the group (admins might want more notifications)
    admin_status = member_info.get('admin_status', '').lower()
    if admin_status == 'true' or admin_status == 'yes':
        return 0.9
    
    # Check group type or name for relevance (simplified)
    group_type = group.get('group_type', '').lower()
    if 'family' in group_type or 'close' in group_type:
        return 0.8
    elif 'work' in group_type or 'office' in group_type:
        return 0.7
    elif 'school' in group_type or 'college' in group_type:
        return 0.6
    else:
        return 0.5  # Default for unknown groups

def _calculate_time_sensitivity(enriched: Dict[str, Any]) -> float:
    """Calculate time sensitivity based on message timestamp and content."""
    timestamp_dt = enriched.get('timestamp_dt')
    if not timestamp_dt:
        return 0.5  # Default if we can't parse time
    
    # Check if it's during business hours (9 AM - 6 PM) on weekdays
    # This is a simplification; in reality, urgency might be higher outside business hours for emergencies
    hour = timestamp_dt.hour
    weekday = timestamp_dt.weekday()  # Monday is 0, Sunday is 6
    
    # Base score
    sensitivity = 0.5
    
    # Adjust for time of day: early morning or late night might be more urgent for personal messages
    if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
        sensitivity += 0.2
    elif 9 <= hour <= 18:  # Business hours
        sensitivity -= 0.1  # Less urgent during business hours for non-work messages
    
    # Adjust for weekend: might be more urgent for family messages
    if weekday >= 5:  # Saturday or Sunday
        sensitivity += 0.1
    
    
    return min(max(sensitivity, 0.0), 1.0)

def _calculate_content_spam_score(enriched: Dict[str, Any]) -> float:
    """Calculate spam/scam score based on message content."""
    # Get message text - we need to know where the text is stored
    # Assuming the enriched message has a 'text' field or similar
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        # If no text, check media type - voice notes and images might have text via OCR/ASR
        # but we don't have that, so we'll return a low score
        return 0.1
    
    text_lower = text.lower()
    
    # Count spam keyword matches
    spam_matches = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
    # Count urgent keyword matches (might reduce spam score if it's a legitimate urgent message)
    urgent_matches = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    
    # Base score from spam matches
    score = min(spam_matches * 0.1, 0.8)  # Cap spam contribution at 0.8
    
    # Reduce score if urgent keywords are present (could be legitimate emergency)
    if urgent_matches > 0:
        score = max(score - urgent_matches * 0.1, 0.0)
    
    # Check for suspicious patterns
    # Multiple exclamation marks or all caps
    if text.count('!!!') > 0 or text.count('!!') > 2:
        score += 0.1
    if text.isupper() and len(text) > 10:
        score += 0.1
    
    # Check for suspicious URLs or phone numbers
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
        score += 0.1
    if re.search(r'\\b\\d{10,}\\b', text):  # Long numbers could be phone numbers or OTPs
        score += 0.05
    
    return min(score, 1.0)

def _calculate_media_risk(enriched: Dict[str, Any]) -> float:
    """Calculate risk score for media content."""
    media_type = enriched.get('media_type')
    if not media_type:
        return 0.0
    
    # Voice notes and images might be higher risk for spam/scam
    # In a real system, we'd run OCR/ASR and then check the content
    # For now, we'll assign a base risk
    if media_type == 'voice_note':
        return 0.3  # Voice notes can be spam but often personal
    elif media_type == 'image':
        return 0.4  # Images can be spam posters or scams
    else:
        return 0.0

def _check_urgency_keywords(enriched: Dict[str, Any]) -> float:
    """Check if urgency keywords are present in the message."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    if not text:
        return 0.0
    
    text_lower = text.lower()
    for keyword in URGENCY_KEYWORDS:
        if keyword in text_lower:
            return 1.0
    return 0.0

def _is_forwarded(enriched: Dict[str, Any]) -> float:
    """Check if the message is forwarded."""
    # Assuming there's a 'forwarded' field in the original message
    forwarded = enriched.get('forwarded', '') or enriched.get('is_forwarded', '')
    if isinstance(forwarded, str):
        return 1.0 if forwarded.lower() in ('true', 'yes', '1') else 0.0
    elif isinstance(forwarded, bool):
        return 1.0 if forwarded else 0.0
    else:
        # Try to infer from content: common forwarded markers
        text = enriched.get('text', '') or enriched.get('message', '') or ''
        if text.startswith('FW:') or text.startswith('Fwd:') or 'forwarded' in text.lower():
            return 1.0
        return 0.0

def _normalize_message_length(enriched: Dict[str, Any]) -> float:
    """Normalize message length to 0-1 range."""
    text = enriched.get('text', '') or enriched.get('message', '') or ''
    length = len(text)
    # Assume max length of 500 characters for normalization
    return min(length / 500.0, 1.0)