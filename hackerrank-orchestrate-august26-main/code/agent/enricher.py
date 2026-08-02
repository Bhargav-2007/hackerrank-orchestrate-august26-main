"""
Enricher module for the Message Notification Router.
Enriches raw message data with contextual information from users, groups, businesses, and media.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def enrich_message(
    message: Dict[str, Any],
    users: dict,
    groups: dict,
    group_members: dict,
    businesses: dict,
    user_business_history: dict,
    images: dict,
    voice_notes: dict
) -> Dict[str, Any]:
    """
    Enrich a message with contextual data.
    
    Args:
        message: Raw message dictionary from CSV
        users: Dictionary of user data keyed by user_id
        groups: Dictionary of group data keyed by group_id
        group_members: Dictionary of group members data (keyed by group_id, but we'll restructure)
        businesses: Dictionary of business account data keyed by business_id
        user_business_history: Dictionary of user-business interaction history
        images: Dictionary of image metadata
        voice_notes: Dictionary of voice note metadata
    
    Returns:
        Enriched message dictionary
    """
    # Create a copy to avoid modifying the original
    enriched = message.copy()
    
    try:
        # Add sender information
        sender_id = message.get('sender_id')
        if sender_id and sender_id in users:
            enriched['sender'] = users[sender_id]
        else:
            enriched['sender'] = {}
        
        # Add group information if it's a group message
        group_id = message.get('group_id')
        if group_id and group_id in groups:
            enriched['group'] = groups[group_id]
            # Get member info for this user in this group
            # Note: group_members.csv has columns: group_id, user_id, admin_status, etc.
            # We'll restructure group_members to be nested: group_id -> list of members
            # But for simplicity, we'll assume group_members is already structured that way
            # If not, we'll restructure on the fly
            if isinstance(group_members, dict) and group_id in group_members:
                # If it's a dict of group_id -> list of members
                if isinstance(group_members[group_id], list):
                    member_info = next((m for m in group_members[group_id] if m.get('user_id') == sender_id), {})
                    enriched['member_info'] = member_info
                else:
                    # If it's a dict of user_id -> member_info (unlikely given our data)
                    enriched['member_info'] = group_members[group_id].get(sender_id, {})
            else:
                enriched['member_info'] = {}
        else:
            enriched['group'] = {}
            enriched['member_info'] = {}
        
        # Add business information if sender is a business
        if sender_id and sender_id in businesses:
            enriched['business'] = businesses[sender_id]
            # Get interaction history with this business
            if sender_id in user_business_history:
                # Assuming user_business_history is keyed by business_id? Let's assume by user_id for simplicity
                # Actually, user_business_history.csv has: user_id, business_id, interaction_count, last_interaction
                # We'll restructure it later if needed, but for now, we'll do a simple lookup
                enriched['business_interaction'] = user_business_history.get(sender_id, {})
            else:
                enriched['business_interaction'] = {}
        else:
            enriched['business'] = {}
            enriched['business_interaction'] = {}
        
        # Add media information if present
        media_id = message.get('media_id')
        if media_id:
            if media_id in images:
                enriched['media'] = images[media_id]
                enriched['media_type'] = 'image'
            elif media_id in voice_notes:
                enriched['media'] = voice_notes[media_id]
                enriched['media_type'] = 'voice_note'
            else:
                enriched['media'] = {}
                enriched['media_type'] = 'unknown'
        else:
            enriched['media'] = {}
            enriched['media_type'] = None
        
        # Add timestamp parsing for time-based features
        timestamp_str = message.get('timestamp')
        if timestamp_str:
            try:
                # Assuming timestamp is in ISO format or similar
                from datetime import datetime
                # Try common formats
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'):
                    try:
                        enriched['timestamp_dt'] = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If none worked, set to None
                    enriched['timestamp_dt'] = None
            except Exception:
                enriched['timestamp_dt'] = None
        else:
            enriched['timestamp_dt'] = None
        
        logger.debug(f"Enriched message {message.get('message_id')}")
        
    except Exception as e:
        logger.error(f"Error enriching message {message.get('message_id')}: {e}")
        # Return the original message with minimal enrichment to avoid breaking the pipeline
        enriched['sender'] = {}
        enriched['group'] = {}
        enriched['member_info'] = {}
        enriched['business'] = {}
        enriched['business_interaction'] = {}
        enriched['media'] = {}
        enriched['media_type'] = None
        enriched['timestamp_dt'] = None
    
    return enriched