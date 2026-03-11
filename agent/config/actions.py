from nemoguardrails.actions import action
import re
import base64


@action(is_system_action=True)
async def normalize_input(user_input: str):
    # look for base64 encoding
    base64_pattern = r'(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
    matches = re.findall(base64_pattern, user_input)
    
    decoded_parts = []
    for m in matches:
        try:
            decoded_parts.append(base64.b64decode(m).decode('utf-8'))
        except:
            pass
            
    return f"{user_input} [DECODED CONTENT: {' '.join(decoded_parts)}]"

@action(is_system_action=True)
async def check_intent(text: str):
    # TODO: Load a quicker, smaller model to summarize intent of normalized input
    # placeholder follows
    suspicious_markers = ['concatenate', 'part 1', 'part a', 'execute', 'combine']
    if any([s in text for s in suspicious_markers]):
        return True
    return False