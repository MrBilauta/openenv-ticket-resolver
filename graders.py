def safe_score(x):
    try:
        x = float(x)
    except:
        return 0.0001

    return max(0.0001, min(0.998, x))


def grade_category(pred, expected):
    return 0.998 if pred == expected else 0.0001


def grade_priority(pred, expected):
    return 0.998 if pred == expected else 0.0001


def grade_action(pred, expected):
    return 0.998 if pred == expected else 0.0001


def grade_response(text):
    text = text.lower()
    score = 0.0001

    if any(w in text for w in ["hello", "hi", "dear"]):
        score += 0.2

    if any(w in text for w in ["sorry", "apologize", "understand"]):
        score += 0.2

    if any(w in text for w in ["please", "try", "reset", "check", "refund", "resolve"]):
        score += 0.3

    if len(text) > 50:
        score += 0.15

    if any(w in text for w in ["thank", "regards", "support"]):
        score += 0.15

    return safe_score(score)
