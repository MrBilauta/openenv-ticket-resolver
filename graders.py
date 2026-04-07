def grade_category(pred, expected):
    return 1.0 if pred == expected else 0.0


def grade_priority(pred, expected):
    return 1.0 if pred == expected else 0.0


def grade_action(pred, expected):
    return 1.0 if pred == expected else 0.0


def grade_response(text):
    text = text.lower()
    score = 0.0

    if "hello" in text or "hi" in text:
        score += 0.25
    if "sorry" in text:
        score += 0.25
    if "resolve" in text or "refund" in text or "support" in text:
        score += 0.25
    if "thank" in text:
        score += 0.25

    return min(score, 1.0)
