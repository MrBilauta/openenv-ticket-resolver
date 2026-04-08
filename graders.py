def safe_score(x):
    return min(max(x, 0.05), 0.95)


def grade_category(pred, expected):
    if pred == expected:
        return 0.95
    return 0.05


def grade_priority(pred, expected):
    if pred == expected:
        return 0.95
    return 0.05


def grade_action(pred, expected):
    if pred == expected:
        return 0.95
    return 0.05


def grade_response(text):
    text = text.lower()
    score = 0.05  # start safely above 0

    # Greeting
    if any(word in text for word in ["hello", "hi", "dear"]):
        score += 0.18

    # Apology
    if any(word in text for word in ["sorry", "apologize", "understand"]):
        score += 0.18

    # Solution clarity
    if any(word in text for word in ["please", "try", "reset", "check", "refund", "resolve"]):
        score += 0.27

    # Professional tone
    if len(text) > 50:
        score += 0.14

    # Closing
    if any(word in text for word in ["thank", "regards", "support"]):
        score += 0.13

    return safe_score(score)
