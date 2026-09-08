def calculate_attention(posture_class, expression_class, alpha=0.7, beta=0.3):
    posture_scores = {
        'listen': 100,
        'write': 70,
        'phone': 10,
        'trance': 20,
        'drink': 40,
        'unknown': 50
    }
    expression_scores = {
        'happy': 100,
        'neutral': 80,
        'surprise': 70,
        'sad': 30,
        'fear': 30,
        'angry': 20,
        'disgust': 20,
        'unknown': 50
    }

    p_score = posture_scores.get(posture_class, 50)
    e_score = expression_scores.get(expression_class, 50)

    # 未知类别兜底：一方未知时，参考另一方
    if posture_class == 'unknown' and expression_class != 'unknown':
        # 姿态未知，表情已知，降低姿态权重
        return round(0.3 * 50 + 0.7 * e_score, 1)
    elif expression_class == 'unknown' and posture_class != 'unknown':
        # 表情未知，姿态已知，提高姿态权重
        return round(0.8 * p_score + 0.2 * 50, 1)
    else:
        return round(alpha * p_score + beta * e_score, 1)