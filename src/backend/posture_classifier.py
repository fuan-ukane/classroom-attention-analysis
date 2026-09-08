# src/posture_classifier.py
import mediapipe as mp

mp_pose = mp.solutions.pose

def classify_posture(landmarks):
    """根据关键点返回姿态类别字符串"""
    if landmarks is None:
        return 'unknown'
    
    nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # 改前：
    # # 举手判断（手腕在肩膀上方）
    # if left_wrist.y < mid_shoulder_y or right_wrist.y < mid_shoulder_y:
    #     return 'hand_raise'
    
    # # 趴桌判断（鼻子比肩膀低很多）
    # if nose.y > mid_shoulder_y + 0.15:
    #     return 'lying'
    
    # # 左顾右盼（鼻子偏离中心）
    # if abs(nose.x - mid_shoulder_x) > 0.15:
    #     return 'look_around'
    
    # # 低头写字（鼻子低于眼睛）
    # eye_avg_y = (left_eye.y + right_eye.y) / 2
    # if nose.y > eye_avg_y + 0.05:
    #     return 'writing'
    
    # return 'sit_up'  # 抬头正坐
    
    
    #phone 和 drink 通过关键点规则较难区分，
    # 可以暂时使用 trance 或 write 代替，或者我们后面用训练一个轻量 CNN 来分类，但目前先用规则完成实验。


    # 举手判断
    if left_wrist.y < mid_shoulder_y or right_wrist.y < mid_shoulder_y:
        return 'hand_raise'   # 注意：原数据集中没有这个类，但我们可以保留，后续可合并到 'write' 或单独处理
    # 趴桌/睡觉 → 走神
    if nose.y > mid_shoulder_y + 0.15:
        return 'trance'
    # 左顾右盼
    if abs(nose.x - mid_shoulder_x) > 0.15:
        return 'trance'
    # 低头写字 → write
    eye_avg_y = (left_eye.y + right_eye.y) / 2
    if nose.y > eye_avg_y + 0.05:
        return 'write'
    return 'listen'
