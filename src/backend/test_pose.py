import cv2
import mediapipe as mp
import os
import sys

# 导入我们自己的分类函数
sys.path.append(os.path.dirname(__file__))
from dl_experiments.course_design.src.backend.posture_classifier import classify_posture

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ========== 选择测试图片 ==========
# 先进入 listen 文件夹，再进入第一个子文件夹，取第一张图片
base_path = '../data/class_analysis/listen'
sub_dirs = sorted(os.listdir(base_path))
test_img_path = os.path.join(base_path, sub_dirs[0], os.listdir(os.path.join(base_path, sub_dirs[0]))[0])

print(f"测试图片: {test_img_path}")

img = cv2.imread(test_img_path)
if img is None:
    print(f"无法读取图片: {test_img_path}")
    exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

if results.pose_landmarks:
    # 打印关键点坐标
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    print(f"鼻子: ({nose.x:.2f}, {nose.y:.2f})")
    print(f"左肩: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f})")
    print(f"右肩: ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")

    # 调用姿态分类
    posture = classify_posture(results.pose_landmarks)
    print(f"姿态分类结果: {posture}")
else:
    print("未检测到人体姿态")