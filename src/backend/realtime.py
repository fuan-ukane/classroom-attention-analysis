import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import time

# 复用已有模块
from posture_classifier_cnn import classify_posture_cnn, load_model as load_posture_model
from expression_classifier import recognize_expression, load_model as load_expression_model
from fusion import calculate_attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型（只加载一次）
posture_model = load_posture_model('../../models/posture_model.pth')
expression_model = load_expression_model('../../models/expression_model.pth')

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# 学生跟踪状态
student_id_counter = 0
student_positions = {}  # {id: (cx, cy, last_seen_frame)}
MAX_DISAPPEARED = 30    # 30帧未出现则删除

def get_center(bbox):
    """计算边界框中心点"""
    x, y, w, h = bbox
    return (x + w//2, y + h//2)

def assign_id(center, frame_num):
    """为新检测到的人分配ID（简单距离匹配）"""
    global student_id_counter
    # 尝试匹配已有ID
    for sid, (old_center, last_frame) in list(student_positions.items()):
        dist = np.sqrt((center[0]-old_center[0])**2 + (center[1]-old_center[1])**2)
        if dist < 50:  # 50像素内视为同一人
            student_positions[sid] = (center, frame_num)
            return sid
    # 新学生
    student_id_counter += 1
    student_positions[student_id_counter] = (center, frame_num)
    return student_id_counter

def cleanup_students(frame_num):
    """清理长时间未出现的学生"""
    to_remove = []
    for sid, (_, last_frame) in student_positions.items():
        if frame_num - last_frame > MAX_DISAPPEARED:
            to_remove.append(sid)
    for sid in to_remove:
        del student_positions[sid]

def detect_pose_and_face(img_rgb):
    """检测所有人体的姿态和人脸"""
    # 姿态检测（MediaPipe 一次只能检测一个人，需遍历）
    pose_results = pose.process(img_rgb)
    
    # 人脸检测
    face_results = face_detector.process(img_rgb)
    
    persons = []
    if pose_results.pose_landmarks:
        # 单人姿态（简化：取第一个检测到的人体）
        posture = classify_posture_cnn(img_rgb, model=posture_model)
    else:
        posture = 'unknown'
    
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = img_rgb.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            # 提取人脸并识别表情
            face_img = img_rgb[y:y+bh, x:x+bw]
            if face_img.size > 0:
                pil_img = Image.fromarray(face_img)
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                pil_img.save(tmp.name)
                try:
                    expression = recognize_expression(tmp.name, model=expression_model)
                except:
                    expression = 'unknown'
                finally:
                    os.unlink(tmp.name)
            else:
                expression = 'unknown'
            
            center = get_center([x, y, bw, bh])
            persons.append({
                'bbox': [x, y, bw, bh],
                'center': center,
                'posture': posture,
                'expression': expression
            })
    
    return persons

def process_frame(img_bgr, frame_num=0):
    """处理单帧图像，返回分析结果"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 清理过期学生
    cleanup_students(frame_num)
    
    # 检测所有人
    persons = detect_pose_and_face(img_rgb)
    
    results = []
    for person in persons:
        # 分配ID
        sid = assign_id(person['center'], frame_num)
        
        # 计算专注度
        attention = calculate_attention(person['posture'], person['expression'])
        
        results.append({
            'id': sid,
            'posture': person['posture'],
            'expression': person['expression'],
            'attention': attention,
            'bbox': person['bbox']
        })
    
    # 课堂整体专注度
    if results:
        overall = round(np.mean([r['attention'] for r in results]), 1)
    else:
        overall = 0
    
    return {
        'students': results,
        'overall': overall,
        'count': len(results)
    }