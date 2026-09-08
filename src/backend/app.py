import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import tempfile
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from PIL import Image
import torch

# 导入多人分析函数
from realtime import process_frame

sys.path.append(os.path.dirname(__file__))

# ------------------ 初始化 Flask 和 SocketIO ------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ------------------ 设备 & 模型 ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=False, device=device, min_face_size=40)

# 加载模型
from posture_classifier_cnn import load_model as load_posture_model
from expression_classifier import load_model as load_expression_model

expression_model = load_expression_model('../../models/expression_model.pth')
posture_model = load_posture_model('../../models/posture_model.pth')

# MediaPipe 备用
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# ------------------ 数据库 ------------------
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  posture TEXT,
                  expression TEXT,
                  attention REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ------------------ 辅助函数 ------------------
def get_pose_landmarks(img_rgb):
    results = pose.process(img_rgb)
    if results.pose_landmarks is None:
        img_flip = cv2.flip(img_rgb, 1)
        results = pose.process(img_flip)
    return results.pose_landmarks

# ------------------ 路由：单张图片上传（改为多人分析） ------------------
@app.route('/api/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # 调用多人分析函数
    result = process_frame(img)

    # 存储每个学生的记录到数据库
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    for student in result['students']:
        c.execute("INSERT INTO detections (posture, expression, attention, timestamp) VALUES (?, ?, ?, ?)",
                  (student['posture'], student['expression'], student['attention'], datetime.now().isoformat()))
    conn.commit()
    conn.close()

    # 返回多人结果，同时保留整体专注度
    return jsonify({
        'students': result['students'],
        'overall': result['overall'],
        'count': result['count']
    })

# ------------------ 路由：历史统计 ------------------
@app.route('/api/stats')
def stats():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT posture, expression, attention, timestamp FROM detections ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    c.execute("SELECT AVG(attention) FROM detections")
    avg = c.fetchone()[0]
    conn.close()

    history = [
        {
            'posture': r[0],
            'expression': r[1],
            'attention': r[2],
            'time': r[3]
        }
        for r in rows
    ]
    return jsonify({
        'history': history,
        'average': round(avg, 1) if avg else 0
    })

# ------------------ 路由：实时分析（保持原样） ------------------
@app.route('/api/realtime', methods=['POST'])
def realtime():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    result = process_frame(img)
    return jsonify(result)

# ------------------ 启动 ------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)