import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  // 原有状态
  const [result, setResult] = useState(null);       // 多人结果：{ students, overall, count }
  const [history, setHistory] = useState([]);
  const [alertMsg, setAlertMsg] = useState('');
  const [preview, setPreview] = useState(null);
  const [records, setRecords] = useState([]);
  const [avgAttention, setAvgAttention] = useState(0);

  // 实时分析状态
  const [isRealTime, setIsRealTime] = useState(false);
  const [overallAttention, setOverallAttention] = useState(0);
  const [studentCount, setStudentCount] = useState(0);
  const realTimeRef = useRef(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await axios.get('/api/stats');
      setRecords(res.data.history || []);
      setAvgAttention(res.data.average || 0);
    } catch (err) {
      console.log('统计接口暂未就绪');
    }
  }, []);

  useEffect(() => {
    fetchStats();
    return () => {
      if (realTimeRef.current) clearInterval(realTimeRef.current);
    };
  }, [fetchStats]);

  // 图片上传处理（适配多人结果）
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const data = res.data;
      // data 格式：{ students: [...], overall: 85, count: 3 }
      setResult(data);

      // 将整体专注度加入曲线
      if (data.overall !== undefined) {
        setHistory(prev => [...prev.slice(-19), data.overall]);
      }

      // 整体专注度过低警报
      if (data.overall < 40) {
        setAlertMsg('⚠ 整体专注度过低，请注意！');
      } else {
        setAlertMsg('');
      }

      fetchStats();
    } catch (err) {
      console.error('上传失败', err);
      alert('上传失败，请确认后端已启动。');
    }
  };

  // 实时分析：开启/停止（保持不变）
  const toggleRealTime = async () => {
    if (isRealTime) {
      setIsRealTime(false);
      if (realTimeRef.current) {
        clearInterval(realTimeRef.current);
        realTimeRef.current = null;
      }
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        setIsRealTime(true);
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        realTimeRef.current = setInterval(async () => {
          if (!video.videoWidth) return;
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);

          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.7));
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');

          try {
            const res = await axios.post('/api/realtime', formData);
            setOverallAttention(res.data.overall || 0);
            setStudentCount(res.data.count || 0);
            if (res.data.overall !== undefined) {
              setHistory(prev => [...prev.slice(-19), res.data.overall]);
            }
          } catch (e) {
            console.error('实时帧发送失败', e);
          }
        }, 1000);
      } catch (err) {
        alert('无法打开摄像头：' + err.message);
      }
    }
  };

  // 曲线数据与选项
  const chartData = {
    labels: history.map((_, i) => i + 1),
    datasets: [{
      label: '专注度分数',
      data: history,
      fill: false,
      borderColor: '#4a90d9',
      backgroundColor: '#4a90d9',
      tension: 0.2,
    }],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: '专注度' } },
      x: { title: { display: true, text: '检测次数' } },
    },
    plugins: {
      legend: { display: false },
      title: { display: true, text: '课堂注意力关注曲线' },
    },
  };

  const getAttentionColor = (score) => {
    if (score < 40) return '#e74c3c';
    if (score < 70) return '#f39c12';
    return '#2ecc71';
  };

  const formatDateTime = (timeStr) => {
    if (!timeStr) return '-';
    const d = new Date(timeStr);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  };

  const thStyle = {
    borderBottom: '2px solid #ddd',
    padding: '6px',
    textAlign: 'center',
    backgroundColor: '#f2f2f2',
    fontSize: 14,
  };
  const tdStyle = {
    borderBottom: '1px solid #ddd',
    padding: '6px',
    textAlign: 'center',
    fontSize: 13,
  };

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gridTemplateRows: '1fr 1fr',
      height: '100vh',
      width: '100vw',
      gap: '24px',
      padding: '10px',
      boxSizing: 'border-box',
      background: 'linear-gradient(135deg, #9b59b6, #f1c40f)',
      fontFamily: 'Arial'
    }}>
      
      {/* 左上：说明 + 上传 + 实时分析 */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}>
        <div>
          <h2 style={{ marginTop: 0, color: '#333' }}>🎓 课堂专注度智能分析系统</h2>
          <div style={{ fontSize: 14, color: '#555', lineHeight: 1.6 }}>
            <p>本系统基于多任务模型，融合<b>姿态估计</b>与<b>表情识别</b>，实时评估学生课堂专注度。</p>
            <p><b>可识别姿态：</b>听课(listen)、写字(write)、玩手机(phone)、走神(trance)、喝水(drink)</p>
            <p><b>可识别表情：</b>高兴、中性、惊讶、悲伤、恐惧、愤怒、厌恶</p>
            <p style={{ marginBottom: 0 }}>上传课堂照片，即可获得专注度评分。</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: 15 }}>
          <label style={{
            display: 'inline-block',
            padding: '12px 28px',
            backgroundColor: '#4a90d9',
            color: 'white',
            borderRadius: 25,
            fontSize: 16,
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'background-color 0.3s',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            textAlign: 'center'
          }}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#357abd'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#4a90d9'}
          >
            📷 上传图片
            <input type="file" accept="image/*" onChange={handleUpload} 
              style={{ display: 'none' }} 
            />
          </label>

          <button 
            onClick={toggleRealTime}
            style={{
              padding: '12px 28px',
              backgroundColor: isRealTime ? '#e74c3c' : '#27ae60',
              color: 'white',
              borderRadius: 25,
              fontSize: 16,
              fontWeight: 'bold',
              cursor: 'pointer',
              border: 'none',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              transition: 'background-color 0.3s'
            }}
          >
            {isRealTime ? '⏹ 停止实时分析' : '▶ 开始实时分析'}
          </button>
          {isRealTime && (
            <div style={{ fontSize: 14, color: '#333', textAlign: 'center' }}>
              <span>课堂整体专注度：<b>{overallAttention}</b></span> | 
              <span> 检测人数：<b>{studentCount}</b></span>
            </div>
          )}
        </div>
      </div>

      {/* 右上：图片预览 + 多人检测结果 */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {!preview && !result && !isRealTime && (
          <p style={{ color: '#999' }}>上传图片后，预览和结果将显示在这里</p>
        )}
        {isRealTime && !preview && (
          <p style={{ color: '#27ae60' }}>📹 实时分析中，整体专注度: {overallAttention}</p>
        )}
        {preview && (
          <img src={preview} alt="预览" style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 8, marginBottom: 15 }} />
        )}
        {result && !isRealTime && (
          <div style={{ width: '100%' }}>
            <p style={{ fontWeight: 'bold', fontSize: 16, marginBottom: 10 }}>
              检测到 {result.count} 人，课堂整体专注度：{result.overall}
            </p>
            {result.students.map((student, index) => (
              <div key={index} style={{ 
                border: '1px solid #eee', 
                padding: 10, 
                marginBottom: 8, 
                borderRadius: 8,
                backgroundColor: '#fafafa'
              }}>
                <p style={{ margin: 2 }}><strong>👤 学生 #{student.id}</strong></p>
                <p style={{ margin: 2 }}>姿态：{student.posture} | 表情：{student.expression}</p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span>专注度：</span>
                  <div style={{ 
                    flex: 1, 
                    height: 20, 
                    backgroundColor: '#eee', 
                    borderRadius: 10, 
                    overflow: 'hidden' 
                  }}>
                    <div style={{
                      width: `${student.attention}%`,
                      height: '100%',
                      backgroundColor: getAttentionColor(student.attention),
                      transition: 'width 0.3s',
                      borderRadius: 10,
                    }}></div>
                  </div>
                  <span style={{ fontWeight: 'bold' }}>{student.attention}</span>
                </div>
              </div>
            ))}
            {alertMsg && <div style={{ color: 'red', fontWeight: 'bold', marginTop: 10 }}>{alertMsg}</div>}
          </div>
        )}
      </div>

      {/* 左下：注意力曲线 */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        position: 'relative',
      }}>
        <h3 style={{ margin: '0 0 10px', color: '#333' }}>📈 注意力关注曲线</h3>
        <div style={{ width: '100%', height: 'calc(100% - 40px)' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* 右下：历史记录 */}
      <div style={{
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h3 style={{ margin: 0, color: '#333' }}>📋 历史记录</h3>
          {avgAttention > 0 && (
            <span style={{ fontSize: 14, backgroundColor: '#f0f0f0', padding: '4px 12px', borderRadius: 12 }}>
              平均专注度：<b>{avgAttention}</b>
            </span>
          )}
        </div>
        {records.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                <th style={thStyle}>姿态</th>
                <th style={thStyle}>表情</th>
                <th style={thStyle}>专注度</th>
                <th style={thStyle}>时间</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r, i) => (
                <tr key={i}>
                  <td style={tdStyle}>{r.posture}</td>
                  <td style={tdStyle}>{r.expression}</td>
                  <td style={tdStyle}>{r.attention}</td>
                  <td style={tdStyle}>{formatDateTime(r.time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: '#999', textAlign: 'center', marginTop: 20 }}>暂无检测记录</p>
        )}
      </div>
    </div>
  );
}

export default App;