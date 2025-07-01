"use client";

import Image from "next/image";
import { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function Home() {
  const [showCover, setShowCover] = useState(true);
  const [apiResponseMessage, setApiResponseMessage] = useState('');
  const [trafficData, setTrafficData] = useState('');
  const controllerRef = useRef(null); // 存储AbortController实例
  const readerRef = useRef(null);    
  const isMountedRef = useRef(true); // 跟踪组件是否已卸载
  const receivedLinesRef = useRef([]); // 使用ref存储行数据，避免闭包问题
  const decoder = new TextDecoder('utf-8');
  const [currentDate, setCurrentDate] = useState('');
  const [datePart, setDatePart] = useState('');
  // 新增：节流时间戳引用
  const throttleTimestampRef = useRef(0);
  // 节流时间间隔（毫秒）
  const throttleInterval = 300; 
  const [lossChartData, setLossChartData] = useState([]); // 定义lossChartData状态

  // 组件卸载时更新标志
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      // 清理所有活动请求
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
      if (readerRef.current) {
        readerRef.current.cancel().catch(e => console.warn("Error canceling reader:", e));
        readerRef.current = null;
      }
    };
  }, []);

  const fetchStreamData = async (data_type) => {
    // 终止之前的请求
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }

    if (readerRef.current) {
      readerRef.current.cancel().catch(e => console.warn("Error canceling previous reader:", e));
      readerRef.current = null;
    }

    let isAborted = false; // 跟踪请求是否被中止
    
    try {
      if (isMountedRef.current) {
      }
      // 2. 创建新的AbortController实例
      controllerRef.current = new AbortController();
      const signal = controllerRef.current.signal;

      const response = await fetch(
        `http://localhost:8000/read_dataset?data_type=${data_type}`,
        { signal }
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 验证响应是否可流式处理
      if (!response.body) {
        throw new Error('响应不包含可读取的body');
      }

      const reader = response.body.getReader();
      readerRef.current = reader; // 存储读取器引用

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const newLines = chunk.split('\n').map(line => line.replace(/^data: /, '')).filter(line => line.trim() !== '');
        newLines.forEach(line => {
          if (line.startsWith('date_part:')) {
            setDatePart(line.replace('date_part:', '').trim());
          } else {
            receivedLinesRef.current = receivedLinesRef.current.concat([line]);
          }
        });
        // 只保留最新的24条数据
        if (receivedLinesRef.current.length > 24) {
          receivedLinesRef.current = receivedLinesRef.current.slice(-24);
        }
        // 更新UI
        if (isMountedRef.current && !isAborted) {
          const coloredLines = receivedLinesRef.current.map(line => {
            if (data_type === 'Dos攻击') return `\x1b[31m${line}\x1b[0m`;
            if (data_type === '模糊攻击') return `\x1b[33m${line}\x1b[0m`;
            if (data_type === 'RPM攻击') return `\x1b[35m${line}\x1b[0m`;
            if (data_type === 'Gear攻击') return `\x1b[34m${line}\x1b[0m`;
            return `\x1b[32m${line}\x1b[0m`;
          });
        setTrafficData(coloredLines.filter(line => line.trim() !== '').join('\n'));
      }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('请求被中止:', error);
      } else {
         console.error('加载数据失败:', error);
        
        if (isMountedRef.current && !isAborted) {
          const errorMessage = error.message || '未知错误';
          setTrafficData(`加载数据失败。错误详情: ${errorMessage}`);
        }
      }
    } finally {
      // 无论如何都清理资源
      controllerRef.current = null;
      if (readerRef.current) {
        try {
          readerRef.current.cancel();
        } catch (e) {
          console.warn("Error canceling reader in finally:", e);
        }
        readerRef.current = null;
        }
    }
  };

  // 将fetchDetectResult函数移到组件内部，以便访问状态
  const fetchDetectResult = async (data_type) => {
    try {
      const response = await fetch(`http://localhost:8000/detect_attack?attack_type=${data_type}`);
      if (!response.body) return;
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        
        lines.forEach(line => {
          const parts = line.split(',');
          if (parts.length >= 2) {
            const timeValue = parts[0];
            const lossValue = parseFloat(parts[1]);
            if (!isNaN(lossValue)) {
              const formattedTime = new Date(parseFloat(timeValue) * 1000).toLocaleTimeString('zh-CN', { hour12: false });
              setLossChartData(prevData => {
                const newData = [...prevData, { time: formattedTime, loss: lossValue }];
                return newData.slice(-50);
              });
            }
          }
        });
      }
    } catch (error) {
      console.error('获取检测结果失败:', error);
    }
  };

  const handleAttackTypeClick = (data_type) => {
    const now = Date.now();
    if (now - throttleTimestampRef.current >= throttleInterval) {
      fetchStreamData(data_type);
      fetchDetectResult(data_type); // 新增：调用数据检测函数
      throttleTimestampRef.current = now;
    }
  };


  if (showCover) {
    return (
      <div style={{
        minHeight: "100vh",
        width: "100vw",
        background: "radial-gradient(circle,rgba(80, 196, 241, 0.23),rgba(0, 105, 252, 0.76))", 
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden"
      }}>
        <div className="flex flex-col items-center justify-center relative z-10" style={{
          background: "rgba(255,255,255,0.92)",
          borderRadius: "2rem",
          boxShadow: "0 8px 32px 0 rgba(2, 2, 6, 0.75)",
          padding: "5rem 7rem",
          minWidth: "350px",
          maxWidth: "90vw"
        }}>
          <div style={{
            width: "100%",
            textAlign: "center",
            fontSize: "4.2rem",
            fontWeight: "bold",
            marginBottom: "2.2rem",
            background: "linear-gradient(90deg,rgb(0, 0, 0) 0%,rgb(1, 6, 10) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            textShadow: "0 4px 24px rgba(33, 150, 243, 0.18)",
            letterSpacing: "2px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            车联网入侵检测系统
          </div>
          <button
            className="px-16 py-7 text-3xl font-bold bg-gradient-to-r from-blue-500 to-cyan-400 text-white rounded-2xl shadow-2xl hover:scale-105 hover:from-blue-600 hover:to-cyan-500 transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-cyan-200"
            style={{ marginTop: "1.5rem", boxShadow: "0 6px 24px 0 rgba(33, 150, 243, 0.13)" }}
            onClick={() => setShowCover(false)}
          >
            开始使用
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-screen p-6 gap-6 bg-blue-50">
      {/* 左侧车辆流量 */}
      <div className="md:w-1/3 bg-white rounded-xl shadow-sm p-6 flex flex-col transition-all duration-300 hover:shadow-md border-4 border-blue-500">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <h2 className="text-xl font-semibold text-gray-800">车辆流量实时监控</h2>
          {datePart && (
            <span className="ml-8 text-base text-gray-500 font-mono relative top-0.5">{datePart}</span>
          )}
        </div>
        
        <div className="flex-1 bg-gray-50 rounded-lg p-4 flex items-center justify-center text-gray-700 text-lg font-mono"  id="traffic-counter">
          <pre className="whitespace-pre-wrap">
            {trafficData.split('\n').reverse().map((line, i) => {
              let color = 'text-black-400';
              if (line.includes('\x1b[31m') & line.includes('T')) color = 'text-red-500';
              else if (line.includes('\x1b[33m')& line.includes('T')) color = 'text-yellow-500';
              else if (line.includes('\x1b[35m')& line.includes('T')) color = 'text-green-500';
              else if (line.includes('\x1b[34m')& line.includes('T')) color = 'text-purple-500';
              
              const cleanLine = line.replace(/\x1b\[[0-9;]*m/g, '');
              return <div key={i} className={`${color}`}>{cleanLine}</div>;
            })}
          </pre>
        </div>
      </div>

      {/* 右侧内容容器 */}
      <div className="md:w-2/3 flex flex-col gap-2">
        {/* 攻击类型按钮组 - 现代胶囊按钮设计 */}
        <div className="flex flex-wrap gap-3">
          {['Dos攻击', '模糊攻击', 'RPM攻击', 'Gear攻击', '正常流量'].map((type) => (
            <button 
              key={type}
              onClick={() => handleAttackTypeClick(type)}
              className="flex-1 min-w-0 px-0 py-4 text-xl font-bold bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-sm hover:shadow transform hover:-translate-y-0.5 transition-all duration-200 text-center"
              style={{maxWidth: '20%'}}
            >
              {type}
            </button>
          ))}
        </div>

        {/* 检测数据展示区 - 现代仪表盘设计 */}
        <div className="flex-1 bg-white rounded-xl shadow-sm p-6 transition-all duration-300 hover:shadow-md">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-800">安全检测数据</h2>
            {apiResponseMessage && (
              <span className="ml-4 p-2 bg-blue-100 text-blue-800 rounded-lg text-base font-semibold">
                  {apiResponseMessage}
                </span>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: '异常流量', value: '0', color: 'red' },
              { label: '可疑连接', value: '0', color: 'yellow' },
              { label: '阻断事件', value: '0', color: 'blue' },
              { label: '系统状态', value: '正常', color: 'green' }
            ].map((item, i) => (
              <div 
                key={i} 
                className="bg-gray-50 p-4 rounded-lg border-l-4 border-l-blue-500"
              >
                <div className="text-sm text-gray-500 mb-1">{item.label}</div>
                <div className="text-2xl font-bold text-gray-800">{item.value.replace(/\d{4}-\d{2}-\d{2}/g, '')}</div>
              </div>
            ))}
          </div>

        </div>
        
        {/* 动态折线图区域 */}
        <div className="flex-1 bg-white rounded-xl shadow-sm p-6 mt-4" style={{ height: '300px' }}>
          <h3 className="text-lg font-semibold mb-2">Loss-时间 动态折线图</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossChartData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} label={{ value: "时间 (秒)", position: "insideBottom", offset: 0 }} />
              <YAxis tick={{ fontSize: 12 }} label={{ value: "Loss", angle: -90, position: "insideLeft" }} domain={[0, 0.08]} />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" dot={true} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}