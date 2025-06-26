"use client";

import Image from "next/image";
import { useEffect, useState, useRef } from "react";
import { useState as useReactState } from "react";

export default function Home() {
  const [showCover, setShowCover] = useReactState(true);
  const [apiResponseMessage, setApiResponseMessage] = useState('');
  const [trafficData, setTrafficData] = useState('');
  const controllerRef = useRef(null); // 存储AbortController实例
  const readerRef = useRef(null);    
  const isMountedRef = useRef(true); // 跟踪组件是否已卸载
  const receivedLinesRef = useRef([]); // 使用ref存储行数据，避免闭包问题
  const decoder = new TextDecoder('utf-8');

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

  const fetchNewData = async (attackType) => {
    try {
      const response = await fetch(`http://localhost:8000/new_api_endpoint?attack_type=${attackType}`);
      const data = await response.json();
      console.log(data);
      if (isMountedRef.current) {
        setApiResponseMessage(data.message);
      }
    } catch (error) {
      console.error('Failed to fetch new data:', error);
      setApiResponseMessage('Failed to fetch new data.');
    }
  };

  const fetchStreamData = async (attackType) => {
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
        setTrafficData('正在加载数据...');
      }
      // 2. 创建新的AbortController实例
      controllerRef.current = new AbortController();
      const signal = controllerRef.current.signal;

      const response = await fetch(
        `http://localhost:8000/read_dataset?attack_type=${attackType}`,
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
        receivedLinesRef.current = receivedLinesRef.current.concat(newLines);

        // 只保留最新的20条数据
        if (receivedLinesRef.current.length > 20) {
          receivedLinesRef.current = receivedLinesRef.current.slice(-20);
        }
        
        // 更新UI
        if (isMountedRef.current && !isAborted) {
          const coloredLines = receivedLinesRef.current.map(line => {
            if (attackType === 'Dos攻击') return `\x1b[31m${line}\x1b[0m`;
            if (attackType === '模糊攻击') return `\x1b[33m${line}\x1b[0m`;
            if (attackType === 'RPM攻击') return `\x1b[35m${line}\x1b[0m`;
            if (attackType === 'Gear攻击') return `\x1b[34m${line}\x1b[0m`;
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

  const handleAttackTypeClick = (attackType) => {
    fetchNewData(attackType);
    fetchStreamData(attackType);
  };

  if (showCover) {
    return (
      <div style={{
        minHeight: "100vh",
        width: "100vw",
        background: "url('/circuit-bg.jpg') center center / cover no-repeat, #e3f2fd",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden"
      }}>
        <div className="flex flex-col items-center justify-center relative z-10" style={{
          background: "rgba(255,255,255,0.92)",
          borderRadius: "2rem",
          boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.13)",
          padding: "3rem 4rem",
          minWidth: "350px",
          maxWidth: "90vw"
        }}>
          <div style={{
            width: "100%",
            textAlign: "center",
            fontSize: "3.2rem",
            fontWeight: "bold",
            marginBottom: "2.2rem",
            background: "linear-gradient(90deg, #6ec6ff 0%, #2196f3 100%)",
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
            className="px-12 py-5 text-2xl font-bold bg-gradient-to-r from-blue-500 to-cyan-400 text-white rounded-2xl shadow-2xl hover:scale-105 hover:from-blue-600 hover:to-cyan-500 transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-cyan-200"
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
    <div className="flex flex-col md:flex-row h-screen p-6 gap-6 bg-gray-50">
      {/* 左侧车辆流量 */}
      <div className="md:w-1/3 bg-white rounded-xl shadow-sm p-6 flex flex-col transition-all duration-300 hover:shadow-md">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <h2 className="text-xl font-semibold text-gray-800">车辆流量实时监控</h2>
        </div>
        
        <div className="flex-1 bg-gray-50 rounded-lg p-4 flex items-center justify-center text-gray-700 text-lg font-mono overflow-auto"  id="traffic-counter">
          <pre className="whitespace-pre-wrap">
            {trafficData.split('\n').map((line, i) => {
              let color = 'text-gray-700';
              if (line.includes('\x1b[31m')) color = 'text-red-500';
              else if (line.includes('\x1b[33m')) color = 'text-yellow-500';
              else if (line.includes('\x1b[35m')) color = 'text-purple-500';
              else if (line.includes('\x1b[34m')) color = 'text-blue-500';
              else if (line.includes('\x1b[32m')) color = 'text-green-500';
              
              const cleanLine = line.replace(/\x1b\[[0-9;]*m/g, '');
              return <div key={i} className={`${color}`}>{cleanLine}</div>;
            })}
          </pre>
        </div>
      </div>

      {/* 右侧内容容器 */}
      <div className="md:w-2/3 flex flex-col gap-6">
        {/* 攻击类型按钮组 - 现代胶囊按钮设计 */}
        <div className="flex flex-wrap gap-3">
          {['Dos攻击', '模糊攻击', 'RPM攻击', 'Gear攻击', '正常流量'].map((type) => (
            <button 
              key={type}
              onClick={() => handleAttackTypeClick(type)}
              className="px-4 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 
              text-white text-sm font-medium rounded-full shadow-sm hover:shadow 
              transform hover:-translate-y-0.5 transition-all duration-200"
            >
              {type}
            </button>
          ))}
        </div>

        {/* 检测数据展示区 - 现代仪表盘设计 */}
        <div className="flex-1 bg-white rounded-xl shadow-sm p-6 transition-all duration-300 hover:shadow-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">安全检测数据</h2>
          {apiResponseMessage && (
            <div className="mb-4 p-3 bg-blue-100 text-blue-800 rounded-lg">
              {apiResponseMessage}
            </div>
          )}
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
                <div className="text-2xl font-bold text-gray-800">{item.value}</div>
              </div>
            ))}
          </div>
          {/* 攻击数据展示容器 */}
          <div className="mt-8">
            <div className="text-base font-semibold text-red-600 mb-2">检测攻击数据</div>
            <div className="p-4 bg-red-50 rounded-lg min-h-[60px] border border-red-200 text-red-700 text-base font-mono shadow-inner">
              {/* 这里将来可动态渲染检测到的攻击数据 */}
              暂无检测到的攻击数据。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
