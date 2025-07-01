"use client";

import React, { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

class AbortError extends Error {
  constructor() {
    super("The operation was aborted");
    this.name = "AbortError";
  }
}

const MemoizedChart = React.memo(({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart 
        data={data} 
        margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 12 }} 
          label={{ value: "时间", position: "insideBottom", offset: 0 }} 
        />
        <YAxis 
          tick={{ fontSize: 12 }} 
          label={{ value: "Loss", angle: -90, position: "insideLeft" }} 
          domain={[0, 'dataMax + 0.01']} 
        />
        <Tooltip 
          formatter={(value) => [value.toFixed(6), 'Loss']}
          labelFormatter={(label) => `时间: ${label}`}
        />
        <Line 
          type="monotone" 
          dataKey="loss" 
          stroke="#8884d8" 
          dot={false}
          activeDot={{ r: 4 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}, (prevProps, nextProps) => {
  const prevData = prevProps.data;
  const nextData = nextProps.data;
  
  if (prevData.length !== nextData.length) return false;
  if (prevData.length === 0) return true;
  
  const lastPrev = prevData[prevData.length - 1];
  const lastNext = nextData[nextData.length - 1];
  
  return (
    lastPrev.time === lastNext.time && 
    lastPrev.loss === lastNext.loss && 
    lastPrev.label === lastNext.label && 
    lastPrev.predict === lastNext.predict
  );
});

export default function Home() {
  const [showCover, setShowCover] = useState(true);
  const [apiResponseMessage, setApiResponseMessage] = useState('');
  const [trafficData, setTrafficData] = useState([]);
  const [datePart, setDatePart] = useState('');
  const [lossChartData, setLossChartData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAttackType, setCurrentAttackType] = useState('');
  
  const isMountedRef = useRef(true);
  const receivedLinesRef = useRef([]);
  const decoder = useRef(new TextDecoder('utf-8'));
  const throttleTimestampRef = useRef(0);
  const throttleInterval = 300;
  const abortControllerRef = useRef(new AbortController());

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortControllerRef.current.abort();
    };
  }, []);

  const fetchNewData = async (attackType, signal) => {
    try {
      const response = await fetch(
        `http://localhost:8000/new_api_endpoint?attack_type=${attackType}`,
        { signal }
      );
      const data = await response.json();
      if (isMountedRef.current) {
        setApiResponseMessage(data.message);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('获取数据失败:', error);
        setApiResponseMessage('获取数据失败');
      }
    }
  };

  const fetchStreamData = async (attackType, signal) => {
    try {
      receivedLinesRef.current = [];
      if (isMountedRef.current) {
        setTrafficData([]);
      }

      const response = await fetch(
        `http://localhost:8000/read_dataset?attack_type=${attackType}`,
        { signal }
      );
      
      if (!response.ok) throw new Error(`HTTP错误! 状态码: ${response.status}`);
      if (!response.body) throw new Error('响应不包含可读取的body');

      const reader = response.body.getReader();

      while (true) {
        if (signal.aborted) throw new AbortError();
        
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.current.decode(value, { stream: true });
        const newLines = chunk.split('\n').filter(line => line.trim() !== '');
        
        const newData = [];
        newLines.forEach(line => {
          if (line.startsWith('date_part:')) {
            setDatePart(line.replace('date_part:', '').trim());
          } else if (line.includes(',')) {
            const parts = line.split(',');
            if (parts.length >= 4) {
              const formattedLine = {
                time: parts[0].trim(),
                id: parts[1].replace('ID:', '').trim(),
                dlc: parts[2].replace('DLC:', '').trim(),
                data: parts.slice(3, parts.length - 1).map(part => part.trim()), // 修改部分
                label: parts[parts.length-1] ? parts[parts.length-1].trim() : 'N/A',
                attackType: attackType
              };
              newData.push(formattedLine);
            }
          }
        });

        receivedLinesRef.current = [
          ...receivedLinesRef.current.filter(item => item.attackType === attackType),
          ...newData
        ].slice(-24);

        if (isMountedRef.current) {
          setTrafficData([...receivedLinesRef.current]);
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError' && isMountedRef.current) {
        console.error('加载数据失败:', error);
        setTrafficData([{ error: `加载失败: ${error.message}` }]);
      }
    }
  };

  const fetchDetectResult = async (attackType, signal) => {
    try {
      const response = await fetch(
        `http://localhost:8000/detect_attack?attack_type=${attackType}`,
        { signal }
      );
      if (!response.ok) throw new Error(`请求失败: ${response.status}`);
      if (!response.body) return;

      const reader = response.body.getReader();
      let buffer = '';
      
      while (true) {
        if (signal.aborted) throw new AbortError();
        
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.current.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        
        for (const line of lines) {
          if (!line.trim()) continue;
          
          try {
            const now = Date.now();
            if (now - throttleTimestampRef.current >= throttleInterval) {
              throttleTimestampRef.current = now;
              
              const data = JSON.parse(line);
              if (data.loss !== undefined && isMountedRef.current) {
                const timeStr = data.time ? 
                  new Date(data.time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}) : 
                  new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
                
                setLossChartData(prev => {
                  const newData = [...prev, { 
                    time: timeStr, 
                    loss: parseFloat(data.loss),
                    label: data.label,
                    predict: data.predict,
                    attackType: attackType
                  }];
                  return newData.filter(item => item.attackType === attackType).slice(-30);
                });
              }
            }
          } catch (e) {
            console.warn('解析JSON失败:', e, '原始数据:', line);
          }
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError' && isMountedRef.current) {
        console.error('获取检测结果失败:', error);
        setLossChartData([]);
      }
    }
  };

  const handleAttackTypeClick = async (attackType) => {
    const now = Date.now();
    if (now - throttleTimestampRef.current < throttleInterval) return;
    
    throttleTimestampRef.current = now;
    
    abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;
    
    setTrafficData([]);
    receivedLinesRef.current = [];
    setLossChartData([]);
    setApiResponseMessage('');
    setCurrentAttackType(attackType);
    
    setIsLoading(false);
    
    try {
      await Promise.all([
        fetchNewData(attackType, signal),
        fetchStreamData(attackType, signal),
        fetchDetectResult(attackType, signal)
      ]);
    } catch (error) {
      if (error.name !== 'AbortError' && isMountedRef.current) {
        console.error('请求出错:', error);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
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
      <div className="md:w-1/3 bg-white rounded-xl shadow-sm p-6 flex flex-col transition-all duration-300 hover:shadow-md border-4 border-blue-500">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <h2 className="text-xl font-semibold text-gray-800">车辆流量实时监控</h2>
          {datePart && (
            <span className="ml-8 text-base text-gray-500 font-mono relative top-0.5">{datePart}</span>
          )}
        </div>
        
        <div className="flex-1 bg-gray-50 rounded-lg p-4 flex items-center justify-center text-gray-700 text-lg font-mono">
          <div className="w-full overflow-y-auto max-h-[500px]">
            {trafficData
              .filter(item => item.attackType === currentAttackType)
              .slice()
              .reverse()
              .map((line, i) => {
                const isAttack=line.label == 0;
                const isNormal = line.label == 1;
                return (
                  <div 
                    key={i} 
                    className={`mb-3 p-3 rounded-lg border ${
                      isAttack 
                        ? 'bg-red-100 border-red-300 text-red-800' 
                        : isNormal 
                          ? 'bg-green-100 border-green-300 text-green-800'
                          : 'bg-gray-100 border-gray-300 text-gray-800'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold">{line.time}</span>
                      {isAttack && (
                        <span className="px-2 py-1 bg-red-200 text-red-800 text-xs font-bold rounded-full">
                          异常
                        </span>
                      )}
                      {isNormal && (
                        <span className="px-2 py-1 bg-green-200 text-green-800 text-xs font-bold rounded-full">
                          正常
                        </span>
                      )}
                    </div>
                    <div className="mt-1 grid grid-cols-2 gap-1">
                      <div><span className="font-semibold">ID:</span> {line.id}</div>
                      <div><span className="font-semibold">DLC:</span> {line.dlc}</div>
                    </div>
                    <div className="mt-1">
                      <span className="font-semibold">数据:</span> {line.data}
                    </div>
                    {line.error && (
                      <div className="mt-1 text-red-500 text-sm">{line.error}</div>
                    )}
                  </div>
                );
              })}
            {trafficData.filter(item => item.attackType === currentAttackType).length === 0 && (
              <div className="text-gray-500">
                {isLoading ? '数据加载中...' : '暂无数据，请选择攻击类型'}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="md:w-2/3 flex flex-col gap-2">
        <div className="flex flex-wrap gap-3">
          {['Dos攻击', '模糊攻击', 'RPM攻击', 'Gear攻击', '正常流量'].map((type) => (
            <button 
              key={type}
              onClick={() => handleAttackTypeClick(type)}
              disabled={isLoading}
              className={`flex-1 min-w-0 px-0 py-4 text-xl font-bold text-white rounded-full shadow-sm hover:shadow transform hover:-translate-y-0.5 transition-all duration-200 text-center disabled:opacity-50 ${
                currentAttackType === type && isLoading 
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-700' 
                  : 'bg-gradient-to-r from-blue-500 to-indigo-600'
              }`}
              style={{maxWidth: '20%'}}
            >
              {currentAttackType === type && isLoading ? '加载中...' : type}
            </button>
          ))}
        </div>

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
              { label: '异常流量', value: lossChartData.filter(d => d.predict === 0 && d.attackType === currentAttackType).length, color: 'red' },
              { label: '可疑连接', value: lossChartData.filter(d => d.label === 0 && d.attackType === currentAttackType).length, color: 'yellow' },
              { label: '阻断事件', value: lossChartData.filter(d => d.predict === 0 && d.label === 0 && d.attackType === currentAttackType).length, color: 'blue' },
              { label: '系统状态', value: lossChartData.length > 0 ? '运行中' : '待机', color: 'green' }
            ].map((item, i) => (
              <div 
                key={i} 
                className="bg-gray-50 p-4 rounded-lg border-l-4 border-l-blue-500"
              >
                <div className="text-sm text-gray-500 mb-1">{item.label}</div>
                <div className={`text-2xl font-bold ${
                  item.color === 'red' ? 'text-red-600' :
                  item.color === 'yellow' ? 'text-yellow-600' :
                  item.color === 'blue' ? 'text-blue-600' : 'text-green-600'
                }`}>
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="flex-1 bg-white rounded-xl shadow-sm p-6 mt-4" style={{ height: '300px' }}>
          <h3 className="text-lg font-semibold mb-2">Loss-时间 动态折线图</h3>
          {lossChartData.filter(item => item.attackType === currentAttackType).length > 0 ? (
            <MemoizedChart data={lossChartData.filter(item => item.attackType === currentAttackType)} />
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              {isLoading ? '数据加载中...' : '暂无数据，请选择攻击类型'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}