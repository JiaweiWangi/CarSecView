from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np
import pandas as pd
from fastapi.responses import StreamingResponse
import datetime
import re
import aiofiles  # 用于异步文件操作
import asyncio
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_file_path(attack_type: str) -> str:
    """根据攻击类型获取对应的文件路径"""
    if attack_type == "Dos攻击":
        return "../test/DoS_dataset.csv"
    elif attack_type == "模糊攻击":
        return "../test/Fuzzy_dataset.csv"
    elif attack_type == "RPM攻击":
        return "../test/RPM_dataset.csv"
    elif attack_type == "Gear攻击":
        return "../test/gear_dataset.csv"
    elif attack_type == "正常流量":
        return "../test/normal_run_data.txt"
    else:
        raise ValueError(f"未知攻击类型: {attack_type}")

async def process_csv_line(line: str) -> list:
    """处理CSV格式的单行数据"""
    parts = line.strip().split(',')
    if len(parts) >= 12:  # 确保行有足够的字段
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc]]
            return [timestamp, _id, dlc] + data_parts
        except ValueError:
            return None
    elif len(parts) >= 9:  # 处理字段较少的行
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc]]
            return [timestamp, _id, dlc] + data_parts
        except ValueError:
            return None
    return None

async def process_txt_line(line: str) -> list:
    """处理TXT格式的单行数据"""
    match = re.match(r'Timestamp:\s*(\d+\.\d+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*(\d+)\s+(.*)', line)
    if not match:
        return None
    
    timestamp = float(match.group(1))
    _id = match.group(2)
    dlc = int(match.group(3))
    data_parts_str = match.group(4).strip()
    data_parts = data_parts_str.split(' ')
    
    # 确保dlc与实际数据长度匹配
    if len(data_parts) != dlc:
        print(f"Warning: DLC mismatch in line: {line.strip()}. Expected {dlc}, got {len(data_parts)}")
        return None
    
    return [timestamp, _id, dlc] + data_parts

@app.get("/read_dataset")
async def read_dataset(attack_type: str = "正常流量"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    async def generate():
        # 初始数据，让客户端知道连接已建立
        yield "data: {\"message\": \"连接已建立，正在加载数据...\"}\n\n"
        
        previous_timestamp = None
        first_data_sent = False
        
        # 根据文件类型选择处理函数
        if file_path.endswith('.csv'):
            process_line = process_csv_line
        else:
            process_line = process_txt_line
        
        # 使用异步文件操作
        async with aiofiles.open(file_path, 'r') as f:
            first_line = True
            line_count = 0
            
            async for line in f:
                # 跳过CSV文件的标题行
                if first_line and file_path.endswith('.csv'):
                    first_line = False
                    continue
                
                line_count += 1
                # 每1000行输出一次加载状态
                if line_count % 1000 == 0:
                    yield f"data: {{\"loading\": \"已加载 {line_count} 行数据\"}}\n\n"
                
                # 处理当前行
                processed_data = await process_line(line)
                if not processed_data:
                    continue
                
                timestamp = processed_data[0]
                
                # 计算时间差
                if previous_timestamp is not None and first_data_sent:
                    time_diff = timestamp - previous_timestamp
                    # 限制最大延迟，避免过长等待
                    if time_diff > 1.0:
                        await asyncio.sleep(1.0)  # 最大延迟1秒
                    else:
                        await asyncio.sleep(time_diff)
                
                previous_timestamp = timestamp
                
                # 格式化时间
                current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                # 构建并发送数据
                # 移除"data: "前缀和JSON结构，直接输出CSV格式
                data_str = (f"{current_time},"
                            f"ID:{processed_data[1]},"
                            f"DLC:{int(processed_data[2])},"
                            f"{','.join(str(processed_data[i]) for i in range(3, int(processed_data[2]) + 3))}\n")
                
                yield data_str
                first_data_sent = True
        
        # 发送结束标记
        yield "data: {\"message\": \"数据传输完成\"}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/new_api_endpoint")
async def new_api_endpoint(attack_type: str):
    print(f"Received request for attack type: {attack_type}")
    if attack_type == "Dos攻击":
        return {"message": "Dos攻击数据已接收并处理！", "data": {"type": "Dos", "status": "active"}}
    elif attack_type == "模糊攻击":
        return {"message": "模糊攻击数据已接收并处理！", "data": {"type": "Fuzzing", "status": "completed"}}
    elif attack_type == "RPM攻击":
        return {"message": "RPM攻击数据已接收并处理！", "data": {"type": "RPM", "status": "completed"}}
    elif attack_type == "Gear攻击":
        return {"message": "Gear攻击数据已接收并处理！", "data": {"type": "Gear", "status": "completed"}}
    elif attack_type == "正常流量":
        return {"message": "正常流量数据已接收并处理！", "data": {"type": "Normal", "status": "completed"}}
    else:
        return {"message": f"未知攻击类型 {attack_type} 已接收！", "data": {"type": "Unknown", "status": "N/A"}}