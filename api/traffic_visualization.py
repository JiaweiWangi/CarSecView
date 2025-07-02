import re
import time
import torch
import asyncio
import datetime
import aiofiles  # 用于异步文件操作

import json #6.30丁

import numpy as np
from torch import nn
from fastapi import FastAPI 
from api.LSTM import LSTMAutoencoder
from api.car_queue import CarQueue,StrideNode
from sklearn.metrics import confusion_matrix
from api.NLT_main import likelihood_transformation
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os
from typing import Dict, List, Any, Optional
# 全局文件缓存
FILE_CACHE = {}
# 缓存过期时间(秒)，0表示永不过期
CACHE_EXPIRE_TIME = 3600  # 1小时
# 缓存状态记录
CACHE_STATUS = {}
#切换后要清空数据积累重新积累
#先塞30秒数据
#为什么只在切换后右边刷新比左边快
#封面组名 校标 制作人
app = FastAPI()

def init_model():   
    hidden_size = 50
    input_size = 1
    model_file = 'car_hacking_model.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_size, hidden_size, 2)
    model.load_state_dict(torch.load(f'./model/{model_file}', map_location=device))
    model.to(device)
    model.eval()
    return model,device
     
model,device = init_model()
transfer=likelihood_transformation()
transfer.set_global_max(0.09638328545934269)
stride_time = 1
size = int(30 / stride_time)
car_queue=CarQueue(max_len=size, stride=stride_time)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#切换或者重启都不传

async def preload_files():#用于预读取文件
    """预加载所有数据集文件到缓存"""
    attack_types = ["Dos攻击", "模糊攻击", "RPM攻击", "Gear攻击", "正常流量"]
    for attack_type in attack_types:
        try:
            file_path = get_file_path(attack_type)
            await preload_file(file_path, attack_type)
            CACHE_STATUS[attack_type] = {
                "status": "loaded",
                "time": time.time(),
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            print(f"成功预加载文件: {file_path}")
        except Exception as e:
            CACHE_STATUS[attack_type] = {
                "status": "failed",
                "error": str(e),
                "time": time.time()
            }
            print(f"预加载文件 {file_path} 失败: {str(e)}")

async def preload_file(file_path: str, cache_key: str):
    """预加载单个文件到缓存"""
    if file_path.endswith('.csv'):
        process_line = process_csv_line_add
    else:
        process_line = process_txt_line_add
    
    data = []
    async with aiofiles.open(file_path, 'r') as f:
        first_line = True
        line_count = 0
        
        async for line in f:
            if first_line and file_path.endswith('.csv'):
                first_line = False
                continue
            
            processed_data = await process_line(line)
            if processed_data:
                data.append(processed_data)
                line_count += 1
            
            # 分批次处理，避免大文件一次性加载导致内存问题
            if line_count % 1000 == 0:
                await asyncio.sleep(0)  # 让出控制权，避免阻塞
    
    # 存储到缓存，包含数据和加载时间
    FILE_CACHE[cache_key] = {
        "data": data,
        "loaded_time": time.time(),
        "line_count": line_count
    }
    print(f"预加载完成: {file_path}, 行数: {line_count}")

def is_cache_valid(cache_key: str) -> bool:
    """检查缓存是否有效"""
    if cache_key not in FILE_CACHE:
        return False
    
    if CACHE_EXPIRE_TIME == 0:
        return True
    
    return time.time() - FILE_CACHE[cache_key]["loaded_time"] < CACHE_EXPIRE_TIME

def get_cached_data(cache_key: str) -> Optional[List[Any]]:
    """从缓存获取数据"""
    if cache_key in FILE_CACHE and is_cache_valid(cache_key):
        return FILE_CACHE[cache_key]["data"]
    return None

# #时间 ID DLC 数据1 数据2 数据3... 数据n
async def generate_detect_result(file_path: str):
    cache_key = os.path.basename(file_path)
    cached_data = get_cached_data(cache_key)
    stride_node = StrideNode(stride_time)
    start_time = time.time()
    result = []
    label = []
    
    if not cached_data:
        # 如果缓存中没有数据，从文件读取并添加到缓存
        async for data in add2queue(file_path):
            new_data = data
            current_time = new_data[0]  # 每次更新为当前数据的时间戳
            
            if start_time + stride_time > new_data[0]:
                stride_node.add_data(new_data)
            else:
                car_queue.append(stride_node)
                if len(car_queue) == size:
                    result, label = car_queue.get_result()
                start_time += stride_time
                while start_time + stride_time < new_data[0]:
                    stride_node = StrideNode(stride_time)
                    car_queue.append(stride_node)
                    start_time += stride_time
                stride_node = StrideNode(stride_time)
                stride_node.add_data(new_data)

                if(result):
                    z=transfer.out(result) 
                    z=torch.from_numpy(z).float()
                    if z.dim() == 1:
                        z = z.view(1, -1, 1) 
                    z=z.to(device)
                    z_hat=model(z)
                    criterion=nn.MSELoss()
                    loss=criterion(z_hat,z)
                    label_predict=1
                    if loss>0.0004979983381813239:
                        label_predict=0

                    if 0 in label:
                        label=0
                    else:
                        label=1
                        
                    yield json.dumps({
                        "time": datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S'),  # 使用当前数据的时间戳
                        "log": f"ID:{new_data[1]} DLC:{new_data[2]} DATA:{new_data[3]}",
                        "loss": float(loss),
                        "label": label,
                        "predict": label_predict
                    }) + "\n"
    else:
        # 从缓存中读取数据
        for data in cached_data:
            new_data = data
            current_time = new_data[0]  # 每次更新为当前数据的时间戳
            
            if start_time + stride_time > new_data[0]:
                stride_node.add_data(new_data)
            else:
                car_queue.append(stride_node)
                if len(car_queue) == size:
                    result, label = car_queue.get_result()
                start_time += stride_time
                while start_time + stride_time < new_data[0]:
                    stride_node = StrideNode(stride_time)
                    car_queue.append(stride_node)
                    start_time += stride_time
                stride_node = StrideNode(stride_time)
                stride_node.add_data(new_data)

                if(result):
                    z=transfer.out(result) 
                    z=torch.from_numpy(z).float()
                    if z.dim() == 1:
                        z = z.view(1, -1, 1) 
                    z=z.to(device)
                    z_hat=model(z)
                    criterion=nn.MSELoss()
                    loss=criterion(z_hat,z)
                    label_predict=1
                    if loss>0.0004979983381813239:
                        label_predict=0

                    if 0 in label:
                        label=0
                    else:
                        label=1
                        
                    yield json.dumps({
                        "time": datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S'),  # 使用当前数据的时间戳
                        "log": f"ID:{new_data[1]} DLC:{new_data[2]} DATA:{new_data[3]}",
                        "loss": float(loss),
                        "label": label,
                        "predict": label_predict
                    }) + "\n"
#Dos攻击 模糊攻击 RPM攻击 Gear攻击 正常流量
@app.get("/detect_attack")
async def detect_attack(attack_type: str="正常流量"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    return StreamingResponse(generate_detect_result(file_path), media_type="text/event-stream")
      
@app.get("/preload")
async def preload():
    """手动触发预加载所有文件"""
    await preload_files()
    return {"status": "preloading", "cache_status": CACHE_STATUS}

@app.get("/cache_status")
async def cache_status():
    """获取当前缓存状态"""
    return {"cache_status": CACHE_STATUS, "cache_size": len(FILE_CACHE)}

@app.get("/clear_cache")
async def clear_cache():
    """清除所有缓存"""
    global FILE_CACHE, CACHE_STATUS
    FILE_CACHE = {}
    CACHE_STATUS = {}
    return {"status": "cleared", "cache_size": 0}

def get_file_path(attack_type: str) -> str:
    """根据攻击类型获取对应的文件路径"""
    if attack_type == "Dos攻击":
        return "./data/DoS_dataset.csv"
    elif attack_type == "模糊攻击":
        return "./data/Fuzzy_dataset.csv"
    elif attack_type == "RPM攻击":
        return "./data/RPM_dataset.csv"
    elif attack_type == "Gear攻击":
        return "./data/gear_dataset.csv"
    elif attack_type == "正常流量":
        return "./data/normal_run_data.txt"
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
            data_parts = [p for p in parts[3:3+dlc+1]]
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
    line = line.strip()
    if not line:
        return []
    
    # 提取时间戳
    timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', line)
    timestamp = [timestamp_match.group(1)] if timestamp_match else []
    timestamp = [float(t) for t in timestamp]
    
    # 提取ID
    id_match = re.search(r'ID: (\w+)', line)
    message_id = [id_match.group(1)] if id_match else []
    # 提取DLC
    dlc_match = re.search(r'DLC: (\d+)', line)
    dlc = [dlc_match.group(1)] if dlc_match else []
    dlc = [int(d) for d in dlc]
    # 提取数据字节
    data_match = re.search(r'DLC: \d+\s+((?:[0-9a-f]{2}\s+)+(?:[0-9a-f]{2}))', line)
    data_bytes = data_match.group(1).strip().split() if data_match else []
    
    label=['R']
    # 合并所有元素到一个数组
    return timestamp + message_id + dlc + data_bytes+label
    
async def process_csv_line_add(line: str) -> list:
    """处理CSV格式的单行数据"""
    parts = line.strip().split(',')
    if len(parts) >= 12:  # 确保行有足够的字段
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc+1]]
            label=parts[-1]
            data_parts_str=""
            for byte in data_parts:
                data_parts_str+=byte
            result=[timestamp, _id, dlc, data_parts_str,label]
            return result
        except ValueError:
            return None
    elif len(parts) >= 9:  # 处理字段较少的行
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc+1]]
            label=parts[-1]
            data_parts_str=""
            for byte in data_parts:
                data_parts_str+=byte
            result=timestamp + _id + dlc
            result.append(data_parts_str)
            result+=label 
            return result
        except ValueError:
            return None
    return None

async def process_txt_line_add(line: str) -> list:
    line = line.strip()
    if not line:
        return []
    
    # 提取时间戳
    timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', line)
    timestamp = [timestamp_match.group(1)] if timestamp_match else []
    timestamp = [float(t) for t in timestamp]
    
    # 提取ID
    id_match = re.search(r'ID: (\w+)', line)
    message_id = [id_match.group(1)] if id_match else []
    # 提取DLC
    dlc_match = re.search(r'DLC: (\d+)', line)
    dlc = [dlc_match.group(1)] if dlc_match else []
    dlc = [int(d) for d in dlc]
    # 提取数据字节
    data_match = re.search(r'DLC: \d+\s+((?:[0-9a-f]{2}\s+)+(?:[0-9a-f]{2}))', line)
    data_bytes = data_match.group(1).strip().split() if data_match else []
    
    data_bytes_str=""
    for byte in data_bytes:
        data_bytes_str+=byte
    label=['R']
    
    
    result=timestamp + message_id + dlc 
    result.append(data_bytes_str)
    result+=label
    return result
    
async def add2queue(file_path: str):
    # 初始数据，让客户端知道连接已建立
    previous_timestamp = None
    first_data_sent = False
    
    # 根据文件类型选择处理函数
    if file_path.endswith('.csv'):
        process_line = process_csv_line_add
    else:
        process_line = process_txt_line_add
    
    # 使用异步文件操作
    async with aiofiles.open(file_path, 'r') as f:
        first_line = True
        line_count = 0
        time_diff=0
        
        async for line in f:
            # 跳过CSV文件的标题行
            if first_line and file_path.endswith('.csv'):
                first_line = False
                continue
            
            line_count += 1
            # 处理当前行
            processed_data = await process_line(line)
            if not processed_data:
                continue
            timestamp = processed_data[0]
            # 计算时间差
            if previous_timestamp is not None and first_data_sent:
                time_diff = timestamp - previous_timestamp
                if(time_diff<0):
                    time_diff=0
                # 限制最大延迟，避免过长等待
                if time_diff > 1.0:
                    await asyncio.sleep(1.0)  # 最大延迟1秒
                else:
                    await asyncio.sleep(time_diff)
            previous_timestamp = timestamp
            #print(processed_data)
            current_time = time.time()
            new_timestamp=current_time+time_diff
        
            
            #切换数据导致的不同数据集的原始时间戳的先后关系 会将datas中的队列清空 试图用当前时间戳代替原始时间戳
            yield [new_timestamp,processed_data[1],processed_data[2],processed_data[3],processed_data[-1]]
        
async def generate(file_path: str):
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
            current_time = datetime.datetime.fromtimestamp(time.time())
            current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            data_part,time_part=current_time.split(' ',1)
            #这个是label
            label=processed_data[-1]
            #print(label,type(label))
            yield f"date_part:{current_time}\n"
            if('T'in label):
                label=0
            else:
                label=1

            # 构建并发送数据
            # 移除"data: "前缀和JSON结构，直接输出CSV格式
            data_str = (f"{time_part},"
                        f"ID:{processed_data[1]},"
                        f"DLC:{int(processed_data[2])},"
                        f"{','.join(str(processed_data[i]) for i in range(3, int(processed_data[2]) + 3))},"
                        f"{label}\n")
            yield f"{data_str}\n"
            #print(data_str)
            first_data_sent = True


@app.get("/read_dataset")
async def read_dataset(attack_type: str = "Dos攻击"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    return StreamingResponse(generate(file_path), media_type="text/event-stream")

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