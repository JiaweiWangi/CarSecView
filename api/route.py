import threading
import time
import queue
import os
import asyncio
import json # 引入 json 库
import datetime
from tracemalloc import start
import torch

from fastapi import FastAPI, HTTPException,Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from api.data_processing import process_data # 假设你的处理函数在这里
from api.LSTM import init_model
from api.car_queue import CarQueue,StrideNode
from api.NLT_main import likelihood_transformation

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title="车联网入侵检测系统后端",
    description="一个使用FastAPI和多线程的流式数据处理服务",
    version="1.0.0",
)

# 2. 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 全局变量
# 标准队列，用于同步的线程间通信 (API -> Worker)
data_in_queue = queue.Queue()
# 异步队列，用于 Worker -> API 的通信
results_out_queue = asyncio.Queue()
# 全局变量，用于存储启动时加载的数据
loaded_data = {}
# 全局变量，用于存储主线程的事件循环，以便工作线程可以安全地调用异步代码
main_loop = None
# 全局变量，用于存储模型和设备
model=None
device=None
stride_time = 1
size = int(30 / stride_time)
car_queue=CarQueue(max_len=size, stride=stride_time)


# --- 工作线程定义 ---
def worker_thread_task():
    """
    工作线程：获取原始数据，进行处理，并将结果放入输出队列。
    
    """
    model,device = init_model()
    print(f"👍 鲁棒的LSTM模型加载完成")
    # print("✅ 工作线程已启动，等待数据...")
    loss=0
    label_predict=0
    result=None
    label=None
    criterion=torch.nn.MSELoss()
    transfer=likelihood_transformation()
    transfer.set_global_max(0.1086178408236927)
    stride_node = StrideNode(stride_time)
    start_time = 0
    while True:
        # 从输入队列获取原始数据项
        # data_type 用于了解上下文, item 是具体的数据行
        data_type, new_data = data_in_queue.get()
        if start_time + stride_time > new_data[0]:
            # print(start_time,new_data[0])
            stride_node.add_data(new_data)
        else:
            car_queue.append(stride_node)
            if len(car_queue) == size:
                result, label = car_queue.get_result()
                stride_node = StrideNode(stride_time)
                stride_node.add_data(new_data)
                z=transfer.out(result) 
                z=torch.from_numpy(z).float()
                if z.dim() == 1:
                    z = z.view(1, -1, 1) 
                z=z.to(device)
                z_hat=model(z)
                loss=criterion(z_hat,z)
                print(loss)
                label_predict=1
                if loss>0.0007360850974392888:
                    label_predict=0
                        # 准备要返回给前端的处理结果
                result_data = {
                    "time": time.time(),
                    "loss": loss,
                    "original_label": data_type,
                    "predicted_label":label_predict
                }
                
                # ‼️ 关键: 从同步的 worker 线程中，安全地将结果放入异步的 results_out_queue
                # 我们必须使用 run_coroutine_threadsafe，因为它能确保线程安全
                if main_loop and not main_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(results_out_queue.put(result_data), main_loop)


            start_time += stride_time
            while start_time + stride_time < new_data[0]:
                stride_node = StrideNode(stride_time)
                car_queue.append(stride_node)
                start_time += stride_time
            
        if new_data is None:
            print("工作线程收到结束信号，正在退出...")
            # 向结果队列也放入一个终止信号，以防有监听器在等待
            if main_loop and not main_loop.is_closed():
                asyncio.run_coroutine_threadsafe(results_out_queue.put(None), main_loop)
            break

    print("🛑 工作线程已停止。")


# --- FastAPI 生命周期事件：应用启动时执行 ---
@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行的初始化函数。
    """
    global main_loop
    print("--- ‼️ 系统初始化阶段开始 ‼️ ---")
    main_loop = asyncio.get_running_loop()
    
    # 任务1: 加载所有数据集
    print("➡️ 步骤 1/2: 正在加载和处理数据集...")
    # 定义数据集路径
    datasets = {
        "正常流量": './data/normal_run_data.txt',
        # "Dos攻击": './data/DoS_dataset.csv',
        # "模糊攻击": './data/Fuzzy_dataset.csv',
        # "Gear攻击": './data/Gear_dataset.csv',
        # "RPM攻击": './data/RPM_dataset.csv'
    }
    
    # 循环加载所有数据
    for name, path in datasets.items():
        if os.path.exists(path):
            # 使用你的 process_data 函数加载数据
            loaded_data[name] = process_data(path)
            print(f"  ✅ {name} - 加载成功")
        else:
            print(f"  ❌ {name} - 文件未找到: {path}")

    print(f"👍 数据集加载完成。已加载的类型: {list(loaded_data.keys())}")

    # 启动工作线程
    print("➡️ 步骤 2/2: 正在启动后台工作线程...")
    worker = threading.Thread(target=worker_thread_task, daemon=True)
    worker.start()
    print("👍 后台工作线程已成功启动。")
    print("--- ✅ 系统初始化阶段完成 ---")

@app.on_event("shutdown")
async def shutdown_event():
    # 向输入队列发送终止信号
    data_in_queue.put((None, None))
    print("\n--- 系统正在关闭，已向工作线程发送停止信号 ---")


# --- API 端点定义 ---

# ✅ 修正后的 /read_dataset 端点
@app.get("/read_dataset")
async def stream_dataset(data_type: str):
    """
    根据数据类型，流式返回对应的数据集内容。
    输出的每一行都包含了 ANSI 颜色代码，以供前端解析。
    """
    if data_type not in loaded_data:
        raise HTTPException(status_code=404, detail=f"请求的数据类型 '{data_type}' 未在后端加载")

    # 定义ANSI颜色代码，用于在文本流中嵌入颜色信息
    COLOR_RED = "\u001b[31m"    # 红色，通常用于表示异常/攻击
    COLOR_GREEN = "\u001b[32m"  # 绿色，通常用于表示正常
    COLOR_RESET = "\u001b[0m"   # 重置代码，恢复默认颜色

    async def data_generator():
        data_to_stream = loaded_data[data_type]
        print(f"\n▶️ 开始推送 '{data_type}' 数据流 (带颜色)...")
        
        previous_item = None # 用于计算与上一个数据点的时间差
        time_start = time.time()
        for item in data_to_stream:
            # --- 1. 计算与上一个数据点的时间差，以模拟真实的时间流 ---
            sleep_duration = 0.05  # 如果是第一行或时间戳无效，则使用一个固定的短间隔
            
            if previous_item is not None:
                try:
                    # 确保时间戳是浮点数类型以便计算
                    current_timestamp = float(item[0])
                    previous_timestamp = float(previous_item[0])
                    delta = current_timestamp - previous_timestamp
                    time_end =time.time()
                    # 确保等待时间非负
                    if time_end - time_start>0:
                        delta-=(time_end-time_start)
                    sleep_duration = max(0, delta)
                    time_start = time.time()

                except (ValueError, TypeError, IndexError):
                    print(f"警告：无法解析时间戳或数据项格式错误: {item}")
                    pass # 使用默认间隔
            
            await asyncio.sleep(sleep_duration)
            
            # --- 2. 准备要发送给前端的单行数据 ---
            
            # 确定标签和对应的颜色 (假设 '1' 或 'R' 代表正常, 其他为异常)
            # 注意：item[-1] 的值应该是字符串 '0' 或 '1'
            label = item[-1].strip() 
            if label == 'R': 
                label = '1' # 将 'R' 统一为 '1'
            else: 
                label = '0'

            color_code = COLOR_GREEN if label == '1' else COLOR_RED
    
            # 构建将发送给前端的原始文本行
            # 格式: 时间戳 ID:xxx DLC:x Data:xx xx xx xx 标签
            # raw_line = f"{time.time()} ID:{item[1]} DLC:{int(item[2])} {item[3]},{item[-1]}"
            raw_line = f"{item[0]} ID:{item[1]} DLC:{int(item[2])} {item[3]},{item[-1]}"

            # --- 3. 组合最终的流式字符串 ---
            # 格式: [颜色代码]原始文本[重置颜色代码][换行符]
            data_str = f"{color_code}{raw_line}{COLOR_RESET}\n"
            data_in_queue.put((data_type, item))
            yield data_str.encode("utf-8")
            
            # 更新 previous_item 以供下一次循环计算时间差
            previous_item = item

        print(f"⏹️ '{data_type}' 数据流推送完毕。")

    # 返回一个流式响应，FastAPI 会自动处理这个生成器
    return StreamingResponse(data_generator(), media_type="text/event-stream")


# ✅ 新增的 /detect_attack 端点 (流式)
@app.get("/detect_attack")
async def stream_detections(request: Request):
    """
    职责:
    1. 作为一个长连接，持续等待 `results_out_queue` 中的新数据。
    2. 将从工作线程收到的处理结果，流式推送到前端。
    """
    async def detection_generator():
        print(f"\n▶️ 前端已连接，开始监听处理结果...")
        try:
            while True:
                # 异步地从结果队列中获取一个项目
                # 如果队列为空，它会在这里等待，而不会阻塞服务器
                result = await results_out_queue.get()

                if result is None: # 如果收到终止信号
                    break

                # 检查客户端是否已断开连接
                if await request.is_disconnected():
                    print("❌ 客户端已断开连接，停止推送检测结果。")
                    break
                
                current_time=result['time']
                loss=result['loss']
                label=result['original_label']
                label_predict=result['predicted_label']
                # 将结果格式化为 JSON 字符串并推送
                # 后端返回格式为 f"{current_time,float(loss),label,label_predict}\n"
                data_line = f"{current_time},{loss},{label},{label_predict}\n"
            
                # 将格式化后的字符串发送给前端
                yield data_line.encode("utf-8")
        finally:
            print("⏹️ 检测结果监听结束。")

    return StreamingResponse(detection_generator(), media_type="application/x-ndjson")