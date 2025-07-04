"""
本代码用于丢给模型数据和提示词让其生成一份报告
目前先用一个agent生成分析
后续再添加一个agent生成报告
目前调用api模型只有基本的聊天能力，因此关于距离计算，速度计算等能力可能需要之后的工具调用

目前加入速度和距离的计算
但是目前token限制还得优化，把json文件中的标准球员数据可以去除，只保留学生测试视频数据

目前想法是不让模型读取json文件的内容后返回，而是让工具函数直接对json文件中的数据进行操作，这样子可以节省token的消耗
"""

import os
import psutil
from memory_profiler import profile
import logging

"""
标准视频球员分析指标:(结合标准视频中球员位置表分析)
球员S1:
100帧左右开始跑动然后170帧左右后结束跑动(完成指标:1.到达S3附近位置(与S3的距离小于60像素)2.速度符合要求(70帧左右抵达S3附近位置))
在原地等待了40帧左右(等待S3球员完成假动作)(完成指标:等待S3球员完成假动作)
210帧左右向篮筐跑去,270帧左右到达篮筐位置(完成指标:1.到达篮筐位置2.速度符合要求)
球员S2:
2帧左右向下开始跑动然后87帧左右到达第一帧初始位置左右。(完成指标:1.完成假动作2.速度符合要求3.回到原位置)
之后位置几乎没有改变(完成指标:保持静止)
球员S3:
155帧左右开始移动,先得向下移动一段距离后再往上移动(为了完成假动作)。(完成指标:1.完成假动作2.速度符合要求3.到达篮筐位置)
169帧后开始绕开ID1然后向篮球框下移动。然后261帧左右到达篮筐位置(完成指标:1.速度符合要求2.到达篮筐位置)
"""
os.environ["OPENAI_API_KEY"] = "sk-6vYqTDflWdNsI0nbHHauJlQNluIgWmkOpi8Nfw4sAm7t3mcZ"
os.environ["OPENAI_BASE_URL"] = "https://xiaoai.plus/v1"
os.environ["LANGSMITH_TRACING"]= "true"
os.environ["LANGSMITH_API_KEY"]="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"
os.environ["TAVILY_API_KEY"] = "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"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph,MessagesState,END
import json
import math
from langchain_core.messages import HumanMessage
import pandas as pd
@tool
def read_json(file_path: str, key: str = None) -> str:
    """
    读取JSON文件内容，可选读取某个key的数据，每次只返回[start, end)帧数据，防止超出tokens限制。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key and isinstance(data, dict):
        frames = data.get(key, [])
    else:
        frames = data
    if isinstance(frames, list):
        frames = frames
    return json.dumps(frames, ensure_ascii=False, indent=2)
@tool
def summarize(text: str, max_tokens: int = 500) -> str:
    """
    对输入的文本进行摘要，保留关键信息，输出不超过 max_tokens tokens 的内容。
    """
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"请将以下内容压缩成不超过{max_tokens} tokens的摘要，只保留关键信息：\n{text}"
    summary = llm.invoke([HumanMessage(content=prompt)]).content
    return summary
@tool
def calc_speed(file_path: str, save_path: str = "speed_result.csv") -> str:
    """
    读取JSON文件，分析所有帧的球员速度，并保存为CSV文件。

    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                         格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        save_path (str): 保存速度结果的CSV文件路径。

    Returns:
        str: 保存的CSV文件路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    all_speeds = []
    player_ids = ["T1", "T2", "T3"]

    for player_id in player_ids:
        prev_x, prev_y = None, None
        for i in range(len(frames)):
            frame = frames[i]
            curr_frame = frame["frame"]
            curr_pos = frame.get(player_id, [None, None])
            curr_x, curr_y = curr_pos

            if prev_x is not None and prev_y is not None and curr_x is not None and curr_y is not None:
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                speed = math.hypot(dx, dy)  # 欧几里得距离
            else:
                speed = None

            all_speeds.append({
                "player": player_id,
                "frame_number": curr_frame,
                "speed": speed
            })

            prev_x, prev_y = curr_x, curr_y  # 更新上一帧坐标

    speed_df = pd.DataFrame(all_speeds)
    speed_df.to_csv(save_path, index=False)
    return f"速度数据已保存到 {save_path}"
@tool 
def detect_player_run_frame(window_size: int = 10, speed_threshold: float = 5,start_frame: int = 1) -> str:
    """
    读取speed_result.csv文件，获得球员跑动的开始和停止帧
    Args:
        window_size (int): 窗口大小
        speed_threshold (float): 速度阈值
        start_frame (int): 开始帧,球员记录的开始帧，默认为1
    Returns:
        list: 球员的开始和停止帧
    """
    if not os.path.exists("speed_result.csv"):
        return "错误：需要先调用calc_speed生成速度数据文件"
    df = pd.read_csv("speed_result.csv")

    results = {}
    for player_id in df["player"].unique():
        player_df = df[df["player"] == player_id].reset_index(drop=True)
        speeds = player_df["speed"].tolist()
        results[player_id] =[]
        is_running = False
        start_frame = None
        run_index = 1
        i = 0
        while i < len(speeds) - window_size:
            if not is_running:
                # 检查是否进入跑动
                if speeds[i] is not None and speeds[i] > speed_threshold:
                    start_frame = i
                    is_running = True
            else:
                # 检查是否停止（窗口中全都低于阈值）
                window = speeds[i:i + window_size]
                if all(s is not None and s < speed_threshold for s in window):
                    end_frame = i
                    if end_frame - start_frame < 15:
                        is_running = False
                        continue
                    results[player_id].append({
                        "run_index": run_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame
                    })
                    run_index += 1
                    is_running = False
                    i += window_size  # 跳过已确认停止段
                    continue
            i += 1
        # 补尾巴：如果最后还在跑动
        if is_running:
            results[player_id].append({
                "run_index": run_index,
                "start_frame": start_frame,
                "end_frame": len(speeds) - 1
            })

    return results
@tool
def calc_distance_with_basket(file_path: str,save_path: str = "basket_distance_result.csv", target_position: list = [230,550], distance_threshold: float = 100.0) -> str:
    """
    读取json文件,分析所有帧球员与篮筐的距离,并保存为CSV文件。同时返回距离小于阈值的帧和移动方向信息。
    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        save_path (str): 保存距离结果的CSV文件路径
        target_position (list): 篮筐的坐标，默认为[230,550]
        distance_threshold (float): 判断到达篮筐的距离阈值，默认为100像素
    Returns:
        str: 分析结果，包括距离小于阈值的帧和移动方向信息。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    
    all_distances = []
    player_ids = ["T1", "T2", "T3"]
    tx, ty = target_position
    
    # 记录每个球员的分析结果
    player_results = {}
    
    # 对每个球员计算与篮筐的距离
    for player_id in player_ids:
        player_results[player_id] = {
            "near_basket_frames": [],  # 距离小于阈值的帧
            "moving_to_basket": []     # 是否向篮筐移动的帧
        }
        
        prev_distance = None
        
        for i, frame in enumerate(frames):
            curr_frame = frame["frame"]
            x, y = frame[player_id]
            
            if None in (x, y):
                distance = None
                moving_to_basket = None
            else:
                distance = math.sqrt((x - tx)**2 + (y - ty)**2)
                
                # 判断是否向篮筐移动
                if prev_distance is not None:
                    moving_to_basket = distance < prev_distance
                    if moving_to_basket:
                        player_results[player_id]["moving_to_basket"].append(curr_frame)
                
                # 判断是否接近篮筐
                if distance < distance_threshold:
                    player_results[player_id]["near_basket_frames"].append(curr_frame)

                
                prev_distance = distance
            
            # all_distances.append({
            #     "player": player_id,
            #     "frame_number": curr_frame,
            #     "distance_to_basket": distance
            # })
    
    # 保存距离数据到CSV
    # 生成结果报告
    result_report = []
    for player_id in player_ids:
        near_frames = player_results[player_id]["near_basket_frames"]
        moving_frames = player_results[player_id]["moving_to_basket"]
        
        if near_frames:
            result_report.append(f"{player_id}在 {near_frames[0]} 帧到达篮筐位置")
        else:
            result_report.append(f"{player_id}没有接近篮筐的帧")
    
    return result_report
@tool
def calc_distance_with_player(player: str,target_player: str,file_path: str,finished_frame: int = None) -> str:
    """
    判断球员是否在完成动作后到达目标球员附近位置（只用与比较球员与球员之间的距离，不用于比较与自身的距离）,读取json文件,分析所有帧球员与目标球员的距离,并保存为CSV文件。
    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        save_path (str): 保存距离结果的CSV文件路径
        target_player (str): 目标球员的ID
        player (str): 球员的ID
        finished_frame (int): 完成动作(可以是跑动后，也可以是假动作后)之后的帧，默认为None
    Returns:
        str: 保存的CSV文件的路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    player_distances = []
    for frame in frames:
        x1, y1 = frame[player]
        x2, y2 = frame[target_player]
        if None in (x1, y1, x2, y2):
            distance = None
        else:
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            player_distances.append({
                "frame_number": frame["frame"],
                "distance": distance
            })
    distance_df = pd.DataFrame(player_distances)
    distance_df.to_csv(f"{player}_{target_player}_distance_result.csv", index=False)
    if distance_df.iloc[finished_frame]["distance"] < 60:
        return f"球员{player}在{finished_frame}帧到达{target_player}附近位置"
    else:
        return f"球员{player}没有到达{target_player}附近位置"
@tool
def calc_distance_with_initial_position(player: str,file_path: str,reference_frame: int = 0,finished_frame: int = None) -> str:
    """
     判断球员是否在完成动作后回到初始位置,读取json文件,分析所有帧球员与初始位置的距离,并保存为CSV文件。
    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        player (str): 球员的ID
        reference_frame (int): 参考帧，默认为0
        finished_frame (int): 完成假动作之后的帧，默认为None
    Returns:
        str: 保存的CSV文件的路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    initial_x, initial_y = frames[reference_frame][player]
    player_distances = []
    for frame in frames:
        x1, y1 = frame[player]
        
        if None in (x1, y1):
            distance = None
        else:
            distance = math.sqrt((x1 - initial_x)**2 + (y1 - initial_y)**2)
            player_distances.append({
                "frame_number": frame["frame"],
                "distance": distance
            })
    distance_df = pd.DataFrame(player_distances)
    distance_df.to_csv(f"{player}_initial_position_distance_result.csv", index=False)
    for i in range(finished_frame,len(distance_df)):
        if distance_df.iloc[i]["distance"] < 50:
            return f"球员{player}在{i}帧回到初始位置"
    return f"球员{player}没有回到初始位置"
@tool
def read_csv(file_path: str) -> str:
    """
    读取CSV文件内容并返回完整的格式化字符串。
    """
    df = pd.read_csv(file_path)
    
    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 加宽显示宽度
    pd.set_option('display.max_colwidth', None)  # 显示完整的列内容
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # 设置浮点数格式，保留3位小数
    
    # 转换为字符串
    data_str = df.to_string(index=False)
    
    # 重置显示选项（可选，防止影响其他代码）
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.float_format')
    return f"CSV文件内容：\n{data_str}"
@tool
def detect_fake_move(
    file_path: str,
    player_id: str,
    start_frame: int = 0,
    min_speed: float = 2.0,
    max_delay: int = 100,
    direction_window: int = 3
) -> str:
    """
    检测篮球球员的假动作：先明显向下移动，再在指定帧数内迅速向上反转。

    参数:
        file_path (str): json文件路径
        player_id (str): 球员ID (T1/T2/T3)
        start_frame (int): 起始帧
        min_speed (float): 有效帧间速度
        max_delay (int): 假动作完成最大帧间隔
        save_path (str): 中间数据保存路径
        direction_window (int): 方向趋势滑窗（平滑判断上下移动趋势）
    
    返回:
        str: 假动作检测结果报告
    """
    # === 读取并构建帧序列 ===
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    positions = []
    for frame in frames:
        if frame["frame"] >= start_frame:
            if player_id in frame and isinstance(frame[player_id], list):
                positions.append({
                    "frame": frame["frame"],
                    "x": frame[player_id][0],
                    "y": frame[player_id][1]
                })

    df = pd.DataFrame(positions)

    if len(df) < 5:
        return "数据不足，无法分析假动作。"

    # === 计算速度/方向 ===
    results = []
    for i in range(1, len(df)):
        x1, y1 = df.iloc[i-1][['x', 'y']]
        x2, y2 = df.iloc[i][['x', 'y']]
        dist = math.hypot(x2 - x1, y2 - y1)
        dy = y2 - y1
        results.append({
            "frame": df.iloc[i]["frame"],
            "speed": dist,
            "dy": dy,
            "direction": "down" if dy > 0 else "up" if dy < 0 else "still"
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{player_id}_fake_move_result.csv", index=False)

    # === 滑窗趋势判断，用于更平稳检测移动方向 ===
    result_df["trend"] = result_df["dy"].rolling(window=direction_window).mean()

    # === 搜索假动作：先向下，再向上，必须都超过最小速度 ===
    for i in range(len(result_df) - max_delay):
        down_trend = result_df.iloc[i:i+direction_window]
        if down_trend["trend"].mean() > 2.0 and down_trend["speed"].mean() > min_speed:
            # 有连续向下趋势
            for j in range(i + direction_window, min(i + max_delay, len(result_df) - direction_window)):
                up_trend = result_df.iloc[j:j+direction_window]
                if up_trend["trend"].mean() < -2.0 and up_trend["speed"].mean() > min_speed:
                    start_frame = result_df.iloc[i]["frame"]
                    end_frame = result_df.iloc[j]["frame"]
                    duration = end_frame - start_frame
                    return (
                        f"✅ 检测到假动作:\n"
                        f"- 向下起始帧: {start_frame}\n"
                        f"- 向上反转帧: {end_frame}\n"
                        f"- 假动作总持续帧数: {duration}\n"
                        f"- 方向: 先明显向下移动，再快速反转向上\n"
                        f"- 中间速度平均值: {round(down_trend['speed'].mean(),2)}↓ / {round(up_trend['speed'].mean(),2)}↑\n"
                        f"- 详细数据已保存至: {f"{player_id}_fake_move_result.csv"}"
                    )

    return f"⚠️ 未检测到明显假动作（先下后上），详细数据已保存至: {f"{player_id}_fake_move_result.csv"}"
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[calc_speed,calc_distance_with_basket,calc_distance_with_player,calc_distance_with_initial_position,detect_fake_move,detect_player_run_frame],
    prompt=ChatPromptTemplate.from_messages([
        ("system", """你现在是一个篮球战术执行分析员,你现在分析的是三人篮球战术
我这里有个记录三人战术视频球员位置的json文件(frame代表视频每帧,S1,S2,S3分别代表标准视频中的三个球员的坐标,T1,T2,T3分别代表学生测试视频中的三个球员的坐标),里面记录了标准视频和学生测试视频中球员在每个帧的位置。我给你一份标准视频球员分析指标,你按照这个指标并根据球员的坐标分析学生测试视频中球员(T1,T2,T3)的战术执行情况。
分析补充(十分重要,分析时一定要考虑):
1.标准视频和学生测试视频中球员所站的位置并不相同,如果需要跟原位置比较以各自坐标的第一帧进行比较(球员T2在39帧才到指定战术位置,所以他39帧之前的位置变化直接忽略!!!)。
3.篮筐固定坐标为 (230, 550)。当球员与篮筐距离小于 100 像素时,视为"到达篮筐"。
4.向下移动即往y轴正方向移动,向上移动即往y轴负方向移动
5.轻微抖动可以当作停止不动
6.跑动意味着x轴和y轴都在变化且变化较大
7.假动作判定：球员先向一个方向运动后快速（小于100帧）向反方向运动时即判定为完成假动作
8.最终要生成一份可读的球员分析报告
你在生成的报告中需要包含这几个具体的分析结果：
对于T1球员：
    1.T1球员的第一次跑动的时间和结束跑动的时间，记录第一次跑动总时长
    2.当T1球员结束第一次跑动时（跑动结束帧）是否到达T2附近位置（与T2的距离小于60像素），如果未到达则记录为未到达，如果到达则记录为到达
    3.T1球员到达T2附近位置后是否完成等待的动作（速度始终小于5像素），如果完成则告诉我具体是多少帧，并计算出完成等待动作所用的时间
    4.T1球员是否在完成等待动作后继续向篮筐进行第二次跑动，如果移动T1是否到达篮筐附近（T1与篮筐距离在100像素以内），如果到达则告诉我具体是多少帧，并计算出到达篮筐附近所用的时间
对于T2球员：
    1.T2球员的第一次跑动的时间和结束跑动的时间，记录第一次跑动总时长
    2.T2球员的第一次跑动是否完成假动作，如果完成则告诉我具体是多少帧，并计算出完成假动作所用的时间
    3.T2是否在完成假动作后到达篮筐附近（T2与篮筐距离在100像素以内），如果到达则告诉我具体是多少帧，并计算出到达篮筐附近所用的时间
对于T3球员：
    1.T3球员的第一次跑动的时间
    2.T3球员的第一次跑动是否完成假动作，如果完成则告诉我具体是多少帧，并计算出完成假动作所用的时间
    3.T3是否在完成假动作后回到初始位置（T3与初始位置距离在100像素以内），如果回到则告诉我具体是多少帧，并计算出回到初始位置所用的时间
    4.T3球员是否在完成上述动作后呆在原地始终没有跑动
        """),
        ("system", """
            你必须按照以下顺序使用工具：
            1. 首先必须调用 calc_speed 生成速度数据文件
            - 输入：json文件路径
            - 输出：speed_result.csv
            2. 然后才能调用其他依赖速度数据的工具：
            - detect_player_run_frame（分析跑动）
            - detect_fake_move（分析假动作）
            3. 最后调用距离相关的工具：
            - calc_distance_with_basket
            - calc_distance_with_player
            - calc_distance_with_initial_position
        严格遵守这个顺序，不要跳过步骤或改变顺序。
        如果工具执行失败，请按以下步骤处理：
            1. 检查是否按正确顺序调用工具
            2. 确认所需的输入文件是否存在
            3. 验证前一个工具的输出是否正确
            4. 如果发现错误，从第一步重新开始
        每个工具都有其特定的用途，你需要根据分析需求选择合适的工具来获取所需的数据。确保你的分析结果准确且全面。
        """),
        ("user", "{messages}")
    ])
)

def print_stream(stream, save_path: str = "analysis_result.txt"):
    """
    打印并保存stream中的消息。

    Args:
        stream: 模型生成的消息流
        save_path: 保存结果的文件路径，默认为'analysis_result.txt'
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for s in stream:
            message = s["messages"][-1]
            
            # 打印到控制台
            message.pretty_print()
            
            # 保存到文件
            if isinstance(message, tuple):
                f.write(str(message) + '\n')
            else:
                # 假设message对象有content属性
                content = message.content if hasattr(message, 'content') else str(message)
                f.write(content + '\n')
            
            # 添加分隔符以区分不同的消息
            f.write('-' * 80 + '\n')
print_stream(agent.stream(input={"messages": [HumanMessage(content="json文件路径是frames_output_new.json，你需要三个球员每一帧的数据,根据需要的数据你可以调用不同的工具，这些工具会生成不同的csv文件，你可以根据生成的csv文件名来读取文件并进行分析,请写一份完整的球员的分析报告必须完成每个球员需要分析的所有步骤，你需要告诉我报告内容都是根据什么文件生成的，告诉我你是如何计算。特别注意！！！你必须严格基于文件中的实际数据给出结论，禁止编造或推测不存在的数据！在给出结论时必须引用具体数据！提供数据的来源和具体数值。回答 用中文")]
},stream_mode="values"),save_path="analysis_result.txt")

