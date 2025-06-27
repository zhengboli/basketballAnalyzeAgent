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
                    if end_frame - start_frame < 10:
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
def calc_distance_with_basket(file_path: str, save_path: str = "basket_distance_result.csv", target_position: list = [230,550]) -> str:
    """
    读取json文件,分析所有帧球员与篮筐的距离,并保存为CSV文件。
    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        save_path (str): 保存距离结果的CSV文件路径
        target_position (list): 篮筐的坐标，默认为[230,550]
    Returns:
        str: 保存的CSV文件的路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    all_distances = []
    player_ids = ["T1", "T2", "T3"]
    tx, ty = target_position
    # 对每个球员计算与篮筐的距离
    for player_id in player_ids:
        for frame in frames:
            x, y = frame[player_id]
            if None in (x, y):
                distance = None
            else:
                distance = math.sqrt((x - tx)**2 + (y - ty)**2)
                
            all_distances.append({
                "player": player_id,
                "frame_number": frame["frame"],
                "distance_to_basket": distance
            })
    distance_df = pd.DataFrame(all_distances)
    distance_df.to_csv(save_path, index=False)
    return f"与篮筐的距离数据已保存到 {save_path}"
@tool
def calc_distance_with_player(player: str,target_player: str,file_path: str) -> str:
    """
     读取json文件,分析所有帧球员与篮筐的距离,并保存为CSV文件。
    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        save_path (str): 保存距离结果的CSV文件路径
        target_player (str): 目标球员的ID
        player (str): 球员的ID
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
    return f"与球员的距离数据已保存到 {f"{player}_{target_player}_distance_result.csv"}"
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
    min_speed: float = 3.0,
    max_delay: int = 100,
    save_path: str = "fake_move_result.csv"
) -> str:
    """
    读取json文件，检测篮球球员的假动作。

    Args:
        file_path (str): 包含球员位置的JSON文件路径。
                        格式: [{"frame": 1, "T1": [x1,y1], "T2": [x2,y2], "T3": [x3,y3]}, ...]
        player_id (str): 球员ID，例如 'T2'
        start_frame (int): 从哪一帧开始分析（如T2应为39）
        min_speed (float): 视为有效运动的最小帧间速度（像素）
        max_delay (int): 假动作的最大延迟帧数，默认100帧内必须完成方向反转
        save_path (str): 保存结果的CSV文件路径

    Returns:
        str: 分析结果说明
    """
    # 读取JSON文件
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    
    # 提取指定球员的位置数据
    positions_data = []
    for frame in frames:
        if frame["frame"] >= start_frame:
            positions_data.append({
                "frame": frame["frame"],
                "x": frame[player_id][0],
                "y": frame[player_id][1]
            })
    
    df = pd.DataFrame(positions_data)
    velocities = []
    results = []
    # 计算速度和方向变化
    for i in range(1, len(df)):
        x1, y1 = df.iloc[i-1]['x'], df.iloc[i-1]['y']
        x2, y2 = df.iloc[i]['x'], df.iloc[i]['y']
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dy = y2 - y1
        velocities.append((dist, dy))
        results.append({
            "frame": df.iloc[i]['frame'],
            "speed": dist,
            "vertical_direction": dy
        })
    
    # 保存详细数据到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    # 检测假动作
    for i in range(len(velocities)):
        dist_i, dy_i = velocities[i]
        if dist_i >= min_speed and dy_i > 0:  # 向下移动
            for j in range(i+1, min(i + max_delay, len(velocities))):
                dist_j, dy_j = velocities[j]
                if dist_j >= min_speed and dy_j < 0:  # 向上移动
                    return f"""检测到假动作:
- 开始帧: {df.iloc[i]['frame']}
- 结束帧: {df.iloc[j]['frame']}
- 持续帧数: {df.iloc[j]['frame'] - df.iloc[i]['frame']}
- 方向: 先下后上
- 详细数据已保存到: {save_path}"""
    
    return f"未检测到假动作，详细数据已保存到: {save_path}"
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[calc_speed,calc_distance_with_basket,calc_distance_with_player,detect_fake_move,detect_player_run_frame,read_csv],
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
        你可以使用以下工具来完成分析任务：

        1. calc_speed: 只用于计算球员速度，生成速度数据文件，不用于分析跑动，跑动分析只用detect_player_run_frame工具
        2. detect_player_run_frame: 这是评判球员跑动必须使用的工具,基于速度数据识别球员的跑动状态，包括跑动的开始和结束时间。
        3. calc_distance_with_basket: 计算球员与篮筐的距离，用于判断球员是否到达篮筐位置
        4. calc_distance_with_player: 计算球员之间的距离，用于分析球员之间的位置关系
        5. detect_fake_move: 检测球员的假动作，判断战术动作的完成情况
        6. read_csv: 读取并分析生成的数据文件
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
print_stream(agent.stream(input={"messages": [HumanMessage(content="json文件路径是frames_output_new.json，你需要三个球员每一帧的数据,根据需要的数据你可以调用不同的工具，这些工具会生成不同的csv文件，你可以根据生成的csv文件名来读取文件并进行分析,请写一份完整的球员的分析报告必须完成每个球员需要分析的所有步骤，你需要告诉我报告内容都是根据什么文件生成的，告诉我你是如何计算。特别注意！！！你必须严格基于文件中的实际数据给出结论，禁止编造或推测不存在的数据！在给出结论时必须引用具体数据！提供数据的来源和具体数值。求其是分析球员与篮筐距离的任务，一定要保证数据准确，回答 用中文")]
},stream_mode="values"),save_path="analysis_result.txt")

