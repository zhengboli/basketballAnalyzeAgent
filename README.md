# basketballAnalyzeAgent
## 本仓库主要用于记录制作篮球分析Agent过程的数据分析，主要考虑的几点：1.设计结构 2.关键实现方式 3.性能指标：（a.token消耗 b.准确率 c.时延）
### 6.18 version：0.1
#### 1.设计结构
当前采用的是singleAgent，使用ReAct框架。模型可以使用多个工具包括：1.读取json文件返回球员数据坐标。2.计算球员的速度。 3.计算球员与球员之间的距离。 4.计算球员与篮筐的距离
用户输入prompt给模型，模型依据prompt开始调用工具然后将工具信息传回模型，模型再根据传回内容和之前的信息继续调用工具直至完成用户需求
#### 2.关键实现方式
采用的是langchain+langgraph实现
#### 3.性能指标
##### a.token消耗
token消耗极大，时常出现token超出120k的情况，导致无法完成进行
<img width="700" alt="截屏2025-06-18 10 56 35" src="https://github.com/user-attachments/assets/ecc723a8-b405-46f8-879d-cfb51da48500" />
<img width="700" alt="截屏2025-06-18 11 02 59" src="https://github.com/user-attachments/assets/76babbd1-ebe4-42b9-87e3-74943d9bc14f" />

图中内容为langsmith捕获的信息，由图中可见为了防止token消耗过大采用每次只读50帧分析，但模型连续调用读取json工具，可见这样的操作的效果并不好。
造成token消耗的原因是模型从json文件读取数据后返回文件内的数据，这就导致模型message中始终带着大量的球员位置数据，由大模型分析后光5帧的数据就需要消耗600tokens因此这种消耗大量token的方式需要更改。
##### b.准确率
当前模型返回的报告内容与使用gpt网页端的4o相比任然差距较大。
##### c.时延
完成一轮生成的时间大概在400～500秒左右
