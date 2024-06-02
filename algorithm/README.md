# RL Logistics Documentation
@author: 朱虹翱
@description: 我把物流系统运输包裹的宏观调控视为一个马克洛夫决策过程，因此，把整个环境包装成了一个加强学习的环境。在这个程序中，环境按照固定的时间频率刷新，每刷新一次，Agent都可以选择临时“阻塞”某些道路，把该道路上即将运送的包裹转移到另一条道路上。运用于训练的强化学习算法为DQN。
### Node: 表示物流环境中的节点，包括节点ID、位置、吞吐量、延迟、成本等属性。
    - id: 节点ID，唯一标识一个节点。
    - pos: 节点在环境中的位置坐标。
    - throughput: 节点的吞吐量，表示节点能够处理包裹的能力。
    - delay: 节点的延迟，表示包裹在节点处停留的时间。
    - cost: 节点的成本，表示包裹在节点处停留的成本。
    - is_station: 是否是station
    - buffer: 仓库，用来存放入站的包裹，用堆存储
    - packages: 运输线，用来处理包裹，用堆存储
    - dones: 完成仓库，用来存放到达终点的包裹，用列表存储
    - history: 收到的包裹，用列表存储
    - package_order: 最新处理的包裹index（用于给堆排序，并不总是等于处理的包裹的总数）

    - reset(): 重置节点状态，清空仓库和运输线上的包裹，清空历史记录
    - add_package(package): 如果是包裹的终点，加入done，否则加入buffer；如果buffer中有其他包裹，判断新包裹的类型是否与buffer顶端的包裹类型一致（一致则插入到最底端，否则对于express包裹，插入到最顶端）。更新包裹的delay为当前节点的delay
    - process_packages(): 自动处理包裹，把buffer顶端的包裹输入到packages中处理
    - remove_package(): 自动从packages中移除delay<=0，即可以转移的包裹

### Route: 表示物流环境中的路由，包括路由ID、源节点、目的节点、延迟、成本等属性。
    - src: 路由的源节点ID。
    - dst: 路由的目的节点ID。
    - id: 路由ID，唯一标识一个路由。f"{src}- >{dst}"
    - time: 路由的延迟，表示包裹在路由上传输的时间。
    - cost: 路由的成本，表示包裹在路由上传输的成本。
    - package_order: 最新处理的包裹index（用于给堆排序，并不总是等于处理的包裹的总数）
    - packages: 正在运输的包裹，用堆存储
    - history: 收到的包裹，用列表存储

    - reset(): 重置路由状态，清空运输线上的包裹，清空历史记录
    - add_package(package): 将包裹加入到packages中，更新包裹的delay为当前路由的time
    - remove_package(): 自动从packages中移除delay<=0，即可以转移的包裹

### Package: 表示物流环境中的包裹，包括包裹ID、延迟、源节点、当前节点、目的节点、路径、完成状态等属性。
    - id: 包裹ID，唯一标识一个包裹
    - time_created: 创建时间
    - time_arrived: 到达时间
    - src: 起点
    - dst: 终点
    - category: 1 for 'Express' and 0 for 'Standard'
    - history: 过去的状态及时间戳
    - path: 规划的路线（当前节点不一定位于开头）
    - delay: 允许状态转移的倒计时
    - done: 是否到达终点


### LogisticsEnv类：
    - init(): 初始化物流环境，创建节点、路由和时间戳。
    - reset(): 重置物流环境，重新创建包裹和时间戳。
    - step(action): 根据Agent的选择执行一步操作，推进时间并获取奖励。
    - get_state(): 获取当前物流环境的state，用于打印状态。
    - get_load(): 获取当前物流环境各个节点和路由的负荷，用于评估状态。
    - get_reward(): 获取当前步的总奖励，用于激励学习。
    - find_shortest_time_path(): express包裹采用的Bellman Ford默认策略
    - find_lowest_cost_path(): standard包裹采用的Bellman Ford默认策略
    - find_alternative_time_path(): express包裹采用的，绕过了下一个节点的Bellman Ford策略
    - find_alternative_cost_path(): standard包裹采用的，绕过了下一个节点的Bellman Ford策略
    - change_route(route): 对于即将发送到某条路由上的包裹，改变它们的未来路径，绕过该路径
    - 
### Agent类：
    - init(self, env): 初始化Agent，传入环境对象
    - choose_action(): 根据当前状态选择阻塞哪些道路
    - train(n_episodes): 训练n_episodes个循环
    - test(): 测试训练后的模型

### test_classic()
    - 默认策略的测试样例

### test_RL()
    - RL策略的测试样例（先训练，再测试效果）
    - 由于环境较为复杂，加之DQN对超参数的高度敏感，1000个包裹的训练1000个episode的通常在48h，且收敛程度受随机环境的位置影响，大部分时候只是比初始策略略优。
    - 包装为强化学习环境的一大难点在于定义Agent的主体，动作空间，以及揣测收敛的超参数条件。本例中，将整个环境视为整体，将Agent的动作空间设置为了控制各个路由上的“红绿灯”。本例并未限制每次控制的路由数量，而是由模型自动学习————如果限制每次控制的路由数量为常数，能大幅减少动作空间，带来更简单的收敛，也节省时间，但其结果必然是局部最优的。因此，为了全局最优的可能性，我们坚持不限制每次控制的路由数量；虽然目前的学习效果比较一般，我们认为这主要是由训练episode不足（受时间、算力限制）导致的；当算力充足且训练episodes足量（上万）的情况下，这样的方向能带来更好的全局优化。