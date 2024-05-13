# RL Logistics Documentation
### Node: 表示物流环境中的节点，包括节点ID、位置、吞吐量、延迟、成本等属性。
    id: 节点ID，唯一标识一个节点。
    position: 节点在环境中的位置坐标。
    throughput: 节点的吞吐量，表示节点能够处理包裹的能力。
    delay: 节点的延迟，表示包裹在节点处停留的时间。
    cost: 节点的成本，表示包裹在节点处停留的成本。
### Route: 表示物流环境中的路由，包括路由ID、源节点、目的节点、延迟、成本等属性。
    id: 路由ID，唯一标识一个路由。
    source_node: 路由的源节点ID。
    destination_node: 路由的目的节点ID。
    delay: 路由的延迟，表示包裹在路由上传输的时间。
    cost: 路由的成本，表示包裹在路由上传输的成本。
### Package: 表示物流环境中的包裹，包括包裹ID、延迟、源节点、当前节点、目的节点、路径、完成状态等属性。
    id: 包裹ID，唯一标识一个包裹。
    delay: 包裹的延迟，表示包裹从源节点到目的节点的总延迟。
    source_node: 包裹的源节点ID。
    current_node: 包裹当前所在的节点ID。
    destination_node: 包裹的目的节点ID。
    path: 包裹的路径，表示包裹从源节点到目的节点的传输路径。
    done: 包裹是否已完成，完成状态为True，未完成状态为False。


### LogisticsEnv类：
    init(): 初始化物流环境，创建节点、路由和时间戳。
    reset(): 重置物流环境，重新创建节点、路由和时间戳。
    step(action): 根据Agent的选择执行一步操作，推进时间并获取奖励。
    get_state(): 获取当前物流环境的state，用于评价策略。
    get_reward(): 获取当前时间戳的总奖励。



### Agent类：
    init(self, env): 初始化Agent，传入环境对象。
    choose_action(): 根据当前状态选择一个操作。
