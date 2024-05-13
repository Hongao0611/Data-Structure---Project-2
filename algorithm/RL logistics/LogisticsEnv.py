import data_gen
import uuid
import heapq
from LogisticsNode import Node
from LogisticsRoute import Route
from LogisticsPackage import Package

#generate stations, centers, packages.
data = data_gen.data_gen()
station_pos = data['station_pos']
station_prop = data['station_prop']
center_pos = data['center_pos']
center_prop = data['center_prop']
edges = data['edges']
packets = data['packets']

class LogisticsEnv:
    def __init__(self):
        self.nodes = {}  # Dictionary of nodes
        self.routes = {}  # Dictionary of routes
        self.packages = {}  # Dictionary of packages
        self.min_delay = float('inf')  # Minimum delay for all packages
        self.TimeTick = 0.0  # Current time tick
        self.done = False

    def reset(self):
        self.nodes = {}  # Dictionary of nodes
        self.routes = {}  # Dictionary of routes
        self.packages = {}  # Dictionary of packages

        # Initialize nodes, routes, and packages
        # Add stations as nodes
        for i in range(len(station_pos)):
            p = self.add_node(f"s{i}", station_pos[i], *station_prop[i], is_station=True)
            #print(f"Center({p.id}, {p.pos}, Throughput: {p.throughput}, Delay: {p.delay}, Cost: {p.cost}), is_station: {p.is_station}")

        # Add centers as nodes
        for i in range(len(center_pos)):
            p = self.add_node(f"c{i}", center_pos[i], *center_prop[i])
            #print(f"Center({p.id}, {p.pos}, Throughput: {p.throughput}, Delay: {p.delay}, Cost: {p.cost}), is_station: {p.is_station}")

        # Add edges as routes
        for edge in edges:
            p = self.add_route(edge[0], edge[1], edge[2], edge[3])
            #print(f"Route({p.src}->{p.dst}, Time: {p.time}, Cost: {p.cost})")

        # Add packets as packages
        for packet in packets:
            p = self.add_package(uuid.uuid4(), *packet)
            print(f"Package {p.id} added, delay={p.delay}, src={p.src}, curr={p.curr}, dst={p.dst}, done={p.done}, path={p.path}")
        
        for package in self.packages.values():
            self.min_delay = package.delay if self.min_delay>package.delay and package.delay>0 else self.min_delay
        for node in self.nodes.values():
            print(node)
        self.TimeTick = 0.0  # Current time tick
        print(f"Env reset. min_delay={self.min_delay}, TimeTick={self.TimeTick}")

        return self.get_state()

    def add_node(self, id, pos, throughput, delay, cost, is_station=False):
        self.nodes[id] = Node(id, pos, throughput, delay, cost, is_station)
        return self.nodes[id]

    def add_route(self, src, dst, time, cost):
        self.routes[(src, dst)] = Route(src, dst, time, cost)
        return self.routes[(src, dst)]

    def add_package(self, id, time_created, src, dst, category):
        package = Package(id, time_created, src, dst, category)
        
        # Calculate the initial optimal路径 based on the package's category
        if category == 'Express':
            # 对于Express包裹，找到最短的总时间路径
            optimal_path = self.find_shortest_time_path(src, dst)
        else:
            # 对于Standard包裹，找到最低的总成本路径
            optimal_path = self.find_lowest_cost_path(src, dst)
        
        package.path = optimal_path  # 将最优路径初始化为包裹的路径
        self.packages[id] = package
        self.nodes[src].add_package(package)  # 添加到优先队列中
        return package

    def find_shortest_time_path(self, src, dst):
        # Initialize distances and visited nodes
        distances = {node: float('inf') for node in self.nodes}  # Use a very large number instead of infinity
        distances[src] = 0
        visited = set()

        # Initialize priority queue with the source node
        pq = [(0, src, 0)]  # (distance, current_node, total_time)

        while pq:
            current_distance, current_node, current_time = heapq.heappop(pq)

            # If the node has already been visited, skip it
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check neighbors of the current node
            for route in self.routes.values():
                if route.src == current_node and route.dst not in visited:
                    # Calculate the new distance and time
                    new_distance = current_distance + route.time
                    new_time = current_time + route.time + self.nodes[current_node].delay

                    # If this is a shorter path, update the distances and add to the priority queue
                    if new_distance < distances[route.dst]:
                        distances[route.dst] = new_distance
                        heapq.heappush(pq, (new_distance, route.dst, new_time))

        # Construct the path from the source to the destination
        path = []
        current_node = dst
        while current_node != src:
            path.append(current_node)
            # 选择下一个节点，不检查是否已经在路径中
            # 使用 self.routes 来获取当前节点的实际对象
            current_node = min(
                [route.dst for route in self.routes.values() if route.src == current_node],
                key=lambda x: distances[x],
                default=None
            )
            if current_node is None:
                # No path found
                return []
        path.append(src)
        path.reverse()

        return path
    
    def find_lowest_cost_path(self, src, dst):
        # Initialize distances and visited nodes
        distances = {node: float('infinity') for node in self.nodes}
        distances[src] = 0
        visited = set()

        # Initialize priority queue with the source node
        pq = [(0, src)]  # (distance, current_node)

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            # If the node has already been visited, skip it
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check neighbors of the current node
            for route in self.routes.values():
                if route.src == current_node and route.dst not in visited:
                    # Calculate the new distance
                    new_distance = current_distance + route.cost + self.nodes[current_node].cost

                    # If this is a lower distance, update the distances and add to the priority queue
                    if new_distance < distances[route.dst]:
                        distances[route.dst] = new_distance
                        heapq.heappush(pq, (new_distance, route.dst))

        # Construct the path from the source to the destination
        path = []
        current_node = dst
        while current_node != src:
            path.append(current_node)
            # 选择下一个节点，使用 (current_node, neighbor) 来检查路由是否存在
            previous_node = min(
                [route.dst for route in self.routes.values() if route.src == current_node],
                key=lambda x: distances[x],
                default=None
            )
            if previous_node is None:
                # No path found
                return []
            current_node = previous_node

        path.append(src)
        path.reverse()

        return path

    def get_policy(self, package):
        if package.category == 'Express':
            # For Express packages, find the shortest total time path
            return self.find_shortest_time_path(package.curr, package.dst)
        else:
            # For Standard packages, find the lowest total cost path
            return self.find_lowest_cost_path(package.curr, package.dst)

    def step(self, actions):
        self.TimeTick += self.min_delay  # 更新时间
        # 更新所有包裹的延迟
        for package in self.packages.values():
            package.delay -= self.min_delay
        # 遍历所有动作
        for package_id, action in actions.items():
            print(f"STEP:::PacakgeID={package_id}, Action={action}")
            package = self.packages[package_id]  # 使用 package_id 获取包裹
            if package.done:
                package.reward = 0
                continue
            elif package.delay <= 0:  # 包裹准备好转移状态
                if action is not None:
                    node = self.nodes[action[0]]
                    next_node = self.nodes[action[1]]
                    route = self.routes[(action[0], action[1])]
                    if package_id in node.package_indices:  # 包裹位于Node
                        print(f"is in Node: {node.id},")
                        node.remove_package(package_id)
                        route.add_package(package)
                        # 更新包裹的历史、当前所在的节点
                        package.history.append((self.TimeTick, package.curr))
                        package.curr = node.id
                        # 计算包裹的reward
                        if package.category == 'Express':
                            package.reward -= self.min_delay
                        else:
                            package.reward -= node.cost
                    elif package_id in route.package_indices:  # 包裹位于Route
                        print(f"is on Route: {route.src}->{route.dst},")
                        if len(node.packages) < node.throughput:
                            route.remove_package(package_id)
                            next_node.add_package(package)
                            # 更新包裹的历史、Route起点的节点
                            package.history.append((self.TimeTick, package.curr))
                            package.curr = next_node.id
                            # 计算包裹的reward
                            if package.category == 'Express':
                                package.reward -= self.min_delay
                            else:
                                package.reward -= route.cost
                        else: # 包裹滞留在当前Route无法进站
                            print(f"waiting..., Node: {route.dst} full")
                            if package.category == 'Express':
                                package.reward -= self.min_delay
                    else:
                        print("Error! Undone package is neither on route or in node!")
                        return 0
                else:  # 包裹准备好转移状态，但动作是None
                    print(f"This should not happen! Package: {package_id} is ready but action is None.")
                    if package.category == 'Express':
                        package.reward -= self.min_delay
            else:
                print(f"Package: {package_id} processing...")
                # package正在Route或者Node中处理
                if package.category == 'Express':
                    package.reward -= self.min_delay
        #判断是否有新的package变成done,顺便计算当前的reward,顺便判断是否done
        self.done = True
        curr_reward = 0
        self.min_delay = float('inf')
        for package in self.packages.values():
            if not package.done: #对于之前没有done的package
                curr_reward +=package.reward #累加reward
                if package.curr == package.dst: #该节点是否刚到done
                    package.history.append((self.TimeTick, package.curr))
                    next_node.remove_package(package_id)
                    package.done = True
                    package.delay = float('inf')
                else:
                    self.done = False #如果还有undone package，那么整个环境就没有done
                    #print(f"undone package {package.id} at {package.curr} has delay: {package.delay}")
                    self.min_delay = package.delay if package.delay>0 and (package.delay < self.min_delay) else self.min_delay
        print(f"the new min_delay: {self.min_delay}")
        return self.get_state(), curr_reward, self.done

    def get_reward(self, package):
        # Calculate the reward based on package category and time passed
        if package.category == 'Express':
            reward = package.time_created - self.TimeTick #TODO re-scale the reward to be compatible with the route cost
        else:
            # Calculate the cost based on package history and node/route costs
            total_cost = 0
            for timestamp,node_id in package.history:
                total_cost += self.nodes[node_id].cost
            for route in package.path:
                total_cost += self.routes[route].cost
            reward = -total_cost  # Negative cost as a reward for minimizing costs

        # If the package has reached its destination, update its delay to zero and mark it as done
        if package.src == package.dst:
            package.delay = 0
            package.mark_done()

        return reward
    
    def get_state(self):
        # Create a state representation that includes relevant information for decision-making
        state = {
            'package_positions': self.get_package_positions(),
            'package_destinations': self.get_package_destinations(),
            'package_categories': self.get_package_categories(),
            'queue_lengths': self.get_queue_lengths(),
            'route_loads': self.get_route_loads(),
            'current_time_tick': self.TimeTick
        }
        return state

    def get_package_positions(self):
        # Return a dictionary with package IDs as keys and their current node IDs as values
        positions = {pkg_id: pkg.src for pkg_id, pkg in self.packages.items()}
        return positions

    def get_package_destinations(self):
        # Return a dictionary with package IDs as keys and their destination node IDs as values
        destinations = {pkg_id: pkg.dst for pkg_id, pkg in self.packages.items()}
        return destinations

    def get_package_categories(self):
        # Return a dictionary with package IDs as keys and their categories as values
        categories = {pkg_id: pkg.category for pkg_id, pkg in self.packages.items()}
        return categories

    def get_queue_lengths(self):
        # Return a dictionary with node IDs as keys and the number of packages in their queues as values
        queue_lengths = {node_id: len(node.packages) for node_id, node in self.nodes.items()}
        return queue_lengths

    def get_route_loads(self):
        # Return a dictionary with route tuples (src, dst) as keys and the number of packages on the route as values
        route_loads = {route_tuple: len(route.packages) for route_tuple, route in self.routes.items()}
        return route_loads
