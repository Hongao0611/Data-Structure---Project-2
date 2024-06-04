import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from sklearn.cluster import KMeans
import heapq

# 定义节点类型常量
STATION = "station"
CENTER = "center"
AIRPORT = "airport"
parameters = {
    "station_num": 25,
    "center_num": 5,
    "packet_num": 1000,
}
def data_gen():
    # Generate Stations
    station_pos = []
    # properties are defined here: throughput/tick, time_delay, money_cost
    station_prop_candidates = [
        (10, 2, 0.5), (15, 2, 0.6), (20, 1, 0.8), (25, 1, 0.9)]
    station_prop = []
    for i in range(parameters["station_num"]):
        # Map size is defined here, which is 100*100
        station_pos.append((random.randint(0, 100), random.randint(0, 100)))
        station_prop.append(
            station_prop_candidates[random.randint(0, len(station_prop_candidates)-1)])
    # Output Stations
    print("Stations:")
    for i in range(len(station_pos)):
        print(f"s{i}", station_pos[i], station_prop[i])

    # Generate Centers by clustering
    kmeans = KMeans(n_clusters=parameters["center_num"])
    kmeans.fit(station_pos)
    station_labels = kmeans.predict(station_pos)
    center_pos = [(int(x[0]), int(x[1])) for x in kmeans.cluster_centers_]
    for i in range(len(center_pos)):
        while center_pos[i] in station_pos:
            # move slightly if center is overlapped with station
            # you can also use other methods to avoid this situation
            print("Warning: Center moved")
            center_pos[i] = center_pos[i][0] + 1, center_pos[i][1] + 1
    # properties are defined here: throughput/tick, time_delay, money_cost
    center_prop_candidates = [
        (100, 2, 0.5), (150, 2, 0.5), (125, 1, 0.5), (175, 1, 0.5)]
    center_prop = []
    for i in range(parameters["center_num"]):
        center_prop.append(
            center_prop_candidates[random.randint(0, len(center_prop_candidates)-1)])
    # Output Centers
    print("Centers:")
    for i in range(parameters["center_num"]):
        print(f"c{i}", center_pos[i], center_prop[i])

    # Draw Stations and Centers
    plt.scatter([x[0] for x in station_pos], [x[1]
                for x in station_pos], c=station_labels, s=50, cmap='viridis')
    plt.scatter([x[0] for x in center_pos], [x[1]
                for x in center_pos], c='black', s=200, alpha=0.5)

    # Generate Edges
    edges = []
    print("Edges (center to center):")      # Airlines
    for i in range(parameters["center_num"]):
        for j in range(parameters["center_num"]):
            if j > i:
                dist = np.linalg.norm(
                    np.array(center_pos[i]) - np.array(center_pos[j]))
                # src, dst, time_cost, money_cost
                # time_cost and money_cost are defined here
                edges.append((f"c{i}", f"c{j}", 0.25 * dist, 0.2 * dist))
                edges.append((f"c{j}", f"c{i}", 0.25 * dist, 0.2 * dist))
                plt.plot([center_pos[i][0], center_pos[j][0]], [
                         center_pos[i][1], center_pos[j][1]], 'r--')
                print(edges[-2])
                print(edges[-1])
    print("Edges (center to station):")     # Highways
    for i in range(parameters["center_num"]):
        for j in range(parameters["station_num"]):
            if station_labels[j] == i:
                dist = np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                # time_cost and money_cost are defined here
                edges.append((f"c{i}", f"s{j}", 0.6 * dist, 0.12 * dist))
                edges.append((f"s{j}", f"c{i}", 0.6 * dist, 0.12 * dist))
                plt.plot([center_pos[i][0], station_pos[j][0]], [
                         center_pos[i][1], station_pos[j][1]], 'b--')
                print(edges[-2])
                print(edges[-1])
    print("Edges (station to station):")    # Roads
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                dist = np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                # time_cost and money_cost are defined here
                edges.append((f"s{i}", f"s{j}", 0.8 * dist, 0.07*dist))
                edges.append((f"s{j}", f"s{i}", 0.8 * dist, 0.07*dist))
                plt.plot([station_pos[i][0], station_pos[j][0]], [
                         station_pos[i][1], station_pos[j][1]], 'g--')
                print(edges[-2])
                print(edges[-1])
    plt.show()

    # Generate Packets
    packets = []
    src_prob = np.random.random(parameters["station_num"])
    src_prob = src_prob / np.sum(src_prob)
    dst_prob = np.random.random(parameters["station_num"])
    dst_prob = dst_prob / np.sum(dst_prob)
    # Package categories are defined here: 0 for Regular, 1 for Express
    speed_prob = [0.7, 0.3]
    print("Packets:")
    for i in range(parameters["packet_num"]):      # Number of packets
        src = np.random.choice(parameters["station_num"], p=src_prob)
        dst = np.random.choice(parameters["station_num"], p=dst_prob)
        while dst == src:
            dst = np.random.choice(parameters["station_num"], p=dst_prob)
        category = np.random.choice(2, p=speed_prob)
        # Create time of the package, during 12 time ticks(hours). Of course you can change it.
        create_time = np.random.random() * 12
        packets.append((create_time, f"s{src}", f"s{dst}", category))
    # Sort packets by create time
    packets.sort(key=lambda x: x[0])
    # Output Packets
    for packet in packets:
        print(uuid.uuid4(), packet)

    M=np.zeros((2*(parameters["center_num"]+parameters["station_num"]),2*(parameters["center_num"]+parameters["station_num"])))
    for i in range(2*(parameters["center_num"]+parameters["station_num"])):
        for j in range(2*(parameters["center_num"]+parameters["station_num"])):
            M[i][j]=np.Infinity
    for i in range(parameters["center_num"]+parameters["station_num"]):
        M[2*i][2*i+1]=0.01
    for i in range(parameters["center_num"]):               #要处理的还有M[2*i][2*i+1]
        for j in range(parameters["center_num"]):
            if j > i:
                M[2*i+1][2*j] = 0.25*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(center_pos[j]))
                M[2*j+1][2*i] = M[2*i+1][2*j]
    for i in range(parameters["center_num"]):
        for j in range(parameters["station_num"]):
            if station_labels[j] == i:
                M[2*i+1][2*j+2*parameters["center_num"]] = 0.6*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                M[2*j+2*parameters["center_num"]+1][2*i] = M[2*i+1][2*j+2*parameters["center_num"]]
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                M[2*i+2*parameters["center_num"]+1][2*j+2*parameters["center_num"]] = 0.8*np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                M[2*j+2*parameters["center_num"]+1][2*i+2*parameters["center_num"]]=M[2*i+2*parameters["center_num"]+1][2*j+2*parameters["center_num"]]
    N=np.zeros((2*(parameters["center_num"]+parameters["station_num"]),2*(parameters["center_num"]+parameters["station_num"])))
    for i in range(2*(parameters["center_num"]+parameters["station_num"])):
        for j in range(2*(parameters["center_num"]+parameters["station_num"])):
            N[i][j]=np.Infinity
    for i in range(parameters["center_num"]):               #要处理的还有M[2*i][2*i+1]
        for j in range(parameters["center_num"]):
            if j > i:
                N[2*i+1][2*j] = 0.2*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(center_pos[j]))
                N[2*j+1][2*i] = N[2*i+1][2*j]
    for i in range(parameters["center_num"]):
        for j in range(parameters["station_num"]):
            if station_labels[j] == i:
                N[2*i+1][2*j+2*parameters["center_num"]] = 0.12*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                N[2*j+2*parameters["center_num"]+1][2*i] = N[2*i+1][2*j+2*parameters["center_num"]]
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                N[2*i+2*parameters["center_num"]+1][2*j+2*parameters["center_num"]] = 0.07*np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                N[2*j+2*parameters["center_num"]+1][2*i+2*parameters["center_num"]]=N[2*i+2*parameters["center_num"]+1][2*j+2*parameters["center_num"]]
    for i in range(parameters["center_num"]):
        N[2*i][2*i+1]=center_prop[i][2]
    for i in range(parameters["station_num"]):
        N[2*i+2*parameters["center_num"]][2*i+2*parameters["center_num"]+1]=station_prop[i][2]

    return {
        "station_pos": station_pos,
        "station_prop": station_prop,
        "center_pos": center_pos,
        "center_prop": center_prop,
        "edges": edges,
        "packets": packets,
        "money cost":M,
        "time cost":N,
    }

import heapq
#generate stations, centers, packages.
data = data_gen()
station_pos = data['station_pos']
station_prop = data['station_prop']
center_pos = data['center_pos']
center_prop = data['center_prop']
edges = data['edges']
packets = data['packets']
moneycost_initial = data["money cost"]
timecost_initial = data["time cost"]
moneycost = data["money cost"]
timecost = data["time cost"]
time_global = 0.0

class Package:
    def __init__(self, id, time_created, src, dst, category):
        self.id = id
        self.time_created = time_created
        self.src = src
        self.dst = dst
        self.category = category
        self.history = []
        self.path = []
        self.delay = float('inf')  # Remaining delay before processing
        self.done = False
        self.reward = 0.0  # Current Reward/cost for this package

    def __str__(self):
        # return f"Package({self.id}, TimeCreated: {self.time_created}, Src: {self.src}, Dst: {self.dst}, Category: {self.category})"
        return f"Package({self.id}, Path: {self.path})"

class Node:
    def __init__(self, id, pos, throughput, delay, cost, node_type):
        self.id = id
        self.pos = pos
        self.throughput = throughput
        self.delay = delay
        self.cost = cost
        self.node_type = node_type
        self.is_station = node_type == STATION
        self.buffer = []
        self.packages = []
        self.dones = []
        self.history = []
        self.package_order = 0
        heapq.heapify(self.packages)  # Convert the list to a heap
        heapq.heapify(self.buffer)

    def reset(self):
        self.buffer = []
        self.packages = []
        self.dones = []
        self.history = []
        heapq.heapify(self.packages)  # Convert the list to a heap
        heapq.heapify(self.buffer)

    def add_package(self, package):
        self.history.append(package.id)
        # 如果是包裹的终点，加入done，而不是buffer
        if package.dst == self.id:
            self.dones.append(package)
            package.done = True
            package.delay = float('inf')
        else:
            # 计数器
            self.package_order += 1
            # 检查buffer堆是否为空
            if not self.buffer:
                heapq.heappush(self.buffer, (self.package_order, package))
                package.delay = float('inf')  # Update package delay
                # print(f"Pack added to Node: {self.id};")
                self.process_packages()
            else:  # 如果buffer有包裹，获取堆顶的包裹
                index, top_package = self.buffer[0]
                if top_package.category or package.category == 0:  # 如果堆顶是express包裹，不做特殊处理；如果堆顶是standard,插入也是standard,不做特殊处理
                    heapq.heappush(self.buffer, (self.package_order, package))
                    package.delay = float('inf')  # Update package delay
                    # print(f"Pack added to Node: {self.id};")
                else:  # 如果堆顶是standard包裹,插入是express
                    heapq.heappush(self.buffer, (index - 1, package))
                    package.delay = float('inf')  # Update package delay
                    # print(f"Pack added to Node: {self.id};")

                self.process_packages()

    def process_packages(self):
        while self.buffer and len(self.buffer) < self.throughput:
            index, package = heapq.heappop(self.buffer)  # Get the package with the highest priority (oldest)
            if package.done == False:
                heapq.heappush(self.packages, (index, package))
                package.delay = self.delay
                package.history.append((time_global, self.id, f"PROCESSING: In Node: {self.id}"))
            else:
                self.dones.append(package)

    def remove_package(self):
        if self.packages and self.packages[0][1].delay <= 0:
            _, package = heapq.heappop(self.packages)  # Remove the package with the least priority (oldest)
            # print(f"package: {package.id} removed from Node: {self.id}.")
            self.process_packages()
            return package
        elif self.packages[0][1].delay > 0:
            print(f"Error! Removing Package: {package.id} not done from Node: {self.id}!")

    def __str__(self):
        return f"Route({self.src}->{self.dst}, Time: {self.time}, Cost: {self.cost})"


class Route:
    def __init__(self, src, dst, route_id, time, cost):
        self.src = src
        self.dst = dst
        self.id = f"{self.src}->{self.dst}"
        self.time = time
        self.cost = cost
        self.package_order = 0
        self.packages = []
        self.history = []
        heapq.heapify(self.packages)  # Convert the list to a heap

    def reset(self):
        self.package_order = 0
        self.packages = []

class LogisticsEnv:
    def __init__(self):
        self.nodes = {}  # Dictionary of nodes
        self.routes = {}  # Dictionary of routes
        self.packages = {}  # Dictionary of packages
        self.TimeTick = 0.0  # Current time tick
        self.done = False
        self.moneycost = 1  # 货币成本
        self.timecost = 1  # 时间成本

        # Add stations as nodes
        for i in range(len(station_pos)):
            p = self.add_node(f"s{i}", station_pos[i], *station_prop[i], node_type=STATION)

        # Add centers as nodes
        for i in range(len(center_pos)):
            p = self.add_node(f"c{i}", center_pos[i], *center_prop[i], node_type=CENTER)

        # Add airports as nodes
        for i in range(len(airport_pos)):
            p = self.add_node(f"a{i}", airport_pos[i], *airport_prop[i], node_type=AIRPORT)

        # Add edges as routes
        for edge in edges:
            p = self.add_route(edge[0], edge[1], edge[2], edge[3])

        # Add packages
        for packet in packets:
            p = self.add_package(uuid.uuid4(), *packet)

    def add_node(self, node_id, pos, throughput, delay, cost, node_type):
        node = Node(node_id, pos, throughput, delay, cost, node_type)
        self.nodes[node_id] = node
        return node

    def add_route(self, src, dst, route_id, time, cost):
        route = Route(src, dst, route_id, time, cost)
        self.routes[route_id] = route
        return route

    def add_package(self, package_id, time_created, src, dst, category):
        package = Package(package_id, time_created, src, dst, category)
        self.packages[package_id] = package
        return package

    def get_shortest_path(self, src, dst):
        # 初始化距离字典和前驱节点字典
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[src] = 0
        prev_nodes = {}

        # 使用优先队列实现堆
        heap = [(0, src)]

        while heap:
            curr_distance, curr_node = heapq.heappop(heap)

            if curr_node == dst:
                # 已找到最短路径，构建路径列表
                path = []
                while curr_node in prev_nodes:
                    path.insert(0, curr_node)
                    curr_node = prev_nodes[curr_node]
                path.insert(0, curr_node)
                return path

            if curr_distance > distances[curr_node]:
                continue

            # 遍历邻居节点
            for neighbor_node in self.get_neighbors(curr_node):
                route = self.get_route(curr_node, neighbor_node)
                distance = curr_distance + route.time

                if distance < distances[neighbor_node]:
                    distances[neighbor_node] = distance
                    prev_nodes[neighbor_node] = curr_node
                    heapq.heappush(heap, (distance, neighbor_node))

        return None

    def get_neighbors(self, node_id):
        neighbors = []
        for route_id in self.routes:
            route = self.routes[route_id]
            if route.src == node_id:
                neighbors.append(route.dst)
        return neighbors

    def get_route(self, src, dst):
        for route_id in self.routes:
            route = self.routes[route_id]
            if route.src == src and route.dst == dst:
                return route
        return None

    def reset(self):
        for node_id in self.nodes:
            node = self.nodes[node_id]
            node.reset()
        for route_id in self.routes:
            route = self.routes[route_id]
            route.reset()
        self.packages = {}

    def plan_delivery(self, package_from, package_to, package_priority):
        # 重置环境
        self.reset()

        # 添加起点和终点包裹
        package_id = uuid.uuid4()
        package = self.add_package(package_id, self.TimeTick, package_from, package_to, package_priority)
        node = self.nodes[package_from]
        node.packages.append(package_id)
        node.buffer.append(package_id)

        # 执行路径规划
        path = self.get_shortest_path(package_from, package_to)
        if path:
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                route = self.get_route(src, dst)
                route.packages.append(package_id)
                package.path.append(dst)
                package.history.append((src, dst))
            package.done = True
            return package.path

        return []

# 根据提供的边缘信息和节点信息生成环境
station_pos = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
center_pos = [(6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
airport_pos = [(11, 11), (12, 12), (13, 13), (14, 14), (15, 15)]

station_prop = [(2, 1, 1), (2, 1, 1), (2, 1, 1), (2, 1, 1), (2, 1, 1)]
center_prop = [(5, 1, 1), (5, 1, 1), (5, 1, 1), (5, 1, 1), (5, 1, 1)]
airport_prop = [(10, 1, 1), (10, 1, 1), (10, 1, 1), (10, 1, 1), (10, 1, 1)]

edges = [("s1", "c1", "r1", 1, 1),
         ("s2", "c1", "r2", 1, 1),
         ("s3", "c2", "r3", 1, 1),
         ("s4", "c2", "r4", 1, 1),
         ("s5", "c3", "r5", 1, 1),
         ("c1", "c4", "r6", 1, 1),
         ("c2", "c4", "r7", 1, 1),
         ("c3", "c5", "r8", 1, 1),
         ("c4", "c5", "r9", 1, 1),
         ("c5", "s9", "r10", 1, 1),
         ("s1", "a1", "r11", 10, 1),
         ("a1", "a2", "r12", 1, 1),
         ("a2", "a3", "r13", 1, 1),
         ("a3", "s9", "r14", 10, 1)]

packets = [("s1", "s9", 0),
           ("s2", "s9", 0),
           ("s3", "s9", 0),
           ("s4", "s9", 0),
           ("s5", "s9", 0)]

#logistics_env = LogisticsEnv()
#package_from = "s1"
#package_to = "s9"
#package_priority = 0
#shortest_path = logistics