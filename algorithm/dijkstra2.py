import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from sklearn.cluster import KMeans
import heapq

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
        self.time_arrived = float('inf')
        self.src = src
        self.dst = dst
        self.category = category  # 1 for 'Express' and 0 for 'Standard'
        self.history = []
        self.path = []  # List of nodes on the expected path
        self.delay = float('inf')  # Remaining delay before processing
        self.done = False
        self.reward = 0.0  # Current Reward/cost for this package
    def __str__(self):
        #return f"Package({self.id}, TimeCreated: {self.time_created}, Src: {self.src}, Dst: {self.dst}, Category: {self.category})"
        return f"Package({self.id}, Path: {self.path})"

class Node:
    def __init__(self, id, pos, throughput, delay, cost, is_station=False):
        self.id = id
        self.pos = pos
        self.throughput = throughput
        self.delay = delay
        self.cost = cost
        self.is_station = is_station
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
        self.package_order = 0
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
            #print(f"package: {package.id} removed from Node: {self.id}.")
            self.process_packages()
            return package
        elif self.packages[0][1].delay > 0:
            print(f"Error! Removing Package: {package.id} not done from Node: {self.id}!")

    def __str__(self):
        return f"Route({self.src}->{self.dst}, Time: {self.time}, Cost: {self.cost})"


class Route:
    def __init__(self, src, dst, time, cost):
        self.src = src
        self.dst = dst
        self.id = f"{self.src}->{self.dst}"
        self.time = time
        self.cost = cost
        self.package_order = 0
        self.packages = []  # Use a list for packages on the route
        self.history = []
        heapq.heapify(self.packages)  # Convert the list to a heap

    def reset(self):
        self.package_order = 0
        self.packages = []  # Use a list for packages on the route
        self.history = []
        heapq.heapify(self.packages)  # Convert the list to a heap

    def add_package(self, package):
        self.history.append(package.id)
        self.package_order += 1
        # Packages are added to the route.packages in the order they arrive
        heapq.heappush(self.packages, (self.package_order, package))
        package.delay = self.time  # Update package delay
        print(f"Pack added to Route: {self.src}->{self.dst};")

    def remove_package(self):
        if self.packages and self.packages[0][1].delay <= 0:
            _, package = heapq.heappop(self.packages)  # Remove the package with the least priority (oldest)
            print(f"package: {package.id} removed from Route: {self.id}.")
            return package
        elif self.packages[0][1].delay > 0:
            print(f"Error! Removing Package: {package.id} not done from Route: {self.id}!")
            return None

    def __str__(self):
        return f"Route({self.src}->{self.dst}, Time: {self.time}, Cost: {self.cost})"


def get_top_package(node_or_route):
    # 检查堆是否为空
    if not node_or_route.packages:
        return None
    # 获取堆顶的包裹
    _, package = node_or_route.packages[0]
    return package


def get_next_node(package_path_list, curr_node):
    if (len(package_path_list) == 0):
        print("Error! package_path_list is empty!")
        return None
    else:
        for i in range(len(package_path_list) - 1):
            if package_path_list[i] == curr_node:
                return package_path_list[i + 1]
        print("Error! curr_node not found in path!")
        return None

class Location:
    def __init__(self, name):
        self.name = name
        self.distance = float('inf')
        self.previous = None
        self.routes = []

# 创建 Location 类的实例
station_pos = [Location('s1'), Location('s2'), Location('s3'), Location('s4'),Location('s5'), Location('s6')]

# 分配初始距离
initial_distances = [0, float('inf'), float('inf'), float('inf')]  # 根据位置数量来设定初始距离
for i, location in enumerate(station_pos):
    if i < len(initial_distances):
        location.distance = initial_distances[i]

start = station_pos[0]
end = station_pos[-1]
moneycost_initial = 0  # 在此处设置初始距离值

start.distance = moneycost_initial
start.previous = None


# 调用dijkstra函数
#path = dijkstra(locations, start, end)
def dijkstra(locations, start, end):
    # 初始化队列
    queue = [start]
    start.previous = None
    distances = {location: float('inf') for location in locations}
    distances[start] = 0
    previous = {}

    while queue:
        # 找到下一个最短路径节点
        current_location = min(queue, key=lambda x: x.distance)
        queue.remove(current_location)

        # 检查是否达到终点
        if current_location == end:
            break

        # 更新当前位置的最短路径
        for neighbor in current_location.routes:
            # 更新路径长度
            new_distance = current_location.distance + neighbor.distance

            if new_distance < neighbor.distance:
                neighbor.distance = new_distance
                neighbor.previous = current_location

    # 构建路径并返回
    path = []
    while current_location != start:
        path.append(current_location)
        current_location = current_location.previous
    path.append(start)
    path.reverse()
    return path

# 根据位置ID获取对应的位置对象
def get_location_by_id(locations, id):
    for location in locations:
        if location.id == id:
            return location
    return None
# 根据包裹ID获取对应的包裹对象
def get_package_by_id(packages, id):
    for package in packages:
        if package.id == id:
            return package
    return None

# 模拟器主函数
def simulate(locations, routes, packages):
    # 定义locations、start和end
    locations = station_pos
    start = station_pos[0]
    end = station_pos[-1]

    # 调用dijkstra函数
    path = dijkstra(locations, start, end)
    for route in routes:
        src_location = get_location_by_id(locations, route.src)
        dst_location = get_location_by_id(locations, route.dst)
        src_location.routes.append(Route(src_location, dst_location, route.time, route.cost))

    for package in packages:
        src_location = get_location_by_id(locations, package.src)
        dst_location = get_location_by_id(locations, package.dst)
        path = dijkstra(locations, src_location, dst_location)
        total_time = package.time_created

        package_path = []
        for i in range(len(path) - 1):
            current_location = path[i]
            package_path.append(current_location.id)
            next_location = path[i + 1]
            route = None

            for r in current_location.routes:
                if r.dst == next_location:
                    route = r
                    break

            if route is not None:
                time_taken = max(total_time, route.time)
                total_time = time_taken + route.delay
                event = {
                    'Time': time_taken,
                    'Location': current_location.id,
                    'Event': 'ARRIVED'
                }
                package.log.append(event)
                event = {
                    'Time': total_time,
                    'Location': current_location.id,
                    'Event': 'PROCESSING'
                }
                package.log.append(event)
                event = {
                    'Time': total_time,
                    'Location': next_location.id,
                    'Event': 'SENT'
                }
                package.log.append(event)

        package.log.append({
            'Time': total_time,
            'Location': dst_location.id,
            'Event': 'ARRIVED'
        })
        package.time_delivered = total_time

        print(f"Pack added to Node: {package_path[0]};")
        print(f"Package {package.id} added, delay={package.delay}, src={package.src}, dst={package.dst}, done={package.done}, path={package_path}")

    # 根据包裹到达时间排序
    packages.sort(key=lambda p: p.time_delivered)

    for package in packages:
        print(f"Package ID: {package.id}")
        print(f"Source: {package.src}")
        print(f"Destination: {package.dst}")
        print(f"Time Created: {package.time_created}")
        print(f"Time Delivered: {package.time_delivered}")
        print("Log:")
        for event in package.log:
            print(f"Time: {event['Time']}, Location: {event['Location']}, Event: {event['Event']}")

    # 遍歷所有路徑並給出每個包裹的用時最短路徑的站點及包裹信息
    for package in packages:
        src_location = get_location_by_id(locations, package.src)
        dst_location = get_location_by_id(locations, package.dst)
        shortest_path = dijkstra(locations, src_location, dst_location)
        path_ids = [location.id for location in shortest_path]
        print(f"Pack added to Node: {path_ids[0]};")
        print(
            f"Package {package.id} added, delay={package.delay}, src={package.src}, dst={package.dst}, done={package.done}, path={path_ids}")

    # 输出包裹列表和跟踪信息
    #for package in packages:
        #print(f"Package ID: {package.id}")
        #print(f"Source: {package.src}")
        #print(f"Destination: {package.dst}")
        #print(f"Time Created: {package.time_created}")
        #print(f"Time Delivered: {package.time_delivered}")
        #print("Log:")
        #print(f"Time: {event['Time']}, Location: {event['Location']}, Event: {event['Event']}")
if __name__ == '__main__':
    # 生成数据
    data = data_gen()
    station_pos = data['station_pos']
    station_prop = data['station_prop']
    center_pos = data['center_pos']
    center_prop = data['center_prop']
    edges = data['edges']
    packets = data['packets']
    #moneycost = data["money cost"]
    #timecost = data["time cost"]
    #parameters = data["parameters"]

    # 调用simulate函数进行模拟
    #simulate(locations, edges, packets)