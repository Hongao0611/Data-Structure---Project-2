import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from sklearn.cluster import KMeans
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
    #plt.show()

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

    M=np.zeros((60,60))
    for i in range(60):
        for j in range(60):
            M[i][j]=np.Infinity
    for i in range(30):
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
                M[2*i+1][2*j+10] = 0.6*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                M[2*j+11][2*i] = M[2*i+1][2*j+10]
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                M[2*i+11][2*j+10] = 0.8*np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                M[2*j+11][2*i+10]=M[2*i+11][2*j+10]
    N=np.zeros((60,60))
    for i in range(60):
        for j in range(60):
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
                N[2*i+1][2*j+10] = 0.12*np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                N[2*j+11][2*i] = N[2*i+1][2*j+10]
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                N[2*i+11][2*j+10] = 0.07*np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                N[2*j+11][2*i+10]=N[2*i+11][2*j+10]
    for i in range(parameters["center_num"]):
        N[2*i][2*i+1]=center_prop[i][2]
    for i in range(parameters["station_num"]):
        N[2*i+10][2*i+11]=station_prop[i][2]

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
moneycost = data["money cost"]
timecost = data["time cost"]

class Package: #TODO add self.history
    def __init__(self, id, time_created, src, dst, category):
        self.id = id
        self.time_created = time_created
        self.src = src
        self.dst = dst
        self.category = category  # 'Express' or 'Standard'
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
        self.package_order = 0
        heapq.heapify(self.packages)  # Convert the list to a heap
        heapq.heapify(self.buffer)

    def reset(self):
        self.packages = []
        self.dones = []
        self.package_order = 0
        heapq.heapify(self.packages)  # Convert the list to a heap
        heapq.heapify(self.buffer)

    def add_package(self, package):
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
                heapq.heappush(self.buffer, (self.package_order ,package))
                package.delay = float('inf')  # Update package delay
                print(f"Pack added to Node: {self.id};")
                self.process_packages()
            else: # 如果buffer有包裹，获取堆顶的包裹
                index, top_package = self.buffer[0]
                if top_package.category or package.category == 0: #如果堆顶是express包裹，不做特殊处理；如果堆顶是standard,插入也是standard,不做特殊处理
                    heapq.heappush(self.buffer, (self.package_order ,package))
                    package.delay = float('inf')  # Update package delay
                    print(f"Pack added to Node: {self.id};")
                else: #如果堆顶是standard包裹,插入是express
                    heapq.heappush(self.buffer, (index-1 ,package))
                    package.delay = float('inf')  # Update package delay
                    print(f"Pack added to Node: {self.id};")

                self.process_packages()
    
    def process_packages(self):
        while self.buffer and len(self.buffer) < self.throughput:
            index, package = heapq.heappop(self.buffer)  # Get the package with the highest priority (oldest)
            if package.done == False:
                heapq.heappush(self.packages, (index, package))
                package.delay = self.delay
            else:
                self.dones.append(package)

    def remove_package(self):
        if self.packages and self.packages[0][1].delay <= 0:
            _, package = heapq.heappop(self.packages)  # Remove the package with the least priority (oldest)
            print(f"package: {package.id} removed from Node: {self.id}.")
            self.process_packages()
            return package
        elif self.packages[0][1].delay > 0:
            print(f"Error! Removing Package: {package.id} not done from Node: {self.id}!")

class Route:
    def __init__(self, src, dst, time, cost):
        self.src = src
        self.dst = dst
        self.id = f"{self.src}->{self.dst}"
        self.time = time
        self.cost = cost
        self.package_order = 0
        self.packages = []  # Use a list for packages on the route
        heapq.heapify(self.packages)  # Convert the list to a heap
    
    def reset(self):
        self.package_order = 0
        self.packages = []  # Use a list for packages on the route
        heapq.heapify(self.packages)  # Convert the list to a heap

    def add_package(self, package):
        self.package_order += 1
        # Packages are added to the route.packages in the order they arrive
        heapq.heappush(self.packages, (self.package_order ,package))
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
    if(len(package_path_list)==0):
        print("Error! package_path_list is empty!")
        return None
    else:
        for i in range(len(package_path_list) - 1):
            if package_path_list[i] == curr_node:
                return package_path_list[i + 1]
        print("Error! curr_node not found in path!")
        return None

class LogisticsEnv:
    def __init__(self):
        self.nodes = {}  # Dictionary of nodes
        self.routes = {}  # Dictionary of routes
        self.packages = {}  # Dictionary of packages
        self.TimeTick = 0.0  # Current time tick
        self.done = False
        self.moneycost=moneycost
        self.timecost=timecost
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
            print(f"Package {p.id} added, delay={p.delay}, src={p.src}, dst={p.dst}, done={p.done}, path={p.path}")
    def reset(self):
        print("Reseting......")
        for node, route in zip(self.nodes.values(),self.routes.values()):
            node.reset()
            route.reset()
        self.packages = {}  # Dictionary of packages
        self.TimeTick = 0.0  # Current time tick
        self.done = False
        self.moneycost=moneycost
        self.timecost=timecost
        # Add packets as packages
        for packet in packets:
            p = self.add_package(uuid.uuid4(), *packet)
            print(f"Package {p.id} added, delay={p.delay}, src={p.src}, dst={p.dst}, done={p.done}, path={p.path}")

        print(f"Env reset. TimeTick={self.TimeTick}")
        return self.get_load()
    
    def get_state(self):
        # 创建一个状态表示，包括决策相关的信息
        state = {
            'current_time_tick': self.TimeTick,
            'nodes': {node_id: {
                'buffer': [(pkg.id, pkg.category) for _, pkg in node.buffer],
                'packages': [(pkg.id, pkg.category) for _, pkg in node.packages],
                'dones': [(pkg.id, pkg.category) for pkg in node.dones]
            } for node_id, node in self.nodes.items()},
            'routes': {(src, dst): tuple(pkg.id for _, pkg in route.packages) for (src, dst), route in self.routes.items()}
        }
        return state

    def get_load(self):
        # 创建一个状态表示，包括决策相关的信息
        state = {
            'nodes': {node_id: {
                'buffer_count': len(node.buffer),
                'packages_count': len(node.packages),
                #'dones_count': len(node.dones)
            } for node_id, node in self.nodes.items()},
            'routes': {(src, dst): len(route.packages) for (src, dst), route in self.routes.items()}
        }
        # 转换状态为数值形式
        converted_state = []
        for node_id, node_info in state['nodes'].items():
            converted_state.extend([node_info['buffer_count'], node_info['packages_count']])
        for route_id, route_info in state['routes'].items():
            converted_state.append(route_info)
        return converted_state

    def get_reward(self):
        reward = 0
        for package in self.packages.values():
            if package.done:
                continue
            elif package.category:
                reward -= 2
            else:
                reward -= 1
        return reward

    def add_node(self, id, pos, throughput, delay, cost, is_station=False):
        self.nodes[id] = Node(id, pos, throughput, delay, cost, is_station)
        return self.nodes[id]

    def add_route(self, src, dst, time, cost):
        self.routes[(src, dst)] = Route(src, dst, time, cost)
        return self.routes[(src, dst)]

    def add_package(self, id, time_created, src, dst, category):
        package = Package(id, time_created, src, dst, category)
        #print(f"Category: {category}")
        # Calculate the initial optimal路径 based on the package's category
        if category:
            #print("find_shortest_time_path")
            # 对于Express包裹，找到最短的总时间路径
            optimal_path = self.find_shortest_time_path(src, dst)
        else:
            #print("find_lowest_cost_path")
            # 对于Standard包裹，找到最低的总成本路径
            optimal_path = self.find_lowest_cost_path(src, dst)
        
        package.path = optimal_path  # 将最优路径初始化为包裹的路径
        assert optimal_path != [], f"Package: {package.id} EMPTY!!!INFO:{id, time_created, src, dst, category}"
        self.packages[id] = package
        self.nodes[src].add_package(package)  # 添加到优先队列中
        return package
   
    def find_shortest_time_path(self, src, dst):
        if src[0]=='s':
            if len(src)==2:
                a=int(src[1])*2+10
            else:
                a=int(src[1:])*2+10
        else:
            a=int(src[1])*2
        if dst[0]=='s':
            if len(dst)==2:
                b=int(dst[1])*2+10
            else:
                b=int(dst[1:])*2+10
        else:
            b=int(dst[1])*2

        n=len(self.timecost)#ordre du graphe
        Delta=[np. Infinity]*n#étape 1
        Chemins=[[]]*n# liste des listes des plus courts chemins
        Delta[a]=0  #étape 1
        Chemins[a]=[a] #plus court chemin de s0 à s0
        for k in range(n-1): #étape 2
            for i in range(n): #étape 3
                for j in range(n): #étape 3
                    if self.timecost[ i ][ j]!=0 and Delta[ i]+self.timecost[ i ][ j]<Delta[ j ]: #ét. 4
                        Delta[ j]=Delta[ i]+self.timecost[ i ][ j ] #étape 4
                        Chemins[ j]=Chemins[ i ]+[j ] #chemin plus court
        d=Chemins
        path=[]
        for i in range(int((len(d[b])+1)/2)):
            if d[b][2*i]>9:
                path.append('s'+str(int((d[b][2*i]-10)/2)))
            else :
                path.append('c'+str(int(d[b][2*i]/2))) 
        return path 
         
    def find_lowest_cost_path(self, src, dst):
        if src[0]=='s':
            if len(src)==2:
                a=int(src[1])*2+10
            else:
                a=int(src[1:])*2+10
        else:
            a=int(src[1])*2
        if dst[0]=='s':
            if len(dst)==2:
                b=int(dst[1])*2+10
            else:
                b=int(dst[1:])*2+10
        else:
            b=int(dst[1])*2

        n=len(self.moneycost)#ordre du graphe
        Delta=[np. Infinity]*n#étape 1
        Chemins=[[]]*n# liste des listes des plus courts chemins
        Delta[a]=0  #étape 1
        Chemins[a]=[a] #plus court chemin de s0 à s0
        for k in range(n-1): #étape 2
            for i in range(n): #étape 3
                for j in range(n): #étape 3
                    if self.moneycost[ i ][ j]!=0 and Delta[ i]+self.moneycost[ i ][ j]<Delta[ j ]: #ét. 4
                        Delta[ j]=Delta[ i]+self.moneycost[ i ][ j ] #étape 4
                        Chemins[ j]=Chemins[ i ]+[j ] #chemin plus court
        d=Chemins
        path=[]
        for i in range(int((len(d[b])+1)/2)):
            if d[b][2*i]>9:
                path.append('s'+str(int((d[b][2*i]-10)/2)))
            else :
                path.append('c'+str(int(d[b][2*i]/2))) 
        return path
    
    def find_alternative_time_path(self, src, dst, avoid_node):
        if src[0]=='s':
            if len(src)==2:
                a=int(src[1])*2+10
            else:
                a=int(src[1:])*2+10
        else:
            a=int(src[1])*2
        if dst[0]=='s':
            if len(dst)==2:
                b=int(dst[1])*2+10
            else:
                b=int(dst[1:])*2+10
        else:
            b=int(dst[1])*2
        M=np.copy(self.timecost)
        if avoid_node[0]=='c':
            for j in range(60):
                if len(src)==2:
                    a=int(src[1])*2   
                else:
                    a=int(src[1:])*2
                M[a][j]=np.Infinity
                M[j][a]=M[a][j]

        if avoid_node[0]=='s':
            for j in range(60):
                if len(src)==2:
                    a=int(src[1])*2+10   
                else:
                    a=int(src[1:])*2+10
                M[a][j]=np.Infinity
                M[j][a]=M[a][j]
        n=len(M)#ordre du graphe
        Delta=[np. Infinity]*n#étape 1
        Chemins=[[]]*n# liste des listes des plus courts chemins
        Delta[a]=0  #étape 1
        Chemins[a]=[a] #plus court chemin de s0 à s0
        for k in range(n-1): #étape 2
            for i in range(n): #étape 3
                for j in range(n): #étape 3
                    if M[ i ][ j]!=0 and Delta[ i]+M[ i ][ j]<Delta[ j ]: #ét. 4
                        Delta[ j]=Delta[ i]+M[ i ][ j ] #étape 4
                        Chemins[ j]=Chemins[ i ]+[j ] #chemin plus court
        d=Chemins
        path=[]
        if Delta[b]>100000:
            path=[]
        else:
            for i in range(int((len(d[b])+1)/2)):
                if d[b][2*i]>9:
                    path.append('s'+str((d[b][2*i]-10)/2))
                else :
                    path.append('c'+str(d[b][2*i]/2)) 
        if len(path)<2:
            return []
        else:
            new_path = [path[0]] + self.find_shortest_time_path(path[1],path[-1])
            return path #TODO new_path

    def find_alternative_cost_path(self, src, dst, avoid_node):
        if src[0]=='s':
            if len(src)==2:
                a=int(src[1])*2+10
            else:
                a=int(src[1:])*2+10
        else:
            a=int(src[1])*2
        if dst[0]=='s':
            if len(dst)==2:
                b=int(dst[1])*2+10
            else:
                b=int(dst[1:])*2+10
        else:
            b=int(dst[1])*2
        M=np.copy(self.timecost)
    
        if avoid_node[0]=='c':
            for j in range(60):
                if len(src)==2:
                    a=int(src[1])*2   
                else:
                    a=int(src[1:])*2
                M[a][j]=np.Infinity
                M[j][a]=M[a][j]

        if avoid_node[0]=='s':
            for j in range(60):
                if len(src)==2:
                    a=int(src[1])*2+10   
                else:
                    a=int(src[1:])*2+10
                M[a][j]=np.Infinity
                M[j][a]=M[a][j]
    
        n=len(M)#ordre du graphe
        Delta=[np. Infinity]*n#étape 1
        Chemins=[[]]*n# liste des listes des plus courts chemins
        Delta[a]=0  #étape 1
        Chemins[a]=[a] #plus court chemin de s0 à s0
        for k in range(n-1): #étape 2
            for i in range(n): #étape 3
                for j in range(n): #étape 3
                    if M[ i ][ j]!=0 and Delta[ i]+M[ i ][ j]<Delta[ j ]: #ét. 4
                        Delta[ j]=Delta[ i]+M[ i ][ j ] #étape 4
                        Chemins[ j]=Chemins[ i ]+[j ] #chemin plus court
        d=Chemins
        path=[]
        if Delta[b]>100000:
            path=[]
        else:
            for i in range(int((len(d[b])+1)/2)):
                if d[b][2*i]>9:
                    path.append('s'+str((d[b][2*i]-10)/2))
                else :
                    path.append('c'+str(d[b][2*i]/2)) 
        if len(path)<2:
            return []
        else:
            new_path = [path[0]] + self.find_shortest_time_path(path[1],path[-1])
            return path #TODO new_path

    def change_route(self, route):#TODO 检查是否正确更改路径
        src_node = self.nodes[route.src]
        # 检查堆是否为空
        if src_node.packages:
            # 获取堆顶的包裹
            for _, package in src_node.packages:
                next_node_id = get_next_node(package.path,src_node.id)
                if package.delay<=0.1 and next_node_id == route.dst and next_node_id != package.dst:
                    print(f"有符合条件的包裹！{package}")
                    print(type(next_node_id),next_node_id)
                    if package.category:
                        # For Express packages, find the shortest total time path
                        new_path=self.find_alternative_time_path(src_node.id, package.dst, next_node_id)
                    else:
                        # For Standard packages, find the lowest total cost path
                        new_path=self.find_alternative_cost_path(src_node.id, package.dst, next_node_id)
                    if new_path == []: #如果有新路线，采用；如果没有，不变
                        print(f"Alternative Route unavailable!")
                    else:
                        print(f"Route for Pack: {package.id} changed from {package.path} to {new_path}!")
                        package.path = new_path
                        
    def step(self, actions=None): 
        self.done = all(package.done for package in self.packages.values())
        if self.done == True:
            print("All packs are done!")
            return self.get_load(), self.get_reward()
        
        if actions != None:
            for route, action in zip(self.routes.keys(), actions):
                if action==1:
                    self.change_route(self.routes[route])
                else:
                    continue
        
        self.TimeTick += 0.1  # 更新时间
        # 更新所有包裹的延迟
        for package in self.packages.values():
            package.delay -= 0.1
        
        for node in self.nodes.values():
            top_package = get_top_package(node)
            while (top_package != None and top_package.delay <= 0):
                # 从Node中删除包裹
                top_package = node.remove_package()
                # 往Route中添加包裹
                next_node_id = get_next_node(top_package.path, node.id)
                route = self.routes[(node.id,next_node_id)]
                route.add_package(top_package)
                # 获取下一个包裹
                top_package = get_top_package(node)

        for route in self.routes.values():
            top_package = get_top_package(route)
            while (top_package != None and top_package.delay <= 0):
                # 从Route中删除包裹
                top_package = route.remove_package()
                # 往Node中添加包裹
                next_node = self.nodes[route.dst]
                next_node.add_package(top_package)
                # 获取下一个包裹
                top_package = get_top_package(node)
        return self.get_load(), self.get_reward()
    
def print_state(state):
    print("State:")
    print("  Current TimeTick:", state['current_time_tick'])
    print("  Nodes:")
    for node_id, node_info in state['nodes'].items():
        if any(node_info['packages']) or any(node_info['buffer']) or any(node_info['dones']):
            print(f"    Node {node_id}:")
            buffer_items = [f"({id}, {category})" for id, category in node_info['buffer']]
            packages_items = [f"({id}, {category})" for id, category in node_info['packages']]
            dones_items = [f"({id}, {category})" for id, category in node_info['dones']]
            print(f"      Buffer: {buffer_items}")
            print(f"      Packages: {packages_items}")
            print(f"      Dones: {dones_items}")
    print("  Routes:")
    for route_id, route_info in state['routes'].items():
        if packages_items:=[f"({id})" for id in route_info]:  # 检查是否有包裹
            print(f"    Route {route_id}:")
            print(f"      Packages: {packages_items}")

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
from collections import deque
class Agent: # TODO 检查程序是否正确
    def __init__(self, env, buffer_size=10000):
        self.env = env
        self.action_n = len(env.routes.keys())  # 0 for unchanged route, 1 for changed route
        self.state_n = 2*len(env.nodes.keys()) + len(env.routes.keys())
        print(f"Action_N: {self.action_n}, State_N: {self.state_n}")
        # 初始化神经网络
        self.model = nn.Sequential(
            nn.Linear(self.state_n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_n)
        )
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1
        # 经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        # 目标网络
        self.target_model = nn.Sequential(*list(self.model.children()))
        self.target_model.eval()
        self.update_target_model()
        
    def update_target_model(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param.data)
        
    def choose_action(self, state):
        #print(f"Model input shape: {self.model[0].in_features}")  # 打印模型的输入维度
        #print(state)
        #print(f"State shape: {state.shape}")  # 打印state的形状

        # 使用ε-greedy策略选择动作
        if random.random() < self.epsilon:
            #print("random action!")
            actions = [random.choice([0, 1]) for _ in range(self.action_n)]
        else:
            #print("NOT random action!")
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            actions = q_values.argmax().item()
        
        #dist = td.Categorical(logits=q_values)
        #actions = [dist.sample().item() for _ in self.env.routes.keys()]
        #print(f"Action:{actions}")
        return actions

    def store_transition(self, state, action, reward, next_state):
        # 确保状态和动作是Tensor格式
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        next_state = torch.FloatTensor(next_state)
        # 存储转换后的Tensor状态和动作
        self.memory.append((state, action, reward, next_state))

    def train(self, n_episodes):
        for episode in range(n_episodes):
            total_reward = 0
            state = self.env.reset()
            while not self.env.done:
                action = self.choose_action(state)
                next_state, reward = self.env.step(action)
                self.store_transition(state, action, reward, next_state)
                state = next_state
                #print(f"Reward: {type(reward)}; {reward}")
                total_reward += reward
            print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")
        
        for _ in range(n_episodes):
            batch = random.sample(self.memory, self.batch_size)  # 随机采样batch_size条经验
            states, actions, rewards, next_states = zip(*batch)
            #print(type(actions))
            #print(type(actions[0]))
            #print(actions[0].size())
            actions = torch.stack(actions, dim=0)
            actions_indices = torch.argmax(actions, dim=1).unsqueeze(1)
            states = torch.stack([torch.FloatTensor(state_values) for state_values in states])
            rewards = torch.FloatTensor(rewards)
            next_states = torch.stack([torch.FloatTensor(state_values) for state_values in next_states])
            
            # 计算当前Q值
            current_q_values = self.model(states).gather(1, actions_indices)
            
            # 计算目标Q值
            with torch.no_grad():
                next_actions = self.target_model(next_states).max(1)[1].unsqueeze(1)  # 获取最大Q值的动作索引
                next_q_values = self.target_model(next_states).gather(1, next_actions)  # 使用gather获取对应的Q值
                target_q_values = rewards + self.gamma * next_q_values

            # 确保target_q_values与current_q_values具有相同的形状
            target_q_values = target_q_values.gather(1, actions_indices)

            # 计算损失
            loss = self.criterion(current_q_values, target_q_values)
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新目标网络
            self.update_target_model()

    def test(self):
        state = self.env.reset()
        total_reward = 0
        while not self.env.done:
            action = self.choose_action(state)
            next_state, reward = self.env.step(action)
            self.store_transition(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        print(f"Total Reward: {total_reward}")

def test_env(n=10):
    # 初始化环境, 打印初始状态
    env = LogisticsEnv()
    print("Initial State:")
    print_state(env.get_state())
    # 模拟
    for _ in range(n):
        while env.done == False:
            env.step() # 获取状态字典
            print_state(env.get_state())
        env.reset()

def test_RL():
    env = LogisticsEnv()
    agent = Agent(env)
    agent.train(3)
    agent.test()

# 运行测试
test_RL()