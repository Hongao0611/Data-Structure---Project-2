import heapq

class Node:
    def __init__(self, id, pos, throughput, delay, cost, is_station=False):
        self.id = id
        self.pos = pos
        self.throughput = throughput
        self.delay = delay  # This delay should be updated based on the queue length and throughput
        self.cost = cost
        self.is_station = is_station
        self.packages = []  # Use a list for packages
        heapq.heapify(self.packages)  # Convert the list to a heap
        self.package_indices = {}  # 辅助字典用于快速查找包裹的索引

    def add_package(self, package):
        heapq.heappush(self.packages, (-package.delay, package))
        package.delay = self.delay
        self.package_indices[package.id] = len(self.packages) - 1  # 更新辅助字典
        print(f"Pack added to Node: {self.id};")

    def remove_package(self, package_id):
        if package_id in self.package_indices:
            index = self.package_indices.pop(package_id)
            # 直接移除指定索引的元素，并重新构建堆
            del self.packages[index]
            heapq.heapify(self.packages)
            self.package_indices = {pkg.id: i for i, (_, pkg) in enumerate(self.packages)}
            print(f"package: {package_id} removed from Node: {self.id}.")

    def __str__(self):
        # 获取所有包裹的 ID
        package_ids = [pkg.id for _, pkg in self.packages]
        # 将包裹 ID 转换为字符串并附加到节点信息中
        return f"Node({self.id}, {self.pos}, Throughput: {self.throughput}, Delay: {self.delay}, Cost: {self.cost}, Packages: {package_ids})"
