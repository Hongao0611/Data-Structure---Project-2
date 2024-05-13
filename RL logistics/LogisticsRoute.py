import heapq

class Route:
    def __init__(self, src, dst, time, cost):
        self.src = src
        self.dst = dst
        self.time = time
        self.cost = cost
        self.packages = []  # Use a list for packages on the route
        heapq.heapify(self.packages)  # Convert the list to a heap
        self.package_indices = {}  # 辅助字典用于快速查找包裹的索引

    def add_package(self, package):
        package.delay = self.time
        heapq.heappush(self.packages, (-package.delay, package))
        self.package_indices[package.id] = len(self.packages) - 1  # 更新辅助字典
        print(f"Pack added to Route: {self.src}->{self.dst};")

    def remove_package(self, package_id):
        if package_id in self.package_indices:
            index = self.package_indices.pop(package_id)
            # 直接移除指定索引的元素，并重新构建堆
            del self.packages[index]
            heapq.heapify(self.packages)
            self.package_indices = {pkg.id: i for i, (_, pkg) in enumerate(self.packages)}
            print(f"package: {package_id} removed from Route: {self.src}->{self.dst}.")

    def __str__(self):
        return f"Route({self.src}->{self.dst}, Time: {self.time}, Cost: {self.cost})"
