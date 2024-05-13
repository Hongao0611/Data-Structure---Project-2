class Package:
    def __init__(self, id, time_created, src, dst, category):
        self.id = id
        self.time_created = time_created
        self.src = src
        self.dst = dst
        self.curr = src
        self.category = category  # 'Express' or 'Standard'
        self.history = []  # List of nodes visited with timestamps
        self.path = []  # List of nodes on the expected path
        self.delay = 0  # Remaining delay before processing
        self.done = False
        self.reward = 0.0  # Current Reward/cost for this package

    def update_path(self, next_node):
        self.path.append(next_node)

    def mark_done(self):
        self.done = True

    def __lt__(self, other):
        # Ensure other is also a Package object
        if not isinstance(other, Package):
            return False

        # Define comparison for priority queue
        # Express packages have higher priority
        if self.category == 'Express' and other.category != 'Express':
            return True
        if self.category != 'Express' and other.category == 'Express':
            return False
        # Otherwise, compare based on time created
        return self.time_created < other.time_created

    def __eq__(self, other):
        # Ensure other is also a Package object
        if not isinstance(other, Package):
            return False

        # Define comparison for priority queue
        # Express packages have higher priority
        if self.category == 'Express' and other.category != 'Express':
            return True
        if self.category != 'Express' and other.category == 'Express':
            return False
        # Otherwise, compare based on time created
        return self.time_created == other.time_created

    def __str__(self):
        return f"Package({self.id}, TimeCreated: {self.time_created}, Src: {self.src}, Dst: {self.dst}, Category: {self.category})"
