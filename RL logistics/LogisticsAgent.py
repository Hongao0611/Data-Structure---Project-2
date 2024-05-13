from LogisticsEnv import *

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = None
        self.total_reward = 0.0
        self.action_history = []

    def reset(self):
        self.state = None
        self.total_reward = 0.0
        self.action_history = []

    def choose_action(self):
        actions = {}
        for package_id, package in self.env.packages.items():
            if package.delay > self.env.min_delay:
                # If the package's delay is positive, do nothing
                actions[package_id] = None
            else:
                # Call the get_policy method to get the next node in the optimal path
                next_node = self.env.get_policy(package)
                # Select the neighboring node to be the destination
                if next_node and next_node != package.src and next_node != package.dst:
                    actions[package_id] = next_node[:2]
                else:
                    actions[package_id] = None  # No valid action found
            #print(f"action for {package_id} is {actions[package_id]}.")
        return actions
