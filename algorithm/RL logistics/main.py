from LogisticsAgent import *

env = LogisticsEnv()
state = env.reset()

# 创建代理
agent = Agent(env)

# 模拟几个时间步
i=0
done = False
while (not done):
    i+=1
    print(f"------------------------------------------------------STEP{i}------------------------------------------------------")
    actions = agent.choose_action()
    next_state, reward, done = env.step(actions)
    print(f"Time tick: {env.TimeTick}, Total reward: {reward}")

# 检查所有包是否已到达目的地
for package_id in env.packages:
    package = env.packages[package_id]
    if not package.done:
        print(f"Package {package_id} did not reach the destination.")
        break
else:
    print("All packages have reached the destination.")