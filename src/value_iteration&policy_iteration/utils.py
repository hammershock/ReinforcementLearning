import os
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson


class JacksCarRental(object):
    def __init__(self, initA=10, initB=10, limit=20, gamma=0.09):
        self.initA, self.initB = initA, initB
        self.total_cars = self.initA + self.initB
        self.limit = limit
        self.cars = [initA, initB]  # 初始时20辆车
        self.max_move = 5
        self.gamma = gamma
        self.in_lambdas = (3, 2)
        self.out_lambdas = (3, 4)
        
        self.action_space = list(range(11))
        self.state_space = [(i, j) for i, j in product(range(limit + 1), repeat=2) if i + j <= self.total_cars]
        self.state_mapping = {state: i for i, state in enumerate(self.state_space)}
        self.move_cost = 2.
        self.rent_earn = 10.
        self.state_shape = (limit + 1, limit + 1)
        
        self.steps = 0  # 计数
        self.net_earnings = 0.
    
    @property
    def state(self):
        return tuple(self.cars)
    
    
poisson_cache = {}


def poisson_pmf(lam, n):
    """计算泊松分布的PMF，给定λ和事件数n。"""
    if (lam, n) in poisson_cache:
        return poisson_cache[(lam, n)]
    poisson_cache[(lam, n)] = poisson.pmf(n, lam)  # 将特定的泊松分布概率值缓存起来，避免重复计算
    return poisson_cache[(lam, n)]


def sum_poisson(env: JacksCarRental, state, action, V, gamma):
    """
    考虑每一个可能的下一个状态，计算一个状态的期望回报
    """
    move = action - 5
    move_cost = abs(move) * env.move_cost  # 移动成本
    # 先移动车
    if move > 0:  # 从A向B移动车
        num_move = min(state[0], env.limit - state[1], move)
    else:  # 从B向A移动车
        num_move = -min(state[1], env.limit - state[0], -move)
    
    cars_1 = state[0] - num_move
    cars_2 = state[1] + num_move
    
    trans_probs = np.zeros((len(env.state_space, )))  # 从state转移到各个state的概率
    probs = []
    returns = []
    
    # 对所有可能的租车和还车事件进行遍历，由于每天租车还车大于20辆是小概率事件，我们将大于20辆进行截尾
    truncation_point = 21
    for rent_1, rent_2, returns_1, returns_2 in product(range(truncation_point), repeat=4):
        new_cars_1 = cars_1 - rent_1 + returns_1
        new_cars_2 = cars_2 - rent_2 + returns_2
        if (new_cars_1, new_cars_2) not in env.state_space:  # 不合法的状态可能性为0，过滤掉。两个租车点的总车数不应该大于总共Jack拥有的总车数
            continue
        
        # 计算状态转移的概率，（假设每个租车点每天租车还车相互独立）
        prob = (poisson_pmf(env.in_lambdas[0], returns_1) *
                poisson_pmf(env.out_lambdas[0], rent_1) *
                poisson_pmf(env.in_lambdas[1], returns_2) *
                poisson_pmf(env.out_lambdas[1], rent_2))
        
        rent_revenue = (rent_1 + rent_2) * env.rent_earn  # 租车收益
        reward = rent_revenue - move_cost  # 计算奖励
        
        # 根据贝尔曼方程更新期望回报
        ret = reward + gamma * V[new_cars_1, new_cars_2]
        probs.append(prob)
        trans_probs[env.state_mapping[(new_cars_1, new_cars_2)]] += prob
        
        returns.append(ret)
    
    probs = np.array(probs)
    probs = 1 / np.sum(probs) * probs  # 由于受到截尾以及不合法的一些情况的影响，我们需要对概率归一化
    expected_returns = np.sum(np.array(returns) * probs)
    trans_probs = 1 / np.sum(trans_probs) * trans_probs  # 对trans_probs也做归一化
    return expected_returns, trans_probs


def plot_policy(jcr: JacksCarRental, policy):
    plt.figure(figsize=(12, 10))
    
    # 定义颜色映射和范围
    norm = plt.Normalize(-5, 5)
    cmap = plt.get_cmap('RdYlGn')
    
    # 使用黑色边框的矩形绘制每个状态的策略
    for x1, x2 in product(range(jcr.limit + 1), repeat=2):
        if x1 + x2 <= jcr.total_cars:
            action = policy[x1, x2]
            move = action - 5  # 从A到B移动的车辆数
            color = cmap(norm(move))
            rect = plt.Rectangle((x1 - 0.5, x2 - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor=color,
                                 label=str(action))
            plt.gca().add_patch(rect)
    
    # 绘制结果图
    plt.title('Optimal Policy for Jack’s Car Rental')
    plt.xlabel('Number of Cars at Location A')
    plt.ylabel('Number of Cars at Location B')
    
    # 调整坐标轴刻度以与色块对齐
    plt.xticks(np.arange(0, jcr.limit + 1, 1))
    plt.yticks(np.arange(0, jcr.limit + 1, 1))
    
    # 添加颜色条以反映移动策略
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Cars moved from A to B (-5 to 5)',
                        ticks=range(-5, 6))
    cbar.set_ticklabels(range(-5, 6))  # 显示所有刻度标签
    
    # plt.grid(True)
    plt.xlim(-0.5, 20.5)  # 限制x轴的范围
    plt.ylim(-0.5, 20.5)  # 限制y轴的范围
    plt.gca().set_aspect('equal', adjustable='box')  # 保持x和y轴的刻度一致
    os.makedirs('output', exist_ok=True)
    save_path = './output/jack_car_rental_policy.png'
    plt.savefig(save_path)
    print(f'save policy to: {save_path}')
    plt.show()


def plot_value(jcr: JacksCarRental, V):
    """
    绘制状态值函数V的热力图，仿照plot_policy的风格。
    """
    plt.figure(figsize=(12, 10))
    filtered_v = [V[i, j] for i, j in jcr.state_space]
    norm = plt.Normalize(min(filtered_v), max(filtered_v))
    
    cmap = plt.get_cmap('viridis')
    
    for x1, x2 in jcr.state_space:
        value = V[x1, x2]
        color = cmap(norm(value))
        rect = plt.Rectangle((x1 - 0.5, x2 - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor=color)
        plt.gca().add_patch(rect)
    
    plt.title('Value Function Heatmap for Jack’s Car Rental')
    plt.xlabel('Number of Cars at Location A')
    plt.ylabel('Number of Cars at Location B')
    plt.xticks(np.arange(0, 21, 1))
    plt.yticks(np.arange(0, 21, 1))
    plt.xlim(-0.5, 20.5)
    plt.ylim(-0.5, 20.5)
    plt.gca().set_aspect('equal', adjustable='box')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Value')
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    save_path = f"./output/value_function_heatmap.png"
    plt.savefig(save_path)
    print("Saved Value Function Heatmap at {}".format(save_path))
    plt.show()

