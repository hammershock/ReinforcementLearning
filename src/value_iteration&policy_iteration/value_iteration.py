"""
价值迭代算法实现
"""
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from utils import JacksCarRental, sum_poisson, plot_policy, plot_value


def value_iteration(jcr: JacksCarRental, theta=0.1, gamma=0.9):
    """
    执行值迭代算法以找到最优策略。
    """
    V = np.zeros((jcr.limit + 1, jcr.limit + 1))  # 初始化状态值函数
    policy = np.zeros(V.shape, dtype=int)  # 初始化策略，尽管在值迭代中策略是隐式更新的
    
    while True:
        delta = 0
        for s in tqdm(jcr.state_space, desc='Value Iteration'):
            v = V[s]
            best_value = -np.inf
            best_action = None
            # best_value, best_action = max((sum_poisson(jcr, s, a, V, gamma)[0], a) for a in jcr.action_space)
            for a in jcr.action_space:
                expected_return, _ = sum_poisson(jcr, s, a, V, gamma)
                if expected_return > best_value:
                    best_value = expected_return
                    best_action = a
            V[s] = best_value  # 直接更新状态值为最佳动作下的期望回报
            policy[s] = best_action  # 更新策略为最佳动作
            delta = max(delta, abs(v - best_value))
        
        if delta < theta:  # 当改进小于一定阈值时停止迭代
            break
    
    return policy, V


def evaluate_state_for_value_iteration(args):
    jcr, s, V, gamma = args
    best_value = -np.inf
    best_action = None
    for a in jcr.action_space:
        expected_return, _ = sum_poisson(jcr, s, a, V, gamma)
        if expected_return > best_value:
            best_value = expected_return
            best_action = a
    return s, best_value, best_action


def value_iteration_concurrent(jcr: JacksCarRental, theta=0.1, num_processes=4):
    V = np.zeros((jcr.limit + 1, jcr.limit + 1))  # 初始化状态值函数
    policy = np.zeros(V.shape, dtype=int)  # 初始化策略，尽管在值迭代中策略是隐式更新的
    iter_num = 0
    while True:
        delta = 0
        iter_num += 1
        args_list = [(jcr, s, V, jcr.gamma) for s in jcr.state_space]
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(evaluate_state_for_value_iteration, args_list), total=len(jcr.state_space),
                                desc='Value Iteration'))
        
        # 更新V表格和策略
        for s, new_value, best_action in results:
            delta = max(delta, abs(V[s] - new_value))
            V[s] = new_value
            policy[s] = best_action
        print(f'Iteration {iter_num}, Current delta: {delta}')
        if delta < theta:  # 当改进小于一定阈值时停止迭代
            break
            
    string_map = {1: 'st', 2: 'nd', 3: 'rd'}
    print(f'{iter_num}{string_map.get(iter_num, "th")} policy iteration stopped at delta {delta}')
    return policy, V, delta


if __name__ == '__main__':
    # 初始化环境
    env = JacksCarRental()
    
    # 执行值迭代
    policy, V, delta = value_iteration_concurrent(env, theta=0.1, num_processes=16)  # 使用16个进程并发计算
    
    # 打印最终的值函数
    print("Value function after convergence:")
    plot_value(env, V)
    plot_policy(env, policy)
    