import os
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import JacksCarRental, sum_poisson, plot_policy, plot_value


def plot_transition(jcr: JacksCarRental, V, gamma=0.09, num_processes=10):
    trans_matrix_for_A = np.zeros((len(jcr.action_space),) + jcr.state_shape)
    trans_matrix_for_B = np.zeros_like(trans_matrix_for_A)
    
    args_list = [(jcr, s, V, gamma) for s in jcr.state_space]
    
    # 使用进程池并行处理每个状态的策略改进
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(improve_policy_for_state, args_list), desc="Calculating transition Matrix...",
                            total=len(jcr.state_space)))
    
    for state, _, trans_probs_of_actions in results:
        for i, trans_probs in enumerate(trans_probs_of_actions):
            for state_next, prob in zip(jcr.state_space, trans_probs):
                trans_matrix_for_A[i, state[0], state_next[0]] += prob
                trans_matrix_for_B[i, state[1], state_next[1]] += prob
    
    np.save('./output/transition_matrix_for_A.npy', trans_matrix_for_A)
    np.save('./output/transition_matrix_for_B.npy', trans_matrix_for_B)
    
    trans_matrix_for_A = np.load('./output/transition_matrix_for_A.npy')
    trans_matrix_for_B = np.load('./output/transition_matrix_for_B.npy')
    
    fig, axs = plt.subplots(len(jcr.action_space), 2, figsize=(12, len(jcr.action_space) * 6))
    
    for index, action in enumerate(jcr.action_space):
        move = action - 5
        for i, (location, mat) in enumerate([('A', trans_matrix_for_A), ('B', trans_matrix_for_B)]):
            mat = mat[action].T
            mat /= mat.sum(axis=0)
            ax = axs[index, i]
            cax = ax.matshow(mat, cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Location {location}, Action: Move {move} cars")
            ax.set_xlabel('Current Number of Cars')
            ax.set_ylabel('Next Day Number of Cars')
            ax.set_xticks(range(len(trans_matrix_for_A[action])))
            ax.set_yticks(range(len(trans_matrix_for_A[action])))
    
    plt.tight_layout()
    plt.savefig('./output/trans_matrix.png')
    plt.show()
    
    
def policy_evaluation(epoch: int, jcr: JacksCarRental, policy, V, theta=0.1):
    iter_num = 0
    
    while True:
        delta = 0
        iter_num += 1
        for s in tqdm(jcr.state_space, desc=f'Policy Evaluation iter {epoch}'):
            old_v = V[s]
            a = policy[s]  # 根据当前策略选取确定性动作
            V[s], _ = sum_poisson(jcr, s, a, V, jcr.gamma)  # 当前状态价值就是依据当前确定性策略，采取确定性行动的动作价值
            delta = max(delta, abs(old_v - V[s]))  # 计算更新量
        print(f'current delta: {delta}')
        
        if delta < theta:  # 更新量小于delta就认为价值收敛
            break

    return V, delta


def evaluate_state(args):
    jcr, s, policy, V, gamma = args
    a = policy[s]  # 根据当前策略选取的动作
    return s, sum_poisson(jcr, s, a, V, gamma)[0]


def policy_evaluation_concurrent(epoch, jcr: JacksCarRental, policy, V, theta=0.1, num_processes=4):
    iter_num = 0
    delta = float('inf')
    
    while delta >= theta:
        iter_num += 1
        delta = 0
        args_list = [(jcr, s, policy, V, jcr.gamma) for s in jcr.state_space]
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(evaluate_state, args_list), total=len(jcr.state_space),
                                desc=f'Policy Evaluation iter {epoch}'))
        
        # 更新V表格
        for s, v in results:
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        
        print(f'Iteration {iter_num}, Epoch {epoch}, Current delta: {delta}')
        
        if delta < theta:
            break
    
    return V, delta


def policy_improvement(jcr: JacksCarRental, policy, V):
    policy_stable = True
    for s in tqdm(jcr.state_space, desc="Improving Policy..."):  # 对于每一个状态
        old_action = policy[s]  # 之前基于策略的行动
        action_values = np.zeros(len(jcr.action_space))
        for a in jcr.action_space:  # 对于该状态下的每一个动作
            action_values[a], _ = sum_poisson(jcr, s, a, V, jcr.gamma)  # 计算状态动作价值（Q值）
        new_action = np.argmax(action_values)
        policy[s] = new_action  # 选取价值最高的动作更新策略
        if old_action != new_action:  # 只要有一个状态的动作成功改进，就认为策略没有达到稳定
            policy_stable = False
    return policy, policy_stable


def improve_policy_for_state(args):
    jcr, s, V, gamma = args
    action_values = np.zeros(len(jcr.action_space))
    trans_probs_of_actions = []
    for a in jcr.action_space:
        action_values[a], trans_prob = sum_poisson(jcr, s, a, V, gamma)
        trans_probs_of_actions.append(trans_prob)
    new_action = np.argmax(action_values)
    return s, new_action, trans_probs_of_actions


def policy_improvement_concurrent(jcr: JacksCarRental, policy, V, num_processes=4):
    policy_stable = True
    # 准备参数列表，每个元素对应一个状态的完整策略改进任务
    args_list = [(jcr, s, V, jcr.gamma) for s in jcr.state_space]
    
    # 使用进程池并行处理每个状态的策略改进
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(improve_policy_for_state, args_list), desc="Improving policy...", total=len(jcr.state_space)))
    
    # 更新策略并检查是否稳定
    for s, new_action, _ in results:
        old_action = policy[s]
        policy[s] = new_action
        if old_action != new_action:
            policy_stable = False
    
    return policy, policy_stable


if __name__ == '__main__':
    # Jack's Car Rental Problem
    jcr = JacksCarRental(initA=15, initB=10, limit=20, gamma=0.09)
    
    # 初始化策略矩阵，index为5对应的action对应着不移动任何车
    policy = np.full(jcr.state_shape, fill_value=5, dtype=int)
    V = np.zeros(jcr.state_shape)  # 初始化价值矩阵
    
    if os.path.exists('./output/policy.npy') and os.path.exists('./output/value.npy'):
        policy = np.load('./output/policy.npy')
        V = np.load('./output/value.npy')
        
    # 开始策略迭代
    iter_num = 0
    policy_stable = False
    num_process = 16  # 使用16个进程并发计算

    while not policy_stable:
        V, delta = policy_evaluation_concurrent(iter_num, jcr, policy, V, theta=0.1, num_processes=num_process)  # 评估当前策略的价值，可以不用到完全收敛
        iter_num += 1
        string_map = {1: 'st', 2: 'nd', 3: 'rd'}
        print(f'{iter_num}{string_map.get(iter_num, "th")} policy iteration converged at delta {delta}')
        policy, policy_stable = policy_improvement_concurrent(jcr, policy, V, num_processes=num_process)  # 改进策略，直到策略不再更新为止
        np.save('./output/policy.npy', policy)
        np.save('./output/value.npy', V)
    print(f'the policy is stable after {iter_num} iterations')
    
    # 绘制
    plot_policy(jcr, policy)  # 绘制策略矩阵
    plot_value(jcr, V)  # 绘制价值矩阵
    plot_transition(jcr, V, num_processes=16)  # 绘制状态转移矩阵
