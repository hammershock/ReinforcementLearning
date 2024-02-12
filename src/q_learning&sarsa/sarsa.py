import cv2
import numpy as np

from grid_world import GridWorld, visualize_grid_world


if __name__ == '__main__':
    grid_size = (8, 8)
    
    # 初始化Q表
    q_table = np.zeros((*grid_size, 4))  # s, a
    grid = GridWorld(grid_size)
    lr = 0.1
    gamma = 0.99
    # train iteration
    epsilon = 0.1
    over = False
    for i in range(1000):
        grid.reset()
        steps = 0
        epsilon *= 0.99
        action = None
        state = grid.state
        while not grid.done:
            action = grid.choose_action_epsilon_greedy(q_table, epsilon=epsilon)
            
            next_state, reward, done, info = grid.move(action)
            # 类似策略迭代方法
            # next action是依据当前策略选择的，而不是最优值
            next_action = grid.choose_action_epsilon_greedy(q_table)  # 随机性策略，有一定的概率随机探索
            
            q_table[state][action] += lr * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])
            action = next_action
            state = next_state
            
            steps += 1
            frame = visualize_grid_world(grid, q_table)
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                over = True
                break
        if over:
            break
        print(steps)
