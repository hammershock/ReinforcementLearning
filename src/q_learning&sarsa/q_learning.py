import cv2
import numpy as np

from grid_world import GridWorld, visualize_grid_world


if __name__ == '__main__':
    grid_size = (8, 8)
    
    q_table = np.zeros((*grid_size, 4))  # s, a # 用于近似贝尔曼最优公式下的最优动作价值
    grid = GridWorld(grid_size)
    lr = 0.1
    gamma = 0.99
    # train iteration
    epsilon = 0.1
    epsilon_decay = 0.99
    over = False
    for i in range(1000):
        grid.reset()
        steps = 0
        epsilon *= epsilon_decay
        while not grid.done:
            action = grid.choose_action_epsilon_greedy(q_table, epsilon=epsilon)
            state = grid.state
            next_state, reward, done, info = grid.move(action)
            # 增量式更新
            # 直接根据贝尔曼最优公式，直接近似最优策略下的动作价值，简称最优动作价值，类似价值迭代法
            q_table[state][action] += lr * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            steps += 1
            frame = visualize_grid_world(grid, q_table)
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                over = True
                break
        if over:
            break
        print(steps)
        