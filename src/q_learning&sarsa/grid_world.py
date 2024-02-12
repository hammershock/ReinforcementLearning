"""
Grid-World, a grid-world environment for exploring Reinforcement Learning algorithms
"""

import random

import cv2
import numpy as np


class GridWorld(object):
    PENALIZE = -20.
    REWARD = 10.
    IDLE = -0.1
    
    def __init__(self, grid_size=(5, 5), int_state=False, max_step=None):
        self.pos = (0, 0)
        self.grid_size = grid_size
        self.action_space = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.reward_map = np.full(self.grid_size, self.IDLE, dtype=float)
        self.reward_map[-1, -1] = self.REWARD
        self.reward_map[3, :-3] = self.PENALIZE
        self.int_state = int_state
        self.step = 0
        self.max_step = max_step
        self.return_value = 0.0
        self._end = True
        
    def move(self, action: int) -> tuple[tuple[int, int], float, bool, None]:
        """
        action:
        :param action:
        :return: new state, reward, done, info
        """
        self.step += 1
        
        dx, dy = self.action_space[action]
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy
        
        if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
            self.pos = (new_x, new_y)
            reward = self.reward_map[new_x, new_y]
        else:
            reward = self.PENALIZE
        self.return_value += reward
        return self.state, reward, self.done, None
    
    @property
    def state(self):
        if self.int_state:
            return self.pos[0] * self.grid_size[0] + self.pos[1]
        return self.pos
    
    def iter_state(self):
        while not self.done and not self._end:
            yield self.state
            
    def end(self):
        self._end = True
        
    def iter_episodes(self, num_episodes):
        self._end = False
        episode = 0
        while episode < num_episodes and not self._end:
            self.reset()
            yield episode
            episode += 1
    
    @property
    def action_mask(self):
        mask = np.array([True, True, True, True])  # [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = self.pos
        if x == self.grid_size[0]:
            mask[1] = False
        if x == 0:
            mask[3] = False
        if y == self.grid_size[1]:
            mask[0] = False
        if y == 0:
            mask[2] = False
        return mask
    
    @property
    def done(self):
        # if self.reward_map[self.pos] == self.PENALIZE:
        #     return True
        if self.max_step is not None and self.step >= self.max_step:
            return True
        return self.pos == (self.grid_size[0] - 1, self.grid_size[1] - 1)
    
    def choose_action_epsilon_greedy(self, q_table, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, len(self.action_space) - 1)
        return np.argmax(q_table[self.state])
    
    def reset(self):
        self.step = 0
        self.return_value = 0.
        self.pos = (0, 0)


def visualize_grid_world(grid, q_table=None):
    cell_size = 50  # 设置每个格子的像素大小
    grid_height, grid_width = grid.grid_size
    canvas_height = grid_height * cell_size
    canvas_width = grid_width * cell_size
    
    # 创建画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 绘制网格
    for i in range(grid_height + 1):
        cv2.line(canvas, (0, i * cell_size), (canvas_width, i * cell_size), (0, 0, 0), 1)
    for j in range(grid_width + 1):
        cv2.line(canvas, (j * cell_size, 0), (j * cell_size, canvas_height), (0, 0, 0), 1)
    
    # 填充格子
    for i in range(grid_height):
        for j in range(grid_width):
            if grid.reward_map[i, j] == GridWorld.PENALIZE:
                cv2.rectangle(canvas, (j * cell_size, i * cell_size), ((j + 1) * cell_size, (i + 1) * cell_size),
                              (0, 0, 255), -1)
            elif grid.reward_map[i, j] == GridWorld.REWARD:
                cv2.rectangle(canvas, (j * cell_size, i * cell_size), ((j + 1) * cell_size, (i + 1) * cell_size),
                              (0, 255, 0), -1)
            # elif grid.reward_map[i, j] == 2:
            #     cv2.rectangle(canvas, (j * cell_size, i * cell_size), ((j + 1) * cell_size, (i + 1) * cell_size),
            #                   (0, 255, 255), -1)
    
    if q_table is not None:
        # 显示Q值
        # 此处代码仅为示例，实际实现时可能需要调整文字大小或位置
        for i in range(grid_height):
            for j in range(grid_width):
                action_value_text = f"{q_table[i, j, np.argmax(q_table[i, j])]:.2f}"  # 示例：显示最大Q值
                cv2.putText(canvas, action_value_text, (j * cell_size + 5, i * cell_size + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 标记当前位置
    cv2.circle(canvas, (int(grid.pos[1] * cell_size + cell_size / 2), int(grid.pos[0] * cell_size + cell_size / 2)), 10,
               (0, 255, 255), -1)
    
    return canvas
