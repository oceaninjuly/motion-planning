import random
import math
import copy
import time

import matplotlib.pyplot as plt
import numpy as np

freq = 0.1
show_info = False

class RRT:

    # 初始化
    def __init__(self,
                 obstacle_list,         # 障碍物
                 rand_area,             # 采样的区域
                 expand_dis=2.0,        # 步长
                 goal_sample_rate=10,   # 目标采样率
                 max_iter=200):         # 最大迭代次数

        self.start = None
        self.goal = None
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None
        self.connect = False

    # RRT*: 重写
    def rewrite(self,new_node,k=1.8):
        old_parent = new_node.parent
        min_dis = float('inf')
        min_index = -1
        for i,old_node in enumerate(self.node_list):
            if old_node is new_node: continue
            dis = self.line_cost(old_node,new_node)
            if dis < k*self.expand_dis and self.check_segment_collision(old_node.x,old_node.y,new_node.x,new_node.y):
                dis += old_node.cost
                if dis < min_dis:
                    min_dis = dis
                    min_index = i

        if min_index != old_parent:
            if show_info:
                print(f"rewrite for newnode {len(self.node_list)-1},old parent:{old_parent},new parent:{min_index}")
            self.node_list[old_parent].kids.remove(len(self.node_list)-1)
            new_node.parent = min_index
            self.node_list[min_index].kids.append(len(self.node_list)-1)
            new_node.cost = min_dis
            return True

        return False 

    def chk(self):
        visited = {}
        repeated = []
        queue = [0]
        while len(queue) != 0:
            ind = queue[0]
            queue.pop(0)
            if ind not in visited.keys():
                visited[ind] = 1
            else:
                repeated.append(ind)
                continue
            current = self.node_list[ind]
            queue += current.kids
        if len(repeated) != 0:
            print(f'repeated:{repeated}')
            print('?????')

    # RRT*: 重连接
    def random_relink(self,new_node,new_ind,k=2.0):
        for i,old_node in enumerate(self.node_list):
            if old_node is new_node: continue
            dis = self.line_cost(old_node,new_node)
            if dis < k*self.expand_dis:
                if old_node.cost > new_node.cost+dis and self.check_segment_collision(old_node.x,old_node.y,new_node.x,new_node.y):
                    #更新old_node及其所有根节点
                    old_node.cost = new_node.cost+dis
                    try:
                        self.node_list[old_node.parent].kids.remove(i)
                    except ValueError:
                        print('?')
                    if show_info:
                        print(f'relink for {i},old parent:{old_node.parent},new parent:{new_ind}')
                    old_node.parent = new_ind
                    new_node.kids.append(i)
                    # visited = {}
                    queue = []+old_node.kids
                    while len(queue) != 0:
                        # if queue[0] not in visited:
                        #     visited[queue[0]] = 1
                        # else:
                        #     print('???')
                        current = self.node_list[queue[0]]
                        current_parent = self.node_list[current.parent]
                        queue.pop(0)
                        current.cost = current_parent.cost+self.line_cost(current,current_parent)
                        queue += current.kids
                            
                            
    def rrt_planning(self, start, goal, animation=True):
        self.min_path_len = float('inf')
        path_list = []
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        self.near_list = []
        path = None
        chg_flag = False
        self.connect = False
        for i in range(self.max_iter):
            if show_info:
                print(i)
            # 1. 在环境中随机采样点
            rnd = self.sample()

            # 2. 找到结点树中距离采样点最近的结点
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]

            # 3. 在采样点的方向生长一个步长，得到下一个树的结点。
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)

            # 4. 检测碰撞，检测到新生成的结点的路径是否会与障碍物碰撞
            no_collision = self.check_segment_collision(new_node.x, new_node.y, nearest_node.x, nearest_node.y)
            if no_collision:
                
                self.node_list.append(new_node)
                self.node_list[new_node.parent].kids.append(len(self.node_list)-1)
                if animation:
                    time.sleep(freq)
                    self.draw_graph(new_node)
                chg_flag = self.rewrite(new_node)
                self.random_relink(new_node,len(self.node_list)-1)
                # self.chk()
                # 一步一绘制
                if animation and chg_flag:
                    time.sleep(freq)
                    self.draw_graph(new_node)

                # 判断新结点是否临近目标点
                if self.is_near_goal(new_node):
                    if self.check_segment_collision(new_node.x, new_node.y,
                                                    self.goal.x, self.goal.y):
                        self.connect = True
                        self.near_list.append((new_node,None))
                
                for i in range(len(self.near_list)):
                    near_node,cost = self.near_list[i]
                    if cost is None or cost < self.min_path_len:
                        path = self.get_final_course(near_node)        # 回溯路径
                        path_length = self.get_path_len(path)           # 计算路径的长度
                        print("当前的路径长度为：{}".format(path_length))
                        self.near_list[i] = (near_node,path_length)
                        self.min_path_len = path_length if path_length < self.min_path_len else self.min_path_len
                        if animation:
                            self.draw_graph(new_node, path)
                        path_list.append((path,path_length))
        
        path,path_length = min(path_list,key=lambda x:x[1])
        print("最短路径长度为：{}".format(path_length))
        if animation:
            self.draw_graph(new_node, path)
        return path

    def sample(self):
        """ 在环境中采样点的函数，以一定的概率采样目标点 """
        if not self.connect:
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd = [random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand)]
            else:
                rnd = [self.goal.x, self.goal.y]
        else:
            # RRT,RRT*
            # rnd = [random.uniform(self.min_rand, self.max_rand),
            #         random.uniform(self.min_rand, self.max_rand)]
            # informed RRT*
            rnd = self.sample_informedRRT()
        return rnd

    def sample_informedRRT(self):
        theta = random.uniform(-np.pi,np.pi)
        k = random.uniform(0,1)
        base_theta = np.arctan2(self.goal.y,self.goal.x)
        mid_x,mid_y=self.goal.x/2,self.goal.y/2
        c = math.sqrt(mid_x**2+mid_y**2)
        a = self.min_path_len/2
        assert(c<a)
        b = math.sqrt(a**2-c**2)
        l = k*math.sqrt((a*np.cos(theta))**2+(b*np.sin(theta)**2))
        return [mid_x+l*np.cos(base_theta+theta),mid_y+l*np.sin(base_theta+theta)]


    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        """ 计算树中距离采样点距离最近的结点 """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2
                  for node in nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        """ 计算新结点 """
        new_node = copy.deepcopy(nearest_node)

        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)

        new_node.cost += self.expand_dis
        new_node.parent = n_ind

        new_node.kids = []

        return new_node

    def check_segment_collision(self, x1, y1, x2, y2):
        """ 检测碰撞 """
        for (ox, oy, radius) in self.obstacle_list:
            dd = self.line_cost_squared_point_to_segment(
                np.array([x1, y1]),
                np.array([x2, y2]),
                np.array([ox, oy])
            )
            if dd <= radius ** 2:
                return False
        return True

    @staticmethod
    def line_cost_squared_point_to_segment(v, w, p):
        """ 计算线段 vw 和 点 p 之间的最短距离"""
        if np.array_equal(v, w):    # 点 v 和 点 w 重合的情况
            return (p - v).dot(p - v)

        l2 = (w - v).dot(w - v)     # 线段 vw 长度的平方
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)
        return (p - projection).dot(p - projection)

    def draw_graph(self, rnd=None, path=None):

        plt.clf()

        # 绘制新的结点
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, '^k')

        # 绘制路径
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y],
                             '-g')

        # 绘制起点、终点
        plt.plot(self.start.x, self.start.y, "og")
        plt.plot(self.goal.x, self.goal.y, "or")

        # 绘制障碍物
        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        # 绘制路径
        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b')

        # 绘制图的设置
        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def get_final_course(self, last_node):
        """ 回溯路径 """
        path = [[self.goal.x, self.goal.y],[last_node.x,last_node.y]]
        last_index = last_node.parent
        while last_index is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        # path.append([self.start.x, self.start.y])
        return path

    @staticmethod
    def get_path_len(path):
        """ 计算路径的长度 """
        path_length = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
        return path_length


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.kids = []


def main():

    print('Start RRT planning!')
    show_animation = True
    start = [0, 0]
    goal = [15, 12]
    # 障碍物 (x, y, radius)
    obstacle_list = [
        (3, 3, 1.5),
        (12, 2, 3),
        (3, 9, 2),
        (9, 11, 2)
    ]

    rrt = RRT(rand_area=[-2, 18], obstacle_list=obstacle_list, max_iter=200)
    path = rrt.rrt_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    print('Done!')
    if show_animation and path:
        plt.show()


if __name__ == '__main__':
    main()
