import random
import sys
import rospy
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../")

from common.src.script.common import *
# from environment.src.ugv_forward_continuous.script.env import env


class env_obstacle(rl_base):
    def __init__(self):
        """

		"""
        super(env_obstacle).__init__()
        '''physical parameters'''
        self.miss = rospy.get_param(param_name='/miss')
        self.rBody = rospy.get_param(param_name='/rBody')
        self.r = rospy.get_param(param_name='/r')
        self.x_size = rospy.get_param(param_name='/x_size')
        self.y_size = rospy.get_param(param_name='/y_size')
        self.wMax = rospy.get_param(param_name='/wMax')

        # self.laserDis = rospy.get_param(param_name='/laserDis')
        # self.laserBlind = rospy.get_param(param_name='/laserBlind')

        self.ex = 0.
        self.ey = 0.
        self.x = 0
        self.y = 0
        self.phi = 0.
        self.dx = 0.
        self.dy = 0.
        self.dphi = 0.
        self.staticGain = 4
        self.start = [self.x, self.y]
        self.terminal = [self.x, self.y]
        self.time = 0
        self.wLeft = 0
        self.wRight = 0
        self.delta_phi_absolute = 0.
        self.timeMax = 15.0
        self.randomInitFLag = 0
        self.dTheta = 0.
        self.ddTheta = 0.
        self.intdTheta = 0.
        '''physical parameters'''

        '''rl_base'''
        self.state_dim = 8  # [ex/sizeX, ey/sizeY, x/sizeX, y/sizeY, phi, dx, dy, dphi]

        self.action_dim = 2
        self.initial_action = [0.0, 0.0]
        self.current_action = self.initial_action.copy()

        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞障碍物
        '''rl_base'''

        self.start = [0, 0]
        self.terminal = [0, 0]
        self.obs = []  # 默认10个障碍物

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        if (self.x + 1.8 * self.rBody > self.x_size) or (self.x - 1.8 * self.rBody < 0) or (self.y + 1.8 * self.rBody > self.y_size) or (self.y - 1.8 * self.rBody < 0):
            return True
        return False

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        if self.collision_check():
            # print('...collision...')
            self.terminal_flag = 4
            return True
        # if self.delta_phi_absolute > 6 * math.pi + deg2rad(0) and dis_two_points([self.x, self.y], [self.initX, self.initY]) <= 1.0:
        # if self.delta_phi_absolute > 6 * math.pi + deg2rad(0):
        #     print('...转的角度太大了...')
        #     self.terminal_flag = 1
        #     return True
        if dis_two_points([self.x, self.y], self.terminal) <= self.miss:
            print('...success...')
            self.terminal_flag = 3
            return True
        if self.is_out():
            # print('...out...')
            self.terminal_flag = 5
            return True
        return False

    def towards_target_PID(self, threshold: float, kp: float, ki: float, kd: float):
        action = [0, 0]
        if dis_two_points([self.x, self.y], self.terminal) <= threshold:
            temp = self.dTheta  # 上一step的dTheta
            self.dTheta = cal_vector_rad([self.terminal[0] - self.x, self.terminal[1] - self.y], [math.cos(self.phi), math.sin(self.phi)])
            if cross_product([self.terminal[0] - self.x, self.terminal[1] - self.y], [math.cos(self.phi), math.sin(self.phi)]) > 0:
                self.dTheta = -self.dTheta
            self.ddTheta = self.dTheta - temp
            self.intdTheta += self.dTheta
            w = kp * self.dTheta + kd * self.ddTheta + ki * self.intdTheta
            w = min(max(w, -self.wMax), self.wMax)  # 角速度
            action = [self.wMax - w, self.wMax] if w > 0 else [self.wMax, self.wMax + w]
        else:
            pass
        return action

    def action2_ROS_so(self, action):
        """
        :brief:             将env的动作输出转换为ROS驱动插件的输入
        :param action:
        :return:
        """
        vx = self.r / 2 * (action[0] + action[1])
        wz = self.r / self.rBody * (action[1] - action[0])
        return vx, wz

    def set_start(self, start):
        self.start = start

    def set_terminal(self, terminal):
        self.terminal = terminal

    def set_obs_random(self):
        safety_dis = 0.7
        safety_dis_ST = 0.3
        r = 0.4
        self.obs = []  # 清空
        for i in range(10):
            flag = False
            while not flag:
                flag = True
                center = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]
                '''检测newObs与起点和终点的距离'''
                if (self.start is not None) and (self.start != []) and (self.terminal is not None) and (self.terminal != []):
                    if (dis_two_points(self.start, center) < r + safety_dis_ST) or (dis_two_points(self.terminal, center) < r + safety_dis_ST):
                        flag = False
                        continue
                '''检测newObs与起点和终点的距离'''

                '''检测障碍物与其他障碍物的距离'''
                if len(self.obs) > 0:
                    for _obs in self.obs:
                        if dis_two_points(center, _obs) < r + r + safety_dis:
                            flag = False
                            break
                '''检测障碍物与其他障碍物的距离'''
            self.obs.append(center.copy())

    def collision_check(self):
        # 假设所有的障碍物都是圆
        for _obs in self.obs:
            if dis_two_points([self.x, self.y], _obs) < 0.4 + self.rBody:
                return True
        return False
