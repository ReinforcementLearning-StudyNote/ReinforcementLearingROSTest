import sys
import rospy

PROJECT_PATH = "/home/yefeng/yefengGithub/ReinforcementLearningROS/src/"
sys.path.insert(0, PROJECT_PATH)

from common.src.script.common import *


class env(rl_base):
    def __init__(self):
        """

		"""
        super(env).__init__()
        '''physical parameters'''
        self.miss = rospy.get_param(param_name='/miss')
        self.rBody = rospy.get_param(param_name='/rBody')
        self.x_size = rospy.get_param(param_name='/x_size')
        self.y_size = rospy.get_param(param_name='/y_size')
        self.wMax = rospy.get_param(param_name='/wMax')
        self.x = 0
        self.y = 0
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
        self.current_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi]

        self.action_dim = 2
        self.initial_action = [0.0, 0.0]
        self.current_action = self.initial_action.copy()

        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞障碍物
        '''rl_base'''

    def is_out(self):
        """
		:return:
		"""
        '''简化处理，只判断中心的大圆有没有出界就好'''
        if (self.x + self.rBody > self.x_size) or (self.x - self.rBody < 0) or (self.y + self.rBody > self.y_size) or (self.y - self.rBody < 0):
            return True
        return False

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        if self.time > self.timeMax:
            print('...time out...')
            self.terminal_flag = 2
            return True
        if dis_two_points([self.x, self.y], self.terminal) <= self.miss:
            print('...success...')
            self.terminal_flag = 3
            return True
        if self.is_out():
            # print('...out...')
            # self.terminal_flag = 1
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

    def reset(self):
        """

		"""
        '''physical parameters'''
        self.x = self.initX  # X
        self.y = self.initY  # Y
        self.phi = self.initPhi  # 车的转角
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        self.delta_phi_absolute = 0.
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''
