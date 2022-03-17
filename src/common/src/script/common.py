#!/usr/bin/env python3

import math
import random
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch
import xml.etree.ElementTree as elementTree
# import cv2 as cv


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / math.pi


def str2list(string: str) -> list:
    """
    :brief:         transfer a string to list，必须是具备特定格式的
    :param string:  string
    :return:        the list
    """
    res = re.split(r'[\[\]]', string.strip())
    inner = []
    outer = []
    for item in res:
        item.strip()
    while '' in res:
        res.remove('')
    while ', ' in res:
        res.remove(', ')
    while ',' in res:
        res.remove(',')
    while ' ' in res:
        res.remove(' ')
    for _res in res:
        _res_spilt = re.split(r',', _res)
        for item in _res_spilt:
            inner.append(float(item))
        outer.append(inner.copy())
        inner.clear()
    return outer


def sind(theta: float) -> float:
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.sin(theta / 180.0 * math.pi)


def cosd(theta: float) -> float:
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.cos(theta / 180.0 * math.pi)


def points_rotate(pts: list, theta: float) -> list:
    """
    :param pts:
    :param theta:   rad, counter-clockwise
    :return:        new position
    """
    if type(pts[0]) == list:
        return [[math.cos(theta) * pt[0] - math.sin(theta) * pt[1], math.sin(theta) * pt[0] + math.cos(theta) * pt[1]] for pt in pts]
    else:
        return [math.cos(theta) * pts[0] - math.sin(theta) * pts[1], math.sin(theta) * pts[0] + math.cos(theta) * pts[1]]


def points_move(pts: list, dis: list) -> list:
    if type(pts[0]) == list:
        return [[pt[0] + dis[0], pt[1] + dis[1]] for pt in pts]
    else:
        return [pts[0] + dis[0], pts[1] + dis[1]]


def cal_vector_rad(v1: list, v2: list) -> float:
    """
    :brief:         calculate the rad between two vectors
    :param v1:      vector1
    :param v2:      vector2
    :return:        the rad
    """
    # print(v1, v2)
    if np.linalg.norm(v2) < 1e-4 or np.linalg.norm(v1) < 1e-4:
        return 0
    cosTheta = min(max(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1), 1)
    return math.acos(cosTheta)


def cross_product(vec1: list, vec2: list) -> float:
    """
    :brief:         cross product of two vectors
    :param vec1:    vector1
    :param vec2:    vector2
    :return:        cross product
    """
    return vec1[0] * vec2[1] - vec2[0] * vec1[1]


def dis_two_points(point1: list, point2: list) -> float:
    """
    :brief:         euclidean distance between two points
    :param point1:  point1
    :param point2:  point2
    :return:        euclidean distance
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_is_in_circle(center: list, r: float, point: list) -> bool:
    """
    :brief:         if a point is in a circle
    :param center:  center of the circle
    :param r:       radius of the circle
    :param point:   point
    :return:        if the point is in the circle
    """
    sub = [center[i] - point[i] for i in [0, 1]]
    return np.linalg.norm(sub) <= r


def point_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point: list) -> bool:
    """
    :brief:                     判断点是否在椭圆内部
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point:               待测点
    :return:                    bool
    """
    sub = np.array([point[i] - center[i] for i in [0, 1]])
    trans = np.array([[cosd(-rotate_angle), -sind(-rotate_angle)], [sind(-rotate_angle), cosd(-rotate_angle)]])
    [x, y] = list(np.dot(trans, sub))
    return (x / long) ** 2 + (y / short) ** 2 <= 1


def point_is_in_poly(center, r, points: list, point: list) -> bool:
    """
    :brief:                     if a point is in a polygon
    :param center:              center of the circumcircle of the polygon
    :param r:                   radius of the circumcircle of the polygon
    :param points:              points of the polygon
    :param point:               the point to be tested
    :return:                    if the point is in the polygon
    """
    if center and r:
        if point_is_in_circle(center, r, point) is False:
            return False
    '''若在多边形对应的外接圆内，再进行下一步判断'''
    l_pts = len(points)
    res = False
    j = l_pts - 1
    for i in range(l_pts):
        if ((points[i][1] > point[1]) != (points[j][1] > point[1])) and \
                (point[0] < (points[j][0] - points[i][0]) * (point[1] - points[i][1]) / (
                        points[j][1] - points[i][1]) + points[i][0]):
            res = not res
        j = i
    if res is True:
        return True


def line_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point1: list,
                       point2: list) -> bool:
    """
    :brief:                     判断线段与椭圆是否有交点
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point1:              待测点1
    :param point2:              待测点2
    :return:                    bool
    """
    if point_is_in_ellipse(long, short, rotate_angle, center, point1):
        return True
    if point_is_in_ellipse(long, short, rotate_angle, center, point2):
        return True
    pt1 = [point1[i] - center[i] for i in [0, 1]]
    pt2 = [point2[j] - center[j] for j in [0, 1]]  # 平移至原点

    pptt1 = [pt1[0] * cosd(-rotate_angle) - pt1[1] * sind(-rotate_angle),
             pt1[0] * sind(-rotate_angle) + pt1[1] * cosd(-rotate_angle)]
    pptt2 = [pt2[0] * cosd(-rotate_angle) - pt2[1] * sind(-rotate_angle),
             pt2[0] * sind(-rotate_angle) + pt2[1] * cosd(-rotate_angle)]

    if pptt1[0] == pptt2[0]:
        if short ** 2 * (1 - pptt1[0] ** 2 / long ** 2) < 0:
            return False
        else:
            y_cross = math.sqrt(short ** 2 * (1 - pptt1[0] ** 2 / long ** 2))
            if max(pptt1[1], pptt2[1]) >= y_cross >= -y_cross >= min(pptt1[1], pptt2[1]):
                return True
            else:
                return False
    else:
        k = (pptt2[1] - pptt1[1]) / (pptt2[0] - pptt1[0])
        b = pptt1[1] - k * pptt1[0]
        ddelta = (long * short) ** 2 * (short ** 2 + long ** 2 * k ** 2 - b ** 2)
        if ddelta < 0:
            return False
        else:
            x_medium = -(k * b * long ** 2) / (short ** 2 + long ** 2 * k ** 2)
            if max(pptt1[0], pptt2[0]) >= x_medium >= min(pptt1[0], pptt2[0]):
                return True
            else:
                return False


def line_is_in_circle(center: list, r: float, point1: list, point2: list) -> bool:
    """
    :brief:             if a circle and a line segment have an intersection
    :param center:      center of the circle
    :param r:           radius of the circle
    :param point1:      point1 of the line segment
    :param point2:      point2 of t he line segment
    :return:            if the circle and the line segment have an intersection
    """
    return line_is_in_ellipse(r, r, 0, center, point1, point2)


def line_is_in_poly(center: list, r: float, points: list, point1: list, point2: list) -> bool:
    """
    :brief:             if a polygon and a line segment have an intersection
    :param center:      center of the circumcircle of the polygon
    :param r:           radius of the circumcircle of the polygon
    :param points:      points of the polygon
    :param point1:      the first point of the line segment
    :param point2:      the second point of the line segment
    :return:            if the polygon and the line segment have an intersection
    """
    if point_is_in_poly(center, r, points, point1):
        # print('Something wrong happened...')
        return True
    if point_is_in_poly(center, r, points, point2):
        # print('Something wrong happened...')
        return True
    length = len(points)
    for i in range(length):
        a = points[i % length]
        b = points[(i + 1) % length]
        c = point1.copy()
        d = point2.copy()
        '''通过坐标变换将a点变到原点'''
        b = [b[i] - a[i] for i in [0, 1]]
        c = [c[i] - a[i] for i in [0, 1]]
        d = [d[i] - a[i] for i in [0, 1]]
        a = [a[i] - a[i] for i in [0, 1]]
        '''通过坐标变换将a点变到原点'''

        '''通过坐标旋转将b点变到与X重合'''
        l_ab = dis_two_points(a, b)  # length of ab
        cos = b[0] / l_ab
        sin = b[1] / l_ab
        bb = [cos * b[0] + sin * b[1], -sin * b[0] + cos * b[1]]
        cc = [cos * c[0] + sin * c[1], -sin * c[0] + cos * c[1]]
        dd = [cos * d[0] + sin * d[1], -sin * d[0] + cos * d[1]]
        '''通过坐标旋转将b点变到与X重合'''

        if cc[1] * dd[1] > 0:
            '''如果变换后的cd纵坐标在x轴的同侧'''
            # return False
            continue
        else:
            '''如果变换后的cd纵坐标在x轴的异侧(包括X轴)'''
            if cc[0] == dd[0]:
                '''k == inf'''
                if min(bb) <= cc[0] <= max(bb):
                    return True
                else:
                    continue
            else:
                '''k != inf'''
                k_cd = (dd[1] - cc[1]) / (dd[0] - cc[0])
                b_cd = cc[1] - k_cd * cc[0]
                if k_cd != 0:
                    x_cross = -b_cd / k_cd
                    if min(bb) <= x_cross <= max(bb):
                        return True
                    else:
                        continue
                else:
                    '''k_cd == 0'''
                    if (min(bb) <= cc[0] <= max(bb)) or (min(bb) <= dd[0] <= max(bb)):
                        return True
                    else:
                        continue
    return False


class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim, action_dim):
        print(state_dim, action_dim)
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.end_mem = np.zeros(self.mem_size, dtype=np.float)
        self.sorted_index = []
        self.resort_count = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.s_mem[index] = state
        self.a_mem[index] = action
        self.r_mem[index] = reward
        self._s_mem[index] = state_
        self.end_mem[index] = 1 - done
        self.mem_counter += 1

    def get_reward_sort(self):
        """
        :return:        根据奖励大小得到所有数据的索引值，升序，即从小到大
        """
        print('...sorting...')
        self.sorted_index = sorted(range(min(self.mem_counter, self.mem_size)), key=lambda k: self.r_mem[k], reverse=False)

    def store_transition_per_episode(self, states, actions, rewards, states_, dones):
        self.resort_count += 1
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i], dones[i])

    def sample_buffer(self, is_reward_ascent=True):
        max_mem = min(self.mem_counter, self.mem_size)
        if is_reward_ascent:
            batchNum = min(int(0.25 * max_mem), self.batch_size)
            batch = random.sample(self.sorted_index[-int(0.25 * max_mem):], batchNum)
        else:
            batch = np.random.choice(max_mem, self.batch_size)
        states = self.s_mem[batch]
        actions = self.a_mem[batch]
        rewards = self.r_mem[batch]
        actions_ = self._s_mem[batch]
        terminals = self.end_mem[batch]

        return states, actions, rewards, actions_, terminals


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class GaussianNoise(object):
    def __init__(self, mu):
        self.mu = mu
        self.sigma = 1 / 3

    def __call__(self, sigma=1 / 3):
        return np.random.normal(self.mu, sigma, self.mu.shape)


class CriticNetWork(nn.Module):
    def __init__(self, beta=1e-3, state_dim=1, action_dim=1, name='CriticNetWork', chkpt_dir=''):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

    def forward(self, state, action):
        state_action_value = func.relu(torch.add(state, action))
        return state_action_value

    def initialization(self):
        pass

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha=1e-4, state_dim=1, action_dim=1, name='ActorNetwork', chkpt_dir=''):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

    def initialization(self):
        pass

    def forward(self, state):
        # x = torch.tanh(state)  # bound the output to [-1, 1]
        # return x
        pass

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class xml_cfg:
    def __init__(self):
        pass

    def XML_Create(self, filename: str, rootname: str, rootmsg: dict, is_pretty: bool = False):
        """
        :brief:                 创建一个XML文档
        :param filename:        文件名
        :param rootname:        根节点名字
        :param rootmsg:         根节点信息
        :param is_pretty:       是否进行XML美化
        :return:                None
        """
        new_xml = elementTree.Element(rootname, attrib=rootmsg)  # 最外面的标签名
        et = elementTree.ElementTree(new_xml)
        et.write(filename,
                 encoding='utf-8',
                 xml_declaration=True)
        if is_pretty:
            self.XML_Pretty_All(filename)

    @staticmethod
    def XML_Load(filename: str) -> elementTree.Element:
        """
        :brief:             加载一个XML文件
        :param filename:    文件名
        :return:            根节点
        """
        xml_root = elementTree.parse(filename).getroot()
        return xml_root

    @staticmethod
    def XML_FindNode(nodename: str, root: elementTree.Element) -> elementTree.Element:
        """
        :brief:             寻找XML文件中的节点
        :param nodename:    节点名
        :param root:        根节点
        :return:            该节点
        """
        for child in root:
            # print(child.tag)
            if child.tag == nodename:
                return child
        print('No node named' + nodename + 'here...')

    @staticmethod
    def XML_GetTagValue(node: elementTree.Element) -> dict:
        """
        :brief:         得到某一个节点的标签
        :param node:    该节点
        :return:        标签信息
        """
        nodemsg = {}
        for item in node:
            nodemsg[item.tag] = item.text
        return nodemsg

    def XML_InsertNode(self, filename: str, nodename: str, nodemsg: dict, is_pretty: bool = False):
        """
        :brief:                 在XML文档中插入结点
        :param filename:        文件名
        :param nodename:        新的节点名
        :param nodemsg:         新的节点的信息
        :param is_pretty:       是否进行XML美化
        :return:                None
        """
        xml_root = elementTree.parse(filename).getroot()
        new_node = elementTree.SubElement(xml_root, nodename)
        for key, value in nodemsg.items():
            _node = elementTree.SubElement(new_node, key)
            _node.text = str(value)
        et = elementTree.ElementTree(xml_root)
        et.write(filename,
                 encoding='utf-8',
                 xml_declaration=True)
        if is_pretty:
            self.XML_Pretty_All(filename)

    def XML_InsertMsg2Node(self, filename: str, nodename: str, msg: dict, is_pretty: bool = False):
        """
        :brief:                 在某一个节点中插入信息
        :param filename:        文件名
        :param nodename:        节点名
        :param msg:             信息
        :param is_pretty:       是否进行XML美化
        :return:                None
        """
        xml = elementTree.parse(filename)
        xml_root = xml.getroot()
        for child in xml_root:
            if child.tag == nodename:
                for key, value in msg.items():
                    _node = elementTree.SubElement(child, key)
                    _node.text = str(value)
                # break
        et = elementTree.ElementTree(xml_root)
        et.write(filename,
                 encoding='utf-8',
                 xml_declaration=True)
        if is_pretty:
            self.XML_Pretty_All(filename)

    def XML_RemoveNode(self, filename: str, nodename: str, is_pretty=False):
        """
        :brief:                 从XML文档中移除某节点
        :param filename:        文件名
        :param nodename:        节点名
        :param is_pretty:       是否进行XML美化
        :return:                None
        """
        xml = elementTree.parse(filename)
        xml_root = xml.getroot()
        for child in xml_root:
            if child.tag == nodename:
                xml_root.remove(child)
                # break
        et = elementTree.ElementTree(xml_root)
        et.write(filename,
                 encoding='utf-8',
                 xml_declaration=True)
        if is_pretty:
            self.XML_Pretty_All(filename)

    def XML_RemoveNodeMsg(self, filename: str, nodename: str, msgname: str, is_pretty=False):
        """
        :brief:             从某一个节点中移除某些信息
        :param filename:    文件名
        :param nodename:    节点名
        :param msgname:     信息的名字
        :param is_pretty:   是否进行XML美化
        :return:            None
        """
        xml = elementTree.parse(filename)
        xml_root = xml.getroot()
        for child in xml_root:
            if child.tag == nodename:
                for msg in child:
                    if msg.tag == msgname:
                        child.remove(msg)
        et = elementTree.ElementTree(xml_root)
        et.write(filename,
                 encoding='utf-8',
                 xml_declaration=True)
        if is_pretty:
            self.XML_Pretty_All(filename)

    def XML_Pretty(self, element: elementTree.Element, indent: str = '\t', newline: str = '\n', level: int = 0):
        """
        :brief:             XML美化
        :param element:     节点元素
        :param indent:      缩进
        :param newline:     换行
        :param level:       缩进个数
        :return:            None
        """
        if element:
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):
                subelement.tail = newline + indent * (level + 1)
            else:
                subelement.tail = newline + indent * level
            self.XML_Pretty(subelement, indent, newline, level=level + 1)

    def XML_Pretty_All(self, filename: str):
        """
        :brief:             XML美化(整体美化)
        :param filename:    文件名
        :return:            None
        """
        tree = elementTree.parse(filename)
        root = tree.getroot()
        self.XML_Pretty(root)
        tree.write(filename)


class rl_base:
    def __init__(self):
        self.state_dim = 0
        """
        The dimension of the state, which must be a finite number.
        For example, the dimension of a 2D UAV (mass point) is [px py vx vy ax ay],
        the 'state_dim' should be 6, but the number of each dimension can be finite (discrete) of infinite (continuous).
        Another example, the dimension of an inverted pendulum is [x, dx, ddx, theta, dtheta, ddtheta],
        the 'state_dim' should be 6.
        """

        self.state_num = []
        """
        The number of each state.
        It is a two dimensional list which includes the number of the state of each dimension.
        For example, for the inverted pendulum model we mentioned before:
        the 'state_num' should be [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        Another example, if we set the maximum of x as 10, minimum of x as -10, and the step of x as 2 and keep the other states unchanged,
        then we have: state_num = [11, math.inf, math.inf, math.inf, math.inf, math.inf].
        """

        self.state_step = []
        """
        It is a two dimensional list which includes the step of each state.
        Set the corresponding dimension as None if the state of which is a continuous one.
        If we set the maximum of x as 10, minimum of x as -10, and the step of x as 2 in the inverted pendulum example, 
        the state_step should be [2]; other wise, it should be [None]
        """

        self.state_space = []
        """
        It is a two dimensional list which includes all dimensions of states.
        Set the corresponding dimension as None if the state of which is a continuous one.
        If we set the maximum of x as 10, minimum of x as -10, and the step of x as 5 in the inverted pendulum example, 
        the state_space should be [[-10, -5, 0, 5, -10]]; other wise, it should be [[None]]
        """

        self.isStateContinuous = []
        """
        Correspondingly,
        the 'isStateContinuous' of inverted pendulum model is [True, True, True, True, True, True], or we can just set it to [True] 
        However, if we discrete the 'x' in the inverted pendulum model, then isStateContinuous = [False, True, True, True, True, True], 
        and we cannot simply set it to [True] or [False].
        """
        '''
        Generally speaking, the continuity of different states is identical.
        Because it is meaningless to make the problem more complicated deliberately without any benefit.
        '''

        self.action_dim = 0
        """
        The dimension of the action, which must be a finite number.
        For example, the dimension of a 2D UAV (mass point) is [px py vx vy ax ay],
        the 'action_dim' should be 2, which are the jerks of X and Y ([jx jy]).
        Another example, the dimension of an inverted pendulum is [x, dx, ddx, theta, dtheta, ddtheta],
        the 'action_dim' should be 1, which is the acceleration or jerk of the base.
        """

        self.action_num = []
        """
        The number of each action.
        It is a two dimensional list which includes the number of the action of each dimension.
        For example, for the inverted pendulum model we mentioned before:
        the 'action_num' should be [np.inf] if the acceleration is continuous.
        Another example, if we set the maximum of acceleration as 2, minimum of acceleration as -2, and the step as 0.5,
        then we have: action_num = [9], which are: 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0.
        """

        self.action_step = []
        """
        It is a two dimensional list which includes the step of each action.
        Set the corresponding dimension as None if the action of which is a continuous one.
        If we set the step of the acceleration of the base in the inverted pendulum as 1, 
        then the action_step should be [1]; other wise, it should be [None]
        """

        self.action_space = []
        """
        It is a two dimensional list which includes all dimensions of action.
        Set the corresponding dimension as None if the action of which is a continuous one.
        If we set the step of the acceleration of the base in the inverted pendulum as 1, minimum value as -2, and maximum value as 2
        the action_space should be [[-2, -1, 0, 1, 2]]; other wise, it should be [[None]]
        """

        self.isActionContinuous = []
        """
        If a DQN-related algorithm is implemented, then 'isActionContinuous' should be False due to the requirement of the RL algorithm.
        If a DDPG-related algorithm is implemented, then 'isActionContinuous' should be True due to the requirement of the RL algorithm.
        The 'isActionContinuous' is defined as a list because the dimension of the action may be larger than one,
        but the attributes of each dimension should be identical unless we insist on doing something 'strange'. For example:
        for a two-wheel ground vehicle, and we insist on only discretizing the torque of the left wheel,
        although unreasonably, the 'isActionContinuous' should be [False, True].
        """

        self.state_range = []
        """
        Formation: [[min1, max1], [min2, max2], ..., [minX, maxX]]
        It is a two dimensional list which includes the minimum and the maximum of each state.
        For example, if the state of a magnet levitation ball system is [x, v]. Then the state_range should be:
        [[-5, 5], [-15, 15]], which means the maximum of the position of the ball is 5, the minimum of the position of the ball is -5,
        the maximum of the velocity of the ball is 15, the minimum of the velocity of the ball is -15.
        But if we don't have a limitation of the velocity, the 'state_range' should be [[-5, 5], [-math.inf, math.inf]]
        """

        self.action_range = []
        """
        Formation: [[min1, max1], [min2, max2], ..., [minX, maxX]]
        It is a two dimensional list which includes the minimum and the maximum of each action.
        For example, if the action of a 3D UAV system is [ax, ay, az]. Then the action_range should be:
        [[-5, 5], [-5, 5], [-2, 2]], which means the maximum of the acceleration of the UAV is 5, 5, and -2 in the direction of X, Y, Z,
        the minimum of the acceleration of the UAV -5, -5, -2, respectively.
        Generally speaking, the range should not be infinite although it is mathematical-reasonable.
        """

        self.initial_state = []
        self.initial_action = []
        self.current_state = []
        self.next_state = []
        self.current_action = []
        self.reward = 0.0
        self.is_terminal = False

    def state_normalization(self, state: list, gain: float = 1.0, index0: int = -1, index1: int = -1):
        """
        :brief:             default for [-gain, gain]
        :param state:       state
        :param gain:        gain
        :param index1:
        :param index0:
        :return:            normalized state
        """
        length = len(state)
        # assert length == self.state_dim
        # assert length >= index1 >= index0
        start = 0 if index0 <= 0 else index0
        end = length - 1 if index1 > length - 1 else index1
        while start <= end:
            bound = self.state_range[start]
            k = 2 / (bound[1] - bound[0])
            b = 1 - bound[1] * k
            state[start] = (k * state[start] + b) * gain
            start += 1

    def step_update(self, action):
        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def get_reward(self, param=None):
        """
        :param param:       other parameters
        :return:            reward function
        """
        '''should be the function of current state, time, or next state. It needs to be re-written in a specific environment.'''
        pass

    def is_Terminal(self, param=None):
        return False

    def reset(self):
        # self.current_state = self.initial_state.copy()
        # self.next_state = []
        # self.reward = 0.0
        # self.is_terminal = False
        pass

    def reset_random(self):
        # self.current_state = []
        # for i in range(self.state_dim):
        #     if self.isStateContinuous[i]:
        #         if self.state_range[i][0] == -math.inf or self.state_range[i][1] == math.inf:
        #             self.current_state.append(0.0)
        #         else:
        #             self.current_state.append(random.uniform(self.state_range[i][0], self.state_range[i][1]))
        #     else:
        #         '''如果状态离散'''
        #         self.current_state.append(random.choice(self.state_space[i]))
        #
        # self.next_state = []
        # self.reward = 0.0
        # self.is_terminal = False
        pass


class Color:
    def __init__(self):
        """
        :brief:     初始化
        """
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)

        self.Blue = (255, 0, 0)
        self.Green = (0, 255, 0)
        self.Red = (0, 0, 255)

        self.Yellow = (0, 255, 255)
        self.Cyan = (255, 255, 0)
        self.Magenta = (255, 0, 255)

        self.DarkSlateBlue = (139, 61, 72)
        self.LightPink = (193, 182, 255)
        self.Orange = (0, 165, 255)
        self.DarkMagenta = (139, 0, 139)
        self.Chocolate2 = (33, 118, 238)
        self.Thistle = (216, 191, 216)
        self.Purple = (240, 32, 160)
        self.DarkGray = (169, 169, 169)
        self.Gray = (128, 128, 128)
        self.DimGray = (105, 105, 105)
        self.DarkGreen = (0, 100, 0)
        self.LightGray = (199, 199, 199)

        self.n_color = 20

        self.color_container = [self.Black,             # 黑色
                                self.White,             # 白色
                                self.Blue,              # 蓝色
                                self.Green,             # 绿色
                                self.Red,               # 红色
                                self.Yellow,            # 黄色
                                self.Cyan,              # 青色
                                self.Magenta,           # 品红
                                self.DarkSlateBlue,     # 深石板蓝
                                self.LightPink,         # 浅粉
                                self.Orange,            # 橘黄
                                self.DarkMagenta,       # 深洋红色
                                self.Chocolate2,        # 巧克力
                                self.Thistle,           # 蓟
                                self.Purple,            # 紫色
                                self.DarkGray,          # 深灰
                                self.Gray,              # 灰色
                                self.DimGray,           # 丁雷
                                self.LightGray,         # 浅灰
                                self.DarkGreen          # 深绿
                                ]

    def get_color_by_item(self, _n: int):
        """
        :brief:         通过序号索引颜色
        :param _n:      序号
        :return:        返回的颜色
        """
        assert 0 <= _n < self.n_color   # 当 _n 不满足时触发断言
        return self.color_container[_n]

    def random_color(self):
        return self.color_container[random.randint(0, self.n_color - 1)]

    @staticmethod
    def random_color_by_BGR():
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
