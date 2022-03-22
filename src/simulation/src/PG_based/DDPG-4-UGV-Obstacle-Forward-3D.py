#!/usr/bin/env python3

import os

import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler  # 欧拉角转四元数

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from common.src.script.common import *
from environment.src.ugv_forward_obstacle_continuous.script.env_obstacle import env_obstacle
from algorithm.src.actor_critic.src.script.DDPG import DDPG

'''some pre-defined parameters'''
robot_state = None
global_ugv_state = [0.0 for _ in range(8)]
global_laser_state = [0.0 for _ in range(37)]
global_time = 0.00
cmd = Twist()
env = env_obstacle()
'''some pre-defined parameters'''


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        # self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(_action))
        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()

        self.action_value.reset_parameters()
        self.q.reset_parameters()

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

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
    def __init__(self, alpha, state_dim1, state_dim2, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.linear11 = nn.Linear(self.state_dim1, 128)  # 第一部分网络第一层
        self.batch_norm11 = nn.LayerNorm(128)
        self.linear12 = nn.Linear(128, 64)  # 第一部分网络第二层
        self.batch_norm12 = nn.LayerNorm(64)
        self.linear13 = nn.Linear(64, 64)
        self.batch_norm13 = nn.LayerNorm(64)  # 第一部分网络第三层

        self.linear21 = nn.Linear(self.state_dim2, 128)  # 第二部分网络第一层
        self.batch_norm21 = nn.LayerNorm(128)
        self.linear22 = nn.Linear(128, 64)  # 第二部分网络第二层
        self.batch_norm22 = nn.LayerNorm(64)
        self.linear23 = nn.Linear(64, 32)
        self.batch_norm23 = nn.LayerNorm(32)  # 第二部分网络第三层

        self.mu = nn.Linear(64 + 32, self.action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization_default(self):
        self.linear11.reset_parameters()
        self.batch_norm11.reset_parameters()
        self.linear12.reset_parameters()
        self.batch_norm12.reset_parameters()
        self.linear13.reset_parameters()
        self.batch_norm13.reset_parameters()

        self.linear21.reset_parameters()
        self.batch_norm21.reset_parameters()
        self.linear22.reset_parameters()
        self.batch_norm22.reset_parameters()
        self.linear23.reset_parameters()
        self.batch_norm23.reset_parameters()

        # self.combine.reset_parameters()
        self.mu.reset_parameters()

    def forward(self, state):
        """
        :param state:
        :return:            output of the net
        """
        if state.dim() == 1:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=0)
        else:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=1)
        x1 = self.linear11(split_state[0])
        x1 = self.batch_norm11(x1)
        x1 = func.relu(x1)

        x1 = self.linear12(x1)
        x1 = self.batch_norm12(x1)
        x1 = func.relu(x1)

        x1 = self.linear13(x1)
        x1 = self.batch_norm13(x1)
        x1 = func.relu(x1)  # 该合并了

        x2 = self.linear21(split_state[1])
        x2 = self.batch_norm21(x2)
        x2 = func.relu(x2)

        x2 = self.linear22(x2)
        x2 = self.batch_norm22(x2)
        x2 = func.relu(x2)

        x2 = self.linear23(x2)
        x2 = self.batch_norm23(x2)
        x2 = func.relu(x2)  # 该合并了

        x = torch.cat((x1, x2)) if x1.dim() == 1 else torch.cat((x1, x2), dim=1)
        # print(x1.size(), x2.size(), x.size())
        # x = self.combine(x)
        # x = func.relu(x)

        x = torch.tanh(self.mu(x))
        return x

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


def robot_state_call_back(data: ModelStates):
    # print(global_model_states)
    """
    :param data:    callback parameter
    :return:        None
    :brief:         get the message of the vehicle
    """
    '''
    0 - ground plane
    1 - 11X11-EmptyWorld
    2 - terminal
    3-12: obs
    13 - UGV
    '''
    if len(data.pose) != 14:
        pass
    else:
        position = data.pose[13].position
        orientation = data.pose[13].orientation
        w = orientation.w
        x = orientation.x
        y = orientation.y
        z = orientation.z
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        # global_ugv_state[0] = (env.terminal[0] - position.x) / env.x_size * env.staticGain
        # global_ugv_state[1] = (env.terminal[1] - position.y) / env.y_size * env.staticGain
        global_ugv_state[2] = position.x / env.x_size * env.staticGain
        global_ugv_state[3] = position.y / env.y_size * env.staticGain
        global_ugv_state[4] = yaw

        twist = data.twist[13]
        global_ugv_state[5] = twist.linear.x
        global_ugv_state[6] = twist.linear.y
        global_ugv_state[7] = twist.angular.z


def robot_laser_call_back(data: LaserScan):
    temp = data.ranges
    for i in range(37):
        global_laser_state[i] = min(temp[i], 2.0)


def set_model(name: str, position, orientation: Quaternion):
    objstate.model_state.model_name = name
    objstate.model_state.pose.position.x = position[0]
    objstate.model_state.pose.position.y = position[1]
    objstate.model_state.pose.position.z = position[2]
    objstate.model_state.pose.orientation.w = orientation.w
    objstate.model_state.pose.orientation.x = orientation.x
    objstate.model_state.pose.orientation.y = orientation.y
    objstate.model_state.pose.orientation.z = orientation.z
    objstate.model_state.twist.linear.x = 0.0
    objstate.model_state.twist.linear.y = 0.0
    objstate.model_state.twist.linear.z = 0.0
    objstate.model_state.twist.angular.x = 0.0
    objstate.model_state.twist.angular.y = 0.0
    objstate.model_state.twist.angular.z = 0.0
    objstate.model_state.reference_frame = "world"
    result = set_state_service(objstate)


def set_obs_in_gazebo():
    for i in range(10):
        name = 'obs' + str(i)
        objstate.model_state.model_name = name
        objstate.model_state.pose.position.x = env.obs[i][0]
        objstate.model_state.pose.position.y = env.obs[i][1]
        objstate.model_state.pose.position.z = 0.4
        objstate.model_state.pose.orientation.w = 1
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = 0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        objstate.model_state.reference_frame = "world"
        result = set_state_service(objstate)


def set_vel_UGV(vx: float, wz: float):
    objstate.model_state.model_name = 'UGV'
    objstate.model_state.twist.linear.x = vx
    objstate.model_state.twist.linear.y = 0.0
    objstate.model_state.twist.linear.z = 0.0
    objstate.model_state.twist.angular.x = 0.0
    objstate.model_state.twist.angular.y = 0.0
    objstate.model_state.twist.angular.z = wz
    objstate.model_state.reference_frame = "world"
    result = set_state_service(objstate)


if __name__ == '__main__':
    rospy.init_node(name='ddpg_ugv_forward_obs', anonymous=False)

    rospy.Subscriber('/gazebo/model_states', ModelStates, robot_state_call_back)  # 回调机器人状态
    rospy.Subscriber('/scan', LaserScan, robot_laser_call_back)  # 回调雷达数据

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
    rate = rospy.Rate(100)
    '''加载DDPG'''
    cfgPath = '/home/yefeng/yefengGithub/ReinforcementLearingROSTest/src/simulation/src/datasave/network/DDPG-UGV-Forward/'
    cfgFile = 'UGV_Forward_Continuous.xml'
    optPath = '/home/yefeng/yefengGithub/ReinforcementLearingROSTest/src/simulation/src/datasave/network/DDPG-UGV-Obstacle-Avoidance/parameters/'
    agent = DDPG(modelFileXML=cfgPath + cfgFile)

    '''重新加载actor和critic网络结构，这是必须的操作'''
    agent.actor = ActorNetwork(1e-4, 8, 45 - 8, 2, 'Actor', '')
    agent.load_actor_optimal(path=optPath, file='Actor_ddpg')
    '''加载DDPG'''

    try:
        rospy.sleep(1.0)
        while not rospy.is_shutdown():
            print('...start reset...')
            env.is_terminal = False
            env.start = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]

            env.terminal = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]
            while dis_two_points(env.start, env.terminal) <= env.miss:
                env.terminal = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]

            phi0 = cal_vector_rad([env.terminal[0] - env.start[0], env.terminal[1] - env.start[1]], [1, 0])
            phi0 = phi0 if env.start[1] <= env.terminal[1] else -phi0
            phi0 = random.uniform(phi0 - deg2rad(60), phi0 + deg2rad(60))
            q = quaternion_from_euler(0, 0, phi0)
            quaternion = Quaternion()
            quaternion.x = q[0]
            quaternion.y = q[1]
            quaternion.z = q[2]
            quaternion.w = q[3]
            env.set_obs_random()
            set_model('UGV', [env.start[0], env.start[1], 0], quaternion)
            set_model('terminal', [env.terminal[0], env.terminal[1], 0.25], Quaternion(x=0, y=0, z=0, w=1))
            set_obs_in_gazebo()
            startTime = rospy.get_rostime().to_sec()
            # print('startTime:  ', startTime)
            print('...finish reset...')
            while not env.is_terminal:
                '''将状态赋值'''
                env.x = global_ugv_state[2] * env.x_size / env.staticGain
                env.y = global_ugv_state[3] * env.y_size / env.staticGain
                env.ex = env.terminal[0] - env.x
                env.ey = env.terminal[1] - env.y
                env.phi = global_ugv_state[4]
                env.dx = global_ugv_state[5]
                env.dy = global_ugv_state[6]
                env.dphi = global_ugv_state[7]
                global_ugv_state[0] = env.ex / env.x_size * env.staticGain
                global_ugv_state[1] = env.ey / env.y_size * env.staticGain
                if dis_two_points([env.x, env.y], env.terminal) > 1.0:
                    action_from_actor = agent.choose_action(global_ugv_state + global_laser_state, True)
                    action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                else:
                    action = env.towards_target_PID(threshold=100, kp=10, kd=0, ki=0)
                vx, wz = env.action2_ROS_so(action)

                '''将状态赋值'''
                env.is_terminal = env.is_Terminal()
                '''publish the velocity command'''
                # vx = 0
                # wz = 0
                cmd.linear.x = vx
                cmd.linear.y = 0
                cmd.linear.z = 0
                cmd.angular.x = 0
                cmd.angular.y = 0
                cmd.angular.z = wz
                pub.publish(cmd)
                '''publish the velocity command'''

                time = rospy.get_rostime().to_sec()
                if time - startTime > 30:
                    print('time out')
                    env.is_terminal = True
                rate.sleep()
            rate.sleep()
    except:
        print('exit...')
    finally:
        print('保存数据')
