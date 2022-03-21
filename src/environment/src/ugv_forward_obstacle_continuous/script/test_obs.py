#!/usr/bin/env python3

import os
import sys
import datetime
import pandas as pd
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
import random
import tf
from tf.transformations import quaternion_from_euler        # 欧拉角转四元数
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../")

from common.src.script.common import *
from environment.src.ugv_forward_obstacle_continuous.script.env_obstacle import env_obstacle

'''some pre-defined parameters'''
robot_state = None
global_ugv_state = [0.0 for _ in range(8)]
global_laser_state = [0.0 for _ in range(37)]
cmd = Twist()
env = env_obstacle()
'''some pre-defined parameters'''


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


if __name__ == '__main__':
    rospy.init_node(name='test_obs', anonymous=False)
    rospy.Subscriber('/gazebo/model_states', ModelStates, robot_state_call_back)
    rospy.Subscriber('/scan', LaserScan, robot_laser_call_back)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
    rate = rospy.Rate(100)

    try:
        rospy.sleep(1.0)
        while not rospy.is_shutdown():
            print('...start reset...')
            env.is_terminal = False
            env.start = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]

            env.terminal = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]
            while dis_two_points(env.start, env.terminal) <= env.miss:
                env.terminal = [random.uniform(1.0, 9.0), random.uniform(1.0, 9.0)]

            # env.start = [8, 1]
            # env.terminal = [8, 8]

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
            set_model('terminal', [env.terminal[0], env.terminal[1], 0.01], Quaternion(x=0, y=0, z=0, w=1))
            set_obs_in_gazebo()
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
                print('laser:  ', global_laser_state)
                action = env.towards_target_PID(threshold=100, kp=10, kd=0, ki=0)
                vx, wz = env.action2_ROS_so(action)

                # print('x: ', env.x, '  y: ', env.y, '  phi:', env.phi * 180 / math.pi)
                '''将状态赋值'''
                env.is_terminal = env.is_Terminal()
                '''publish the velocity command'''
                cmd.linear.x = vx
                cmd.linear.y = 0
                cmd.linear.z = 0
                cmd.angular.x = 0
                cmd.angular.y = 0
                cmd.angular.z = wz
                pub.publish(cmd)
                '''publish the velocity command'''
                rate.sleep()
            rate.sleep()
    except:
        print('exit...')
    finally:
        print('保存数据')
