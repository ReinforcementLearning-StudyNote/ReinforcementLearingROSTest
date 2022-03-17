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
import random
import tf
import copy

PROJECT_PATH = "/home/yefeng/yefengGithub/ReinforcementLearningROS/src/"
sys.path.insert(0, PROJECT_PATH)
from algorithm.src.actor_critic.src.script.DDPG import DDPG
from common.src.script.common import *

cfgPath = './src/environment/src/twoD/src/config/'
cfgFile = 'UGV_Forward_Continuous.xml'
show_per = 1
robot_state = None
global_ugv_state = [0 for _ in range(8)]

def rpy2quad(r: float, p: float, y: float) -> Pose:
	cy = math.cos(y * 0.5)
	sy = math.sin(y * 0.5)
	cp = math.cos(p * 0.5)
	sp = math.sin(p * 0.5)
	cr = math.cos(r * 0.5)
	sr = math.sin(r * 0.5)
	q = Pose()
	q.orientation.w = cy * cp * cr + sy * sp * sr
	q.orientation.x = cy * cp * sr - sy * sp * cr
	q.orientation.y = sy * cp * sr + cy * sp * cr
	q.orientation.z = sy * cp * cr - cy * sp * sr
	return q


def robot_state_call_back(data: ModelStates):
	# print(global_model_states)
	position = data.pose[1].position
	orientation = data.pose[1].orientation
	w = orientation.w
	x = orientation.x
	y = orientation.y
	z = orientation.z
	yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

	global_ugv_state[0] = (terminal[0] - position.x) / 10 * 4
	global_ugv_state[1] = (terminal[1] - position.y) / 10 * 4
	global_ugv_state[2] = position.x / 10 * 4
	global_ugv_state[3] = position.y / 10 * 4
	global_ugv_state[4] = yaw

	twist = data.twist[1]
	global_ugv_state[5] = twist.linear.x
	global_ugv_state[6] = twist.linear.y
	global_ugv_state[7] = twist.angular.z


def set_model(name:str, position, orientation: Quaternion):
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


cmd = Twist()
done = False

if __name__ == '__main__':
	rospy.init_node(name='DDPG_UGV_Forward_3D', anonymous=False)
	rospy.Subscriber('/gazebo/model_states', ModelStates, robot_state_call_back)
	pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)
	rospy.wait_for_service('/gazebo/set_model_state')
	set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
	objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
	rate = rospy.Rate(100)

	try:
		start = [5, 5, 0]
		terminal = [random.uniform(0.5, 9.5), random.uniform(0.5, 9.5)]
		phi0 = cal_vector_rad([terminal[0] - start[0], terminal[1] - start[1]], [1, 0])
		phi0 = phi0 if start[1] <= terminal[1] else -phi0
		# phi0 = random.uniform(phi0 - deg2rad(60), phi0 + deg2rad(60))
		temp = rpy2quad(0, 0, phi0).orientation
		set_model('UGV', start, temp)
		set_model('terminal', [terminal[0], terminal[1], 0.25], Quaternion(x=0, y=0, z=0, w=1))
		while not rospy.is_shutdown():
			done = False
			while not done:
				cmd.linear.x = 1.0
				cmd.linear.y = 0
				cmd.linear.z = 0
				cmd.angular.x = 0
				cmd.angular.y = 0
				cmd.angular.z = 0
				pub.publish(cmd)
				rate.sleep()
			rate.sleep()
	except:
		print('exit...')
	finally:
		print('保存数据')
