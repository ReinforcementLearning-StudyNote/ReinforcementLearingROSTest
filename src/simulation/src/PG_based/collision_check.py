#!/usr/bin/env python3

import rospy
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


class StateValidity:
    def __init__(self):
        # subscribe to joint joint states
        rospy.Subscriber("joint_states", JointState, self.jointStatesCB, queue_size=1)
        # prepare service for collision check
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is avaiable')
        # prepare msg to interface with moveit
        self.rs = RobotState()
        self.rs.joint_state.name = ['joint1', 'joint2']
        self.rs.joint_state.position = [0.0, 0.0]
        self.joint_states_received = False

    def checkCollision(self):
        '''
        check if robotis in collision
        '''
        if self.getStateValidity().valid:
            rospy.loginfo('robot not in collision, all ok!')
        else:
            rospy.logwarn('robot in collision')

    def jointStatesCB(self, msg):
        '''
        update robot state
        '''
        self.rs.joint_state.position = [msg.position[0], msg.position[1]]
        self.joint_states_received = True

    def getStateValidity(self, group_name='acrobat', constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result

    def start_collision_checker(self):
        while not self.joint_states_received:
            print('ggggg')
            rospy.sleep(0.01)
        rospy.loginfo('joint states received! continue')
        self.checkCollision()
        rospy.spin()


if __name__ == '__main__':
    print('start')
    rospy.init_node('collision_checker_node', anonymous=False)
    collision_checker_node = StateValidity()
    print('start')
    collision_checker_node.start_collision_checker()
