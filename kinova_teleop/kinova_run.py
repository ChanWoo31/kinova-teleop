import time
import sys
import os
import threading

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray



def deg2rad(x):
    y = x * np.pi / 180
    return y

class KinovaRun(Node):
    def __init__(self):
        super().__init__('Kinova_Run')
        self.kinova_run = self.create_subscription(UInt32MultiArray, 'topic', self.subscribe_topic, 10)

        self.d_ur5e = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        self.a_ur5e = [0, -0.425, -0.3922, 0, 0, 0]
        self.alpha_ur5e = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

        self.theta_kinova = [0, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi, np.pi/2]
        self.d_kinova = [0.2433, 0.03, 0.02, 0.245, 0.057, 0.235]
        self.a_kinova = [0, 0.28, 0, 0, 0, 0]
        self.alpha_kinova = [np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, 0]

        self.dynamixel_angle = np.zeors(6)
        
        self.TIMEOUT_DURATION = 20
    
    def subscribe_topic(self, msg):
        q = msg.data
        dir = [1, 1, -1, 1, 1, 1]
        offset = [0, 0, 0, 0, 0, 180]

        for i in range(6):
            self.dynamixel_angle[i] = deg2rad((q[i] - 2048) * dir[i] * 360 / 4096 + offset[i])

        self.x_d = self.forward_kinematic(self.dynamixel_angle, self.d, self.a, self.alpha)

    def forward_kinematic(self, theta, d, a, alpha):
        T = np.eye(4)
        for i in range(6):
            T_i = self.Trans_mat(theta[i], d[i], a[i], alpha[i])
            T = T @ T_i
        return T
    
    def Trans_mat(self, theta, d, a, alpha):
        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
             [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
             [0, np.sin(alpha), np.cos(alpha), d],
             [0, 0, 0, 1]])
        return T
    
    def check_for_end_or_abort(self, e):
        def check(notification, e = e):
            print("EVENT : " + \
                  Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
            return check
        
    def move_to_home_position(self, base):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)

        print("Moving the arm to a safe positioin")
        action_type = Base_pb2.RequestedActionType()
        action_list = base.ReadAllAction(action_type)
        action_handle = None
        for action 

def main():
    rclpy.init()
    node=KinovaRun()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()