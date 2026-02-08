import time
import os
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32MultiArray

import mujoco
import mujoco.viewer

xml_path = 'kinova_description/scene.xml'

def deg2rad(x):
    y = x * np.pi / 180
    return y

class MujocoKinovaRun(Node):
    def __init__(self):
        super().__init__('Mujoco_Kinova_Run')
        self.mujoco_kinova_run = self.create_subscription(UInt32MultiArray, 'topic', self.subscribe_topic, 10)

        self.m_mujoco = mujoco.MjModel.from_xml_path(xml_path)
        self.d_mujoco = mujoco.MjData(self.m_mujoco)
        
        # x_d : 다이나믹셀(목표값), x_c : kinova
        self.dynamixel_angle = np.zeros(6)
        self.x_d = np.eye(4)
        self.x_c = np.eye(4)

    def subscribe_topic(self, msg):
        q = msg.data
        dir = [1, 1, -1, 1, 1, 1]
        offset = [0, -90, 0, -90, 0, 180]

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
        T = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return T
    
    def main():
        rclpy.init()
        node=MujocoKinovaRun()
        with mujoco.viewer.launch_passive(node.m_)