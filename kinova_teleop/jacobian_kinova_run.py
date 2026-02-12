import time
import sys
import os
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

class JacobianKinovaRun(Node):
    def __init__(self):
        super().__init__('Jacobian_Kinova_Run')

        self.kinova_run = self.create_subscription(Float32MultiArray, 'dynamixel_pose', self.subscribe_topic, 10)

        class ConnectionArgs:
            def __init__(self):
                self.ip = "192.168.1.10"
                self.username = "admin"
                self.password = "admin"
                self.session_inactivity_timeout_ms = 60000
                self.connection_inactivity_timeout_ms = 2000

        args = ConnectionArgs()

        self.TIMEOUT_DURATION = 20

        try:
            self.device_connection = utilities.DeviceConnection.createTcpConnection(args)
            self.router = self.device_connection.__enter__()
            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)

            print("연결 성공")

            self.move_to_home_position(self.base)

        except Exception as e:
            print(f" 연결 실패: {e}")
            raise e

        self.x_d = None
        
        # 키노바 dh파라미터
        self.theta_kinova = [0, np.pi/2, np.pi/2, np.pi/2, np.pi, np.pi/2]
        self.d_kinova = [0.2433, 0.03, 0.02, 0.245, 0.057, 0.235]
        self.a_kinova = [0, 0.28, 0, 0, 0, 0]
        self.alpha_kinova = [np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, 0]

        self.timer = self.create_timer(0.01, self.control_loop)
    
    def subscribe_topic(self, msg):
        pose = np.array(msg.data).reshape(4, 4)

        self.x_d = pose
        
    def control_loop(self):

        delta_x_1d = np.zeros(6)
        kp = 1
        ko = 1
        epsilon = 0.02
        damping_factor_max = 0.01

        if self.x_d is None:
            return

        try:
            rot_mat_x_d = self.x_d[:3, :3]
            rot_mat_d = R.from_matrix(rot_mat_x_d)
            
            feedback = self.base_cyclic.RefreshFeedback()
            x_l_c = feedback.base.tool_pose_x
            y_l_c = feedback.base.tool_pose_y
            z_l_c = feedback.base.tool_pose_z
            position_c = [x_l_c, y_l_c, z_l_c]

            x_a_c = feedback.base.tool_pose_theta_x
            y_a_c = feedback.base.tool_pose_theta_y
            z_a_c = feedback.base.tool_pose_theta_z

            euler_angles_c = [x_a_c, y_a_c, z_a_c]
            rotation_c = R.from_euler('xyz', euler_angles_c, degrees=True)
            # rot_mat_c = rotation_c.as_matrix()

            # rot_mat_c_inv = linalg.inv(rot_mat_c)

            rot_mat_err = rot_mat_d * rotation_c.inv()
            rot_e = rot_mat_err.as_rotvec()

            x_l_d = self.x_d[0, 3]
            y_l_d = self.x_d[1, 3]
            z_l_d = self.x_d[2, 3]
            position_d = [x_l_d, y_l_d, z_l_d]

            error_position = position_d - position_c
            delta_x_1d[:3] = np.transpose(kp * error_position)

            command = Base_pb2.TwistCommand()
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            command.duration = 0

            
        
        except Exception as e:
            print("error")

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
        action_list = base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle
        
        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False
        
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        base.ExecuteActionFromReference(action_handle)
        finished = e.wait(self.TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

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
    
    def get_jacobian(self, theta, d, a, alpha):
        T01 = self.Trans_mat(theta[0], d[0], a[0], alpha[0])
        T12 = self.Trans_mat(theta[1], d[1], a[1], alpha[1])
        T23 = self.Trans_mat(theta[2], d[2], a[2], alpha[2])
        T34 = self.Trans_mat(theta[3], d[3], a[3], alpha[3])
        T45 = self.Trans_mat(theta[4], d[4], a[4], alpha[4])
        T56 = self.Trans_mat(theta[5], d[5], a[5], alpha[5])
        T02 = T01 @ T12
        T03 = T02 @ T23
        T04 = T03 @ T34
        T05 = T04 @ T45
        T06 = T05 @ T56
        
        # 왜 외적하면 1차원으로 나오지
        Jv1 = np.cross(np.transpose(np.array([0, 0, 1])) , T06[:3, 3])
        Jomega1 = np.eye(3) @ np.transpose(np.array([0, 0, 1]))
        Jv2 = np.cross(T01[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T01[:3, 3])
        Jomega2 = T01[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv3 = np.cross(T02[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T02[:3, 3])
        Jomega3 = T02[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv4 = np.cross(T03[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T03[:3, 3])
        Jomega4 = T03[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv5 = np.cross(T04[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T04[:3, 3])
        Jomega5 = T04[:3, :3] @ np.transpose(np.array([0, 0, 1]))
        Jv6 = np.cross(T05[:3, :3] @ np.transpose(np.array([0, 0, 1])), T06[:3, 3] - T05[:3, 3])
        Jomega6 = T05[:3, :3] @ np.transpose(np.array([0, 0, 1]))

        Jacobian_matrix = np.array([
            [Jv1[0], Jv2[0], Jv3[0], Jv4[0], Jv5[0], Jv6[0]],
            [Jv1[1], Jv2[1], Jv3[1], Jv4[1], Jv5[1], Jv6[1]],
            [Jv1[2], Jv2[2], Jv3[2], Jv4[2], Jv5[2], Jv6[2]],
            [Jomega1[0], Jomega2[0], Jomega3[0], Jomega4[0], Jomega5[0], Jomega6[0]],
            [Jomega1[1], Jomega2[1], Jomega3[1], Jomega4[1], Jomega5[1], Jomega6[1]],
            [Jomega1[2], Jomega2[2], Jomega3[2], Jomega4[2], Jomega5[2], Jomega6[2]],
        ])
        
        return Jacobian_matrix


def main():
    rclpy.init()
    node=JacobianKinovaRun()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()