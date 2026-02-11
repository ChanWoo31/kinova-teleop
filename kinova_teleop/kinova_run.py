import time
import sys
import os
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

class KinovaRun(Node):
    def __init__(self):
        super().__init__('Kinova_Run')

        self.kinova_run = self.create_subscription(Float32MultiArray, 'dynamixel_pose', self.subscribe_topic, 10)

        class ConnectionArgs:
            def __init__(self):
                self.ip = "192.168.1.10"
                self.username = "admin"
                self.password = "admin"
                self.session_inactivity_timeout_ms = 60000
                self.connection_inactivity_timeout_ms = 2000

        args = ConnectionArgs()

        try:
            self.device_connection = utilities.DeviceConnection.createTcpConnection(args)
            self.router = self.device_connection.__enter__()
            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)

            print("연결 성공")

        except Exception as e:
            print(f" 연결 실패: {e}")
            raise e

        self.x_d = np.eye(4)
        
        self.TIMEOUT_DURATION = 20
        self.x_rotation = np.eye(3)

        self.timer = self.create_timer(0.01, self.control_loop)
    
    def subscribe_topic(self, msg):
        pose = np.array(msg.data).reshape(4, 4)

        self.x_d = pose
        
    def control_loop(self):

        if self.x_d is None:
            self.move_to_home_position(self.base)

        try:
            rot_matrix = self.x_d[:3, :3]
            rotation = R.from_matrix(rot_matrix)
            euler_angles = rotation.as_euler('zyx', degrees=True)

            feedback = self.base_cyclic.RefreshFeedback()
            x_l_c = feedback.base.tool_pose_x
            y_l_c = feedback.base.tool_pose_y
            z_l_c = feedback.base.tool_pose_z

            x_a_c = feedback.base.tool_pose_theta_x
            y_a_c = feedback.base.tool_pose_theta_y
            z_a_c = feedback.base.tool_pose_theta_z

            x_l_d = self.x_d[0, 3]
            y_l_d = self.x_d[1, 3]
            z_l_d = self.x_d[2, 3]

            x_a_d = euler_angles[2]
            y_a_d = euler_angles[1]
            z_a_d = euler_angles[0]

            kp = 1
            vel_x_l = (x_l_d - x_l_c) * kp
            vel_y_l = (y_l_d - y_l_c) * kp
            vel_z_l = (z_l_d - z_l_c) * kp
            
            vel_x_a = (x_a_d - x_a_c) * kp
            vel_y_a = (y_a_d - y_a_c) * kp
            vel_z_a = (z_a_d - z_a_c) * kp

            command = Base_pb2.TwistCommand()
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
            command.duration = 0

            twist = command.twist
            twist.linear_x = vel_x_l
            twist.linear_y = vel_y_l
            twist.linear_z = vel_z_l
            twist.angular_x = 0
            twist.angular_y = 0
            twist.angular_z = 0

            self.SendTwistCommand(command)
        
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
        action_list = base.ReadAllAction(action_type)
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

def main():
    rclpy.init()
    node=KinovaRun()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()